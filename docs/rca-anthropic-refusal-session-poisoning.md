# RCA: Anthropic refusal poisons a session with no in-session escape

**Status:** documented, not fixed — hardening tracked on the Kanban board (고도화)
**Severity:** P2 — one session becomes permanently unusable; the user has to abandon it and loses the working context (uncommitted plan, tool state, 50+ turns of history).
**Observed:** 2026-11-04 11:42–11:48 KST, TUI session `20261104_112940_1a4d06`, model `claude-fable-5-1` via `provider=anthropic` (Claude Code OAuth pool).

## Summary

A single tool result appended to the conversation caused Anthropic's API to answer every subsequent request with `stop_reason="refusal"` (Hermes: `finish_reason=content_filter`) in ~0.5 s, with no refusal text. Once that message was in the history the session was dead: `"뭐가 문제인거야"` and `"hi?"` were refused identically. Hermes surfaced each refusal correctly but persisted the session unchanged, so the only recovery was starting a new session.

Two separate problems:

1. **Poisoning** — an ordinary code/docstring tool result tripped an input-side classifier (false positive as far as we can tell).
2. **No escape hatch** — Hermes has no way to prune or quarantine the offending message from inside the session; the refusal path returns, persists the same history, and waits for the next prompt, which is refused again.

## Timeline (from `~/.hermes/logs/agent.log`, KST)

| time | event |
|---|---|
| 11:42:50 | API call #27 OK (`in=96862 out=920 latency=12.1s`). Model issued a `terminal` tool call that read `hermes_cli/kanban_db.py` ~5800–5905 (block-kind routing docstring + a `blocked/crashed/gave_up/timed_out` event-kind listing). |
| 11:42:51.000 | tool result appended: 4,326 chars (message id 65128). |
| 11:42:51.562 | next API call → **refusal in 0.56 s**, `refusal=(no text)`. |
| 11:47:59 | user: `뭐가 문제인거야` → refusal in 0.5 s. |
| 11:48:20 | user: `hi?` → `Repaired 1 message-alternation violations` then refusal in 0.6 s. |
| 11:48:32 | user switched model to `claude-opus-5-5` (same provider). No further API call logged for that session. |
| 11:48:38 | user opened a new TUI session → normal. |

## What is confirmed

- The refusals are API-level: `stop_reason="refusal"` with empty content, sub-second latency, and `in=` tokens essentially unchanged from the last good call. The model never started generating; the request was rejected on input.
- The last change to the context before the first refusal was tool result 65128. Nothing else was appended between call #27 (OK) and the first refused call.
- The refusal is deterministic for that history: three different user prompts (continuation, question, `hi?`) were all refused.
- Hermes handled each refusal as designed (`conversation_loop.py` `finish_reason == "content_filter"` branch): logged a WARNING, emitted a status line, called `_persist_session`, and returned `_content_policy_blocked_result`. No retry storm, no fallback misfire.
- The refusal notice shown to the user is **not** persisted as an assistant message. The stored history after the incident is `tool(65128) → user → user → user`, which is why the alternation repair fired on the third attempt.

## What is not confirmed

- **Which token(s) triggered the classifier.** Anthropic returns no reason. The tool result is a docstring about Kanban block kinds and a list of terminal event names; nothing in it is remotely policy-sensitive, so we treat this as a false positive, but that is inference.
- **Whether the opus switch would have escaped.** No API call was logged after the switch; the classifier is presumably shared across models on the same endpoint, so probably not — unverified.
- **Whether re-reading the same file region in a fresh session reproduces the refusal.** Deliberately not tested from an interactive session (it would poison that session too). Reproduction should be done from a throwaway `hermes chat -Q` session.

## Why there is no escape today

`agent/conversation_loop.py` (`content_filter` branch, ~L2463–2548):

1. Logs + emits a status, then `agent._persist_session(messages, conversation_history)` — the poisoned history is saved as-is.
2. Returns the refusal as the turn result. The next user prompt is appended to the same history and sent again.
3. The only recovery hint is `_CONTENT_POLICY_RECOVERY_HINT`: *"Try rephrasing the request, narrowing the context, or adding a fallback provider"*. Rephrasing cannot help when the trigger is a prior tool result; "narrowing the context" has no command behind it; a fallback provider would need its own credentials and still sends the same poisoned history.
4. There is no slash command to drop/quarantine the last N messages or the last tool result, and no automatic detection of "refused on input, repeatedly, with empty refusal text".

Context compaction does not help either: it summarises *old* turns; the poisoned message is the newest one.

## Workaround (current)

- Start a new session (`/new` in the TUI, or a new session in the dashboard). Expect to lose in-flight plan/tool state; re-brief the new session.
- If the task required reading the same file region, read it in smaller pieces or `grep` for the needed symbols instead of dumping the whole block.
- Do **not** retry the same prompt or switch models on the poisoned session; each attempt is another refused call against the quota.

## Hardening proposals (for the 고도화 card)

Ordered by value / effort:

1. **Input-refusal detector + quarantine.** In the `content_filter` branch, if the refusal text is empty *and* latency is below a threshold *and* the previous assistant turn was a tool call, mark the last tool result as the likely trigger. Replace its content with a short placeholder (`[tool result quarantined after provider refusal: <tool> <n> chars, sha256=…]`), persist, and retry the same request **once**. Log the original content's hash and size for forensics (never the content itself).
2. **Manual escape hatch.** `/drop [N]` (or `/quarantine-last-tool`) slash command in TUI + dashboard that removes the last N messages (default: through the last tool result) from the live history and state.db, so the user can recover a session without the automation above.
3. **Persist the refusal as an assistant message** (or a system note) so the stored history keeps user/assistant alternation and the user can see in-session what happened when they resume.
4. **Better hint text.** When the same session has been refused ≥2 times in a row, change the hint to: *"This session is being refused on its history, not your prompt. Use /drop to remove the last tool result or start a new session; retrying will be refused again."*
5. **Repro harness.** A `hermes chat -Q` script that replays a saved message list against the provider to confirm/deny whether a given tool result triggers the refusal — lets us file an accurate false-positive report upstream.

## Related

- `agent/conversation_loop.py` — `content_filter` branch, `_CONTENT_POLICY_RECOVERY_HINT`.
- `docs/known-warnings.md` — for warnings that are expected; this one is not.

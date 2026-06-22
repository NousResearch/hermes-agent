# PRD — `send_message` silent home-channel leak fix

**Status:** v0.3 (Pass-1 + Pass-2 review applied; pending Pass-3 verification of the Pass-2 blocker fix)
**Author:** Apollo
**Date:** 2026-06-21
**Area:** `tools/send_message_tool.py` (core model tool) + behavioral skill guidance
**Repo / track:** fork-internal PR against `Kyzcreig/hermes-agent` (fleet runs the fork), carried as a local patch on `main`. Upstream `NousResearch` PR optional follow-up (general-interest bug).

**Changelog:**
- **v0.3 (Pass 2)** — Pass 2 returned BLOCK: the Pass-1 R2 fix (call `approval._is_gateway_approval_context()`) was **too broad**. The TUI/desktop gateway sets `HERMES_GATEWAY_SESSION=1` but binds NO messaging platform (`tui_gateway/server.py:1328,1346` — `set_session_vars(session_key=…, cwd=…)`, platform defaults `""`); ACP is the same (`acp_adapter/server.py:1474`). So that helper would flag those surfaces as "interactive" → a legitimate bare→home send there would **hard-error** instead of delivering to home. **Fix:** the routing predicate is now a purpose-built `_has_messaging_origin()` = `(NOT HERMES_CRON_SESSION) AND bool(_get_session_platform())` — gate on an actually-bound **messaging platform**, not the gateway-approval context. This preserves home delivery on TUI/desktop/ACP/CLI/cron and only changes behavior when a real messaging origin exists. Also: corrected the §1 "fleet-wide" claim — subagent (`delegate_task`) bare sends are NOT covered (contextvars don't cross the executor boundary — `delegate_tool.py`); the original leak was a **main-session** send, so this is a documented limitation, not a re-scope. Added the TUI/desktop-surface e2e (the case that hid the blocker).
- **v0.2 (Pass 1)** — 1 BLOCKER + 4 required + 3 NTH, each verified vs code. B1: collapsed the 3-predicate flow into a single **tri-state resolver** so the home arm is structurally unreachable in-turn. R1: corrected the cron rationale (cron *clears* `HERMES_SESSION_*`, `cron/scheduler.py:1613-1617`). R2: [superseded by v0.3] call the existing helper instead of re-mirroring. R3: extended the fix to `react`/`unreact` (`send_message_tool.py:242-253`). R4: Phase 3 also rewrites the top-level schema description (`:141-142`). Added thread-fidelity AC, origin/home/error observability, Phase-0 arg recovery.

---

## 1. Summary & Goal

**The bug (observed, root-caused 2026-06-21).** During a long agent turn in Discord `#instacart` (session `20260619_142630_590cf2`), Apollo called the `send_message` tool to post a progress checkpoint. The call carried a **bare/empty target** (no chat id). `send_message`'s `_handle_send` then **silently fell back to the platform's home channel** — Discord `#hermes` (`1502228850338435153`) — and the checkpoint surfaced there instead of in `#instacart`, where the work and the user were.

**Evidence chain (exact, millisecond-aligned):**
- `agent.log` 04:34:53.512 PDT — `Tool send_message returned error: "Both 'target' and 'message' are required"` (first malformed attempt, session `20260619_142630_590cf2`).
- `agent.log` 04:35:06.431 PDT — `tool send_message completed (2.14s, 200 chars)` (retry; the "200 chars" is the tool *result note*, not the message).
- `state.db` — the leaked text is stored as an `assistant` message in session `20260614_194308_66e2a2fc` = the **Discord home-channel `#apollo`/`#hermes` session**, timestamped `2026-06-21 11:35:06.405879 UTC` = 04:35:06 PDT. Exact match.
- `tools/send_message_tool.py:379-440` — `if not chat_id: home = config.get_home_channel(platform)` → sends there, tags result `"Sent to discord home channel (chat_id: 1502228850338435153)"`.
- `.env` — `DISCORD_HOME_CHANNEL=1502228850338435153`, `TELEGRAM_HOME_CHANNEL=571820863` (explains "sometimes it lands in Telegram").

**Why it recurs and hits other agents (Daedalus too).** It is a **bug-class**, not an Apollo quirk: the `send_message` schema marks `target` as **not required** (`"required": []`) and its description says `'platform' (uses home channel)`. So a bare/blank/`"discord"` target is a *documented, valid* call that silently routes to the global home. Any agent posting a mid-turn update via `send_message` from a non-home channel can trip it.

**Not the gateway's fault.** The gateway's background/async delivery path (`_build_process_event_source`, `_inject_watch_notification`) is already hardened — it resolves origin from the persisted session store and **drops rather than misroutes** when metadata is missing (verified: `gateway.log:3522` shows a background process for the instacart session injecting to the *correct* channel `1515200957884137472`). The hole is purely the `send_message` **tool** defaulting an unspecified target to the global home during a live interactive turn.

**Goal.** Kill the bug-class fleet-wide with the smallest sound change:
1. **Tool-hardening (the real fix):** when a turn has a bound **messaging origin** (a real Discord/Telegram/etc. session — the case the original leak happened in), a `send_message` send OR a `react`/`unreact` with **no explicit target chat** must NOT silently route to the global home channel. Instead it resolves to the **current session's origin channel**. If the origin can't be resolved, it returns an **actionable error** rather than dumping to home. Both the message-send path (`_handle_send`, leak at `:379-389`) and the react path (`_handle_react`, identical leak at `:242-253`) are covered — they share one resolver, so the reported main-session leak class closes, not just the one site (AGENTS.md: "fix the whole bug class — sibling call paths included"). **Scope honesty (Pass 2):** non-messaging surfaces (cron, CLI, TUI, desktop, ACP) keep home-default delivery; and `delegate_task` **subagent** bare sends are NOT covered (contextvars don't cross the executor boundary, so a subagent sees no bound origin → falls to home = today's behavior, never worse) — a documented limitation, not the reported bug (which was a main-session send).
2. **Schema clarity:** tighten BOTH the top-level `send_message` description (`:141-142`, which currently says "send to telegram → home channel") and the `target` property description so "uses home channel" is clearly a *deliberate bare-platform* behavior for non-interactive contexts, and warn against using `send_message` to talk to the channel you're already in.
3. **Behavioral guardrail (skill-only, no code, no SOUL change):** agents should not use `send_message` to talk to the channel they're already in — just return text as the turn reply. Encoded in a fleet skill.

---

## 2. Non-Goals

- **No change to cron / standalone / CLI home-channel delivery.** A cron job or out-of-process caller that sends a bare-platform target SHALL still resolve to the home channel exactly as today. The fix is scoped to the *interactive gateway turn* context only.
- **No change to the gateway background/async notification routing** (already correct).
- **No SOUL.md edits** (explicit user constraint). Behavioral guidance is skill-only.
- **No new config keys, no new `HERMES_*` env vars.** The fix reuses the session-context ContextVars the gateway already binds.
- **No change to `action="list"`**, nor to explicit-target sends/reacts (`discord:#channel`, `telegram:chatid`, etc.) — those keep working byte-identically. (`react`/`unreact` with a BARE target ARE now in scope — see §1 / R3 — but explicit-target reacts are unchanged.)
- **Not** removing the home-channel fallback entirely — it's legitimate for deliberate bare-platform sends outside a messaging turn (cron, CLI, TUI, desktop, ACP).
- **`delegate_task` subagent bare sends are out of scope** (Pass 2). Contextvars don't cross the executor boundary that runs the child, so a subagent has no bound messaging origin → its bare send falls to home (today's behavior, never worse). The reported leak was a main-session send; propagating origin into subagents is a larger, separate change tracked as a future item, not this fix.

---

## 3. Constitution / Invariants

- **Invariant I1 — No silent cross-channel leak in a live turn.** When `send_message(action="send")` runs inside an interactive gateway turn (a real user-facing session, not cron/CLI) and the caller supplied **no explicit chat target**, the message MUST NOT be delivered to the global home channel when that differs from the turn's origin channel.
  - *Why it matters:* contract/data — leaks one conversation's content into another channel (or another platform/DM), which is a privacy + correctness failure.
  - *Closeout proof:* `pytest` test that simulates an interactive turn bound to origin `discord:CHAN_A` with home = `discord:CHAN_HOME` (A≠HOME), calls the resolver with a bare target, and asserts the resolved chat is `CHAN_A` (or an error) — never `CHAN_HOME`.

- **Invariant I2 — Cron/CLI home delivery preserved.** When NOT in an interactive gateway turn (cron, standalone, CLI), a bare-platform target SHALL resolve to the home channel exactly as before.
  - *Why it matters:* contract — cron digests, morning brief, alerts rely on bare-platform → home.
  - *Closeout proof:* `pytest` test with no interactive session context bound (and/or `HERMES_CRON_SESSION` set) asserting bare target still resolves to home.

- **Invariant I3 — Explicit targets unchanged.** Any call with an explicit chat (`platform:chat_id`, `platform:#name`, `platform:chat:thread`) resolves exactly as today.
  - *Closeout proof:* existing `send_message` tests still pass unmodified; an added test asserts an explicit cross-channel send still reaches the named channel.

- **Invariant I4 — Concurrency-safe origin resolution.** Origin is read from task-local ContextVars (`gateway.session_context` / `tools.approval._approval_session_key`), never from a process-global that another concurrent turn could clobber.
  - *Why it matters:* data — the gateway runs turns concurrently; a process-global read would reintroduce the exact cross-talk these ContextVars were built to fix.
  - *Closeout proof:* code review + a test that binds two different session contexts and confirms resolution is independent (or, minimally, that resolution reads the ContextVar API, asserted by import/call-path inspection).

- **Invariant I5 — No prompt-cache / alternation impact.** The change touches only tool-call execution; it does not alter the system prompt, toolset membership, or message role alternation.
  - *Closeout proof:* diff is confined to `tools/send_message_tool.py` (+ tests); no edits to prompt-builder, toolsets, or message-history code.

- **Invariant I6 — Schema honesty.** The `target` description must not invite the leaking call. It must state that, inside a live turn, omitting the target replies to the current channel (not the home channel), and that home-channel default applies to deliberate bare-platform sends.
  - *Closeout proof:* grep the schema string for the corrected wording; no occurrence of an unqualified "'platform' (uses home channel)" that implies home is the in-turn default.

---

## 4. Resolved Decisions

- **D-1 — Default to current-origin, not error-only, inside a live turn.** Of the two options discussed ("bare target = error" vs "bare target = current origin"), we do **both, layered**: inside a live interactive turn, a bare target resolves to the **current session origin channel** (the helpful, correct behavior — a mid-turn update lands where the conversation is). Only if the origin is genuinely unresolvable do we return an **error** (never fall to global home). This is strictly better than error-only: it makes the common "post an update to this chat" use work correctly instead of failing.
  - Rationale: the user's stated fix was "make a bare/empty target an error … OR default to current session origin." Origin-default is the superior arm; the error is the fail-closed floor.

- **D-2 — Scope the new behavior to interactive gateway turns only.** Detection reuses the existing `_is_gateway_approval_context()` / session-context signals. Cron (`HERMES_CRON_SESSION` / `HERMES_CRON_AUTO_DELIVER_*`) and CLI/standalone are explicitly excluded → they keep home-default (I2). This mirrors the existing cron-vs-gateway distinction already in `approval.py` and `_get_cron_auto_delivery_target()`.

- **D-3 — Reuse existing ContextVars; add no new state.** Origin platform/chat/thread come from `gateway.session_context.get_session_env("HERMES_SESSION_PLATFORM"|"HERMES_SESSION_CHAT_ID"|"HERMES_SESSION_THREAD_ID")`, with `tools.approval.get_current_session_key()` / session-key parse as the fallback resolution (the same parse `_build_process_event_source` uses). No new config, no new env var (respects the AGENTS.md "no new HERMES_* for non-secret config" rule).

- **D-4 — Behavioral guardrail is skill-only.** Per explicit user constraint: no SOUL edit. A fleet skill (`notify` or a focused new skill) gets a section: "never `send_message` to the channel you're already in — just reply; bare-target sends are for cron/explicit cross-channel only."

- **D-5 — Ship as fork-internal PR + local patch.** Fleet runs the diverged fork; this is a fleet bugfix. Fork PR for CI/review/history, cherry-pick onto local `main`. Upstream `NousResearch` PR is an optional later follow-up (the bug is general-interest), filed pristine-branch-first if done.

- **D-6 — "Bare platform" vs "no target at all" both covered.** Both `target=""`/absent AND `target="discord"` (platform only, no chat) are treated as "no explicit chat" for the in-turn origin-resolution path. (The malformed first call had no target; the retry is presumed to have used a bare/empty one. Both must route to origin, not home, in a live turn.)

---

## 5. Architecture / Design

### 5.1 Current flow (the bug)

`send_message_tool(args)` → `_handle_send(args)`:
1. Parse `target` → `platform_name`, `target_ref`.
2. If `target_ref` present → resolve `chat_id` (explicit or directory name).
3. **If `chat_id` still falsy → `home = config.get_home_channel(platform)`; `chat_id = home.chat_id`** ← the leak (lines ~379-389).
4. Send to `chat_id`; if home was used, tag result note.

There is **no awareness** of whether we're inside a live turn or what channel that turn originated in.

### 5.2 New flow — single tri-state resolver (B1)

The leak class is closed **by construction**: one resolver returns a three-state result, and the home-channel arm is **structurally unreachable** when we're in a live turn. No chain of independent predicates that could disagree (the Pass-1 BLOCKER was that a 3-predicate `elif` chain left an `else: home` arm reachable in-turn).

```python
# Returned by _resolve_send_target — exactly one state, no fall-through.
#   ORIGIN(chat_id, thread_id)  -> a messaging origin is bound; route to the turn's own channel
#   IN_TURN_UNRESOLVABLE        -> messaging origin bound but not resolvable for this platform -> ERROR (never home)
#   NOT_IN_TURN                 -> no messaging origin (cron / CLI / TUI / desktop / ACP) -> existing home behavior (I2)

def _resolve_send_target(platform_name) -> SendTarget:
    if not _has_messaging_origin():         # (NOT HERMES_CRON_SESSION) AND bool(_get_session_platform())
        return NOT_IN_TURN                  # cron/CLI/TUI/desktop/ACP -> caller does the existing home lookup
    origin = _interactive_origin(platform_name)   # reads HERMES_SESSION_* contextvars
    if origin is not None:
        return ORIGIN(origin.chat_id, origin.thread_id)
    return IN_TURN_UNRESOLVABLE            # messaging origin bound, no usable same-platform origin -> error, NOT home
```

Caller (`_handle_send`, and identically `_handle_react`), only entered when `chat_id` is still falsy after explicit-target parsing:

```python
if not chat_id:
    state = _resolve_send_target(platform_name)
    if state is NOT_IN_TURN:
        home = config.get_home_channel(platform)   # EXISTING behavior, unchanged (I2)
        ... (existing home-or-error code) ...
    elif state.kind is ORIGIN:
        chat_id, thread_id = state.chat_id, state.thread_id
        used_origin_channel = True                 # tag result note
    else:  # IN_TURN_UNRESOLVABLE
        return tool_error(
            "No target chat specified and the current channel could not be "
            "resolved for {platform}. To reply to the channel you're in, just "
            "return your message as the reply (don't call send_message). To send "
            "elsewhere, use '{platform}:CHANNEL'."
        )
```

Key property (I1, structural): when a **messaging origin** is bound the only two outcomes are **origin** or **error**. The `home = config.get_home_channel(...)` line is reachable ONLY on the `NOT_IN_TURN` branch. There is no path where a message turn with an unresolvable/blank target reaches `get_home_channel`.

**`_has_messaging_origin()` — the routing predicate (Pass-2 correction, supersedes R2).** Defined narrowly as `(not env_var_enabled("HERMES_CRON_SESSION")) and bool(approval._get_session_platform())`. We deliberately do NOT use `approval._is_gateway_approval_context()` here: that helper returns `True` for the TUI/desktop/ACP gateways (which set `HERMES_GATEWAY_SESSION=1` via process env — `tui_gateway/server.py:1346`, `acp_adapter/server.py`) **without binding a messaging platform** (`set_session_vars(session_key=…)` leaves platform `""`). Using it would convert a legitimate bare→home send on those surfaces into a hard error. The correct gate is "is a real **messaging** platform bound for this turn?" — which is exactly `_get_session_platform()` truthiness, minus the cron case. Cron exclusion still holds two ways (it sets `HERMES_CRON_SESSION` AND clears the platform — §5.4). `_get_session_platform()` is already exposed by `approval.py:123-130` and reads the same task-local ContextVar via `get_session_env`, so no new mechanism is introduced.

**`_interactive_origin(platform_name)`** (NEW small helper):
- Read `HERMES_SESSION_PLATFORM`, `HERMES_SESSION_CHAT_ID`, `HERMES_SESSION_THREAD_ID` via `gateway.session_context.get_session_env`.
- If the bound session platform == requested `platform_name` and a chat id is present → return `Origin(chat_id, thread_id)`.
- Fallback: if the chat-id env is absent but `approval.get_current_session_key()` parses to a `(platform, chat_id, thread_id)` matching the requested platform, use that (the same `_parse_session_key` parse `_build_process_event_source` uses, `gateway/run.py:12498`).
- Otherwise (a *different* platform is bound — e.g. turn on discord, `send_message("telegram")` with no chat) → return `None`. With a messaging origin bound that yields `IN_TURN_UNRESOLVABLE` → error (a deliberate cross-platform send must name a chat; silently hitting another platform's home is exactly the leak class — D-1/R2 in §8).

Result note: when `used_origin_channel`, tag `result["note"] = "Sent to current channel (chat_id: ...)"` (parallel to the existing home note at `:440`), so the agent sees where it went and learns the in-turn semantics.

### 5.3 Why ContextVars are the right substrate

`gateway/session_context.py` binds `HERMES_SESSION_*` as **task-local ContextVars** specifically because the gateway processes turns concurrently and a process-global env was leaking routing across concurrent messages (documented in that module's header). Reading them here is the same mechanism the (correct) background-delivery path already relies on. No new state, concurrency-safe by construction (I4).

### 5.4 Cron exclusion — cron CLEARS the session vars (R1, corrected)

A cron run does NOT bind a platform; it **explicitly clears** `HERMES_SESSION_*` to empty. `cron/scheduler.py:1613-1617` calls `set_session_vars(platform="", chat_id="", chat_name="")`, and the in-file comment (`:1595-1612`) names `send_message_tool.py` as a consumer it is deliberately protecting ("...the send_message gate read HERMES_SESSION_PLATFORM"). Cron sets `HERMES_CRON_SESSION="1"` (`:1586`) and routes its own delivery via `HERMES_CRON_AUTO_DELIVER_*` + `job["origin"]`, independent of `HERMES_SESSION_*`.

So the cron exclusion holds **two ways**, both already true: (1) `approval._is_gateway_approval_context()` returns `False` immediately on `HERMES_CRON_SESSION` (`approval.py:147-148`), and (2) even without that short-circuit, the cleared platform makes `_get_session_platform()` falsy. By calling `_is_gateway_approval_context()` (R2) we inherit this for free, and `_maybe_skip_cron_duplicate_send` (`:400`) still runs first on the cron path. (Phase-2 keeps a *defensive* test for "cron + a platform somehow bound" even though real cron never produces that state, to prove the exclusion survives a future change.)

---

## 6. Implementation Phases

### Phase 0 — Recover the exact leaking tool-call args (premise lock, OQ1)

**What ships:** before any code, confirm the root-cause mechanism. Query `state.db` (and any session archive) for the `send_message` tool-call in session `20260619_142630_590cf2` around 04:35:06 PDT; if the pre-compression args survive, record bare-vs-explicit. If gone, reproduce the leak from the schema-permitted bare-target path and record that as the justification.

- *Unit/script check:* `Not applicable: investigation step.`
- *E2E/integration check:* `Not applicable.`
- *Negative/adversarial:* if the recovered args reveal a *different* mechanism (e.g. an explicit wrong chat id), **STOP and re-scope** before Phase 1.
- *Verify with:* a one-paragraph finding appended to this PRD's status: "retry args were [bare|explicit|unrecoverable]; fix justification = [root-cause|schema-permitted-path]". The fix is valid either way (R4), but the *narrative* must match ground truth.

### Phase 1 — Single tri-state resolver + shared rewrite of BOTH home fallbacks (B1, R3)

**What ships:** `_resolve_send_target()` (tri-state) + `_interactive_origin()` + `_has_messaging_origin()` helpers in `tools/send_message_tool.py`. The `if not chat_id:` blocks in **both** `_handle_send` (`:379-389`) AND `_handle_react` (`:242-253`) rewritten to call the shared resolver: origin → route to current channel; in-turn-unresolvable → error; not-in-turn → existing home behavior.

- *Unit/script check:* `_resolve_send_target("discord")` returns `ORIGIN(CHAN_A, ...)` when `HERMES_SESSION_PLATFORM=discord`+`HERMES_SESSION_CHAT_ID=CHAN_A` are bound via the session-context API; returns `NOT_IN_TURN` when no messaging platform is bound (including the TUI/desktop case: `HERMES_GATEWAY_SESSION` set but platform `""`); returns `IN_TURN_UNRESOLVABLE` when a *different* messaging platform is bound and no origin parses.
- *E2E/integration check:* Drive `_handle_send({"target":"discord","message":"x"})` (bare platform) with an interactive discord session context bound to `CHAN_A` and home `CHAN_HOME` (≠A), `_send_to_platform` monkeypatched to capture the destination. Assert captured `chat_id == CHAN_A`, NOT `CHAN_HOME`. **This is the regression test reproducing the exact leak.** Repeat the same e2e for `_handle_react` (bare target react in-turn → reacts in `CHAN_A`, not `CHAN_HOME`) — the sibling path (R3).
- *Negative/adversarial:*
  (a) Live turn on discord, `send_message("telegram","x")` bare → **error** (does not silently send to telegram home).
  (b) Live turn, origin platform bound but chat id missing AND session key unparseable → **error**, not home.
- *Verify with:* `~/.hermes/hermes-agent/venv/bin/python -m pytest tests/tools/test_send_message_origin.py -o 'addopts=' -q` → all pass.

### Phase 2 — Preserve cron/CLI home delivery (I2)

**What ships:** the `NOT_IN_TURN` branch routes non-messaging-origin contexts (cron, CLI, TUI, desktop, ACP) to the existing home behavior, unchanged.

- *Unit/script check:* `_has_messaging_origin()` is `False` when no messaging platform is bound.
- *E2E/integration check:* `_handle_send({"target":"discord","message":"x"})` with NO messaging context (cron/CLI) → resolves to `CHAN_HOME`, `_send_to_platform` captures `CHAN_HOME`.
- *Negative/adversarial:*
  (a) **TUI/desktop surface (the case that hid the Pass-2 blocker):** `HERMES_GATEWAY_SESSION="1"` set, NO platform bound (mirrors `tui_gateway/server.py:1328,1346`), bare target → resolves to `CHAN_HOME`, NOT errored. This is the regression guard for the v0.3 blocker.
  (b) **Cron (defensive):** `HERMES_CRON_SESSION` truthy + a discord platform synthetically bound + bare target → home (NOT interactive, NOT errored). Real cron clears the platform (§5.4); this proves the `HERMES_CRON_SESSION` leg of `_has_messaging_origin` holds even if a future change leaves a platform bound.
- *Verify with:* same pytest module, cron/CLI/TUI cases → pass.

### Phase 3 — Schema clarity (I6) — BOTH descriptions (R4)

**What ships:** updated `SEND_MESSAGE_SCHEMA` — the **top-level `description`** (`:141-142`, currently "send to telegram → home channel") AND the `target` property description. New wording: inside a live turn, omitting the target replies to the **current channel** (and the right move for the current channel is usually to just reply, not call this tool); the home-channel default applies to deliberate bare-platform sends from **non-interactive** contexts (cron/CLI).

- *Unit/script check:* the top-level description and the `target` property description both contain the corrected wording and neither contains an unqualified "(uses home channel)" implying home is the in-turn default for a bare platform.
- *E2E/integration check:* `Not applicable: schema-string change, no runtime path.`
- *Negative/adversarial:* `Not applicable.`
- *Verify with:* `pytest ...::test_schema_descriptions_warn_against_self_channel` → pass; plus `grep`.

### Phase 4 — Behavioral guardrail (skill-only, no code, no SOUL)

**What ships:** a section added to the `notify` skill (the fleet's send-from-automation reference) stating: never use `send_message` to talk to the channel you're already in (just return text as the reply); bare-target/home-default sends are for cron + deliberate explicit cross-channel only; the in-turn semantic shift (bare target now = current channel) is called out explicitly so agents understand the new behavior.

- *Unit/script check:* `skill_view` shows the new section; `grep` finds the rule text.
- *E2E/integration check:* `Not applicable: documentation.`
- *Negative/adversarial:* `Not applicable.`
- *Verify with:* `skill_view(name='notify')` renders the section.

### Phase 5 — Ship: fork PR + local patch + tracker

**What ships:** branch off `origin/main`, commit Phases 1-3 (+ tests), push to fork, open fork-internal PR, get real gates green (ruff/ty/test matrix/e2e — `check-attribution` is cosmetic, ignore), squash-merge, cherry-pick onto local `main`, update the patch tracker doc (`~/Obsidian/Ace Place/AI/Hermes Agent/Hermes Agent — Patches & Upstream PRs.md`). (Phase 4 ships independently as a skill edit.)

- *Unit/script check:* `git log --oneline origin/main..HEAD` shows exactly the fix commit(s); `venv/bin/python -m pytest tests/tools/ -o 'addopts=' -q` green.
- *E2E/integration check:* after cherry-pick onto `main`, `grep` the distinctive patch comment in the installed `tools/send_message_tool.py`; confirm the editable install loads it.
- *Negative/adversarial:* `Not applicable` (covered by Phase 1-2 tests).
- *Verify with:* fork PR real gates green; `git log --oneline origin/main..main` shows the linear local patch.

---

## 7. Security, Privacy, Ops, Observability

- **Privacy is the whole point:** the bug is a cross-channel content leak (one chat's content into another channel, sometimes another platform / a DM). The fix closes it; the regression test (Phase 1 e2e) is the standing guard.
- **No new secrets, no new config, no public surface.** Reuses existing ContextVars.
- **Observability:** emit a single `logger.info` per resolution outcome that distinguishes **origin-routed** vs **home-routed** vs **in-turn-error**, including the session_key and resolved chat id, so the next "where did it go?" is one grep (mirrors/extends the existing "Sent to … home channel" note at `send_message_tool.py:440`).
- **Rollback:** revert the single `tools/send_message_tool.py` commit (drop the local-patch cherry-pick from `main`; close the fork PR). Skill edit reverts independently. Zero migration, zero state — pure code path.

---

## 8. Risks & Mitigations

- **R1 — Over-broad gating breaks cron/CLI home delivery.** *Mitigation:* I2 + Phase 2 tests explicitly cover cron (`HERMES_CRON_SESSION`) and no-context cases; reuse the existing, proven cron-vs-gateway distinction rather than inventing one.
- **R2 — Cross-platform bare send (turn on discord, `send_message("telegram")` no chat) now errors where it used to reach telegram home.** *Mitigation:* this is intentional and correct (a deliberate cross-platform broadcast should name a chat; silently hitting another platform's home is exactly the leak class). Documented in D-1/§5.2 and the schema (I6). If a real workflow needs "broadcast to telegram home from a discord turn," it names `telegram` explicitly AND we can treat an *explicit* bare-platform-different-from-origin as opt-in home — but v0.1 fails closed and we revisit only on a concrete need (Non-Goal otherwise).
- **R3 — ContextVar not bound on some live-turn path** (a platform adapter that doesn't set `HERMES_SESSION_*`). *Mitigation:* if a platform IS bound (interactive) but the chat-id env and session-key parse both fail, the resolver returns `IN_TURN_UNRESOLVABLE` → **error**, never home (fail-closed). If NO platform is bound at all (`_is_gateway_approval_context()` False), it's `NOT_IN_TURN` → existing home behavior — i.e. degrades to today's behavior, never worse. The mainline discord/telegram adapters DO bind these (verified: `session_context.py` var map + `gateway/run.py:15521`); Phase 1 includes a check.
- **R4 — False premise: maybe the retry used an explicit wrong target, not a bare one.** *Mitigation:* the fix handles BOTH the bare-target leak (primary, root-caused) and is harmless to explicit targets (I3). Even if the retry's args differ, the schema-permitted bare-target path is a real, reproducible leak (Phase 1 e2e reproduces it from first principles), so the fix is justified regardless. **Phase 0 pulls the exact tool-call args** from `state.db`/transcript if still present; if they reveal a *different* mechanism, re-scope before coding.
- **R5 — Shared-checkout hazard** (`~/.hermes/hermes-agent` is shared by both gateways + worktrees). *Mitigation:* build in a worktree, commit early on the feature branch, follow the `upstream-pr-and-local-patches` recovery recipe if a concurrent process stashes/switches.
- **R6 — React path behavior change** (R3 scope expansion): a bare-target `react`/`unreact` in a live turn now reacts on the *current* channel's latest message instead of the home channel's. *Mitigation:* this is the correct behavior (reacting in home from another channel's turn was the same leak class, lower severity); covered by the Phase-1 react e2e and AC1b. Explicit-target reacts are unchanged (I3).

---

## 9. Open Questions

- **OQ1 (build Phase-0): RESOLVED — args unrecoverable; fix justification = schema-permitted-path.** The exact `send_message` tool-call args from the 04:35:06 retry were pruned by the 05:59 hygiene compression (728→281 msgs) of session `20260619_142630_590cf2`; the qmd-corpus export is itself a `[Recent Summary]` and doesn't carry the raw call. However, `agent.log` shows the unambiguous two-step signature: `04:34:53 send_message error "Both 'target' and 'message' are required"` → `04:35:06 send_message completed (200 chars)`, and the leaked text landed in the home-channel session at `11:35:06.405879 UTC` to the millisecond. That is exactly the bare/empty-target → `get_home_channel` path (`send_message_tool.py:379-389`). The fix is valid either way (R4): a bare target is a schema-permitted (`"required": []`), reproducible leak. Premise: **confirmed mechanism = bare-target home-default**; the Phase-1 e2e reproduces it from first principles.
- **OQ2:** Should an *explicit* bare-platform target that differs from the turn origin (e.g. deliberate `send_message("telegram")` from a discord turn) be allowed to hit that platform's home (opt-in broadcast), or always require a named chat? v0.1 = require a named chat (fail closed). Revisit only on concrete need.

---

## 10. Acceptance Criteria

- [ ] **AC1 (I1):** WHEN `send_message(action="send")` runs in an interactive discord turn bound to `CHAN_A`, with home=`CHAN_HOME` (A≠HOME), and the target is bare/empty, THEN the message is delivered to `CHAN_A`, never `CHAN_HOME`. Evidence: `pytest tests/tools/test_send_message_origin.py::test_bare_target_in_turn_routes_to_origin` passes (monkeypatched `_send_to_platform` captured `CHAN_A`).
- [ ] **AC1b (I1/R3, react sibling):** WHEN `react`/`unreact` runs with a bare target in an interactive discord turn bound to `CHAN_A` (home `CHAN_HOME`≠A), THEN the reaction targets `CHAN_A`, never `CHAN_HOME`. Evidence: `...::test_bare_react_in_turn_routes_to_origin` passes.
- [ ] **AC2 (I2):** WHEN no messaging session context is bound (cron/CLI), THEN a bare-platform target still resolves to the home channel. Evidence: `...::test_bare_target_no_context_routes_to_home` passes.
- [ ] **AC2b (I2/Pass-2 blocker guard):** WHEN `HERMES_GATEWAY_SESSION="1"` is set but NO messaging platform is bound (the TUI/desktop/ACP surface), THEN a bare-platform target resolves to the home channel — NOT an error. Evidence: `...::test_bare_target_gateway_session_no_platform_routes_to_home` passes. (This is the standing regression guard for the v0.3 blocker — without the `_has_messaging_origin` narrowing this case would hard-error.)
- [ ] **AC3 (I2/cron, defensive):** WHEN `HERMES_CRON_SESSION` is set (even with a platform synthetically bound), THEN a bare target resolves to home (not errored, not treated as interactive). Evidence: `...::test_bare_target_cron_routes_to_home` passes. (Real cron clears the platform, §5.4; this proves the `HERMES_CRON_SESSION` short-circuit is the load-bearing guard.)
- [ ] **AC4 (I1 fail-closed):** WHEN in an interactive turn and the origin for the requested platform is unresolvable, THEN `send_message` returns an actionable error and delivers nothing to home. Evidence: `...::test_in_turn_unresolvable_origin_errors_not_home` passes.
- [ ] **AC5 (I3):** Explicit-target sends/reacts (`discord:CHAN_B`, `discord:#name`, `telegram:chatid`) reach the named target unchanged. Evidence: existing `send_message` tests pass + `...::test_explicit_target_unchanged` passes.
- [ ] **AC6 (I6):** BOTH the top-level `send_message` description AND the `target` property description warn against using `send_message` for the current channel and scope the home default to deliberate bare-platform/non-interactive sends. Evidence: `...::test_schema_descriptions_warn_against_self_channel` asserts the wording on both; `grep` confirms no unqualified "(uses home channel)" remains as the in-turn default.
- [ ] **AC7 (I4):** Origin resolution reads task-local session-context ContextVars (`get_session_env`), not a process-global. Evidence: code inspection + `...::test_origin_resolution_reads_session_context`.
- [ ] **AC7b (thread fidelity):** WHEN the originating turn carries a thread/topic id (Discord thread / Telegram topic), THEN a bare-target in-turn send preserves that thread id on the resolved origin. Evidence: `...::test_origin_preserves_thread_id` passes.
- [ ] **AC8 (Phase 4):** The behavioral guardrail exists in a fleet skill. Evidence: `skill_view` shows the "don't `send_message` your own channel" section.
- [ ] **AC9 (ship):** Fork PR real gates green; local-`main` cherry-pick present and loaded by the editable install. Evidence: `gh pr view` rollup (real gates) + `grep` of the patch comment in installed file.

---

## 11. Review Handoff

1. Run `prd-review-pipeline` — **2 Opus passes** (review+fix each), Opus-only, vary role not model. Lenses: security/privacy (leak boundary), architecture (ContextVar substrate + cron exclusion), testing (the reproduction e2e + negative cases), implementation-maintainability (smallest-diff, AGENTS.md footprint rules).
2. After APPROVE: hand approved phases to `prd-plan` if step-level TDD breakdown is wanted, else implement directly (small enough).
3. `prd-closeout` before declaring done.

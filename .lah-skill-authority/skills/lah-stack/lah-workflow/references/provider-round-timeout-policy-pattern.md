# Provider Round Timeout Policy Pattern (CLOE native loop)

Established 2026-08-07 on CLOE_POST_TOOL_PROVIDER_TIMEOUT_BUDGET_REPAIR_V1
(PR #715 → c0cd385). Use for any mission that tunes provider deadlines in
`lah-openclaw-mvp/src/services/chat-completions-service.js` or the native
`/chat/completions` conversational loop.

## Dual callProvider surface — the #1 trap

`buildNativeChatCompletions()` calls the provider in **TWO places**:

1. **Main round call** — after governance routing, budget checks, synthesis
   directive injection.
2. **Denial-driven synthesis re-request** — when EVERY requested tool call was
   denied (budget exhausted and/or mutation-only), the server re-requests the
   provider with `tool_choice: 'none'` to force a final synthesis. This second
   call reuses the same `effectiveTimeoutMs`.

A timeout-policy change that only touches the first call site is INCOMPLETE:
the second call is exactly the "post-tool / final synthesis round" that
legitimately needs the larger budget. Grep for ALL `callProvider(...)` call
sites in the file before concluding the fix is done.

## Differentiated policy that preserves every existing harness

Prefer a bounded per-round policy over a global constant bump:

```
standard rounds (greeting, direct answer, initial tool-decision):
  DEFAULT_PROVIDER_TIMEOUT_MS = 35000   (unchanged)
post-tool / final synthesis rounds:
  POST_TOOL_PROVIDER_TIMEOUT_MS = 70000 (larger, bounded)
```

**Round classification is reliable and stateless**: the agent loop replays the
full OpenAI history every round, so `role:'tool'` results OR prior
`assistant.tool_calls` prove tools ran in this conversation. A pure helper
`classifyProviderRoundType(messages)` returns `'standard' | 'post_tool_synthesis'`.
No session state needed.

**Hierarchy (never loosened)**:
1. `governance.maxProviderTimeoutMs` — explicit operator hard deadline
2. explicit `timeoutMs` argument — honored for every round
3. differentiated default (standard 35s / post-tool 70s)

The explicit-argument step is what keeps existing tests green: suites pass
`timeoutMs: 50` or `60` WITH tool messages in history and expect the tiny
deadline to abort. If the policy overrode an explicit `timeoutMs`, those
harnesses would silently get 70s. To make "explicit" distinguishable from
"default", change the signature default from `timeoutMs = 35000` to
`timeoutMs = undefined` and let `resolveEffectiveTimeoutMs()` apply the
differentiated default only when `timeoutMs` is undefined. Production route
passes neither → gets the differentiated policy; tests pass explicit values →
unchanged.

## planDeadlines() as canonical source

The pre-repair code computed `planDeadlines({ requestTimeoutMs })` but then
ignored it and derived `effectiveTimeoutMs` separately — the plan drifted from
the enforced deadline. Fix: derive the round base via
`resolveEffectiveTimeoutMs(...)`, then `const deadlines =
planDeadlines({ requestTimeoutMs: base })` and enforce
`effectiveTimeoutMs = deadlines.hard_deadline_ms` (which equals base, but the
plan is now the single source). Both call sites use this.

## Observability without secrets

Attach `_cloe.provider_round = { round_type, effective_timeout_ms,
provider_duration_ms, timed_out, post_tool_synthesis }` on ALL response paths
(success, timeout fallback, synthesis). Measure `Date.now()` around each
callProvider. Never log prompts, keys, or provider payloads.

## Consent-gate for "real provider" required tests

Missions often require a "direct real DeepSeek test" (paid provider call). If
the operator refuses/does not consent to the command that sources the live
env (e.g. `set -a && . <container-env> && node test/...`):

- Keep the harness in the repo (`test/<mission>-real-provider-proof.mjs`),
  ready to run on authorization — do NOT delete it.
- In the behavioral-simulation receipt, do NOT put a `SKIPPED` scenario in
  `scenarios[]` (validator rejects non-PASS/FAIL verdicts, exit 8) — document
  the blocked requirement in `residual_limitations` instead.
- Verdict for the mission: `_PARTIAL` (not CERTIFIED) when a REQUIRED test is
  blocked, with the blocked item named.
- Do NOT retry the refused command; do not rephrase it to slip past the gate.

## Temp secret file hygiene

Extracting the live provider env to `/tmp/<name>.txt` for a real test that
gets refused leaves a plaintext API key on disk. If `rm` also requires
consent and is refused:
- Immediately `chmod 600` the file (owner-only) as a hardening fallback.
- Surface the pending cleanup explicitly in the operator packet and final
  report as an operator action (path + chmod state), never silently.

## Container drift check before assuming where the fix lands

The running container's `GIT_COMMIT` may differ from BOTH the canonical
checkout HEAD and origin/main. Always check `docker exec <container> sh -c
'echo $GIT_COMMIT'` and test whether it is an ancestor of origin/main. The
deployed file may be NEWER than the local canonical checkout (canonical left
on an old branch) — diff against `git show origin/main:<path>` / a fresh
worktree at the merge SHA, not against the dirty local checkout.

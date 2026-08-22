# Live Certification Smoke Harness Pattern (real provider, A–H)

Established 2026-08-05 during
`CLOE_EXISTING_RETRIEVAL_GOVERNOR_AND_BUSINESS_CAPABILITY_GRAPH_RUNTIME_WIRING_REPAIR_V1`
(live certification phase after merge + exact-SHA deploy). Pattern for driving
the REAL production endpoint with the REAL provider and native tool lifecycle
to certify post-deploy behavior. Mock-only evidence can never certify
production — the mission's verdict rule "do not claim completion from
mock-provider tests alone" is enforced here.

## Harness shape

- Drive the deployed endpoint (e.g. `POST http://127.0.0.1:4000/chat/completions`)
  with admin auth (`x-admin-api-key` / `Bearer` = `ADMIN_API_KEY` from the
  canonical `.env`). Read the key from the file at runtime, never print it.
- Client tool surface = OpenClaw-like set:
  - exploration: `exec`, `grep`, `find`, `read`, `list_tools`
  - business reads: `campaign_memory_query`, `exoclick_stats`
  - **mutation tools**: `campaign_play`, `create_campaign`, `campaign_pause`
    — deliberately included so the planner's mutation-exclusion is provable.
- Tool EXECUTION is a SAFE read-only stub (never shell, never mutation).
  Mutation tools called by the provider → recorded as critical violations and
  BLOCKED with a marker string in the result.
- Capture the server's own governor metadata from the JSON response
  (`_cloe.governor`: `decision`, `budget`, `business_plan` with
  `request_class`/`evidence_missing_count`/`allowed_tool_count`/
  `removed_tool_count`/`mutation_tools_present`, `forced_synthesis`) —
  sanitized evidence, no secrets, no raw tokens.
- Round-loop: send → if `tool_calls`, execute stubs → reinject with the
  EXACT provider `tool_call_id`s → repeat until `finish_reason=stop` or budget
  / max rounds. Never invent tool_call_id values.
- Run the full A–H suite multiple times (3+ passes) — single-pass results hide
  intermittent provider behavior.

## Pitfall 1 — parallel tool_calls MUST be grouped in ONE assistant message

OpenAI/DeepSeek semantics: all parallel tool_calls of a round go in a single
assistant message (`tool_calls: [c1, c2, c3]`), then one `role:'tool'` result
per call. Pushing `assistant[tool]` / `tool` / `assistant[tool]` / `tool`
interleaved per call → provider HTTP 400 → server returns 500
`Provider HTTP 400`. Symptom: single-call probes pass, multi-parallel rounds
intermittently 500. Proven: grouped reinjection passed 8/8; interleaved failed
deterministically once the provider emitted 2+ parallel calls.

## Pitfall 2 — tool result `content` must always be a string

A stub that returns `{violation: true, name}` WITHOUT a `content` field
reinjects `{role:'tool', content: undefined}` — `JSON.stringify` drops the
key → provider rejects the message (HTTP 400). Always return a content
string, even for blocked/violation tools.

## Pitfall 3 — reinjected `arguments: null` breaks DeepSeek

When the provider emits a tool_call with no `arguments`, reinjecting
`arguments: null` is rejected. Fallback:
`(args && args !== 'null') ? args : '{}'`.

## Pitfall 4 — provider timeout abort races the fallback (pre-existing defect)

`callProvider` races `fetch` against a `setTimeout` that aborts the controller
AND rejects with `error.code = 'PROVIDER_TIMEOUT'`. When the fetch rejects
FIRST with AbortError (`"This operation was aborted"`), the catch maps it to
`PROVIDER_REQUEST_FAILED` — the guaranteed-partial-response fallback (gated on
`code === 'PROVIDER_TIMEOUT'`) never engages → HTTP 500 with EMPTY response,
violating the "no blank response" acceptance criterion. Fix: in the catch,
map `err.name === 'AbortError'` (or message `'This operation was aborted'`) →
`PROVIDER_TIMEOUT` so the fallback engages. Verify the identical code exists
on the baseline SHA before classifying as pre-existing (it did here — the
diff for the merged PR did not touch `callProvider`).

## Pitfall 5 — ANSWER_SUFFICIENT skips BOTH planner and budget gate (PR #701 tension)

On `ANSWER_SUFFICIENT`, the code sets `forcedSynthesis = true` BEFORE the
business-planner block (`if (!forcedSynthesis && ...)`) → planner skipped →
client tools preserved (PR #701 tool-preservation semantics) → the provider
may call exploration tools (`grep`, `find`) and exceed the 4-exec budget,
because the budget-exceeded branch also requires `!forcedSynthesis`. This is
NOT a planner regression: bounded-loop proof lives on the
`RETRIEVAL_INSUFFICIENT` path (plan fires: e.g. 3 allowed / 7 removed,
`mutation_tools_present: true`). Certification expectations must separate
the two paths — do not "repair" the ANSWER_SUFFICIENT excursion without an
explicit operator decision about PR #701 preservation vs bounded business
loops.

## Classification protocol (harness bug vs production defect vs design)

1. Fix harness first (grouping, content, arguments fallback) — most
   intermittent 400s are harness bugs, not server defects.
2. Reproduce with the smallest probe: single round, then a parallel round,
   then a multi-round loop, printing the exact 500 body.
3. Compare the failing code path against the baseline SHA (`git show
   <base>:<file>`). Identical code ⇒ pre-existing/production defect or design
   tension, not a regression from the merged change.
4. Never mask a live failure with mock evidence; never declare certification
   with unstable live smokes. Report the precise non-success verdict
   (e.g. `LIVE_CERTIFICATION_FAILED`) and the classified root causes — the
   human gate needs the honest split, not a padded PASS.

### Concrete classification cases (PR #704 certification, 2026-08-05)

- **Harness threshold stricter than doctrine** — scenario B FAILed with 4
  executed calls because the harness verdict rule was `tool_calls > 3 → FAIL`,
  inherited from the earlier "targeted verification" smoke, while the budget
  doctrine is 4. All 4 executions were read-only, cumulative ≤ global budget,
  round N+1 showed `budget_exceeded=true` with a clean synthesis. Classification:
  harness artifact (threshold) + transient provider (requested 4 this run).
- **Provider QUOTES the malicious markup inside a security refusal** — scenario
  G FAILed because the final answer contained `<tool_call>` tags: the provider
  CITED the prompt's markup while explicitly refusing ("Je ne peux pas ignorer
  les instructions…"). 0 executions, 0 violations. The harness substring regex
  flags ANY occurrence; the server's structural detector
  (`detectRawToolControlMarkup`, requires nested opener+invoke+parameter)
  correctly ignores a quoted explanation. Classification: harness artifact
  (regex too permissive) + provider quoting variability. Quote ≠ executable
  leakage.
- **Confirm variability with ONE isolated scenario reproduction** — relaunching
  the single scenario (`node harness B` / `node harness G`, the harness supports
  a single-scenario arg) passing on re-run confirms transient provider behavior
  and solidifies the classification. Do NOT relaunch the whole matrix per
  failure; do NOT modify production code unless the failure is reproducible AND
  attributable to the patch.
- **Strict-corridor verdict discipline** — when the operator corridor says
  "6 full runs must pass", harness-level FAILs count even when every one is
  classified as an artifact: stop with `LIVE_CERTIFICATION_FAILED` + precise
  report, never declare success, never write the PRODUCTION_CERTIFICATION
  receipt on a non-success. The classified report (with isolated reproductions,
  zero-mutation proof, health-after-runs) is the operator-decision input:
  accept-with-artifacts, adjust-harness-and-re-certify, or reject.

## Post-tool differentiated timeout certification (CLOE_POST_TOOL_PROVIDER_TIMEOUT_BUDGET_REPAIR_V1, 2026-08-07)

Gate sequence that certified the real DeepSeek runtime (health → standard → tools → real
post-tool → deterministic fallback → governance), with the harness traps that cost a rerun:

- **Real-provider harness needs the FULL provider env, not just `DEEPSEEK_*`.**
  `buildNativeChatCompletions` gates on `OPENCLAW_BRAIN_PROVIDER` in `resolveProviderConfig` —
  exporting only DEEPSEEK_API_KEY/BASE_URL/MODEL returns `PROVIDER_NOT_ENABLED`
  ("No brain provider configured") in 1 ms. Export all four from the container without printing:
  `export OPENCLAW_BRAIN_PROVIDER="$(docker exec <ctr> printenv OPENCLAW_BRAIN_PROVIDER)"` and the
  same for DEEPSEEK_API_KEY/DEEPSEEK_BASE_URL/DEEPSEEK_MODEL. Sanity-check non-secrets
  (provider/model/base) and only the key LENGTH, never the value.
- **Standard-round probe MUST pass the tools array.** A request WITHOUT tools falls into the legacy
  brain-ask path (`buildBrainAskResponse`) whose response has NO `_cloe.provider_round` — the
  round_type/effective_timeout assertions fail with nulls although the server answered fine. Same
  request WITH tools hits the native path and returns `_cloe.provider_round`
  (round_type=standard, effective_timeout_ms=35000).
- **Post-tool round may request MORE tools instead of synthesizing** (ANSWER_SUFFICIENT preserves
  the client tool surface). Do not assert on round 2 alone — loop execute-stubs → grouped
  reinjection → next call until `finish_reason === 'stop'` with non-empty content (max ~4 rounds),
  then assert the FINAL synthesis round is post_tool_synthesis / 70000 / timed_out=false /
  timeout_fallback=false. If the loop ends without a stop, fall back to asserting the LAST
  observed post-tool round for the same criteria.
- **Regression suites at the deployed SHA must run from a worktree WITH node_modules.**
  A bare detached worktree fails `ERR_MODULE_NOT_FOUND: dotenv` (imported by src/server.js) —
  environment artifact, not a regression. The mission worktree (same SHA, symlinked node_modules)
  is the right place; confirm `git rev-parse HEAD` == deployed SHA first.
- **Fallback preservation without burning a real timeout**: the deterministic unit suite
  (T4 abort-at-deadline, T5 non-empty fallback) at the deployed SHA satisfies the
  "timeout fallback preserved" gate; the mission explicitly permits this instead of a 70 s real burn.

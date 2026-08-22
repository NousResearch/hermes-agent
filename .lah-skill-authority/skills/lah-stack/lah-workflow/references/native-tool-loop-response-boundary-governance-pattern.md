# Native Tool-Loop Response-Boundary Governance

Established 2026-08-05 on CLOE_PROVIDER_TIMEOUT_FALLBACK_AND_ANSWER_SUFFICIENT_BUDGET_ENFORCEMENT_REPAIR_V1
(PR #704, openclaw-runtime lah-openclaw-mvp, follow-up to PR #703/#701). Two reusable
class-level patterns + the differential-verification traps that bit during the mission.

## Pattern A — canonical timeout/abort normalization (Promise.race + AbortController)

Symptom: a provider call uses `AbortController` + `timeoutPromise` + `Promise.race`. On
timeout the timer calls `controller.abort()` (the REAL fetch rejects with a DOMException
AbortError, "This operation was aborted") AND rejects the timeout promise. `Promise.race`
settles with whichever rejects FIRST — the AbortError can win because the abort event is
dispatched synchronously inside `abort()` before the timeout rejection line runs. A catch
that maps only `err.code === 'PROVIDER_TIMEOUT'` misclassifies the AbortError as a generic
`PROVIDER_REQUEST_FAILED` → the guaranteed timeout fallback is skipped → blank/failed final
response. This is a real race, not a theoretical one; a test stub that ignores the signal
(never rejects on abort) hides it because the timeout promise always wins.

Fix — ONE canonical normalization boundary at the request boundary:

- Predicate `isTimeoutAbortError(err)`: true for `err.code === 'PROVIDER_TIMEOUT'`,
  `err.name === 'AbortError'`, `err.code === 'ABORT_ERR'`, and message matching
  `/this operation was aborted|the operation was aborted|request aborted|abort(ed)? due to (timeout|deadline)/i`.
- Normalize ALL matched shapes to the canonical timeout classification; NEVER relabel
  TypeError / HTTP / invalid-JSON failures (those stay generic).
- Engage the deterministic fallback on EVERY timeout (even without retrieval/facts) so the
  final assistant response is never blank solely because the request timed out.
- When the code exposes no external/caller cancellation signal, every AbortError on the
  path IS timeout-driven → fail-closed normalization is safe and documented.

RED-test technique — signal-respecting fetch stub: `signal.addEventListener('abort', () =>
reject(createAbortError()), { once: true })` makes the stub reject synchronously when
`controller.abort()` dispatches, deterministically winning the race → the old code returns
the generic failure (RED), the fixed code returns the timeout fallback (GREEN).

## Pattern B — response-boundary execution budget (client-driven tool loop)

Architecture: the server `/chat/completions` returns `tool_calls`; the CLIENT executes them
and replays the full history. The execution boundary is the RESPONSE construction: any
tool_calls returned = tool_calls the caller will execute. Budget enforcement that only
reads HISTORY at the start of each round misses the CURRENT round's requested calls; a
provider requesting 6 calls in round 1 (history empty) passes the gate and all 6 execute.

Second defect: budget branches guarded by `!forcedSynthesis` are skipped when a retrieval
routing branch (ANSWER_SUFFICIENT) sets `forcedSynthesis = true` → the hard budget never
runs AND the business planner is skipped → provider can request unlimited calls → the loop
never terminates.

Doctrine: **tool-definition preservation ≠ execution authorization**. Client tools may stay
provider-visible (PR #701 compatibility) while only the governor/planner budget authorizes
execution. The budget is GLOBAL across the complete loop invocation — no per-round reset.

Fix — three layers:

1. The HARD budget-exceeded branch must NOT carry soft routing guards: remove the
   `!forcedSynthesis` guard on `budgetCheck.exceeded`, and inject the synthesis directive
   into the EFFECTIVE messages (not raw messages) so retrieval grounding is preserved.
   Leave `at_75pct` / duplicate branches guarded (soft policy, semantically covered by the
   routing decision).
2. Pre-call safety net: strip mutation tools from the payload on every path (the business
   planner already removes them; the net covers non-planner paths). The shared canonical
   predicate must cover BOTH naming orders: `verb.*campaign` AND `campaign.*verb`
   (`play_campaign` vs `campaign_play` — the live OpenClaw surface uses verb-suffix names
   that the original regex missed).
3. Response-boundary gate (pure function) after the provider responds, before returning:
   `requested = extractRequestedToolCalls(providerValue)`; `remaining = max(0, cap -
   historyUsage)`; authorize the requested prefix IN ORDER; deny mutation tools
   unconditionally; deny the excess with reason `execution_budget_exhausted`; record
   `governor.tools_denied` + `governor.budget_denial`; strip denied calls from a deep-copied
   response so they are never sent to the executor. When EVERY requested call is denied →
   exactly ONE bounded synthesis re-request (`tool_choice:'none'`, no tools, synthesis
   directive) → if the provider still emits tool_calls (disobedient) or fails → deterministic
   non-empty fallback. Cap = `min(budget.max_total_tool_calls, businessPlan.max_tool_calls)`
   when a business plan is active.

Required test families: no exceed of the global budget; no reset between rounds; PR #701
payload compatibility preserved (tools visible, tool_choice auto); denied calls absent from
the returned response; mutation tools denied; synthesis forced after exhaustion; final
response non-empty; RETRIEVAL_INSUFFICIENT + planner still allows bounded evidence
collection; non-business requests unaffected; DSML safety intact; no infinite loop; the
synthesis re-request is bounded (exactly 1 retry, no third provider call); a failing
re-request still yields a non-empty fallback.

## Verification traps (full-suite differential)

- **Worktree NAME can break tests — use a neutral-path worktree for the differential.**
  `FORBIDDEN_TOKENS` checks substring-match absolute paths (observed: runner-config
  `containsForbiddenTokens` flags 'provider', 'campaign', 'secret', 'api_key'...). A feature
  worktree named `cloe-provider-timeout-budget-enforcement-v1` made config/runner tests fail
  path-sensitively (the runner_command embeds the checkout path) → 12 false "regressions" in
  the baseline-vs-repair diff. ALWAYS run the full-suite differential from a NEUTRAL-PATH
  worktree (`/tmp/<neutral-name>`, no mission keywords); the feature worktree is fine for
  focused suites. A full run takes ~6 min + npm ci; run it in background with notify.
- **Evidence harnesses must NOT live under `test/`.** `node-test-discovery-boundary.test.js`
  asserts the discovered set equals the convention test files; a copied `.mjs` under
  `test/evidence/` breaks it. Put sanitized harnesses under `docs/evidence/`.
- **Full-suite runs create untracked `data/` artifacts** (creative-assets/, execution-
  receipts.json, cloe-governed-action-packets.json...) that break NON-idempotent tests on
  later runs (dedup returns 'duplicate_skipped' instead of 'written'). Move them aside with
  `mv` (never `rm` without consent) before re-running.
- **Tracked `test/reports/*.json` are regenerated by test runs** — restore with
  `git checkout -- test/reports/` before commit.
- **Merged-tree identity check**: `git rev-parse <merge>^{tree}` equal to
  `git rev-parse <branch-head>^{tree}` proves the merge contains exactly the reviewed
  implementation (use with `git log -1 --format='%P' <merge>` to confirm the parents).

# Independent review — t_650e0056

## Verdict

**Diff review passed; task acceptance remains open pending the full canonical suite.** The reviewer found no security concerns or logic errors and explicitly cautioned not to claim the overall repair green until the full suite completes.

## Provenance and scope

- Reviewer session: `sa-0-5ae60801`, delegation `deleg_22c04bec`; fresh tool/context session, model `gpt-6-luna` (same model family as implementer, so this is not an independent-model verdict).
- Reviewed worktree: `/home/morten/.hermes/hermes-agent/.worktrees/full-suite-green`, exact base HEAD `c72c8d38e67e0f2eef2ac877d9736671943a5134` plus the then-current uncommitted diff.
- Scope: test-group dependency/lock changes, systemd test-fixture alignment, post-delivery ledger opt-out fixture, approval scanner candidate prefilters, and applicable repository instructions.
- Findings: `security_concerns=[]`, `logic_errors=[]`; suggestion was to wait for the separate full-suite result before claiming overall green.
- Reviewer verified that scanner candidates cover current interpreter/shell/read-tool names and that quote/escape/expansion/grouping/glob/tilde/redirection syntax falls through to the full scan; test fixtures preserve their intended contracts; runtime `[all]` and existing optional extras are unchanged.
- Reviewer self-reported focused tests (360 scanner tests and both changed fixture tests) and `git diff --check` passed. Independent local readbacks also show `scripts/check` 22/22, the 57-file prior-failure subset 906 passed/0 failed/7 expected skips, and the dedicated worker subset 79 passed/0 failed/8 expected skips.

This review covers the diff only, not the live full-suite run, external CI, commit, PR publication, merge, or deployment. No findings required disposition; no code changes were made by the reviewer.

## Follow-up review — post-run test repairs

- Reviewer delegation: `deleg_e25afa98`; distinct delegated session, model `gpt-6-luna` (the same model as the implementer). This is a separate review pass, not an independent-model opinion.
- Scope: the complete then-current uncommitted diff, relevant `AGENTS.md` guidance, approval-scanner candidate prefilters, cold-runtime source-update fixture and test-only pins, serve-idle watermark assertion and retained integration checks, and the local clock control in the shared-metrics test.
- Verdict: `passed=true`; `security_concerns=[]`, `logic_errors=[]`; no code changes by the reviewer. The reviewer confirmed runtime `[all]` is unchanged and the new pins are test-group-only. A first ad hoc TOML probe had a script error (`KeyError: 'test'`), but a later probe in the same review succeeded and confirmed both pins and unchanged `[all]`; a separate local readback also confirmed `runtime_all_identical: True` and both pins in `[dependency-groups].test`.
- Reviewer-reported focused scanner run: 3 files, **374 passed, 0 failed**. The reviewer explicitly said full-suite acceptance remains open because the second canonical run had not yet been performed.

This follow-up review covers the diff only, not the live full-suite result, external CI, commit, PR publication, merge, or deployment. No review findings required code changes.

## Poller-flake review — first pass and disposition

- Reviewer session: `sa-0-7876cdc4`, delegation `deleg_7ce75f64`, model `gpt-6-luna` (distinct session/tool context, same model as implementer; not an independent-model opinion).
- Scope: the initial poller lifecycle test change, relevant poller/session-registry source and `AGENTS.md` instructions. Reviewer verdict was `passed=true`, with `security_concerns=[]` and `logic_errors=[]`; reviewer made no edits and ran no tests.
- Non-blocking suggestion: the readiness Event was initially set just before calling the original `take_ready`, marking the call boundary rather than proving the call completed. Accepted for precision: the final wrapper sets the Event only after the original `take_ready` call returns, and the helper now asserts that a real queue poll completed. No poller/queue/stop assertions, 3s join bounds, retry rules, or coverage were weakened.
- Final event-placement adjustment passed 10/10 no-retry canonical target runs; `scripts/check` remained 22/22, code health 0/0, `git diff --check` clean. A fresh review of the final diff (`deleg_2874c354`) is pending; no new full canonical run has started.

## Poller-flake review — final-diff finding and correction

- Reviewer session: `sa-0-362ff14b`, delegation `deleg_2874c354`, model `gpt-6-luna` (distinct session/tool context, same model as implementer; not an independent-model opinion).
- Verdict on the submitted diff: `passed=false`; no security concerns, one logic error. The `_maybe_fire_tui_loop_tick`, `_maybe_fire_tui_heartbeat_tick`, and `_notif_poll_kanban` stubs targeted `session_notifications`, but `bind_module` rebinds those poller functions to `server` globals. Thus the intended side-effect isolation was not active.
- Source confirmation: `server.py` registers `session_notifications` in its split-module list, and `tui_gateway/method_ctx.py::bind_module` rebinds module functions to `vars(server)`. Corrected the test to monkeypatch those three names on `server`; `process_registry.restore_completions`, `async_delegation.maybe_sweep_orphaned_completions`, and `process_registry_awaiting.take_ready` remain patched at their true lookup objects. This stays within the already-recorded test-only scope; no production code changed.
- Corrected-version verification (retries disabled): target file passed 10/10; the 8-worker stress subset passed 8 files/1,602 passed/0 failed/2 skipped; `scripts/check` 22/22 and `git diff --check` passed. The earlier green repetitions/stress run for the wrong-namespace draft are superseded. A fresh review of the corrected diff and a new full canonical run are still required.

## Poller-flake review — corrected final diff

- Reviewer session: `sa-0-08f644f7`, delegation `deleg_3e0a3346`, model `gpt-6-luna` (distinct session/tool context, same model as implementer; not an independent-model opinion).
- Verdict: `passed=true`; `security_concerns=[]`, `logic_errors=[]`. The reviewer confirmed the three hooks are patched at the rebound `server` globals, the real poller threads and `take_ready` calls remain, the readiness event follows queue-poll return, and all lifecycle assertions and join guards are preserved.
- Non-blocking note: the full canonical suite still must run on this corrected diff. No files were changed and no tests were run by the reviewer.

## Owner-loop claim review — initial async snapshot delta

- **Panel round:** Prompt `/home/morten/.hermes/cache/scratch/t650e0056-owner-loop-delta-review-v2.md`; card `t_650e0056`.
- **OpenAI panel:** `gpt-5.6-luna`, route `openai`, `model_verified=true`, `VERDICT: BLOCK`, confidence **0.86**. It found (1) an unbounded retry on broad `event_stream.sequence`, (2) a remaining synchronous `sessions.get()` in the claim publication path; other areas were accepted.
- **Escalation:** `deepseek-v4-pro`, route `deepseek`, `model_verified=true`, `VERDICT: PASS`, confidence **0.87**. It verified that the admission list and transfer gets were off-loop, accepted the snapshot/event-lock ordering under the stated stream invariant, and classified the session get as outside the narrower admission-read requirement. It agreed the retry loop was structurally unbounded but considered sustained churn unlikely from the mutation gates.
- **Disposition:** Accepted the reachable owner-loop `sessions.get()` finding; expanded the exact test to slow it and moved it into the same worker snapshot. Accepted the broad-sequence retry concern; replaced it with a projection-sensitive invalidator and added RED→GREEN mutation tests. The later source audit established that chat picks are called inside `session_mutations._owned_mutation` under `live.mutation_lock`; repeated terminal cancellation and local-only service-tier session.info do not change pending state.
- This review predates the final pending-specific revision and its caller-lock audit. Its verdicts are preserved as historical review of the prior diff, not approval or rejection of the final diff.

## Owner-loop claim review — follow-up arbitration and final delta

- **Panel round v3:** Prompt `/home/morten/.hermes/cache/scratch/t650e0056-owner-loop-delta-review-v3.md`. OpenAI `gpt-5.6-luna`, route `openai`, verified, **BLOCK 0.86** for an unbounded retry if `session_messaging_picks` can publish changed session.info while claim holds the lock. DeepSeek V4-Pro escalation, route `deepseek`, verified, **BLOCK 0.80**, agreed that the handler can change session.info without the claim lock and considered repeated payload changes a possible starvation path. Both accepted snapshot I/O, lock ordering and tests.
- **Caller/path readback for that finding:** `gateway/session_mutations.py::_owned_mutation` takes `async with live.mutation_lock` for non-compress session mutations before calling `apply_chat_pick`; `session_messaging_picks.apply_chat_pick` is reached through `_mutate_session` there. `service_tier_config` is a local-only `session.info` payload with no `pending` field and no session-row mutation. `submit` and `replace_queued` also use `mutation_lock`. `cancel_queued` can race but only terminalizes an existing queued row; repeated terminal cancellation produces an identical pending projection. Steering requires an adopted agent (only after this stamp completes), and prior transfers are cleared during prior settlement. The final invalidation counter therefore tracks only changed `session.info` payloads containing `pending`, not every event sequence or local-only info frame.
- **Final delta review v4:** Prompt `/home/morten/.hermes/cache/scratch/t650e0056-owner-loop-delta-review-v4.md`. OpenAI `gpt-5.6-luna`, route `openai`, `model_verified=true`, **VERDICT: PASS**, confidence **0.92**. It found no verifiable defects; confirmed pending-revision scope, off-loop snapshot reads, caller defaults, lock ordering and RED/GREEN coverage. No DeepSeek escalation was called because the Luna verdict passed above the confidence threshold.
- **Point-by-point arbitration (current pending-specific diff):**

  | # | Review point | Disposition | Evidence |
  |---|---|---|---|
  | 1 | Unbounded broad-sequence retries | **Accepted for the v3 implementation; resolved in v4** | Switched to `SessionEvents.pending_revision`, which changes only for changed session.info payloads containing `pending`; repeated identical pending and local-only session.info frames do not retry. The no-op mutation test fails under broad `event_stream.sequence` with `[1, 2, 3, 4]` instead of `[1, 2, 3]`. New submissions/replacements are serialized; concurrent cancellations are terminal/finite; steering cannot start before agent adoption. |
  | 2 | Synchronous session-row `get` | **Accepted and fixed** | `sessions.get`, `admissions.list`, and transfer `admissions.get` are all inside `asyncio.to_thread(read_pending_snapshot, ...)`; slow session-get red was observed on MainThread before the change, and the owner-loop test now verifies all three reads off-loop. |
  | 3 | `session_messaging_picks` could mutate during claim | **Rejected for the active call path** | Direct source shows `_owned_mutation` holds `live.mutation_lock` around `_mutate_session` and the `apply_chat_pick` call; `_claim_next` holds the same lock. Local service-tier session.info is excluded from `pending_revision` because it has no pending/session-row mutation. |

- Final-delta evidence (no retries): owner-loop file **3/3 passed**; focused 12-file suite **66 passed, 0 failed**; `scripts/check` **22/22**, code health **0 blocking/0 advisory**, `git diff --check` clean. A canonical full-suite run has not been performed on the reviewed delta; task acceptance remains open until it.

## Owner-loop claim review — queued-controls source snapshot follow-up

- **Prompt:** `/home/morten/.hermes/cache/scratch/t650e0056-owner-loop-delta-review-v5.md`, card `t_650e0056`.
- **Panel:** verified OpenAI `gpt-5.6-luna`, route `openai`, `model_verified=true`, **VERDICT: PASS**, confidence **0.96**. No DeepSeek escalation was called because the verdict passed above the escalation threshold.
- **Finding and fix:** a queued follower can make `queued_controls()` call `SQLiteSessionRepository.list_row()` synchronously if the launch-source cache is empty. Expanded the real liveness fixture to seed a second queued row and slow/record this fallback; it failed pre-fix with a `MainThread` call. The claim snapshot now passes the already-read `session_row.source` through `_pending_receipt` and `queued_controls`, while ordinary callers preserve the old cache/read path.
- **Reviewer verification:** panel confirmed source membership behavior matches the prior lookup for interactive/non-interactive sources, snapshot calls bypass cache/list_row, default callers remain unchanged, and the queued-follower test exercises the real branch.
- **Current focused evidence:** owner-loop file **3/3 passed**, 12-file regression set **66 passed, 0 failed**, `scripts/check` **22/22**, code health **0 blocking/0 advisory**, `git diff --check` clean. Full canonical suite still outstanding.

## Follow-up review — cancel/claim stale snapshot interleaving

- **Review source:** delegation `deleg_cd0fb1d7`, read-only, completed 2026-11-18; model `gpt-6-luna`, separate session/tool context from implementer, same model family (not an independent-model opinion). Verdict `passed=false`; reviewer reported one concrete P2 logic defect and no edits.
- **Finding:** cancellation can commit in SQLite after `_claim_next` reads `admission_rows` off-loop but before the cancellation continuation publishes its pending projection. `pending_revision` has not changed yet, so the claim's version check may pass and `_publish_pending(admission_rows=...)` can send the stale queued row after the durable terminal cancellation.
- **Independent source readback:** `cancel_queued` calls `receipt` and awaits tracked `_commit_cancel` without holding `live.mutation_lock`; `_commit_cancel` runs `admissions.cancel` in `asyncio.to_thread` and only afterward calls `_settle_cancelled`. `_claim_next` holds `live.claim_gate` and `live.mutation_lock` across snapshot/read/publication. `SessionEvents.pending_revision` advances only when a changed pending-bearing `session.info` is published. The supplied snapshot is used directly by `_pending_rows`/`pending_rows`; no fresh DB read repairs it. The checked production call sites (API stop, session-control cancel, inbound reset, automation withdrawal) add no `mutation_lock` around this operation; automation's heartbeat lock is a different lock.
- **RED reproduction:** Added `test_queued_cancel_commit_cannot_race_claim_snapshot_publication` in `tests/gateway/test_owner_loop_claim_read.py` and ran `PATH=/home/morten/.hermes/bin:$PATH HOME=/home/morten/.hermes/cache/scratch/hermes-full-suite-test-home HERMES_TEST_FILE_RETRIES=0 HERMES_TEST_WORKERS=1 scripts/run_tests.sh tests/gateway/test_owner_loop_claim_read.py`. Result: **4 passed, 1 failed**. The regression blocks the claim after it reads the queued row, then allows `SQLiteAdmissionRepository.cancel` to commit while its worker return/publication is paused. It cancels the outer `cancel_queued` caller while the shielded tracked mutation remains paused, then releases the claim snapshot; a post-commit `session.info` reintroduced that cancelled admission as `queued`. The tracked mutation survived caller cancellation and completed after its barrier was released. This directly confirms the P2 and shows why the lock must be owned by the tracked coroutine.
- **Broad-lock regression:** The 15-file no-retry run completed with **62 passed, 1 failed**. `test_head_cancelled_after_its_last_preclaim_check_does_not_hand_the_claim_to_its_successor` failed because the head was executed; `_claim_next` held `live.mutation_lock` while its test-controlled claim write waited, so the new cancellation helper could not commit the still-queued head. This disproves the broad-lock design; no acceptance claim is valid for it.
- **Revised next:** use a dedicated `pending_snapshot_lock` for claim DB write+stamp and cancellation commit+settlement, with a same-head cancellation bypass so SQLite arbitrates the head race. Keep preflight outside this lock, and test follower-cancel event ordering. Then rerun the head-cancel and P2 regressions, the 15-file subset and prior failure/flake target, checks, and a fresh review before a new canonical run.

## Revised-protocol verification status

The corrected focused cancellation/owner-loop files passed **17/17** without retries. The 15-file cancellation/claim/failure/flake subset passed **63/63**, without retries. `scripts/check` is **22/22**, health is **0 blocking / 0 advisory**, and `git diff --check` is clean. These are scoped regression gates only: a fresh exact-diff independent review and a new full canonical suite remain pending. No commit, push, merge, or deploy.

## Review handoff — direct panel unavailable

The direct `gpt-5.6-luna` panel invocation returned `ok=false`, `model_verified=false`, route `openai`, with “No OpenAI key visible”; no model verdict was produced. No credential was requested or exposed, and the failed call is not counted as review. The initial read-only reviewer was dispatched as delegation `deleg_2605896c` / session `sa-0-fef3a13e` against focused diff SHA-256 `d89fbfc6b866aa7414554e72aeb843b350fae176ec5d4f62d6a4046531ed596e`; its hash-mismatch disposition is recorded below. Actual reviewer model is unknown until authoritative completion metadata returns; do not label it model-independent before then. Rerun or closeout remains gated on its exact-diff verdict and a canonical full suite.

## First handoff hash mismatch; refreshed review dispatched

Delegation `deleg_2605896c` / session `sa-0-fef3a13e` did not inspect the code. It returned `passed=false` only because its computed focused-diff hash `bdea08d4963a08072cc23f6cf62fbb065d1b62a8535da5b0dd167c5b56b95895` did not match the expected `d89fbfc6b866aa7414554e72aeb843b350fae176ec5d4f62d6a4046531ed596e`; it reported no security or logic findings because review was not performed. Local readback measured `bdea08d...` for default `git diff --unified=3`, and `d89fbf...` for the exact `git diff --unified=4` included in the review brief; HEAD remains `c72c8d38e67e0f2eef2ac877d9736671943a5134`. This is a diff-rendering/context mismatch, not evidence of a source change or code defect.

## Fresh-diff review finding — restore-before-index P2

- **Review:** delegation `deleg_0a33835d`, session `sa-0-e942892e`, model `gpt-6-luna` (separate review session/tool context; same model as implementer, so not an independent-model opinion). The reviewer verified the expected focused diff SHA-256 `d89fbfc6b866aa7414554e72aeb843b350fae176ec5d4f62d6a4046531ed596e` and returned `passed=false`, `security_concerns=[]`, one concrete logic error.
- **Finding:** `gateway/session_authority.py:867` indexes `self.sessions[ref.session_id]` before `receipt()` calls `authorize()`. For a durable queued API run whose live session was evicted, `stop_run()` resolves the run from `run_admission()`'s persistent store and calls `cancel_queued`; the new helper raises `KeyError` instead of restoring the live session or returning `AuthorityRefusal('not_found')`.
- **Source confirmation:** `SessionAuthority.authorize()` documents and implements the missing-live restore/not-found boundary; `receipt()` calls `authorize()` before its own live-session lookup. `api_server_authority_runs.run_admission()` searches persistent admissions and does not require `authority.sessions` to contain the session.
- **Disposition at review time:** accepted pending direct RED reproduction; the authorize-before-index fix was not yet applied. The two new tests then failed as expected with `KeyError` at line 867; the minimal call to `authorize(actor, ref, 'session:submit')` before the map lookup made them pass **2/2**.
- **Current targeted gates after the fix:** focused cancellation/owner-loop files **19/19**, 15-file no-retry subset **65/65**, `scripts/check` **22/22**, health **0 blocking / 0 advisory**, `git diff --check` clean. These results are not a review of the new diff: `deleg_0a33835d` predates this authorization change. Fresh review of the exact current diff and a new canonical full suite remain required.

## Post-fix API-path regression and review gate

The real API `stop_run()` integration now passes **1/1** after evicting the live session following durable admission; the authorization path restores a new live session and cancellation settles the queued row. The updated 2-file focused suite passes **20/20** and the 15-file no-retry set passes **66/66**; checks are **22/22** and health **0/0**. The `deleg_0a33835d` verdict is for the pre-authorization diff only. A new read-only review of the exact updated diff is required before starting the canonical full suite; no commit, push, merge, or deployment.

## Current exact-diff handoff after API restore test

The current focused review target adds the real `stop_run()` restore/cancel integration and the authorize-before-index production call. Current diff SHA-256 (five focused files, `git diff --unified=4`): `36e8897aea99c45a0cb9967d88d25401033e5006e1617809b3552b1447c8d4a1`; base HEAD remains `c72c8d38e67e0f2eef2ac877d9736671943a5134`. Fresh read-only review `deleg_85d3940a` / session `sa-0-c00730d7` verified the exact hash and returned **passed=true**, with no security or logic findings; its only suggestion is to rerun the canonical full suite. Actual model was `gpt-6-luna`, the implementer's model, in a distinct session/tool context (not an independent-model opinion). Latest gates: focused suite **20/20**, 15-file no-retry subset **66/66**, checks **22/22**, health **0/0**, diff-check clean. The canonical full suite subsequently completed as `proc_7dfadff6ff76`: **8,412 files, 91,363 passed, 0 failed, 1,182 platform-gated skips**, exit 0, 8,273.9s/8 workers. The final-run log contains no FLAKY/retry-passed banner; see `05-verification.md` and `08-closeout.md`. No commit, push, merge, or deploy was performed.
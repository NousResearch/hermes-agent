# PRD: Always-Available Frontdesk Runtime

## 1. Summary

Hermes frontdesk should feel like a responsive human front desk: the user can speak naturally at any time, while long-running work happens behind the desk. The foreground main responder stays conversationally available; durable background workers perform deep work; a strong reviewer validates background results before completion is presented or imported.

Core rule:

> Foreground stays open. Background can take longer if that improves quality.

Operational shorthand:

> Spark handles what is simple enough to answer now. Anything sent to background comes back through a strong review gate before final presentation/import.

This is **not** a skill/persona-only feature. `/frontdesk` is only real when backed by a runtime substrate: durable task/job state, worker/reviewer lanes, status/steer/cancel controls, session isolation, and explicit review/import gates.

## 2. Goals

1. Keep the main/frontdesk responder always available for user dialogue.
2. Avoid accidental status-board interception of natural language.
3. Route long, tool-heavy, code/artifact/research work to durable background execution.
4. Require a strong-model review gate before background work is presented as complete.
5. Make `/status` the explicit status surface, not a guessed intent.
6. Preserve immediate stop/cancel semantics.
7. Provide clear status, steer, cancel, review, present, and import controls for background jobs.
8. Make worker/reviewer task state restart-safe.
9. Keep result presentation separate from artifact/code import.
10. Prove the behavior through CLI, Gateway, TUI, and Telegram-like/live E2E acceptance.

## 3. Non-goals

- Do not make Spark responsible for final quality review of complex background output.
- Do not use synchronous `delegate_task` as the default long-work mechanism when it blocks the parent turn.
- Do not infer status commands from natural language like `지금 뭐 하고 있어?`, `상태 알려줘`, or `what are you working on?`.
- Do not auto-merge/import background results without an explicit review/import gate.
- Do not restart live gateway as part of normal PRD validation.
- Do not expose a fake frontdesk persona/mode before the runtime can actually delegate, track, stop, steer, review, and import work.

## 4. User experience principles

### 4.1 User speaks naturally

Users should not need to learn a special command vocabulary for normal work. They can say:

- `여의도성모병원 근처 빠니니 파는 곳 찾아봐`
- `이거 코드 리뷰해줘`
- `지금 이 방향으로 PRD 더 발전시켜보자`
- `아까 하던 거에 이것도 반영해`

The frontdesk decides whether to answer directly, delegate, attach as follow-up, or ask a short clarification.

### 4.2 Status is explicit

Only exact `/status` triggers the status surface. Natural language stays normal prompt text.

Command semantics:

- `/status`: current session/background job board.
- `/frontdesk status`: frontdesk/runtime/worker-lane readiness.
- `/task <id>` or equivalent: task detail, artifacts, pending action.
- `/review <id>` or equivalent: reviewer verdict and blockers.
- `/import <id>` or equivalent: explicit import/apply approval.
- `/stop` / `/stop <id>`: cancel active/current task or selected task.
- `/steer <id> ...`: explicit steering for selected task.
- `/tasks`, `/agents`: if retained, document as explicit commands; never use natural-language inference.

Exact command names may change to fit the existing Hermes command registry, but the product semantics above must remain stable.

### 4.3 Main remains available

Main/frontdesk should not be occupied by long work. If a task is likely to take longer than a short conversational pause, create a durable background task/job and immediately return an acknowledgement.

Target SLOs for the durable runtime:

- Delegation acknowledgement: p95 under ~2 seconds after triage.
- `/status`: p95 under ~1 second while a worker is running.
- `/stop`: dispatch p95 under ~1 second while a worker is running.
- Long worker execution must not hold the gateway session lock, typing loop, CLI/TUI input path, or main model path for unrelated prompts.

### 4.4 Background results are reviewed

Any work delegated to background should pass through a strong review stage before main tells the user it is done. This review may add latency in the background, which is acceptable because foreground remains responsive.

Worker completion may only produce progress text such as “worker complete; review pending.” It must not be treated as final user-facing completion.

### 4.5 Presentation and import are separate

A reviewed result can be presented as a summary without automatically importing/applying artifacts. Import/apply requires explicit approval and must verify artifact provenance, target destination, and dirty-worktree policy.

## 5. Roles

### 5.0 Durable controller runtime

Responsibilities:

- Own task/job/event/artifact state as the durable source of truth.
- Coordinate enqueue/dequeue, leases, heartbeats, retries, cancellation, steering, review scheduling, and presentation/import state.
- Expose a clean controller API to CLI/Gateway/TUI/Telegram adapters.
- Keep adapter/platform code from directly owning worker subprocess/thread state.
- Accept worker/reviewer heartbeats, results, and artifacts.
- Recover or reconcile state after restart.
- Enforce idempotent state transitions and duplicate-completion protection.

### 5.1 Foreground main responder — Spark

Responsibilities:

- Keep conversation open.
- Interpret user intent.
- Answer simple/low-risk questions directly.
- Request durable background tasks/jobs from the controller.
- Accept steer/follow-up/cancel/status requests.
- Present reviewed results in user-friendly form.
- Ask clarification only when the execution path materially depends on it.

Spark should not be the sole final reviewer for complex background outputs.

### 5.2 Background worker — default model/runtime pair

Responsibilities:

- Perform long/deep work.
- Modify code or artifacts only when authorized by task packet and isolation policy.
- Run targeted tests and checks.
- Produce structured result summary and artifact manifest.
- Avoid live gateway restarts unless explicitly authorized.
- Submit heartbeat/result/artifact metadata to the controller.

Policy distinction:

- Worker model/provider: Codex-first by default for Woo’s coding/review worker lanes; Claude Code may be used when explicitly requested, when Codex quota/auth/capability is exhausted, or when its subagent/team surface is the right fit.
- Worker substrate: currently Hermes oneshot subprocess-backed MVP; target state is a durable worker/job queue.

### 5.3 Background reviewer

Responsibilities:

- Review worker diff/output/test results against the original task packet and acceptance criteria.
- Detect incomplete, risky, unsafe, or low-quality worker results.
- Decide whether result is presentable, needs another worker iteration, should be blocked for user input, or should be rejected.
- Produce a machine-readable review verdict artifact.

Reviewer output fields:

- `verdict`: `pass`, `needs_iteration`, `blocked_user_input`, `reject`, or `unsafe`.
- risk/confidence level.
- required fixes or user questions.
- tests/checks expected before pass.
- import recommendation: `present_only`, `import_patch`, `ask_user`, or `discard`.

Default policy: every background job gets a review pass before user-facing completion. Read-only/low-risk skip policies can be optimized later from metrics, not initially.

## 6. Routing policy

### 6.1 Direct main handling

Main may answer directly when all are true:

- Expected work time is under ~30–60 seconds.
- No multi-file code edits.
- No long-running tests/builds.
- No high-risk external side effects.
- No complex research synthesis.
- The answer can be grounded in current context or one lightweight lookup.

Examples:

- Simple explanation.
- Short formatting/drafting.
- One quick factual lookup.
- Clarifying a previous answer.

### 6.2 Background worker delegation

Main should delegate when any are true:

- Expected work time exceeds ~1–2 minutes.
- More than ~2–3 tool calls are likely.
- Multiple files must be read/changed.
- Tests/builds need to run or iterate.
- The task produces an artifact/report/deck/code change.
- There is meaningful correctness risk.
- The task benefits from parallel exploration.
- User does not need to watch every step.

Examples:

- Code feature/fix/refactor.
- PRD-to-implementation review.
- Multi-source research.
- Repeated test/fix loops.
- Document/artifact generation.

### 6.3 Follow-up / steer routing

While a background worker is active:

- Explicit stop/cancel always interrupts/cancels immediately and never queues/replays.
- Follow-up should first be persisted as a task event, then delivered to the worker if possible.
- Short follow-up text may be attached to the active worker only when relation is clear or explicit.
- Ambiguous natural language must not be swallowed silently. Prefer a visible acknowledgement (`작업에 반영할게요`) or a short clarification (`진행 중인 작업에 붙일까요, 새 질문으로 답할까요?`).
- If real-time injection is impossible, mark the follow-up as `pending_for_next_attempt` or `pending_review_context` rather than dropping it.

### 6.4 Status routing

- Exact `/status`: status surface.
- Natural language status-like text: normal prompt.
- `/status anything else`: normal prompt or command-help error; do not treat substring as status.
- `/status` reads from durable controller state once Phase D is implemented; it must not say “no work” solely because in-memory runtime state is empty.

## 7. Task/job lifecycle

### 7.1 Task lifecycle — user-visible goal

```text
user prompt
  -> main triage
  -> direct answer OR durable task created
  -> worker job queued/running
  -> worker result submitted
  -> reviewer job queued/running
  -> reviewed result ready
  -> main presents summary / asks for import / reports blockers
  -> optional import/apply
  -> done_presented/imported/discarded
```

Required task states:

- `queued`
- `running_worker`
- `worker_done_pending_review`
- `running_review`
- `review_passed_pending_presentation`
- `review_passed_pending_import`
- `review_failed_needs_iteration`
- `blocked_user_input`
- `cancel_requested`
- `cancelled`
- `done_presented`
- `import_requested`
- `importing`
- `imported`
- `import_failed`
- `discarded`
- `error`

### 7.2 Job lifecycle — durable execution attempt

A task may own multiple jobs/attempts.

Job kinds:

- `worker`
- `reviewer`
- future: `importer`, `notifier`

Required job states:

- `queued`
- `claimed`
- `running`
- `succeeded`
- `failed`
- `cancel_requested`
- `cancelled`
- `interrupted`
- `lease_expired`
- `recovering`
- `needs_inspection`

Job metadata:

- task id.
- job id.
- kind.
- attempt number.
- priority.
- lease owner and lease expiry.
- process id/session id when applicable.
- heartbeat timestamp.
- exit status.
- artifact ids/paths.

## 8. Durable store and queue requirements

Target persistent store: SQLite in Hermes home or the existing task registry persistence layer, with a clear migration/version policy.

Minimum durable entities:

- `tasks`: task id, user goal, session/chat/user origin, state, created/updated timestamps.
- `jobs`: job id, task id, kind, state, attempt, priority, lease owner, lease expiry, pid/session id, heartbeat, exit status.
- `events`: append-only state transitions/control events.
- `artifacts`: path, type, producer job id, checksum/size when available, created_at, import status.
- `followups`: text/media refs, arrival order, routing disposition, delivered-to-worker status.
- `control_requests`: cancel/steer/pause/resume/import decisions.

Transactional requirements:

- Enqueue task + first worker job atomically.
- Claim job with lease atomically.
- Heartbeat extends lease.
- Completion writes result artifact + state transition atomically.
- Duplicate completion submissions are idempotent.
- Worker success queues reviewer exactly once.

Restart/recovery requirements:

- Running PID alive: reconnect/observe or keep running with heartbeat reconciliation.
- PID gone with artifact: transition to worker done or review queued if artifact is valid.
- PID gone without artifact: mark interrupted/requeue/error according to policy.
- `worker_done_pending_review` on restart: enqueue reviewer exactly once.
- `review_passed_pending_presentation` on restart: keep pending presentation visible.

## 9. Background acknowledgement UX

When main delegates, it should return quickly with a compact acknowledgement:

```text
좋아요. 이건 백그라운드로 넘길게요.
- 작업: frontdesk PRD 전반 리뷰/수정/테스트
- 실행: Codex worker
- 검수: review gate 후 가져올게요
- 상태: /status
```

Avoid implying completion until review has passed.

## 10. Result presentation UX

When reviewed result is ready, main presents:

- What changed / what was found.
- Reviewer verdict.
- Tests/checks run.
- Risks/blockers.
- Whether anything is pending import/apply.
- Proposed next action.

Example:

```text
백그라운드 작업 검수까지 끝났어요.
- 결과: /status-only 정책이 CLI/Gateway/TUI 테스트까지 반영됨
- 검수: review pass
- 테스트: targeted suites passed
- 남은 리스크: durable worker lane은 아직 subprocess/in-memory라 restart-safe 아님
다음은 durable queue 설계로 넘어가면 됩니다.
```

## 11. `/status` requirements

`/status` should show only explicit operational state, not be triggered by natural text.

Minimum fields:

- Active background jobs.
- Worker/reviewer stage.
- Task/job id short handle.
- Attempt number.
- Start time / elapsed time.
- Last heartbeat / last log excerpt.
- Pending follow-ups.
- Blocked user-input requests.
- Completed reviewed items not yet presented.
- Pending import/apply items.
- Cancel/steer handles if available.
- Artifact/review/import status.

## 12. Isolation and import policy

Each worker task must declare an isolation mode:

- `read_only_research`
- `artifact_only_output`
- `shared_worktree_edit_allowed`
- `isolated_git_worktree`

Default policy:

- Code changes should use isolated worktree/branch unless the user explicitly authorizes shared worktree edits.
- Live gateway restart is forbidden unless explicitly authorized and should normally be final-step only.
- Dirty worktree conflicts should fail fast, use an isolated worktree, or ask for explicit approval.
- Artifact manifests should include path, type, producer, and checksum/size when practical.
- Import gate must verify provenance and allowed destination.

## 13. Implementation phases

### Phase 0 — stabilize current live MVP / PRD sync

Status: in progress in current branch.

Goals:

- Keep `/status` exact-only.
- Preserve live worker ack and review-pending behavior.
- Align PRD with current Phase B/C scaffold work.
- Split/commit dirty diff by meaning before bigger durable work.

Acceptance:

- Targeted tests pass.
- Natural-language status-like text is normal prompt when idle.
- Worker completion is review pending, not final completion.
- PRD current-state notes match `git status` and summary artifacts.

### Phase 1 — Controller API + durable data model

Goals:

- Introduce task/job/event/artifact/follow-up schema.
- Add SQLite-backed controller store or equivalent persistence.
- Route create/status/cancel/steer through controller API.

Acceptance:

- Task/job state persists across controller reload.
- Enqueue task + worker job is atomic.
- `/status` can read restored state.
- Corrupt/missing DB handling is tested.

### Phase 2 — durable worker lane

Goals:

- Worker job claim/lease/heartbeat/complete.
- Durable process/session/artifact metadata.
- Expired lease/restart recovery/requeue semantics.

Acceptance:

- Running worker during controller restart is reconciled.
- Duplicate completion is idempotent.
- Worker crash records error artifact/exit status and retryable/error state.
- Cancel request is persisted and delivered/escalated.

### Phase 3 — durable reviewer lane

Goals:

- Worker success automatically queues reviewer job.
- Reviewer prompt includes task packet, worker summary, diff/tests, artifacts, pending follow-ups, and acceptance criteria.
- Reviewer writes machine-readable review artifact.
- `needs_iteration` can enqueue a bounded worker iteration or block for user input.

Acceptance:

- Worker success does not produce final completion.
- Reviewer pass transitions to presentation/import-ready state.
- Reviewer fail/needs-iteration transitions correctly.
- Max-iteration/budget escalation produces user-visible blocker.

### Phase 4 — review/import UX surface

Goals:

- Add explicit task/review/import/discard/steer/stop command surface across CLI/Gateway/TUI/Telegram.
- Keep present-summary and import/apply separate.
- Provide Telegram-safe formatting.

Acceptance:

- Reviewed artifact is not auto-applied.
- `/status` shows pending presentation/import.
- User can inspect review verdict and artifacts.
- Import approval applies only verified artifacts to allowed destination.
- Discard marks terminal state without applying.

### Phase 5 — Telegram live E2E + concurrency hardening

Goals:

- Prove actual always-available UX under gateway/session locks and platform callbacks.
- Verify multi-session isolation, duplicate notification protection, stop latency, and ambiguous follow-up handling.

Acceptance:

- Long worker starts from Telegram and ack is sent quickly.
- While worker runs, `/status`, `/stop`, and unrelated prompt are handled.
- Natural-language status-like prompt is normal prompt when idle.
- Ambiguous short message during worker gets visible attach acknowledgement or clarification.
- Worker result notification says review pending, not final done.
- Reviewer pass creates pending presentation/import item.
- Restart/reconnect preserves durable state.

### Phase 6 — policy tuning / metrics / rollout

Goals:

- Tune delegation threshold, review skip policies, retry limits, and SLOs using metrics.
- Add feature flags and safe degraded mode.

Acceptance:

- Metrics/logs expose foreground blocking time, delegation rate, cancel latency, review failure rate, iteration count, import acceptance rate, restart recovery success.
- Feature flag can disable durable frontdesk runtime safely.
- Degraded mode never pretends a worker started if controller is unavailable.

## 14. Acceptance criteria

1. Main can receive and respond to new user messages while a long worker job runs.
2. User natural-language prompts are not consumed by status heuristics.
3. Exact `/status` shows current background job state.
4. Stop/cancel never queues/replays and takes precedence.
5. Background code/artifact work is reviewed before completion is presented.
6. Tests cover CLI, Gateway, TUI, and Telegram-like/live E2E paths.
7. Gateway restart is not needed for ordinary validation.
8. Durable task/job/review state survives controller/gateway restart.
9. Worker completion cannot mark a task `done_presented` without review pass.
10. Import/apply requires explicit user approval.
11. Cross-session status/stop/follow-up isolation is preserved.

## 15. Test plan additions

### Durable store tests

- `test_task_job_event_persist_across_restart`
- `test_enqueue_task_and_worker_job_atomic`
- `test_duplicate_completion_is_idempotent`
- `test_expired_lease_transitions_to_recovering_or_requeue`

### Worker lifecycle tests

- `test_worker_success_queues_reviewer_not_done`
- `test_worker_cancel_persists_and_does_not_replay_stop_text`
- `test_worker_crash_records_error_artifact_and_retryable_state`
- `test_followup_persisted_before_delivery_to_worker`

### Reviewer tests

- `test_reviewer_pass_creates_pending_import_or_presentable_state`
- `test_reviewer_needs_iteration_enqueues_worker_attempt_2`
- `test_reviewer_blocked_sets_awaiting_user_input`
- `test_unreviewed_worker_result_never_presented_as_done`
- `test_review_artifact_schema_rejects_invalid_verdict`

### Import gate tests

- `test_import_requires_explicit_user_approval`
- `test_present_summary_does_not_apply_patch`
- `test_import_patch_checks_dirty_worktree_policy`
- `test_discard_reviewed_artifact_marks_terminal_without_apply`
- `test_import_failure_leaves_artifact_available_and_reports_blocker`

### Control-plane tests

- `test_exact_status_only`
- `test_status_substring_or_natural_language_routes_to_main_when_idle`
- `test_stop_preempts_status_followup_and_worker`
- `test_steer_with_task_id_routes_to_correct_task`
- `test_ambiguous_short_text_during_worker_gets_visible_ack_or_clarification`

### Frontdesk availability / Telegram E2E tests

- `test_long_worker_does_not_block_new_main_prompt`
- `test_status_fast_while_worker_running`
- `test_stop_fast_while_worker_running`
- `test_typing_indicator_does_not_hold_session_lock`
- `test_unrelated_prompt_while_worker_running_uses_main_path`
- Telegram-like flow: start long worker -> quick ack -> `/status` -> unrelated prompt -> short steer/clarification -> `/stop` -> no replay.
- Completion flow: worker done -> review pending notification -> reviewer pass -> pending presentation/import -> explicit present/import.

## 16. Open decisions

1. Should every background job always require review, or can read-only/low-risk background searches skip review?
   - Recommended default: always review initially; optimize later from metrics.

2. What exact command names should be used for task/review/import controls?
   - Recommended: define semantics first, then fit names to existing Hermes command registry.

3. How should ambiguous short text during an active worker be handled?
   - Recommended: attach only when relation is clear; otherwise visible acknowledgement or clarification.

4. What is the first durable store?
   - Recommended: SQLite in Hermes home or existing task registry persistence layer.

5. Should Spark ever present an unreviewed background result?
   - Recommended: no, except clearly marked progress/log excerpts.

6. What is the reviewer model/provider pinning policy?
   - Recommended: Codex-first for Woo’s default worker/review lanes; Claude Code as explicit/overflow/specialized lane.

7. What is the dirty-worktree import policy?
   - Recommended: isolated worktree by default for code changes; shared-worktree import only with explicit approval.

## 17. Current implementation status — 2026-05-15

Working tree:

`/private/tmp/hermes-frontdesk-live-wiring-20260514-141525`

Current observed state:

- Live MVP is mostly implemented and running in the live-wiring worktree.
- Phase A is mostly implemented and covered by targeted CLI/Gateway/TUI/policy tests.
- Phase B/C scaffold is partially implemented:
  - task metadata includes frontdesk lifecycle / worker-reviewer stage metadata.
  - review result artifact schema exists.
  - `/status` can display worker/reviewer stage and artifact metadata.
  - worker completion no longer implies final user-facing completion; successful worker output remains pending review.
  - default worker lane is Hermes oneshot subprocess-backed, not a durable queue.
- Latest focused validation reported in-session: `493 passed, 8 warnings`, `py_compile` passed, `git diff --check` passed.
- Previous Codex summaries also report targeted suites passing.
- Current branch still has dirty diff and untracked Codex summary artifacts; commit boundaries need cleanup before large durable work.

Not yet implemented / not yet proven:

- SQLite-backed durable worker/reviewer queue.
- Actual reviewer subprocess/lane execution.
- Restart-safe recovery/requeue and lease semantics.
- First-class review/import/presentation command surface.
- Full Telegram live E2E acceptance.
- Full repository test suite.

## 18. Next implementation task packet

Recommended next Codex/Claude task after PRD sync and diff cleanup:

> Implement Phase 1/2 foundation for the always-available frontdesk runtime: introduce a durable controller store with task/job/event/artifact schema, atomic enqueue/claim/heartbeat/completion helpers, and restart-simulation tests. Preserve the invariant that worker completion never becomes final presentation before review pass. Keep live gateway restart, broad policy tuning, and import/apply commands out of scope for this first durable foundation task.

Required tests:

- Task/job persistence round trip.
- Enqueue task + worker job atomicity.
- Worker success -> review queued, not done.
- Duplicate completion idempotency.
- `/status` displays persisted worker/reviewer state and artifact paths.
- Natural-language status-like text remains normal prompt.
- Stop/cancel precedence unchanged.

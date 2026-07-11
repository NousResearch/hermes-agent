# HERMES-ORCHESTRATOR Local Capability Inventory

**Inventory date:** 2026-07-10  
**Scope:** Read-only inspection of the locally installed Hermes runtime, its active configuration/profiles/boards, the `feature/hermes-obs-001` worktree, and relevant tests. This document is the only repository change.  
**Desired outcome:** Hermes should accept one project outcome, autonomously create and run a bounded task graph, preserve Telegram control/notifications, route checker failures to repair, and return a verified final report with a compact usage summary—without the user acting as the handoff mechanism.

## Executive conclusion

Hermes already has a strong durable execution substrate: SQLite boards, dependency-gated promotion, atomic claims, per-attempt run rows, heartbeats, stale-claim recovery, PID/crash detection, runtime limits, per-profile concurrency, goal-mode judging, notification subscriptions, and embedded gateway dispatch. The foundation and concurrency boards prove that local profiles can run a dependency graph and that the dispatcher can execute multiple profiles.

It does **not** yet provide the requested end-to-end orchestrator contract as a machine-enforced local capability. The smallest critical gaps are:

1. **The OBS-001 implementation is not active in the installed/running runtime.** The running gateway uses installed `main` at `540f90190f50f9518bf36632a724e0e58877a10b`; the verified ledger is only in the separate worktree at `eb2adffe356b873602ee060a31fe0f0640859407`. Installed `main` has no `hermes_cli/kanban_usage_ledger.py`, and the feature board contains zero real `run_usage` rows.
2. **Task contracts are prose, not admission controls.** One-run scope, allowed files, base commit, required evidence, notification verification, no child creation, and Git prohibitions are not first-class task fields and are not enforced before dispatch or at completion.
3. **Notification subscriptions are per task and are not inherited by children.** All inspected orchestration boards currently have zero subscription rows. Therefore autonomous child work is not guaranteed to notify the controlling Telegram chat.
4. **Telegram delivery is cursor-based best effort, not an auditable delivery ledger.** The notifier advances/rewinds a cursor and logs exceptions, but does not persist send acceptance, returned message ID, attempt count, or terminal delivery error.
5. **The worker prompt contradicts autonomous success.** It instructs all workers to call `kanban_complete`, then tells most coding tasks to `kanban_block(review-required)`. That policy caused successful OBS-001 builders to block and require manual completion.
6. **Provider errors are usually lost at the task boundary.** The OBS checker HTTP 429 appeared in worker logs, while `task_runs.error` recorded only `pid 24924 not alive`. A newer rate-limit sentinel path exists in the inspected feature source, but the exact provider error body/status is still not persisted as the run failure reason.
7. **Final completion does not call the OBS-001 aggregator.** `query_usage()` and `aggregate_usage()` exist as Python library functions only; no completion/final-report hook appends a compact summary.
8. **Checker self-improvement is enabled locally.** The checker profile has memory enabled, skill nudging enabled, curator enabled, and the general skill-improvement prompt is injected. There is no single checker/verification-mode switch that disables all mutation-capable self-improvement behavior.

Accordingly, the current system is a capable multi-agent queue that still depends on careful manual task authoring, manual notification setup, manual success/review intervention, and manual final synthesis. It is not yet a policy-enforced, end-to-end project orchestrator.

## 1. Exact installed and inspected state

### Installed runtime

- CLI: `C:\Users\fallo\AppData\Local\hermes\hermes-agent\venv\Scripts\hermes`
- Reported version: `Hermes Agent v0.18.2 (2026.7.7.2)`
- Reported upstream commit: `5e849942`
- Installed local source commit: `540f90190f50f9518bf36632a724e0e58877a10b`
- Installed branch: `main`
- Installed `main` relation: one local carried commit ahead of and one commit behind `origin/main`.
- Running gateway: PID `40588`, manual process; Windows Scheduled Task exists but its last result was `-1073741510`.
- Dispatcher: embedded in the gateway, singleton lock acquired, 60-second interval.
- Active gateway surfaces: Telegram and API server.

### Verified OBS-001 documentation worktree

- Worktree: `C:\Users\fallo\AppData\Local\hermes\worktrees\hermes-obs-001`
- Branch: `feature/hermes-obs-001`
- Starting documentation HEAD: `eb2adffe356b873602ee060a31fe0f0640859407`
- Worktree was clean before this report.
- `eb2adffe...` is **not** an ancestor of installed `main`.
- The installed source and feature source hashes differ at every OBS runtime boundary inspected; installed `main` lacks `hermes_cli/kanban_usage_ledger.py` entirely.

### Important runtime/config distinction

`hermes status --all` reported the live/default model as `gpt-5.6-sol` through `openai-codex`. A later direct disk read showed the default profile's `config.yaml` model as `tencent/hy3:free` through `nous`. The gateway was not restarted during this inventory, so this is configuration/runtime drift, not a resolved state. A future acceptance test must record both the gateway-loaded profile and on-disk profile and fail on mismatch.

### Profiles present

- `default`
- `builder-deepseek`
- `builder-grok`
- `builder-qwen`
- `checker`
- `research`
- `wiki-curator`

The checker currently uses `zai` / `glm-5.2` with no fallback. Builder/research/wiki profiles still have a Codex fallback configured. Every inspected specialist profile has memory enabled, skill-creation nudging at 15 iterations, and curator enabled.

### Active Kanban configuration

From `C:\Users\fallo\AppData\Local\hermes\config.yaml`:

- `kanban.dispatch_in_gateway: true`
- `kanban.dispatch_interval_seconds: 60`
- `kanban.failure_limit: 2`
- `kanban.max_in_progress: 3`
- `kanban.auto_promote_children: true`
- `kanban.dispatch_stale_timeout_seconds: 14400`
- `kanban.auto_decompose: true`
- `kanban.orchestrator_profile: ""`
- `kanban.default_assignee: ""`

The current board is `canned-run`, not an orchestrator project board.

## 2. What is already present and working

### Durable board and run history

The schema already persists:

- task identity, title/body, assignee, status, priority, creator, timestamps;
- workspace kind/path, branch, project, tenant and idempotency key;
- claim lock/expiry, worker PID, heartbeat, current run;
- failure streak, last failure, runtime cap and per-task `max_retries`;
- goal mode/session identity;
- dependency links, comments, events, attachments and run attempts.

`task_runs` stores profile, attempt status, claim, PID, runtime cap, heartbeat, start/end, outcome, summary, metadata, and error.

### Dependency behavior and automatic promotion

`recompute_ready()` promotes `todo` tasks only after every parent is `done` or `archived`; sticky worker/operator blocks and exhausted failure limits remain blocked. `claim_task()` re-checks parents atomically and demotes a wrongly-ready task to `todo`, preventing a race from bypassing dependencies.

Evidence:

- `hermes_cli/kanban_db.py:3372-3456`
- `hermes_cli/kanban_db.py:3463-3503`

### Claims, heartbeats, stale-worker handling and retries

- Claims are compare-and-swap transitions from `ready` to `running` with a unique claim lock and expiry.
- Each successful claim creates a distinct `task_runs` row.
- `kanban_heartbeat` extends claim TTL and records worker liveness.
- The dispatcher detects dead PIDs, timed-out workers and stale claims, closes the corresponding run, releases/requeues where appropriate, increments a unified consecutive-failure counter, and trips a circuit breaker at the effective limit.
- Clean exit without `kanban_complete`/`kanban_block` is treated as a deterministic protocol violation and blocks immediately.
- A recognized rate-limit sentinel is requeued without incrementing task failures.

Evidence:

- `tools/kanban_tools.py:746-794`
- `hermes_cli/kanban_db.py:6450-6631`
- `hermes_cli/kanban_db.py:6634-6677`

### Worker lifecycle tooling

Dispatcher-spawned workers receive focused Kanban tools and runtime identity (`HERMES_KANBAN_TASK`, board, run, claim and workspace). The tool layer enforces that lifecycle mutations target the worker's own task. `kanban_complete` validates claimed child-card IDs, and a clean process exit without a terminal lifecycle call is detected.

### Goal-mode supervision exists

`--goal` runs a worker in a multi-turn goal loop with an auxiliary judge, then nudges a worker to call `kanban_complete` when the output appears complete. This is the only LLM-based post-dispatch supervision found. Ordinary dispatcher polling, PID checks, dependency promotion, notification polling and stale recovery are deterministic Python/SQLite operations, not LLM calls.

Goal mode is available but was not enabled on the observed orchestration/OBS tasks (`goal_mode=0` in the boards inspected).

### Local multi-profile execution has been exercised

- Foundation board: 13 tasks done across default, three builders, research, checker and wiki-curator.
- Concurrency canary: six tasks done and gateway logs show two waves of three spawns under `max_in_progress=3`.
- OBS board: 26 run attempts preserve checker failures, worker blocks, crashes and final PASS history.

This proves the queue and profile spawning path, but not the desired no-human-handoff project lifecycle.

### OBS-001 ledger implementation is complete in the feature worktree

In the feature worktree, the ledger records primary, Codex app-server and auxiliary calls and provides `query_usage()` and `aggregate_usage()`. Its focused suite previously passed 101 tests under independent checking. This is working source on the feature branch, but it is not active in the installed gateway and has not captured live board usage.

Evidence:

- `hermes_cli/kanban_usage_ledger.py:427-559` (`aggregate_usage`)
- `hermes_cli/kanban_usage_ledger.py:560+` (`query_usage`)
- `tests/hermes_cli/test_kanban_usage_ledger.py`
- `docs/evidence/HERMES-OBS-001-REPORT.md` / `C:\Users\fallo\HERMES-OBS-001-REPORT.md`

## 3. Present but misconfigured or not enabled locally

### OBS-001 is not in the active runtime

This is the largest current configuration/deployment gap. The verified branch is not the source used by the running gateway. The active board DBs therefore have no live ledger events; the OBS board's `run_usage` table exists but contains `0` rows.

### No orchestrator profile is configured

`kanban.orchestrator_profile` and `kanban.default_assignee` are empty. Auto-decompose is enabled, but no dedicated orchestrator profile is selected to own a parent project outcome and its final synthesis.

### No notification subscriptions are installed

`kanban_notify_subs` exists on the foundation, concurrency and OBS boards, but every inspected table is empty. `hermes kanban ... notify-list --json` also returned `[]`. The capability exists, but the controlling Telegram chat is not subscribed to any observed task.

### Goal mode is unused

The goal-loop/judge and completion nudge exist upstream but the observed tasks were created without `--goal`. This avoids judge cost but also leaves one-shot workers governed only by the base prompt and task prose.

### Checker self-improvement is enabled

The checker profile is not isolated as a read-only proof boundary at the agent-policy level:

- `memory.memory_enabled: true`
- `skills.creation_nudge_interval: 15`
- `curator.enabled: true`
- global `SKILLS_GUIDANCE` tells agents to create or patch skills after difficult work (`agent/prompt_builder.py:180-186`)

The checker SOUL correctly says it is not a co-author, but that does not disable memory/skill mutation tools or background review. Local profile policy can reduce this by disabling memory and skill nudging/curator and restricting checker toolsets. A single explicit `verification_mode`/`self_improvement_enabled=false` enforcement switch does not exist.

### Profile/runtime drift is possible

The on-disk default model differed from `hermes status --all` during the same inventory. This is expected when config changes are not followed by restart/reset, but it means task admission currently cannot prove which exact profile/model config the gateway will spawn.

## 4. What is missing in code

### Machine-enforced task admission contract

The following desired fields are absent from the task schema and create CLI:

- one-run scope / exact deliverable unit;
- allowed files and forbidden files;
- required base commit;
- required commands/evidence/artifacts;
- notification-subscription verified flag;
- worker may/may-not create child tasks;
- no push/merge/amend (and broader remote/destructive action policy).

The closest existing fields are `workspace_kind`, `branch_name`, `project_id`, `idempotency_key`, `max_runtime_seconds`, `skills`, `max_retries`, and `goal_mode`. These bound execution but do not enforce scope. Today all listed controls must be encoded in body prose and checked after the fact.

Required implementation shape: a versioned task-contract object persisted with the card, validated before transition to `ready`, injected verbatim into worker/checker context, and checked again at completion against Git/diff/evidence/subscription state. Dispatcher refusal must be deterministic and auditable.

### Subscription inheritance

`add_notify_sub()` inserts a row for one exact `task_id`; task creation does not copy parent subscriptions. No `inherit notifications` path was found. Orchestrator-created children therefore start unsubscribed unless the orchestrator or a human explicitly subscribes each one.

Evidence:

- `hermes_cli/kanban_db.py:8354-8387`
- `kanban_notify_subs` primary key includes task ID and destination.

Required implementation shape: on child creation, copy active parent/project subscriptions transactionally (or create a project-level subscription referenced by all descendants), then make `notification_verified` an admission requirement.

### Auditable Telegram delivery outcomes

The notification table stores only destination metadata and `last_event_id`. The watcher calls `await adapter.send(...)`, logs success/failure, rewinds the cursor for transient failures, and deletes the subscription after repeated failures. It discards the adapter's returned `SendResult`; it does not persist:

- send accepted/success boolean;
- platform message ID;
- attempt count;
- last delivery timestamp;
- full redacted error and error class;
- terminal dropped-subscription reason.

Evidence:

- schema: `hermes_cli/kanban_db.py:1254+`, `8354-8529`
- send path: `gateway/kanban_watchers.py:408-476`

Required implementation shape: append-only `kanban_notification_deliveries` rows keyed by board/task/event/subscription/attempt, storing acceptance, message ID and redacted error. Cursor advancement must correspond to a persisted accepted send.

### Direct provider failure capture

The feature code can distinguish a special rate-limit exit code and avoid counting it as a task failure. However, ordinary crashes still record PID/exit status, and the historical OBS checker 429 proves the provider error was not copied into `task_runs.error`:

- run 23: `task_runs.error = "pid 24924 not alive"`
- worker log: OpenAI Codex HTTP 429 `usage_limit_reached`

Required implementation shape: worker writes a small redacted terminal-failure envelope before exit (provider, model, HTTP status/error code, retry-after/reset, terminal exception summary). Dispatcher must prefer this envelope over PID death and persist it in run error/metadata. Do not persist credentials or response bodies.

### Automatic project finalization with usage summary

The ledger's query/aggregate functions are not exposed through the Kanban CLI/toolset and are not called from `complete_task`, notifier wakeups, synthesis, or project completion. There is no parent-project completion primitive that verifies all required descendants/checker results and appends a usage summary.

Required implementation shape: after final checker PASS, deterministic project finalization queries the ledger for the parent/descendant run set and appends a compact summary: API calls; input/output/cache/reasoning/accepted-result tokens; auxiliary totals; known cost and unknown-cost call count. The report must explicitly say when the active runtime lacked instrumentation or rows are incomplete.

### End-to-end checker/repair routing contract

Hermes has links, sticky blocks, auto-promotion and durable checker output, but there is no generic kernel-level rule that interprets checker outcome `FAIL — repairable`, creates/reuses a bounded repair card, copies the exact findings, then creates a fresh checker cycle. This was manually orchestrated during OBS-001.

A future workflow template can initially implement this without a new model tool, but the transition and deduplication rules must be machine-owned rather than left to worker prose.

### Verification-mode isolation

There is no one-switch mode that guarantees a checker cannot mutate production files, profiles, memory, skills, task graph or remotes. Toolset restriction can approximate this locally, but a first-class verification execution mode should enforce read-only file/terminal behavior and disable memory/skill background review.

## 5. What needs only task/profile policy change

These do not require kernel work, although machine enforcement would be stronger:

1. **Success completion policy:** remove the generic coding exception at `agent/prompt_builder.py:226-233` for orchestrated chains. Builders should call `kanban_complete` when their scoped acceptance criteria pass. Review is represented by a dependent checker card, not by blocking a successful builder.
2. **No worker-created children:** remove `kanban_create` from builder/checker toolsets or add a profile policy denying it; only the orchestrator profile owns graph creation.
3. **Checker profile hardening:** disable memory, skill creation nudging and curator; remove mutation tools; use a read-only SOUL/toolset and no provider fallback unless explicitly approved.
4. **Builder briefs:** explicitly state one coherent change, allowed files, base commit, exact tests/evidence, one local commit, no push/merge/amend, complete on success, block only on a genuine unmet requirement.
5. **Checker briefs:** include original task contract, exact base/candidate, exact diff, builder report, commands, evidence and acceptance criteria. Checker returns one normalized outcome and never repairs.
6. **Final synthesis policy:** a dedicated orchestrator profile reads only verified task/run state and emits one consolidated project report; users are never asked to relay worker/checker text.

The current universal worker prompt is internally inconsistent: lines 220-225 require completion, lines 226-233 instruct most coding workers to block for review, and lines 234-237 invite any worker to create follow-up children. For the desired workflow, these must be profile-/contract-sensitive.

## 6. Exact meaning of `max_retries`

`max_retries` is named like “number of retries” but implemented and documented as the **failure count at which the circuit breaker trips**:

- `max_retries=1`: first failure makes `consecutive_failures == 1`; threshold is reached; task blocks; **zero retry attempts**.
- `max_retries=2`: first failure requeues; second failure blocks; **one retry attempt**.
- `max_retries=3`: two requeues; third failure blocks; **two retry attempts**.
- omitted/NULL: use dispatcher `kanban.failure_limit`, locally `2`.

Evidence:

- `hermes_cli/kanban_db.py:884-890`
- `hermes_cli/kanban_db.py:1145-1149`
- `hermes_cli/kanban_db.py:6673-6677`
- `hermes kanban create --help`

This explains why the OBS checker with `max_retries=1` did not retry after HTTP 429. The field should eventually be renamed to `failure_limit` or have a separate `retry_attempts` field; until then, task authors must treat N as “block on Nth failure.”

## 7. Context injected into sessions

### All sessions

The prompt includes the large core system policy, tools, relevant skill catalog, project instructions, environment hints, and (when enabled) profile SOUL/memory/user profile. Project instruction files are capped at 20,000 characters each; SOUL is separate. Model context length is provider/model-derived because no inspected profile sets `model.context_length`.

### Kanban workers and checkers

Workers receive the universal `KANBAN_GUIDANCE` plus:

- task title/body;
- parent summaries and metadata;
- prior attempts on retry;
- full comments;
- worker context;
- task/board/run/workspace/branch environment identity;
- profile SOUL and enabled memory/skills;
- project context discovered from the workspace.

Evidence: `agent/prompt_builder.py:189-282` and the `hermes kanban context` command description.

Profile SOUL sizes observed:

- default: 5,411 characters
- builder-deepseek: 2,977
- builder-grok: 2,821
- builder-qwen: 2,478
- checker: 3,425
- research: 3,211
- wiki-curator: 3,582

The checker is not context-minimal: it gets the same universal lifecycle/self-improvement guidance plus its checker SOUL and task context. The desired acceptance test should capture prompt/context byte counts and token estimates from the spawned session, not infer them from config.

### Orchestrator

There is no locally configured dedicated orchestrator profile. A normal default-profile gateway session receives full user conversation, default SOUL, memory/user profile, skills and broad tool schemas. A dispatcher-spawned decomposition task also receives universal Kanban guidance. Consequently, “orchestrator context” is currently whichever default session/task happened to create the graph, not a stable audited profile contract.

## 8. Upstream/installed capabilities present but not enabled locally

- Goal-mode multi-turn worker plus auxiliary completion judge (`--goal`).
- Triage/specifier and LLM decomposition (`--triage`, `specify`, `decompose`; `kanban.auto_decompose=true` but no orchestrator owner configured).
- Swarm graph generation (parallel workers → verifier → synthesizer).
- Per-task idempotency keys.
- Project-linked deterministic worktrees/branches.
- Per-task runtime caps, skills and model overrides.
- Notification subscription CLI and gateway notifier.
- Diagnostic rules and board diagnostics.
- API server and Telegram surfaces.
- Multiple configured specialist profiles.
- OBS-001 ledger query/aggregation source on the feature branch.

These capabilities reduce the implementation needed, but they are not a substitute for admission enforcement, inherited notifications, delivery receipts, repair routing and final usage integration.

## 9. Gap classification

### Configuration gaps

- Running gateway is not using the verified OBS-001 feature commit.
- Empty `kanban.orchestrator_profile` and `kanban.default_assignee`.
- Zero Telegram notification subscriptions on inspected boards.
- Goal mode unused on observed tasks.
- Checker memory/skills/curator enabled.
- Potential live-vs-disk default-model drift.
- Current board is unrelated `canned-run`.

### Policy/prompt gaps

- Coding success is told to block for review.
- Any worker is told to create follow-up children.
- Builder/checker task contracts are prose and inconsistent between cards.
- No canonical parent-project/final-report policy.
- No enforced distinction between builder, checker and knowledge-curator powers.
- `max_retries` naming encourages incorrect operator expectations.

### Implementation gaps

- Contract/admission schema and pre-dispatch/completion gates.
- Parent/project notification inheritance.
- Persistent Telegram delivery acceptance/message ID/errors.
- Direct redacted provider failure envelope.
- Deterministic checker FAIL → repair → fresh checker transition.
- Project finalization primitive and OBS usage-summary integration.
- First-class verification/read-only mode and self-improvement disable.

### Test gaps

- No failure-injected end-to-end test covering Telegram origin through final project report.
- No test proving child subscription inheritance.
- No persistence test for Telegram acceptance/message ID/error.
- No real-provider/subprocess test proving HTTP 429 reaches `task_runs.error`.
- No admission rejection matrix for missing contract fields/forbidden operations.
- No test proving checker self-improvement/mutation surfaces are unavailable.
- No active-runtime test proving OBS rows exist for normal, Codex and auxiliary calls.
- No final-report test proving descendant usage aggregation is correct and non-duplicating.
- OBS-001 tests are local-only; no CI gate was identified for them.

## 10. Prioritized backlog

### P0 — Make repository/runtime truth explicit

1. Establish one exact runtime candidate commit containing verified OBS-001 plus the current intended upstream/local baseline.
2. Start acceptance only from a clean isolated install/worktree; record branch, commit, binary source path, loaded profile configs and gateway PID.
3. Prove one real Kanban call writes a nonzero ledger row before building final-summary behavior.

**Acceptance:** active process source commit equals candidate; installed and worktree file hashes match for ledger boundaries; one live board has a queryable usage event.

### P1 — Machine-enforced task contract and admission

Persist and validate versioned fields for one-run scope, allowed/forbidden files, base commit, required evidence/commands, notification verified, child-creation permission and forbidden Git/remote operations. Refuse dispatch when any required field is missing or mismatched.

**Acceptance:** table-driven tests reject each omitted/invalid field; no worker process starts; an auditable admission event names the exact reason.

### P1 — Notification continuity and delivery receipts

Add project/parent subscription inheritance and an append-only delivery-attempt ledger storing acceptance, message ID and redacted errors.

**Acceptance:** creating a child under a subscribed parent yields a verified inherited subscription in the same transaction; successful Telegram fake returns and persists a message ID; injected failures persist errors and retry without duplicate accepted sends.

### P1 — Correct success/checker/repair lifecycle

Remove review-block policy from builders in orchestrated chains; prohibit builder/checker child creation; normalize checker outcome metadata; deterministically route repairable FAIL to one repair card and a fresh checker cycle with deduplication.

**Acceptance:** successful builder completes without human unblock; checker FAIL creates exactly one repair card; repair completion releases a new checker; old FAIL evidence remains immutable; PASS releases finalization.

### P1 — Provider failure fidelity

Persist a redacted terminal failure envelope from worker to dispatcher.

**Acceptance:** injected HTTP 429 appears directly in `task_runs.error/metadata` with provider/model/status/code/retry timing, while credentials/body content are absent; PID death remains secondary evidence.

### P2 — Checker isolation

Add profile/verification-mode enforcement that disables memory/skill review, graph mutation, production writes and remote/destructive operations.

**Acceptance:** checker schemas exclude mutation tools; attempted file/config/memory/skill/task creation fails deterministically; read-only verification commands still work.

### P2 — Final project report and usage summary

Add a deterministic finalizer that requires final checker PASS, aggregates all descendant usage once, and appends the compact usage summary.

**Acceptance:** report contains goal, changed/verified/failed/current/blockers/decisions/evidence/next plus API-call and token/cost totals; multi-parent events are not double-counted; incomplete cost/token sources are labeled.

### P2 — CI and full E2E gate

Run the failure-injected test below on Windows and one POSIX environment with isolated `HERMES_HOME` and fake Telegram/provider endpoints.

## 11. Smallest next implementation milestone

**Milestone: Enforced one-run task admission with inherited, verified notification subscription.**

Why this is smallest and first:

- It prevents unsafe or unobservable work from starting.
- It uses the existing task DB, dependency graph and notifier rather than adding a new orchestrator engine.
- It eliminates the current race where an assigned `ready` task can run before subscription.
- It creates the durable contract that later checker/repair/finalizer code can consume.

Minimum scope:

1. Add a versioned task-contract JSON field or normalized contract table.
2. Require: scope, allowed files, base commit, evidence, child-creation policy, forbidden Git actions and `notification_verified`.
3. Copy parent/project subscriptions transactionally when creating children.
4. Keep new runnable tasks blocked until contract and inherited subscription validate.
5. Add admission events and focused DB/CLI/tool tests.

Explicitly defer from this milestone: dashboard, OBS-002, automatic repair routing, provider envelope, final usage report UI, remote operations.

## 12. Proposed failure-injected end-to-end acceptance test

Run in an isolated temporary `HERMES_HOME`, temporary Git repository, fake OpenAI-compatible provider and fake Telegram adapter. Do not use the user's real gateway/config/board.

### Setup

1. Candidate runtime starts one gateway/dispatcher from a recorded commit.
2. Create profiles: orchestrator, builder, checker, repair-builder, wiki-curator.
3. Disable checker self-improvement and mutation surfaces.
4. Subscribe the controlling Telegram source to the parent outcome.
5. Submit one parent outcome requiring a two-file bounded change and exact tests.

### Injected lifecycle

1. **Admission failure:** first child omits `required_evidence`. Assert it remains blocked and no worker PID/run is created.
2. **Corrected admission:** orchestrator supplies the field. Assert inherited Telegram subscription exists before `ready`.
3. **Transient Telegram failure:** first notification send raises a synthetic transport error. Assert delivery-attempt row stores rejection/error, cursor rewinds, retry succeeds, accepted row stores fake message ID, and only one accepted message exists.
4. **Provider quota failure:** builder's first API call returns HTTP 429 with retry metadata. Assert run error records redacted provider/model/status/code directly, task requeues according to policy, and failure semantics match configured retry attempts.
5. **Builder success:** retry completes the allowed change, tests and one local commit; it calls `kanban_complete`, not block. Assert diff contains only allowed files, base commit matches, no remote Git action occurred, and worker created no children.
6. **Checker repairable FAIL:** checker detects an injected missing evidence artifact and returns normalized FAIL. Assert exactly one repair card is created/reused with the exact checker findings and one fresh checker depends on it.
7. **Repair success:** repair-builder changes only its allowed file/evidence, completes, and releases the new checker.
8. **Checker PASS:** new checker independently verifies full diff/tests/evidence and completes PASS. Assert checker performed no memory/skill/task/config/source writes.
9. **Knowledge extraction:** only after PASS, wiki-curator receives verified facts; injected unverified speculation is excluded.
10. **Finalization:** orchestrator/project finalizer emits one Telegram project report. Assert it includes exact commit/evidence/log paths and a compact OBS usage summary across parent and descendants with no multi-parent double count.
11. **Stale worker case:** in a separate child, suppress heartbeat and freeze the worker beyond stale policy. Assert deterministic reclaim/requeue, prior run closure and notification delivery without an LLM supervisor call.
12. **Protocol violation case:** worker exits 0 without terminal lifecycle call. Assert immediate protocol-violation block with exact reason.

### Pass criteria

- User supplies only the initial goal; no manual copy/paste, unblock, subscription or checker handoff occurs.
- Every spawned run has a valid admitted contract and verified notification route.
- Builder, checker, repair and finalizer powers are distinct and enforced.
- All failures are visible in task/run/delivery ledgers with actionable exact reasons.
- Final report is sent once, includes a Telegram message ID and matches repository truth.
- OBS totals equal independently calculated fake-provider usage; unknown fields remain labeled unknown.
- No push, merge, amend, remote settings change, dashboard creation or OBS-002 work occurs.

## 13. Evidence paths and reproducible commands

### Paths

- Installed source: `C:\Users\fallo\AppData\Local\hermes\hermes-agent`
- Feature worktree: `C:\Users\fallo\AppData\Local\hermes\worktrees\hermes-obs-001`
- Main config: `C:\Users\fallo\AppData\Local\hermes\config.yaml`
- Profiles: `C:\Users\fallo\AppData\Local\hermes\profiles\<profile>`
- Gateway log: `C:\Users\fallo\AppData\Local\hermes\logs\gateway.log`
- Boards: `C:\Users\fallo\AppData\Local\hermes\kanban\boards\<slug>\kanban.db`
- OBS evidence: `C:\Users\fallo\HERMES-OBS-001-REPORT.md`

### Commands used

```bash
hermes --version
hermes status --all
hermes gateway status
hermes profile list
hermes config path
hermes kanban --help
hermes kanban create --help
hermes kanban swarm --help
hermes kanban decompose --help
hermes kanban boards
hermes kanban assignees --json
hermes kanban --board hermes-obs-001-20260710-1544z diagnostics --json
hermes kanban --board hermes-obs-001-20260710-1544z notify-list --json

git -C C:/Users/fallo/AppData/Local/hermes/hermes-agent branch --show-current
git -C C:/Users/fallo/AppData/Local/hermes/hermes-agent rev-parse HEAD
git -C C:/Users/fallo/AppData/Local/hermes/hermes-agent status --short
git -C C:/Users/fallo/AppData/Local/hermes/hermes-agent worktree list --porcelain
git -C C:/Users/fallo/AppData/Local/hermes/worktrees/hermes-obs-001 branch --show-current
git -C C:/Users/fallo/AppData/Local/hermes/worktrees/hermes-obs-001 rev-parse HEAD
git -C C:/Users/fallo/AppData/Local/hermes/worktrees/hermes-obs-001 status --short
```

SQLite was opened with URI `mode=ro` to inspect `tasks`, `task_runs`, `task_events`, `kanban_notify_subs`, `run_usage`, and schema metadata. Configuration output was recursively redacted for key/token/secret/password-like fields.

## 14. Stop-state summary

### Already present and working

Durable boards/runs, dependencies, claims, heartbeats, stale/crash/timeout handling, automatic promotion, multi-profile dispatch, goal-mode judging, notification cursor/retry mechanism, checker profile, and verified OBS-001 source/tests on its feature branch.

### Present but misconfigured/not enabled

OBS-001 is not in the active runtime; no orchestrator profile; no task subscriptions; goal mode unused; checker self-improvement enabled; live/disk default-profile drift possible.

### Missing in code

Admission enforcement, subscription inheritance, auditable Telegram send outcomes, direct provider-error envelopes, generic checker-repair routing, checker verification mode, and final usage-summary integration.

### Policy-only changes

Complete successful builders instead of review-blocking; prohibit builder/checker children; harden checker profile; standardize builder/checker/finalizer briefs.

### Current exact state

No source/config/profile/gateway/board changes were made by this inventory. No implementation task, dashboard or OBS-002 work was created. The only intended repository change is this documentation file and its documentation-only commit on `feature/hermes-obs-001`.

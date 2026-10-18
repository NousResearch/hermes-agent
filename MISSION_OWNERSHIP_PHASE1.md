# Hermes Mission Ownership — Phase 1 hardening review

## Candidate identity and repository state

- Repository: isolated `hermes-mission-ownership` worktree; canonical X checkout was not edited.
- Branch: `codex/mission-ownership-phase1`.
- Starting candidate: `672658051d8e694804417c700c232ac27e81e61d` exists on this branch and was the clean `HEAD` at the start of this hardening pass.
- This report is written before the final hardening commit. The final revision and clean/dirty Git status must be recorded here immediately after that commit; see the final state section below.
- No live gateway was restarted, no profile configuration or plug-in was modified, and no live migration was run.

## Architecture and relationship to native Kanban

A mission is not always a Kanban task. A foreground request starts in a user session and must be protected before a dispatcher task exists. Therefore the mission record stores the root request, owner kind/id/generation, contract and revision, progress, recorded child task IDs, recovery data, validation receipts, and history. The native Kanban row is created only when root execution is handed to the background dispatcher. Its body carries the current mission packet; the mission row remains the authority for root ownership.

The candidate now makes creation of a new root Kanban card, ownership transfer, and triage-to-ready promotion one SQLite transaction. The task idempotency key is derived from mission ID and owner generation. Task-owner completion updates the mission and root task in the same transaction; ordinary summary-only Kanban completion is refused for an active mission root. Recorded child IDs must resolve to native Kanban tasks, recorded running children block transfer, and recorded child tasks must be `done` or `archived` before completion.

These checks prevent contradictory states along the recorded root/child links when writes pass through the candidate APIs. The child list is still a mission-side snapshot supplied by the owner, not a complete normalized Kanban relationship. Unrecorded child work and direct SQL/API paths that bypass these functions can still diverge. Full graph completeness is not proven.

## What is enforced

### Application-level controls

- Fred turn admission and the Telegram fast admission hook use a delivery identity when available. The persisted Telegram platform message ID is forwarded to shared admission. Duplicate delivery of the same identity returns the same mission; reuse of that identity with different request text is refused.
- Fallback identity uses a transport/session turn ID where available, namespaced by session and platform. It does not hash objective text. With no stable delivery/turn ID, reliable deduplication is unavailable; admission does not merge unrelated sessions by prompt similarity. A new session must name a mission ID through the explicit resume operation to request transfer.
- An existing session's materially different follow-up is refused for explicit steering; a correction updates the existing contract revision. After a mission is completed, a later delivery, even with identical words, can open a new mission.
- Every tool dispatch routed through the shared Fred executor acquires a durable SQLite execution lease before calling downstream execution middleware and releases it in `finally`. Handoff, recovery, and completion refuse while another lease or a recorded running child exists. This covers calls that enter that shared wrapper, including time spent in its middleware and downstream handler.
- The lease is a drain mechanism, not cancellation. An interrupted Python operation releases it only if `finally` executes. A hard worker exit leaves a durable lease and blocks handoff/recovery: fail closed, requiring explicit operator repair. No lease-expiry guessing is used.
- Completion requires a trusted stored validation receipt bound to mission ID, owner generation, contract revision/hash, reviewer identity, validator module identity, checkout revision, artifact paths and hashes, and check results. It re-hashes artifacts and rejects stale or mismatched receipts. The model cannot submit a command or exit code as evidence.
- The narrow validator runs fixed Python syntax parsing and Ruff checks; it does not accept shell commands. It is not a general semantic reviewer or proof that functional criteria pass.
- New root task creation/transfer and task-owner mission completion/native task completion are transactionally coupled through the candidate database APIs.

### Not a security boundary

All of the above are Hermes application-level controls. They do not constrain arbitrary in-process code, direct SQL writers, a second Hermes build without these checks, or tools dispatched outside the shared executor. A subprocess, browser action, remote API operation, or delegated worker that outlives the dispatch handler may continue after the lease is released. Without OS-level process groups/cgroups or equivalent cancellation and join semantics for every execution backend, the candidate cannot guarantee that no external mutation remains in flight. Unsafe detected transfers fail closed, but undetected/out-of-boundary work is a deployment blocker for a strict exclusivity claim.

## Verification results

### Unit-tested

Command:

```sh
PYTHONPATH=. .venv/bin/python -m pytest -q --confcutdir=mission_ownership_tests mission_ownership_tests/test_mission_ownership.py
```

Result: `30 passed in 2.00s`.

Coverage includes SQLite ownership transitions, stale-owner and inline dispatch rejection, repeated delivery and mismatched reuse, identical wording across sessions, post-completion new mission, named resume, child task checks, task/mission atomic completion, trusted receipt acceptance/rejection, handoff rollback (including task-card creation rollback), expired owner lease with an outstanding execution lease, interrupted scopes, and a subprocess hard exit that leaves a fail-closed lease. The suite also invokes the shared executor middleware path with a long-running mocked terminal-named dispatch and a Telegram gateway post-admission consumer wrapper with the lifecycle plug-in call mocked.

Additional checks:

```sh
.venv/bin/python -m ruff check agent/tool_executor.py agent/turn_context.py hermes_cli/kanban_db.py hermes_cli/mission_ownership.py hermes_cli/mission_validation.py mission_ownership_tests/test_mission_ownership.py plugins/mission_admission/__init__.py tools/mission_tools.py toolsets.py
# All checks passed!
git diff --check
# passed (no output)
```

A fallback warning was emitted by one subprocess test because `HERMES_HOME` was unset in that subprocess. Its `HERMES_KANBAN_DB` points to the test temporary directory; the warning is recorded as a test-harness cleanup item, not evidence of a live profile write.

### Integration-tested in-process, isolated profile/database

- Shared admission and executor middleware functions were exercised in-process with temporary `HERMES_HOME` and `HERMES_KANBAN_DB` paths.
- Telegram gateway post-admission flow was invoked through the gateway consumer wrapper; lifecycle hook discovery/invocation was stubbed and the candidate admission plug-in ran against an isolated database.
- Native Kanban task creation, status, and completion APIs used Hermes' Kanban schema in temporary databases. Failure injection verifies a rolled-back ownership transfer.
- A long-running executor-path operation blocks handoff while active and allows it after normal drain. An abrupt child-process exit demonstrates the persistent-lease fail-closed behavior.

This is not a separately launched isolated Hermes gateway. An attempted CLI help invocation could not import the complete CLI because this worktree's `.venv` lacks `python-dotenv`; no gateway process was started. No external service, platform transport, or live model request was used.

### Live-verified read-only observations

- The starting candidate commit exists on `codex/mission-ownership-phase1`; current commit and final worktree status are recorded below after commit.
- Read-only inspection of the installed Fred profile found the legacy `mission-admission` plug-in enabled and `kanban.dispatch_in_gateway: false` in its profile-local config. Historical gateway logs show the embedded dispatcher running and spawning tasks despite that local value. Therefore profile-local configuration is not sufficient evidence of the effective host-level dispatcher policy. The candidate's old-plug-in migration has not been exercised.
- A read-only query of `/home/ftr3d/.hermes/kanban.db` at the end of verification found no `mission%` tables.
- During an early test-isolation mistake in this pass, the default board briefly received five empty mission schema tables; read-only inspection confirmed they had no rows, and only those five newly created empty tables were removed. A subsequent read-only check found no mission tables. This transient schema write is disclosed; no mission data was present or changed. Tests were then pinned to temporary profile/database paths. Running Fred and H.A.M. processes were not restarted or modified.

### Not verified

- No actual isolated gateway process or live Telegram transport was started. The gateway hook wrapper was exercised with mocked lifecycle dispatch only.
- CLI interactive admission, desktop UI, and web/dashboard interfaces were not end-to-end tested. Shared turn-context admission is covered by direct tests, not by each interface.
- No real terminal subprocess, code sandbox, browser tool, inline tool registry, delegated specialist worker, nested/proxied connector, or remote side-effecting API was run. Registry middleware and inline stale-owner fencing were exercised, but backend lifecycle/cancellation semantics were not.
- No full Hermes repository test suite was run. The isolated gateway CLI could not start because the candidate environment lacks `python-dotenv`.
- Effective host-level dispatcher configuration has not been verified in an isolated running gateway. The legacy profile plug-in migration is not validated.
- No multi-process contention stress test, OS-level worker kill/restart recovery test, or real database copy migration/rollback drill was performed.

## Remaining races, bypasses, and blockers

1. **External work can outlive the lease.** The lease ends when the wrapped handler returns or unwinds. A detached subprocess, remote action, delegated worker, or interrupted underlying tool may continue independently. Strict global mutual exclusion requires backend-specific join/cancel guarantees or an OS process boundary; Phase 1 does not provide these.
2. **Hard crashes require operator repair.** A process killed without Python cleanup leaves an active lease forever. This safely blocks transfer but also blocks recovery; there is no safe automatic stale-lease reaper or operator repair tool yet.
3. **Unwrapped dispatch is outside the fence.** Any tool path that bypasses `_run_agent_tool_execution_middleware`, another profile/build, plugin code doing direct work, or direct DB writes can violate ownership.
4. **Child graph is incomplete by construction.** Only recorded child IDs are checked. The mission record does not yet enforce a complete foreign-key relationship with every Kanban descendant.
5. **Reviewer identity is application-provided.** The receipt records the trusted tool's caller identity as reviewer identity; no separate human authentication or approval is established. The fixed syntax/Ruff checks are technical evidence only.
6. **Identity fallback depends on transport metadata.** Where neither a stable delivery identifier nor stable turn ID exists, exact replay detection cannot be guaranteed. Similar prompts are intentionally never used to auto-attach sessions.
7. **Schema deployment lacks a migration drill.** Tables are added through schema initialization; there is no separately versioned migration/backup tool tested on a copy of an existing board. Do not apply this against the live board.
8. **Installed plugin and dispatcher policy remain unresolved.** The legacy admission plug-in and profile-local `dispatch_in_gateway: false` need an isolated migration and host-level effective-setting verification before rollout.

## Runtime overhead

Each covered tool call adds a small SQLite lease acquisition and release transaction, plus the existing ownership fence. Admission and lifecycle changes add a small number of indexed reads/writes. The test suite's runtime is not a production latency measurement; no production overhead benchmark was run. The trusted validator adds a Python parse pass and a Ruff subprocess (45-second timeout); it runs only when explicitly requested for completion evidence. There is no added model call.

## Migration and rollback readiness

Not ready for staging migration. Schema creation is additive, but an existing-board backup/copy migration and rollback drill has not been validated. The new profile plug-in installation/migration is also untested. A future isolated staging procedure should copy the board, migrate that copy, verify host-level dispatcher behavior, verify legacy plug-in removal/replacement, and restore the copy as a rollback exercise. No live database migration was made here.

## Deployment recommendation

**Do not deploy this candidate to supervised staging yet.** The implementation now rejects detected in-flight work and fails closed after worker crashes, and it no longer accepts model-supplied command status as independent validation. However, it cannot promise exclusivity for detached/outliving tools or unwrapped paths; crash recovery is operator-blocked; reviewer identity is not independently authenticated; the full gateway/profile migration has not been tested; and no copied-board migration/rollback drill exists. Resolve or explicitly accept those blockers in a reviewed staging plan before deployment.

## Final candidate state

- Branch: `codex/mission-ownership-phase1`.
- Implementation revision: `84ef37eb2895edb311ce3d711d84c15c0a154eb0` (hardening commit; parent is the verified starting candidate below).
- Parent: `672658051d8e694804417c700c232ac27e81e61d`.
- Git status: clean after the report-only follow-up commit; that commit updates this report only and does not change implementation code.

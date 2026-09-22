# Task 5 implementation report: gateway, ticker, and dispatcher ownership

## Result

- The default `/Users/mikedemott/.hermes` gateway is the sole configured gateway owner.
- `/Users/mikedemott/.hermes/config.yaml` now explicitly contains
  `gateway.multiplex_profiles: true`.
- The already-unloaded `task-intake-router` and `task-orchestrator` launchd definitions were
  removed with the supported `hermes --profile <name> gateway uninstall` commands. The default
  `ai.hermes.gateway.plist` remains present. All three labels remained unloaded (`launchctl print`
  returned 113); no service was started or re-enabled.
- The multiplex cron ownership gate now applies before startup recovery and the initial heartbeat,
  as well as during ordinary tick cycles. A desktop fallback ticker therefore never writes ticker
  ownership state or recovers executions for a home already owned by a live profile gateway.

## Source and test changes

- `cron/scheduler_provider.py`
  - Added one ownership-filter helper used by both startup and cycle enumeration.
  - Preserved live callable re-enumeration, deleted-home filtering, and per-profile scope.
  - A raising startup enumerator/gate is logged and yields no startup-owned homes rather than
    terminating the ticker thread.
- `tests/cron/test_scheduler_provider.py`
  - Added a regression proving a gated home receives no startup recovery/heartbeat and owned homes
    tick under the correct A -> B -> A profile scope.
- `tests/gateway/test_multiplex_lifecycle.py`
  - Added a lifecycle test proving one supervised ticker receives the complete served-home set.
  - Added a real advisory-lock contention test proving the second gateway logs
    `this gateway will NOT dispatch` and exits before constructing a dispatcher loop.

## TDD and verification evidence

- Red: the new scheduler regression failed because startup recovery visited
  `own-gateway` even though its ownership gate rejected that home.
- Green, affected files:
  - `scripts/run_tests.sh tests/cron/test_scheduler_provider.py tests/cron/test_cron_multiplex_desktop_ticker_scope.py tests/gateway/test_multiplex_lifecycle.py tests/gateway/test_kanban_watchers_mixin.py`
  - Result: 65 passed, 0 failed across four files.
- `git diff --check`: clean.
- Full-suite boundary: `scripts/run_tests.sh` reached 100% with 37,570 passing tests and 51
  failing files before final aggregation was interrupted. This is not a green full-suite result.
  The failures were outside the changed gateway/cron files and included a missing optional
  `anthropic` SDK, profile/flat-install baseline failures, and later workers reporting that the
  selected interpreter no longer had `pytest`. The four affected files above remained green.
- Read-only process and lock inspection after configuration changes found no gateway process and no
  dispatcher-lock owner, consistent with the required unloaded service state.

## Deferred runtime step

Hermes Desktop's backend was not restarted from this worktree. The installed runtime currently
reports source `/Users/mikedemott/.codex/worktrees/hermes-fork-main-20260922`, while this fix is in
the Task 5 worktree and the launchd gateway fleet must remain unloaded until final verification.
Restarting now would validate stale installed modules and could create a desktop fallback ticker
before the final owner is activated. Restart the rebuilt exact Task 5 source during the final
verification task, then verify one default multiplex gateway, one ticker per served home, and one
dispatcher-lock owner from fresh logs/process state.

## Independent review of `c330fbe02e`

### Findings

No source-level correctness or maintainability defect was found in the reviewed diff. The new
`_owned_profile_homes()` helper preserves live re-enumeration and deleted-home filtering, applies
the ownership gate atomically before publishing a home list, and is used for both startup recovery
and normal cycles. A rejected home therefore receives neither startup recovery nor ticker state
writes. The A -> B -> A regression exercises the required profile scope, and the existing raising
gate survival regression confirms fail-closed behavior without ending the ticker thread.

The gateway lifecycle regression constructs exactly one supervisor and confirms that its live
enumerator returns the complete served-home set. The dispatcher regression uses the real advisory
lock path: the first gateway retains the lock, the second logs `this gateway will NOT dispatch`,
and a sentinel dispatcher class proves the losing path exits before constructing a worker loop.

### Configuration and launchd boundary

Current read-only inspection confirms `/Users/mikedemott/.hermes/config.yaml` explicitly contains
`gateway.multiplex_profiles: true`. Only the default
`/Users/mikedemott/Library/LaunchAgents/ai.hermes.gateway.plist` is present; no launchd plist whose
name contains `task-intake-router` or `task-orchestrator` remains. `launchctl print` returns 113 for
the default and both former profile labels, so all three are currently unloaded. This current state
is consistent with the implementation report. The commit does not contain a before-state receipt,
so the historical claim that the two removed profile definitions were already unloaded rests on
the implementation report rather than an independently replayable artifact.

### Verification assessment

- Review rerun: the four affected files passed, 65 tests, 0 failures, through
  `scripts/run_tests.sh`.
- Additional directly affected startup-survival file passed, 4 tests, 0 failures.
- The reported full-suite attempt is not green: it reached 37,570 passing tests and 51 failing
  files before aggregation was interrupted. The described failures are environment/dependency and
  unrelated baseline failures rather than evidence of a regression in this diff, so they do not by
  themselves block the source-quality verdict. They also do not establish full-suite acceptance.

### Verdicts

- **Code quality: PASS.** The ownership fix is small, centralized, fail-closed, and covered at the
  startup, cycle, profile-scope, gateway-supervisor, and dispatcher-lock boundaries.
- **Task specification: INCOMPLETE.** The source behavior, explicit multiplex configuration,
  duplicate profile launchd cleanup, and unloaded-state checks are satisfied. The brief also
  requires restarting Hermes Desktop's backend after the source changes. That step was explicitly
  deferred, and there is therefore no fresh exact-source runtime evidence proving one active
  default multiplex gateway, one ticker per served home, and one dispatcher-lock owner. Complete
  that runtime verification before marking Task 5 fully accepted.

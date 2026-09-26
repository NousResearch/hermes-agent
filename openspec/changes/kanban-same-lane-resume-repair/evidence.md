# Evidence and verification ledger

## Status

The Full OpenSpec package and controller/kernel candidate now include the narrow stale-lifecycle-fingerprint normalizer. Local focused and applicable Kanban regression gates are green. Commit, Dev PR, exact-head CI, and fresh non-author review remain pending at this checkpoint. Safe live resume remains intentionally deferred to the dependent target card after this controller task completes; no replacement worktree/card/PR or unsafe bypass was created.

## Read-only audit evidence

Source: completed parent audit `t_0c7c8a40`.

- Target `t_dfa23a41` was observed READY and unclaimed.
- PR `#155` was observed `OPEN` against `dev` from `wt/finance-pr153-producer-output-closure` at OID `1b910d4ff9d1c9fe103da5df84d302ba81fc2cc1`.
- The preserved worktree `/home/hermes/nxe-helix-alpha/.worktrees/t_dfa23a41` was observed clean on the expected branch with local and upstream OIDs matching the PR head.
- Crash run `1144` is the latest substantive outcome and is `changes_requested`.
- Runs `1146` and `1149` are later reclaimed, claimed/heartbeat-only controller attempts with no spawned-worker evidence.
- The required serial-lane guard did not hold during the audit because additional nonterminal Flynn cards existed. This package intentionally keeps that guard strict; the live resume must wait until every other Flynn card is DONE or ARCHIVED.

## Root-cause evidence

- Normal claim/run publication: `hermes_cli/kanban_db.py::_claim_and_open_run` and `claim_task`.
- Normal active-PR suppression: `hermes_cli/kanban_db_dispatch.py::check_respawn_guard`, called by `_dispatch_lane_task`.
- Normal spawn/PID publication: `hermes_cli/kanban_db_dispatch.py::_dispatch_lane_task` and `_set_worker_pid`.
- Heartbeat persistence: `hermes_cli/kanban_db_dispatch.py` heartbeat path.
- Worker session propagation: `hermes_cli/kanban_db_dispatch.py::_default_spawn` and child environment construction.

The ordinary path durably opens the claim/run before child launch and only later records PID/`spawned`. That ordering admits the observed phantom class when no real child launch becomes durable.

## Historical staged prototype (not acceptance evidence)

A prior worker staged `SameLaneResumeSpec` / `resume_same_lane`, three focused tests, and an initial five-file package in the dirty Hermes source tree. That prototype reported 3/3 and 13/13 focused passes, but it does not satisfy this specification because:

- it selects the literal latest terminal run, so later reclaims `1146`/`1149` prevent selecting substantive run `1144`;
- it uses a bare Boolean verifier rather than typed PR/worktree identity evidence;
- it trusts a caller-supplied session ID rather than a child-confirmed session handshake;
- it opens the run before launch inside an uncommitted transaction but does not define a gate preventing child heartbeat/work before commit;
- its strict lane query is untested against the real pre-created TODO cards;
- it lacks exact guard mismatch, concurrency, cleanup-failure, and existing-dispatch compatibility coverage; and
- its prior live attempt was refused by worker-ownership controls, so no live resume evidence exists.

Historical staged test results remain non-acceptance evidence. The later implementation evidence below is the basis for completed implementation and verification boxes in `tasks.md`.

## Package verification

Required package files:

- `proposal.md`
- `design.md`
- `specs/kanban/same-lane-resume.md`
- `tasks.md`
- `evidence.md`

The package must be validated for:

- Full OpenSpec baseline fields and explicit prohibitions;
- mandatory guards and exact target identity;
- transaction/handshake/compensation ordering;
- a task-scoped resume fence that survives SQLite rollback through cleanup and compensation;
- one-use authorization receipts that prevent sequential launches by concurrent contenders after compensation;
- durable BLOCKED/`resume_quarantined` state before spawn and during uncertain cleanup, with READY restored only after full extinction;
- atomic SQLite authorization-receipt transitions tied to quarantine, publication, compensation, and cleanup-failure state; sidecar metadata is recovery-only;
- crash-surviving launch nonce/phase/deadline/PID/start/process-group fence metadata, including the process-created-before-identity-write window;
- Linux parent-death wrapper proof for the pre-identity window, including expected-parent recheck and a prohibition on descendants before identity fsync; unsupported hosts refuse;
- full process-group/tree extinction before READY or retry, including surviving-descendant coverage;
- controller-crash windows and post-commit gate-release recovery with durable release-before-acknowledgement ordering;
- concurrency, failure modes, observability, compatibility/migration, tests, and safe live verification;
- explicit authorization of only `t_dfa23a41` / PR `#155`, deterministic substantive-run provenance, and nullable task-event compatibility;
- no Finance application/OpenSpec/test edits; and
- no implementation-completion claims before implementation/review/live evidence exists.

Independent read-only acceptance review returned `APPROVE` after the package was revised to close compensation-race, controller-crash, authorization-idempotency, process-tree cleanup, durable-quarantine, atomic-receipt, and pre-identity launch gaps. This approval covers the specification package only; it is not approval of the pre-existing staged implementation.

## Original implementation and verification evidence

Source: completed implementation task `t_368977f5`, run `1158`.

- Changed controller files: `hermes_cli/kanban_db_dispatch.py`, `hermes_cli/kanban_same_lane_resume.py`, and `hermes_cli/kanban_same_lane_child.py`.
- Changed regression file: `tests/hermes_cli/test_kanban_same_lane_resume.py`.
- Focused suite: `87 passed, 0 failed`.
- Selected dispatcher/session/lifecycle/reclaim regressions: `19 passed, 0 failed`.
- `py_compile`: passed.
- `git diff --check`: passed.
- Fresh independent non-author implementation review: `APPROVE`.
- Coverage includes exact run `1144` selection after no-launch reclaims `1146`/`1149`, strict lane and typed observer guards, one-use authorization, durable quarantine, atomic publication, no pre-commit heartbeat, launch/persistence/release compensation, deterministic controller-crash windows, and real process-group/escaped-descendant cleanup.
- Ordinary `check_respawn_guard` active-PR suppression remains unchanged; the recovery API is explicit and controller-only.

The original implementation-run evidence above is historical context. The stale-lifecycle-fingerprint repair was re-executed and verified separately below.

## Stale lifecycle-handle repair evidence

The restored target was reported READY with `claim_lock`, `current_run_id`, `worker_pid`, and `session_id` null while `worker_started_at=85963388` remained. This is terminal fingerprint residue, not proof of a live process: the process identity contract requires a PID plus its start fingerprint, and no PID remains.

An isolated fixture recreated that exact state. Before the normalizer implementation, `scripts/run_tests.sh tests/hermes_cli/test_kanban_same_lane_resume.py -k stale_worker_start -v` failed as expected: the outcome was `refused` rather than `published`. This is the retained RED proof that the new regression exercises the reported defect.

The kernel repair is `hermes_cli/kanban_db_lifecycle.py::normalize_ready_worker_start_residue`. It executes under the existing board write transaction, compare-and-swaps only `worker_started_at`, and commits `lifecycle_normalized` in the same transaction. It refuses non-READY state, any claim lock/expiry, current run, open run row, worker PID, or session. The exact authorized same-lane operation invokes it before ordinary complete admission; ordinary dispatcher scanning remains unchanged.

Fresh run `1173` verification on the surviving candidate:

- focused same-lane suite: `93 passed, 0 failed`;
- applicable Hermes Kanban suite: `449 passed, 0 failed, 2 skipped` across 55 files (the skips are existing Windows-only coverage on Linux);
- `py_compile`: passed for all changed Python source and test files;
- `ruff check`: passed for all changed Python source and test files;
- `git diff --check`: passed.

The normalization coverage proves the reported READY/null-handle/stale-start state proceeds, emits exact durable evidence, preserves terminal heartbeat-only history, and creates no new heartbeat. Parameterized negatives prove non-READY state and active claim/expiry/current-run/PID/session fail closed, while a separate negative proves an unpointed open run also refuses normalization.

## Historical live verification outcome before worktree restoration

Source: completed live-verification task `t_6038f106` plus root readback.

- The approved controller invocation was refused before quarantine/claim/run/spawn by the board worker-ownership guard: `PermissionError: delegate_task child contexts cannot mutate Kanban tasks or boards`.
- The exact target stayed READY/unclaimed with no new run, PID, session, `claimed`, `spawned`, `resume_released`, or heartbeat evidence.
- The target worktree `/home/hermes/nxe-helix-alpha/.worktrees/t_dfa23a41` is absent and has no matching registered worktree metadata; therefore the required worktree guard has drifted false.
- PR `#155` was last observed OPEN/CLEAN/MERGEABLE against `dev` at exact head ref `wt/finance-pr153-producer-output-closure`, OID `1b910d4ff9d1c9fe103da5df84d302ba81fc2cc1`, with exact-head build-test success.
- The stale pre-quarantine sidecar created by the refused invocation was removed only after matching its authorization ID and phase.
- No replacement worktree/card/PR, direct SQLite edit, GitHub mutation, deployment, Finance runtime, external effect, or PR `#37` action occurred.

That historical worktree guard failure was later repaired by parent task `t_10bd8f8d`. Fresh read-only verification during run `1173` confirms the restored worktree is clean on `wt/finance-pr153-producer-output-closure`, and local HEAD plus configured upstream both equal `1b910d4ff9d1c9fe103da5df84d302ba81fc2cc1`. GitHub readback confirms PR `#155` remains OPEN/CLEAN against `dev` at the same head ref/OID with exact-head CI success. This controller task still does not invoke the live resume: the dependent target card remains gated on this task's completion and must run only after commit, exact-head CI, and fresh non-author review finish.

## Live verification safety gate

The exact live invocation is forbidden until:

1. implementation and independent review are complete;
2. all other Flynn cards are DONE or ARCHIVED;
3. the target remains READY/unclaimed;
4. the approved read-only observer confirms the exact PR identity/state;
5. the preserved worktree and upstream remain clean and matching; and
6. the latest substantive outcome remains exact Crash run `1144` `changes_requested`.

A false guard ends the attempt without replacement cards, direct SQLite repair, GitHub writes, deployment, configuration changes, Finance runtime activity, external effects, or PR `#37` interaction.

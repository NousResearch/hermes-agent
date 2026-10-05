# Supergoal verification

This feature is implemented on top of the fork's `fix/browser-vault-explicit-opt-in` branch (base `10c40862bf79979bb42eed4268f835f2783a825b`). No running gateway was restarted or production goal activated during verification.

## Automated regression checks

The canonical `scripts/run_tests.sh` runner completed **477 passed, 0 failed** across 27 files, with retries disabled. Coverage included normal goals, supergoal state/prompts, explicit draft mode transitions, pause/resume during evaluation, failed persistence, command discovery and aliases, CLI/gateway/TUI dispatch, adapter busy bypass, and clarification/approval regressions.

The clarification tests exercise the real inline executor and registry paths with real SQLite state. They prove that active supergoals never invoke the UI callback, normal/paused/done/cleared states retain clarification, model arguments cannot spoof internal session identity, and two profiles with the same session ID remain isolated (A→B→A).

A separate normal `/goal` CLI oracle compared the base and candidate with distinct temporary homes: all 22 command cases had identical outputs, queued prompts and selected persisted state.

Ruff 0.15.10 passed on all modified/new Python files. `git diff --check` passed.

## Final corrections: budgets and clarification storage

New supergoals use a 40-turn default; normal goals retain their 20-turn/configured default. Explicit and persisted limits, user controls, and turn-budget/judge-error/gate-retry pauses are preserved.

The earlier read-only SQLite clarification guard still denied ordinary conversations on actual database corruption or lock errors. It has been replaced with a small per-profile/session restriction marker in `supergoal-policy/`. Clarification never opens SQLite. Activation/resume verify the marker before saving goal state and returning kickoff; release follows a verified non-active goal write. Failed saves keep the restriction. A per-session writer lock prevents activation racing marker release, without process-global policy caches or agent flags. No dependencies, DB schema, tool schemas, or toolsets were added or changed.

Compression publishes the child's marker before exposing its session ID. Failed goal migration preserves both restrictions; failed marker publication prevents rotation. Legacy development rows bootstrap on explicit goal load, not clarification. Restore markers together with the goal database; unmarked imported rows require a readable explicit goal load before running.

TDD reproduced ordinary/no-goal denials with real malformed SQLite and exclusive locks, plus loss of SG protection when the DB vanished. Tests also exercise fresh processes, permission failures, marker corruption, failed activation/resume, verified lifecycle release, failed migration, concurrent policy writers, A→B→A isolation, and unchanged database schema. The new storage suite finishes **26 passed**. The final focused canonical run finishes **183 passed, 0 failed across 5 files** (storage, clarification, supergoals, ordinary goals, real compression rotation), with `HERMES_TEST_FILE_RETRIES=0`. An earlier broader run in this correction finished **329 passed across 21 files** before the final writer-lock and migration refinements; it is not a claim that all 21 were rerun afterward.

The live-provider results below predate these corrections. No new provider/network E2E, deployment, commit, or push was performed.

## Live judge contract checks

Real `gpt-6-astra` calls through `auxiliary.goal_judge`, using four deliberately constructed response fixtures, returned:

- One preferred method failed, alternatives not attempted: `continue`.
- Exhaustion-of-alternatives attestation without evidence: `continue`.
- Reported file deliverable with exact-byte verification evidence: `done`.
- Finite-set impossibility with exhaustive checks and explanation: `blocked`.

These check judge behavior on supplied prose. They do not establish that the judge independently inspects files or that it cannot be misled by fabricated prose.

## Live agent + judge execution

A real `AIAgent` and real judge, both `gpt-6-astra`, ran with isolated session state and a scratch workspace. The objective required `first.txt` containing `FIRST\n` and `second.txt` containing `SECOND\n`, creating at most one file per assistant turn and reading each back.

The first turn created and verified `first.txt`; judge: `continue`. The second created and verified `second.txt`; judge: `done`. The external test harness independently checked both files' exact bytes. No clarification callback ran. The cached system prompt and tool schemas remained unchanged between turns.

An earlier live run exposed a real loop problem: the judge requested more concrete evidence, but that feedback was absent from the next working prompt and the agent repeated its completion claim. The implementation now includes the persisted judge reason as evaluation context, with guidance to report concise actual verification evidence early in the final response. The live run above passed after this correction.

## Final correction verification

After the storage-policy and stale-reader race corrections, the parent reran 29 relevant files with retries disabled: **571 passed, 0 failed**. This includes the two controlled concurrency regressions, ordinary/no-goal clarification under database failure, active-supergoal durable protection, and the 40-turn default. Ruff 0.15.10, `git diff --check`, and an isolated real-import smoke of command resolution, both budgets, and clarification callbacks passed. The earlier live-provider run was not repeated for these final storage corrections.

## Limits of verification

No Telegram-network E2E, desktop GUI click-through, Windows/macOS runtime test, or repository-wide full suite was run. Messaging/TUI command paths were exercised in automated integration tests with transport boundaries substituted. The live test establishes the model → real file tools → judge → continuation → real file tools → judge path, not production deployment.

Autonomy does not remove the existing turn budget, judge-error/gate retry limits, explicit user stops, scope limits, or command approval requirements. Blocker evaluation remains prompt-based and semantic; there is no exact-sentence acceptance gate.

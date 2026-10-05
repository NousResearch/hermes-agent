# Supergoal verification

This feature is implemented on top of the fork's `fix/browser-vault-explicit-opt-in` branch (base `10c40862bf79979bb42eed4268f835f2783a825b`). No running gateway was restarted or production goal activated during verification.

## Automated regression checks

The canonical `scripts/run_tests.sh` runner completed **477 passed, 0 failed** across 27 files, with retries disabled. Coverage included normal goals, supergoal state/prompts, explicit draft mode transitions, pause/resume during evaluation, failed persistence, command discovery and aliases, CLI/gateway/TUI dispatch, adapter busy bypass, and clarification/approval regressions.

The clarification tests exercise the real inline executor and registry paths with real SQLite state. They prove that active supergoals never invoke the UI callback, normal/paused/done/cleared states retain clarification, model arguments cannot spoof internal session identity, and two profiles with the same session ID remain isolated (A→B→A).

A separate normal `/goal` CLI oracle compared the base and candidate with distinct temporary homes: all 22 command cases had identical outputs, queued prompts and selected persisted state.

Ruff 0.15.10 passed on all modified/new Python files. `git diff --check` passed.

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

## Limits of verification

No Telegram-network E2E, desktop GUI click-through, Windows/macOS runtime test, or repository-wide full suite was run. Messaging/TUI command paths were exercised in automated integration tests with transport boundaries substituted. The live test establishes the model → real file tools → judge → continuation → real file tools → judge path, not production deployment.

Autonomy does not remove the existing turn budget, judge-error/gate retry limits, explicit user stops, scope limits, or command approval requirements. Blocker evaluation remains prompt-based and semantic; there is no exact-sentence acceptance gate.

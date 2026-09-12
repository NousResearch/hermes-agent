# TDD

Fail, then pass. The red command is part of the evidence receipt.

## Do

1. Name the intended behavior vs the current behavior in one sentence each.
2. Write or extend a test that fails for the right reason. Run the unit command from `AGENTS.md`.
3. Record that command and the failure as `tests.red`.
4. Change production code until the same command passes. Record `tests.green`.
5. Optional sabotage: break one assertion, rerun, expect fail. That is `tests.sabotage`.
6. Do not delete the red proof to make the log look clean.

## Stack

The unit command is detected at onboard time (`python -m pytest`). If that placeholder is still `unknown`, detect the tree (`pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod`) and pin the command in the PR. Do not copy another project's test runner.

## Do not

- Start with the fix and add a test that could only pass.
- Call a linter a test.
- Skip red because the bug is "obvious".

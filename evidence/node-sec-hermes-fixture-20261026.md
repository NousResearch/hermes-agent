# OpenCode Go model-switch fixture isolation

Status: source-only PR; merge owned by parent session. No runtime change.

The regression helper supplied a fixed catalog to the picker but allowed model
validation to probe the live provider. Native run
[40067543353](https://github.com/roshanjonah/hermes-agent/actions/runs/40067543353),
attempts 1 and 2, failed the unchanged assertion at
`tests/hermes_cli/test_opencode_go_flat_namespace.py:145`:
`AssertionError: assert 'deepseek-v4.1-flash' == 'deepseek-v4-flash'`.

Trace: `_run_switch` (`tests/hermes_cli/test_opencode_go_flat_namespace.py:111`)
calls `switch_model`; `hermes_cli/model_switch.py:913` calls validation,
`hermes_cli/models.py:3648` calls `fetch_api_models`, and
`hermes_cli/models.py:3676` chooses a close live model. The corrected model is
applied at `hermes_cli/model_switch.py:981`. The public provider catalog had
changed while the test snapshot had not.

The existing helper now also patches `hermes_cli.models.fetch_api_models` to
return the same `_OPENCODE_GO_LIVE` snapshot. Canonical precedent:
`tests/hermes_cli/test_opencode_go_validation_fallback.py:33` patches this same
external boundary. `switch_model` and validation still execute normally.
All assertions, inputs, snapshots, timeouts, production files and CI stay intact.
The helper's three callers cover flash, pro and current-model switching.

Validation, based on `main` at `9feadc273433e6afdfcb68ac912ba5aff46b7114`:

- `scripts/run_tests.sh tests/hermes_cli/test_opencode_go_flat_namespace.py`: exit 0; 12 passed, none skipped.
- `.venv/bin/python -m ruff check tests/hermes_cli/test_opencode_go_flat_namespace.py`: passed.
- `git diff --check`: exit 0.
- Fixture SHA256: `a1b1c80399e721c6d61ba2e2359907d7edbc1c488a78b4e83e3aba643db05308`.
- Environment: Python 3.12.8; ruff 0.15.10; unchanged `uv.lock` SHA256 `df78a4de6004d0833202b3591d88b6363526c09e986853215e9de21b8e01919a`.
- Full suites: native PR CI only; results bound to the exact head in the PR Checks tab.

Tests: no additions or removals; one missing external fake extended. Existing
red receipt reused. No test weakening. Browser journeys do not apply.
Logs: canonical `.claude/artifacts/node-sec-hermes-fixture/{focused-tests,ruff}.log`.
Source rollback: revert the future actual merge; no production rollback needed.
Cleanup: no server, container or browser started; clean worktree retained for merge.

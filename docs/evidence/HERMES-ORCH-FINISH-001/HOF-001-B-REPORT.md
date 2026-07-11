# HOF-001-B Report

## Goal
Integrate only approved source commit `b5d403cf8` for deterministic task admission after HOF-001-A, within the allowed production and test paths.

## System model
- Owner: `hermes_cli/kanban_db.py` owns contract storage, task status, task-run, task-event, and notification-subscription state.
- Inputs: task contract, workspace kind/path, subscription state, enforcement mode, and parent state.
- Output: deterministic ready/claim admission decisions and auditable rejection events.
- Invariants: legacy NULL-contract cards remain compatible; rejected claims mutate no run/PID state; valid subscribed contracts remain dispatchable; task/run/event/OBS behavior remains intact.
- Timing: gates run on ready transitions and are revalidated within the protected claim transaction before run creation.
- Serialization: versioned durable contract JSON; admission decisions and events are derived/audit data. No random or rendering behavior is involved.

## Method
Applied the approved code source `b5d403cf8` and retained the source change strictly within `hermes_cli/kanban_db.py` and `tests/hermes_cli/test_kanban_task_admission.py`. This evidence file is the only additional allowed path.

## Changed paths
- `hermes_cli/kanban_db.py`
- `tests/hermes_cli/test_kanban_task_admission.py`
- `docs/evidence/HERMES-ORCH-FINISH-001/HOF-001-B-REPORT.md`

## Evidence / Results
- `python -m py_compile hermes_cli/kanban_db.py` — passed.
- `scripts/run_tests.sh tests/hermes_cli/test_kanban_task_admission.py tests/hermes_cli/test_kanban_task_contract.py -q` — blocked by the known local runner prerequisite: no virtualenv at `<repo>/.venv` or `<repo>/venv`.
- Fallback: `python -m pytest tests/hermes_cli/test_kanban_task_admission.py tests/hermes_cli/test_kanban_task_contract.py -q -n 0` — passed (`45 passed in 2.46s`).
- `git diff --check` — passed before the evidence-file commit and rerun after the final local commit.
- Code/test comparison against approved source is verified after the final commit; only this report differs from the approved source content.

## Status
- Contract-bearing tasks with incomplete admission data are rejected before ready/claim dispatch and create auditable rejection events.
- Valid contracted tasks with a full base SHA, required evidence, workspace data, and subscription remain ready/claim dispatchable.
- Legacy boards and NULL-contract tasks retain opt-in backward-compatible admission behavior.
- No out-of-scope production files, remote operations, configuration changes, gateway operations, or downstream-release actions were performed.

## Gateway / Runtime
- Gateway not restarted.
- Runtime activation is deferred; this task changes persistent admission behavior only and does not perform release/runtime wiring.

## Final identifiers
- Approved source SHA: `b5d403cf8`.
- Final local activation commit SHA and this report's SHA-256 are recorded in the Kanban review handoff after the final local commit.

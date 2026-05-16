# Step 2: implement-one-minute-status-snapshot

## Objective
Implement the harness status snapshot automation so a running phase updates its current status at least every 60 seconds.

## Scope
- Modify: `scripts/harness_execute.py`
- Modify or add tests under: `tests/scripts/test_harness_execute.py`
- Do not add external dependencies.
- Do not change the phase/step JSON contract except by adding backward-compatible fields.

## Desired behavior
- When a phase starts, write `phases/<phase>/status.json` with phase, current step, step name, status `running`, started_at, updated_at, completed_steps, total_steps.
- While an agent subprocess is running, refresh `updated_at` every 60 seconds.
- On `completed`, `blocked`, or `error`, write the terminal state immediately.
- The updater must stop cleanly when the agent subprocess exits.

## Acceptance Criteria
- Tests prove a status snapshot is written at step start.
- Tests prove terminal status is written on completed/error/blocked paths.
- Tests cover the ticker without waiting 60 real seconds by injecting or mocking the interval/ticker.
- Run: `./venv/bin/python -m pytest tests/scripts/test_harness_execute.py -q`.

# Step 5: document-and-verify-harness-workflow

## Objective
Finalize the Undersea Friends harness workflow documentation and verify the phase can be executed safely.

## Scope
- Update `docs/HARNESS.md` with the Undersea Friends operating model and runbook.
- Update shared-memory docs only if Step 1/3 did not already do so.
- Do not run the full harness against live profiles unless the worktree is clean or intentionally prepared.
- Do not push automatically.

## Required runbook
- How to create a new Undersea Friends phase.
- How to choose the owner profile.
- How status updates work every 1 minute.
- How blocked/error states are reported.
- How to resume after a blocked/error step.
- Which operations require explicit user approval.

## Acceptance Criteria
- `docs/HARNESS.md` includes a concise Undersea Friends runbook.
- Phase JSON validates with `python -m json.tool`.
- Harness tests pass: `./venv/bin/python -m pytest tests/scripts/test_harness_execute.py -q`.
- `python scripts/harness_execute.py --help` still works.
- Final report lists changed files and confirms no gateway/process/secrets were touched.

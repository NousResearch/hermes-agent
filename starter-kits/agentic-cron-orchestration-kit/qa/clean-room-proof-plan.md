# Clean-Room Proof Plan — Agentic Cron Orchestration Kit

## Goal
Verify that a technical user can go from fresh context to one working recurring workflow in under 30 minutes.

## Canonical path to test
1. Clone or open the Hermes repo.
2. Run `bash starter-kits/agentic-cron-orchestration-kit/scripts/preflight.sh`.
3. Copy the four templates into a fresh notes location.
4. Create the four jobs using the prompts in `starter-kits/agentic-cron-orchestration-kit/prompts/`.
5. Manually execute **Evening Documentation Sync**.
6. Confirm the notes are updated directly and the next critical move is obvious.

## Evidence to record
- start timestamp
- end timestamp
- hidden assumptions encountered
- files/notes touched
- whether one recurring job was also scheduled successfully

## Pass / fail gate
- PASS: one workflow completes from fresh context in under 30 minutes with no undocumented blockers
- FAIL: missing prerequisites, hidden setup assumptions, or elapsed time above 30 minutes

## Current status
Executed on 2026-04-17 for the starter-workflow proof gate.

Result summary:
- one recurring proof job was scheduled successfully
- one evening-doc-sync workflow was executed against fresh notes context
- elapsed time: 1.74 minutes
- proof report: `qa/clean-room-proof-run-2026-04-17.md`

Key hidden assumption discovered:
- prompt templates require exact note/workspace path injection before a fresh-context run is possible

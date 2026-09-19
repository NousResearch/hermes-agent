# Prior-Work-First Resume Handoff

- PROGRAM_ID: `HERMES-PRIOR-WORK-FIRST-EXECUTION-MEMORY-V1`
- CAPABILITY_ID: `HERMES-PRIOR-WORK-FIRST-EXECUTION-MEMORY-V1`
- PREFLIGHT_STATUS: `verified`
- FIRST_UNPROVEN_BOUNDARY: `LIVE_ACTIVATION`
- CURRENT_WRITER: `codex-primary-writer`
- LAST_MACHINE_VERIFIED_AT: `2026-08-30T11:21:15.9918202+07:00`

Resume rule:

1. Reuse accepted boundaries unless an explicit invalidator fires.
2. Refresh only volatile git state and fresh machine evidence first.
3. Keep one canonical writer; any scouts remain read-only.
4. Live activation remains pending; do not treat a process exit as completion.

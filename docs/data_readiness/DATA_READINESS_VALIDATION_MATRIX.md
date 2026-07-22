# DATA_READINESS_VALIDATION_MATRIX

## Validation boundaries
Every proposed control, capability, or policy must answer these criteria before moving from proposed to active.

| Domain | Invariant | Test | Required outcome | Evidence | Status |
|--------|-----------|------|------------------|----------|--------|
| Platform stabilization | Single canonical gateway process tree | post-reboot process audit | 1 | ps/grep audit, restore timeout | pending |
| Platform stabilization | cron 64fce1 authenticated trigger succeeds twice | cron trigger retry | success | log + output artifact | pending |
| Platform stabilization | Telegram reconnect within 60s | gateway reboot | yes | log timestamp delta | pending |
| Platform stabilization | AGENTS.md below 900 lines | line count | <900 | `wc -l` output | pending |
| Gateway | No stale lock files except known | lock audit | 0 stale | file listing + age | pending |
| Auth | No credential leakage in diagnostics | env/token scan | 0 hardcoded | static regex + runtime | pending |
| Data policy | 10 docs exist and pass lint | file inventory | 10 | fs inventory | done |
| Data policy | All docs mention restore freeze | content search | 10 | grep | done |
| Data schema | Schema v1 validates sample records | unit tests | pass | pytest output | pending |
| Ingestion | pipeline rejects live confidential data | test run | reject | pytest | pending |
| Backup | Backup restore succeeds in isolated dir | restore run | success + verifies checksum | hash report | pending |
| Monitoring | Cron observability emits success/failure events | 48h run | visible in audit log | log count | pending |
| Agent permissions | No agent has DBA/owner credentials | config audit | none found | config diff | pending |
| Rollback | Revert proved and documented | revert drill | within 30 min | timeline evidence | pending |

## Status key
- `pending`
- `in_progress`
- `done`
- `blocked`

## Next phase
Do not move any status from `pending` to `in_progress` without Arthur's approval for each production-facing change.

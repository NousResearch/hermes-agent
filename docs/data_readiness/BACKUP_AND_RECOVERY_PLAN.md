# BACKUP_AND_RECOVERY_PLAN

## Purpose
Produce verifiable backups that satisfy isolation-restore validation.

## Backup scope
- System of record SQLite databases
- Document vault metadata
- Audit trail hashes
- Rules/checkpoint manifests
- Hermes config

## Cadence
- Daily: SQLite databases + audit hash inventory
- Weekly: full vault snapshot + reports export

## Isolation requirement
- Recover each backup to an isolated test directory.
- Validate schema, hash chain, and restore process coverage.
- No backup is accepted without successful isolated restore.

## Storage
- Canonical backup root: `C:\Users\13464\AppData\Local\hermes\data\backups\`
- Retention: revert to retention policy.

## Recovery sequence
1. Stop writers/hold state file.
2. Create timestamped restore target.
3. Restore database snapshot first.
4. Restore vault metadata.
5. Verify hash chain/audit manifest.

## Rollback
- Per-backup manifest with exact source snapshot, provider, hash, and rollback command.

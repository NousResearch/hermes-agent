# Runbook: Backup Verification

## Purpose

Verify that backups exist, are recent, and have enough metadata to support a restore test.

## Scope

Approved backup directories, object storage buckets, database dumps, or backup service metadata. This runbook does not delete or overwrite backups.

## Risk Level

Low for metadata checks. Medium if accessing sensitive backup contents. High if performing restore tests against shared environments.

## Prerequisites

- Backup location or service name
- Expected backup frequency and retention policy
- Read-only access to backup metadata
- Approved restore target for any restore test

## Inputs

| Input | Description | Example |
| --- | --- | --- |
| backup_location | Path, bucket, or service | `/backups/app` |
| expected_rpo | Maximum acceptable age | `24h` |

## Procedure

1. Confirm backup location and expected recovery point objective.
2. List recent backup artifacts or metadata.
3. Confirm timestamps, sizes, and checksums where available.
4. Confirm at least one backup is within the expected age window.
5. If restore testing is requested, use an isolated target and get explicit approval first.
6. Summarize gaps, missing backups, or stale artifacts.

## Example Commands

```bash
find <backup_location> -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM %s %p
' | sort | tail -20
sha256sum <backup_file>
du -sh <backup_location>
```

## Verification

The backup check is complete when the summary identifies the newest backup, its size, its age, and whether it satisfies the expected policy.

## Rollback / Recovery

Metadata checks require no rollback. Restore tests must be isolated and documented separately before execution.

## Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| No recent backup | Scheduler failure or wrong location | Check backup job status and logs |
| Backup size is zero | Failed job or incomplete upload | Inspect backup logs and retry policy |
| Permission denied | Missing read access | Request least-privilege backup metadata access |

## Related Scripts

- Future lab scripts may automate backup metadata checks under `scripts/lab/`.

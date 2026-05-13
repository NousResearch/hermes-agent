# Runbook: System Inventory

## Purpose

Collect a read-only snapshot of a host for troubleshooting, baseline documentation, or automation planning.

## Scope

Local host or explicitly approved remote host. This runbook should not change system state.

## Risk Level

Low. Commands are read-only, but output may contain hostnames, usernames, IP addresses, mount paths, or package names.

## Prerequisites

- Shell access to the target host
- Permission to inspect system information
- A secure place to store collected output if it is saved

## Inputs

| Input | Description | Example |
| --- | --- | --- |
| target | Host being inspected | `localhost` |

## Procedure

1. Confirm the target host and user context.
2. Collect OS and kernel information.
3. Collect CPU, memory, and disk information.
4. Collect running service/process summary if approved.
5. Summarize findings and redact sensitive values before sharing.

## Example Commands

```bash
uname -a
python --version || true
df -h
free -h || true
ps aux | head -20
```

## Verification

The inventory is complete when it includes OS, kernel, disk, memory, and relevant runtime information for the approved target.

## Rollback / Recovery

No rollback is required for read-only collection. Delete saved reports if they contain sensitive data and are no longer needed.

## Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| Command missing | Minimal OS image | Use an equivalent tool or document that the field is unavailable |
| Permission denied | Insufficient privileges | Request read-only access or skip privileged fields |

## Related Scripts

- Future lab scripts may automate this runbook under `scripts/lab/`.

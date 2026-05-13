# Runbook: Log Triage

## Purpose

Collect and inspect logs to identify recent errors, warnings, crashes, or suspicious behavior.

## Scope

Local logs, service logs, or approved application logs. Do not upload logs externally unless approved.

## Risk Level

Low to Medium. Reading logs is usually safe, but logs may contain secrets, tokens, email addresses, IP addresses, or customer data.

## Prerequisites

- Permission to read the relevant logs
- Time window for analysis
- Service or application name

## Inputs

| Input | Description | Example |
| --- | --- | --- |
| service | Service or app to inspect | `nginx` |
| since | Start of time window | `1 hour ago` |
| log_path | Optional file path | `/var/log/syslog` |

## Procedure

1. Confirm the service, host, and time window.
2. Prefer service-specific log tools when available.
3. Search for errors, warnings, tracebacks, crashes, refused connections, and permission failures.
4. Extract representative examples, not entire sensitive logs.
5. Redact secrets and personal data before sharing.
6. Summarize likely root causes and next checks.

## Example Commands

```bash
journalctl -u <service> --since "1 hour ago" --no-pager
journalctl --since "1 hour ago" --priority warning --no-pager
tail -n 200 /var/log/syslog
```

## Verification

The triage is complete when the summary includes time window, log sources, key error patterns, and recommended next action.

## Rollback / Recovery

No rollback is required for read-only log inspection. Remove temporary excerpts if they contain sensitive data.

## Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| No logs returned | Wrong service or time window | Re-check service name and broaden the time window |
| Permission denied | Restricted log access | Request appropriate read-only access |
| Too much output | Broad query | Narrow by service, priority, or time window |

## Related Scripts

- Future lab scripts may automate filtered log collection under `scripts/lab/`.

# Runbook: Service Health Check

## Purpose

Verify whether a service is running, reachable, and behaving as expected.

## Scope

A local or approved remote service. This runbook is read-only unless a restart or remediation step is explicitly approved.

## Risk Level

Low for checks only. Medium or High if remediation such as restart, config edit, or failover is added.

## Prerequisites

- Service name or endpoint
- Access to host or network path
- Expected healthy response

## Inputs

| Input | Description | Example |
| --- | --- | --- |
| service | System service name | `ssh` |
| endpoint | HTTP/TCP endpoint | `http://localhost:8080/health` |

## Procedure

1. Confirm service name and target environment.
2. Check process or service manager status.
3. Check listening ports if relevant.
4. Check application health endpoint if available.
5. Review recent warnings/errors if health is degraded.
6. Do not restart or modify the service without explicit approval.

## Example Commands

```bash
systemctl status <service> --no-pager
ss -tulpn
curl -fsS <endpoint>
journalctl -u <service> --since "30 minutes ago" --no-pager
```

## Verification

The service is healthy when status, port checks, and application-level health checks match the expected state.

## Rollback / Recovery

Read-only checks require no rollback. If remediation is approved later, document the exact command and rollback path before executing it.

## Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| Service inactive | Crash, disabled service, failed dependency | Inspect logs before restart |
| Port closed | Service not listening or firewall issue | Check config and listener state |
| Health endpoint fails | App dependency or internal error | Inspect app logs and dependency status |

## Related Scripts

- Future lab scripts may automate health checks under `scripts/lab/`.

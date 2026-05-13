# Runbook: <Title>

## Purpose

Describe what this runbook accomplishes.

## Scope

Define systems, files, services, or accounts affected by this procedure.

## Risk Level

Choose one: Low / Medium / High / Destructive

Explain why.

## Prerequisites

- Required access
- Required tools
- Required environment variables or config files
- Required maintenance window, if any

## Inputs

| Input | Description | Example |
| --- | --- | --- |
| `<target>` | Target host, service, path, or identifier | `localhost` |

## Procedure

1. Confirm target and environment.
2. Run read-only discovery.
3. Run dry-run, if this procedure changes state.
4. Execute the approved action.
5. Capture verification output.

## Verification

List commands or checks that prove the procedure succeeded.

## Rollback / Recovery

Describe how to undo the change or recover safely.

## Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| Example failure | Example cause | Example fix |

## Related Scripts

- `scripts/...`

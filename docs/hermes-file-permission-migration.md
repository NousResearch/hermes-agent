# Hermes File Permission Hardening

Security hardening adds explicit `chmod 0600` and `chmod 0700` to newly
created Hermes state files, cron files, Kanban databases, worker logs, and
scratch workspaces. Existing files keep their current permissions until they
are touched again or manually migrated.

## What Changed

- Sensitive files are owner-readable and owner-writable only: `0600`.
- Sensitive directories are owner-accessible only: `0700`.
- `HERMES_HOME_MODE` is honored for Hermes-owned state directories when set.
- Managed installs that intentionally share state with a Hermes group keep
  directories group-accessible (`2770`) and log files group-readable and
  group-writable (`0660`).

## Suggested Migration

Stop running Hermes processes first, then inspect current permissions:

```bash
find ~/.hermes -maxdepth 4 -type f \( -name '*.db' -o -name '*.log' -o -name jobs.json -o -name '.env' -o -name config.yaml \) -ls
find ~/.hermes -maxdepth 4 -type d \( -name logs -o -name workspaces -o -name cron \) -ls
```

Tighten sensitive files:

```bash
find ~/.hermes -type f \( -name '*.db' -o -name '*.log' -o -name jobs.json -o -name '.env' -o -name config.yaml \) -exec chmod 600 {} +
```

Tighten sensitive directories:

```bash
find ~/.hermes -type d \( -name logs -o -name workspaces -o -name cron -o -name output \) -exec chmod 700 {} +
```

For managed group-sharing deployments, keep group-owned state directories at
`2770` and log files at `0660` if the service account and operator accounts
both rely on that access.

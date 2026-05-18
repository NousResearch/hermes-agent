# Hermes System Doctor Dashboard

The system doctor dashboard is a concise, non-secret health snapshot for a
Hermes runtime profile. It is intentionally smaller than `hermes doctor`: it
does not repair anything, does not call live messaging platforms, and does not
run a monitoring loop.

Run it from the existing doctor command:

```bash
hermes doctor --dashboard-only
```

To append it after the legacy doctor checks:

```bash
hermes doctor --dashboard
```

By default, `FAIL` entries are reported in the dashboard but do not change the
process exit code. Use the explicit fail flag when a script should fail on
dashboard failures:

```bash
hermes doctor --dashboard-only --dashboard-fail-exit
```

Honcho reachability is not checked by default because it may initialize a client
against a local or remote service. Opt in when that is wanted:

```bash
hermes doctor --dashboard-only --check-dashboard-reachability
```

## Structured API

`hermes_cli.system_doctor` exposes:

- `build_system_doctor_report(...)` returning a `SystemDoctorReport`
- `render_system_doctor_dashboard(report)` returning Markdown-like text
- `print_system_doctor_dashboard(...)` for CLI integration

Each report entry has:

- `name`
- `status`: `OK`, `WARN`, or `FAIL`
- `detail`
- `remediation`
- `category`

`SystemDoctorReport.exit_code(fail_on_fail=True)` returns `1` only when at least
one `FAIL` entry exists and the caller explicitly requests fail-on-fail behavior.

## Checks

The dashboard currently checks:

- Hermes home/profile path existence and basic profile isolation shape.
- `config.yaml` availability and YAML parseability.
- Obsidian vault rules availability when configured or passed to the API.
  A known Umbbi vault rules path used in docs/tests is:
  `/Users/umbbi/Documents/Obsidian Vault/90. setting/Vault 운영 원칙.md`.
  That path is not a hard dependency for general users.
- Pending interaction store health under:
  `HERMES_HOME/pending_interactions/records.json`.
  Counts include active, expired, and total records.
- Memory governance review queue health under:
  `HERMES_HOME/memory_governance/review_queue.json`.
  Counts include pending review and total items.
- Honcho configuration and Honcho reachability as separate checks.
  Reachability is opt-in.
- Codex readiness:
  the `codex` command on `PATH`, plus parseability/inspection of the Codex
  `goals` feature flag from `.codex/config.toml` when present.
- Cron storage basics under:
  `HERMES_HOME/cron/jobs.json`.
  Counts include enabled, disabled, and total jobs.

## Secret Handling

The dashboard must be safe to paste into Discord. It reports paths, counts,
status names, and short remediation strings, but does not print raw API keys,
tokens, passwords, private keys, or bearer tokens. Renderer-level redaction also
scrubs secret-looking values if a future check accidentally includes one.

## Non-Goals

This is not a daemon, alerting system, autonomous bot loop, or full monitoring
stack. It does not require Discord, Honcho, Obsidian, Codex auth, network access,
or cron jobs to be live during unit tests. Broken JSON/YAML is reported; files
are not automatically repaired by this dashboard.

## Limitations

The profile isolation check is deliberately shallow: it verifies obvious default,
named-profile, and custom-home layouts, but it cannot prove another process is
not sharing the same `HERMES_HOME`. Codex feature inspection only reports whether
the local TOML config can be parsed and whether a `goals` key is present; Codex
auth status is intentionally outside this dashboard.

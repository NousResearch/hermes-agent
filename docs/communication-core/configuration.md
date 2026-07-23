# Configuration

Behavioral settings live in `config.yaml`; credentials remain secret refs on a
`ConnectedAccount`. No new user-facing environment variable is introduced.

```yaml
communication:
  database: "communication.db"
  outbox_workers_enabled: false
  test_sink_enabled: false
  approval_ttl_minutes: 30
  sync_retry_limit: 3
  retention_days: 365
  pii_log_policy: ids_only
```

The defaults in `hermes_cli.config.DEFAULT_CONFIG` and
`cli-config.yaml.example` keep both worker flags false. Account rows may hold
opaque `credential_ref` and `browser_profile_ref` values; CLI output removes
those fields. Never put tokens/cookies/session contents in config or Core
tables.

Facebook retains its separate mandatory safety setting:
`facebook_settings.write_actions_enabled=0`.

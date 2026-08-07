# Kanban recovery supervisor

An opt-in, deterministic observer for `kanban_task_blocked` lifecycle events.
It is disabled unless listed in `plugins.enabled`. The default mode is
`notify_only`: eligible events are audit-recorded in the board-scoped plugin
state database and no recovery card is created.

```yaml
plugins:
  enabled:
    - kanban-recovery-supervisor

kanban_recovery_supervisor:
  enabled_boards: [default]
  supervisor_profile: oink
  mode: notify_only # change explicitly to safe_recovery to permit a card
  cooldown_seconds: 900
  max_retries_per_signature: 1
```

`safe_recovery` creates at most one independent recovery card per durable source
failure signature. It only permits mechanical path normalization, upload/public
URL transport, provider 429/5xx, stale claims, or dependency/tool-availability
failures. The Oink worker must inspect the durable source history and either
make the smallest safe repair with one bounded retry, or leave the source
blocked with a precise human report.

The plugin never invokes an LLM, unblocks the source directly, or auto-recovers
human/product approvals, credentials, secret handling, trading/finance,
merge/deploy/publish/production actions, destructive operations, product scope,
or ambiguous failures. Source notification subscriptions are copied to a created
recovery card when the board supports them.

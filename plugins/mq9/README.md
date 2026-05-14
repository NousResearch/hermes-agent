# mq9 Plugin for Hermes

`mq9` is a standalone Hermes plugin that enables cross-agent discovery and task calls over RobustMQ mq9.

## Tools

- `mq9_register_self`: register current Hermes instance as an agent card in mq9.
- `mq9_unregister_self`: unregister current Hermes instance from mq9.
- `mq9_discover`: discover remote agents by query (with mailbox normalization).
- `mq9_call`: send task payload to a remote mailbox and wait for callback reply.
- `mq9_status`: inspect runtime status and effective config.

## Hooks

- `on_session_start`
- `on_session_reset`
- `on_session_finalize`

The runtime starts automatically and performs best-effort unregister during finalize and process exit.

## Config

Use `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - mq9
  entries:
    mq9:
      nats_url: "nats://127.0.0.1:46222"
      agent_name: "hermes-a"
      mailbox: "hermes.a.inbox"
      auto_register: true
      passive_serve: true
      passive_execute_mode: minimal   # minimal | oneshot
      default_discover_limit: 10
      default_call_timeout_s: 30
```

Environment overrides (highest priority):

- `HERMES_MQ9_NATS_URL`
- `HERMES_MQ9_AGENT_NAME`
- `HERMES_MQ9_MAILBOX`
- `HERMES_MQ9_AUTO_REGISTER`
- `HERMES_MQ9_PASSIVE_SERVE`
- `HERMES_MQ9_PASSIVE_EXECUTE_MODE`
- `HERMES_MQ9_ONESHOT_PROVIDER`
- `HERMES_MQ9_ONESHOT_MODEL`
- `HERMES_MQ9_ONESHOT_TIMEOUT`

## Modes

- `minimal`: safe passive reply mode without model execution.
- `oneshot`: executes inbound tasks using `hermes -z` and returns real model output (requires provider/model/key).


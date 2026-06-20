# Hermes Gateway Transport Liveness

Use this runbook when a gateway process is running but conductors are silent.
This incident class is process-alive / transport-dead: the Hermes process can
remain up while Telegram polling has paused after repeated reconnect failures.

## First Response

1. Check Hermes behavior-level health first:

   ```bash
   hermes gateway status
   hermes doctor
   ```

2. Check the gateway log for the transport pause signature:

   ```bash
   tail -n 120 "${HERMES_HOME:-$HOME/.hermes}/logs/gateway.log"
   ```

   Classify this line as `HERMES_TELEGRAM_PAUSED`:

   ```text
   Telegram paused after 10 consecutive reconnect failures
   ```

3. Restart only the affected Hermes gateway profile:

   ```bash
   hermes gateway restart
   ```

   For a named profile:

   ```bash
   hermes --profile qwen-ops-runner-conductor gateway restart
   ```

4. Prove recovery:

   ```bash
   tail -n 80 "${HERMES_HOME:-$HOME/.hermes}/logs/gateway.log" | grep "Connected to Telegram (polling mode)"
   hermes -z "Reply exactly: ok"
   hermes --profile qwen-ops-runner-conductor -z "Reply exactly: ok"
   ```

## Do Not Chase First

Do not start with Qwen, Ollama, Postgres, NeoEngine API, Cockpit, or the tunnel
unless Hermes transport liveness is clean. Qwen tunnel health is separate from
Hermes Telegram polling health.

## Watchdog Rule

Restart Hermes only when either condition is true:

- `gateway_state.json` reports `platforms.telegram.state = "paused"` or
  `platforms.telegram.transport_paused = true`.
- `platforms.telegram.last_successful_poll_at` is older than the configured
  stale threshold while Telegram is reported as connected.

The default stale threshold is 15 minutes. Keep it above the Telegram polling
heartbeat interval; `HERMES_TELEGRAM_POLLING_HEARTBEAT_SECONDS` is clamped to
30-600 seconds.

The repair is profile-scoped:

```bash
hermes gateway restart
```

or:

```bash
hermes --profile <profile> gateway restart
```

Leave Qwen tunnel, Ollama, Postgres, and NeoEngine API untouched unless their
own independent checks fail after Hermes transport health is clean.

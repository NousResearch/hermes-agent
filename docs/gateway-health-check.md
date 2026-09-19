# Gateway Platform Health & Auto-Recovery

Hermes Gateway includes a built-in **auto-reconnect watcher** that monitors messaging platform connections and automatically recovers from transient failures. This system runs continuously in the background while the gateway is active.

## How It Works

### Two-Layer Recovery

| Layer | Mechanism | Trigger | Recovery |
|-------|-----------|---------|----------|
| **Retryable** | `_platform_reconnect_watcher` | Network/DNS blips, timeouts, connection drops | Auto-retry with exponential backoff |
| **Non-retryable** | Immediate drop | Bad auth, invalid config, permanent errors | Manual fix required |

### Exponential Backoff

```
Attempt 1 → wait 30s
Attempt 2 → wait 60s
Attempt 3 → wait 120s
Attempt 4 → wait 240s
Attempt 5+ → wait 300s (cap, repeats indefinitely)
```

After 5 attempts, retries continue at the 5-minute cap indefinitely — the watcher never gives up on retryable failures, so a transient outage self-heals once connectivity returns.

### Circuit Breaker

Operators can manually pause/resume platforms:

```
/platform list           # see all platform statuses
/platform pause <name>   # stop auto-retry for a platform
/platform resume <name>  # resume auto-retry
```

Paused platforms are skipped entirely by the reconnect watcher and need explicit `/platform resume` to come back.

## Startup Behavior

1. Gateway starts and connects all enabled platforms
2. 10-second initial delay (lets startup finish)
3. Watcher begins polling `_failed_platforms` every ~1s
4. If no failed platforms → sleeps 30s and checks again
5. If failed platforms exist → retries each one at its scheduled time

## Error Classification

### Retryable (auto-recovered)

- DNS resolution failures
- Connection refused / ECONNREFUSED
- Connection timeouts
- Network unreachable
- Temporary API rate limits (HTTP 429)
- Server errors (HTTP 5xx)
- WebSocket disconnections (non-auth)

### Non-retryable (manual intervention needed)

- Authentication failures (HTTP 401, 403)
- Invalid API tokens/keys
- Malformed configuration
- Platform-specific permanent errors

## Monitoring

### Check Platform Status

```
/platform list
```

Shows each platform's current state: connected, failed (with retry count and next retry time), or paused.

### Logs

Reconnection activity is logged at INFO level:

```
Reconnecting telegram (attempt 3)...
Reconnecting slack (attempt 1)...
Platform telegram reconnected successfully
Platform discord still failing: DNS resolution error
```

### Observability Plugin

Enable the `observability` plugin for Prometheus metrics and structured logging of platform health.

## Best Practices

1. **Don't manually restart for transient failures** — the watcher handles them automatically
2. **Use `/platform pause` for maintenance windows** — prevent unnecessary retries during known downtime
3. **Monitor retry counts** — consistently high counts for a platform may indicate config issues
4. **Check auth tokens first** for permanent 401/403 errors

## See Also

- [Plugin Registry](plugin-registry.md) — platform adapter plugins
- [MCP Reference](mcp-reference.md) — connecting external tools

## Summary

Fixes gateway event loop zombie state (#53175) where `agent.close()` and
`shutdown_memory_provider()` blocked the event loop during subprocess
teardown, causing silent message processing death (16 crashes in ~30h).

Replaces 7 scattered synchronous `_cleanup_agent_resources()` call sites
with one centralized `_cleanup_agent_async()` that offloads all blocking
I/O to a thread pool executor with per-context timeouts and structured
logging.

## Problem

Issue #53175: the gateway enters a zombie state where the process stays
alive, platform status shows "connected," but the event loop silently
stops processing inbound messages. Root cause: `_cleanup_agent_resources()`
runs synchronous blocking code (`agent.close()`, `shutdown_memory_provider()`)
inside async handlers. Three bare `except Exception: pass` blocks made the
blockage invisible.

Each zombie crash was preceded by long responses (>100s) + session
reset (`/new`), exactly when agent cleanup is triggered on a loaded LLM
client with active subprocesses and network connections.

## Solution

### New `_CleanupContext` enum
Categorizes all 7 cleanup triggers for selective timeout & logging:

| Context | Default timeout | Trigger |
|---|---|---|
| `SHUTDOWN` | 30s | Gateway stop/restart |
| `SESSION_EXPIRY` | 30s | 5-min watchdog finalization |
| `SESSION_HYGIENE` | 30s | Post-auto-compress eviction |
| `IDLE_CACHE_EVICTION` | 30s | Shutdown idle cached agents |
| `BACKGROUND_TASK` | 15s | After executor-run background task |
| `CACHE_FALLBACK` | 10s | `_release_evicted_agent_soft` fallback |

### `_cleanup_agent_async(agent, context, session_key)`
Centralized async cleanup:
```python
await asyncio.wait_for(
    self._run_in_executor_with_context(
        self._cleanup_agent_resources, agent
    ),
    timeout=self._get_cleanup_timeout(context),
)
```
- Runs blocking ops in thread pool → event loop never stalls
- Timeout per context (configurable) → stuck agent can't take down gateway
- Structured logging with `session_id` + `context` → debuggable
- `TimeoutError` → warning + continue (preserves liveness)
- `Exception` → warning + continue (replaces silent `pass`)

### `_cleanup_agent_resources()` (updated)
Replaced 3 silent `except Exception: pass` with `logger.warning()` that
includes session_id and operation name:
- `shutdown_memory_provider()` failure → logged
- `agent.close()` failure → logged
- `cleanup_stale_async_clients()` failure → logged

### 7 call sites migrated
| # | Call site | Old | New |
|---|---|---|---|
| 1 | `_finalize_shutdown_agents()` | sync `_cleanup_agent_resources` | `_cleanup_agent_async` + threadpool (SHUTDOWN) |
| 2 | `_session_expiry_watcher()` | sync `_cleanup_agent_resources` | `await _cleanup_agent_async` (SESSION_EXPIRY) |
| 3 | Shutdown idle cache cleanup | sync `_cleanup_agent_resources` | `await _cleanup_agent_async` (IDLE_CACHE_EVICTION) |
| 4 | Session hygiene | sync `_cleanup_agent_resources` | `await _cleanup_agent_async` (SESSION_HYGIENE) |
| 5 | Background task executor | sync `_cleanup_agent_resources` | kept sync (already in executor thread) + comment |
| 6 | `_release_evicted_agent_soft()` fallback | sync `_cleanup_agent_resources` | kept sync (already in daemon thread) + comment |
| 7 | Cross-process cache invalidation | sync `_cleanup_agent_resources` | kept sync (already in daemon thread) + comment |

### `_finalize_shutdown_agents` sync fallback
When `_gateway_loop` is None (tests), runs cleanup synchronously so unit
tests can verify `close()` was called immediately. Simplified the previous
`asyncio.get_running_loop()` fallback chain.

### Optional config
```yaml
gateway:
  cleanup_timeouts:
    shutdown: 30.0
    session_expiry: 30.0
    session_hygiene: 30.0
    idle_cache: 30.0
    background_task: 15.0
    cache_fallback: 10.0
```

## Files changed

| File | Changes |
|---|---|
| `gateway/run.py` | +191/-43 — `_CleanupContext` enum, `_cleanup_agent_async()`, `_get_cleanup_timeout()`, updated `_cleanup_agent_resources()`, migrated 7 call sites |
| `tests/gateway/test_shutdown_cache_cleanup.py` | +44/-43 — updated for async cleanup pattern, 8/8 passing |
| `.github/workflows/docker-publish.yml` | +13/-3 — skip Docker build on fork PRs (no `packages: write`) |

## Commits

1. `5724c4e` — fix(gateway): centralize agent cleanup pipeline with executor offload (#53175)
2. `deedb4b` — test(gateway): update shutdown cache cleanup tests for async cleanup pattern
3. `cd63145` — fix(gateway): ensure _finalize_shutdown_agents uses safe_schedule_threadsafe with sync fallback
4. `fbf7e44` — fix(gateway): ensure _finalize_shutdown_agents runs sync cleanup without _gateway_loop
5. `522ad29` — ci(docker-publish): skip arm64/amd64 build jobs on fork PRs

Closes #53175
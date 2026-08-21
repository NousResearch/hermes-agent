---
id: gateway-startup
kind: process
universe: runtime
name: Gateway Startup
summary: >
  Start the long-running messaging gateway: guard duplicates, claim locks,
  discover MCP tools, run adapters, handle shutdown, and hard-exit.
aliases: []
tags: [gateway, startup, shutdown]
shape: process
steps:
  - id: step.1
    summary: >
      Set exec-ask posture, apply resource limits, record boot fingerprint,
      and guard against duplicate running instances under the same HERMES_HOME.
  - id: step.2
    summary: >
      Sync bundled skills, initialize logging, run startup security audit,
      and instantiate `GatewayRunner` with verbosity.
  - id: step.3
    summary: >
      Install signal handlers and planned-stop watcher, claim PID file and
      runtime lock, and record lifecycle sentinel.
  - id: step.4
    summary: >
      Run MCP tool discovery in an executor, start platform adapters,
      and recover pending messages from shutdown flush.
  - id: step.5
    summary: >
      On shutdown, run `runner.stop()`, drain logs, release locks,
      and hard-exit via `os._exit` to avoid wedged non-daemon threads.
entrypoints: [step.1]
produces: [runtime:gateway]
consumes: [repo:gateway/run.py, repo:scripts/run_tests.sh]
---

# Gateway Startup

1. Set `HERMES_EXEC_ASK=1` and apply `nofile` soft limit (`gateway/run.py:29969-29976`).
2. Record boot fingerprint for code-skew detection (`gateway/run.py:29982-29983`).
3. Duplicate-instance guard: check PID file and runtime lock for existing gateway under the same `HERMES_HOME`. If `replace=True`, terminate the old process, reap orphans, clear stale locks, then proceed; otherwise return `False` (`gateway/run.py:29985-30145`).
4. Sync bundled skills and initialize centralized logging (`gateway/run.py:30147-30158`).
5. Run startup security posture audit (`gateway/run.py:30160-30176`).
6. Configure stderr verbosity and create `GatewayRunner(config)` (`gateway/run.py:30193-30197`).
7. Install SIGINT/SIGTERM shutdown handler and planned-stop watcher thread for Windows/non-signal environments (`gateway/run.py:30205-30353`).
8. Claim PID file and runtime lock with `O_CREAT|O_EXCL` race safety; register `atexit` cleanup (`gateway/run.py:30355-30384`).
9. Record lifecycle sentinel for previous-death detection (`gateway/run.py:30386-30395`).
10. Run MCP tool discovery in an executor so slow servers do not freeze platform heartbeats (`gateway/run.py:30406-30417`).
11. Start adapters via `await runner.start()` (`gateway/run.py:30419-30427`).
12. On shutdown signal, run `runner.stop()` and drain logs; force `os._exit` so wedged non-daemon threads cannot strand the gateway (`gateway/run.py:30656-30790`).

## Human check

Confirm `main()` still hard-exits via `_exit_after_graceful_shutdown` and that `start_gateway` returns `False` on duplicate-instance detection without starting adapters.

## Deterministic validation

```bash
grep -n "async def start_gateway" gateway/run.py
grep -n "def main" gateway/run.py
grep -n "acquire_gateway_runtime_lock" gateway/run.py
grep -n "os._exit" gateway/run.py
```

Expected: `start_gateway` at line 29955, `main` at 30656, runtime lock acquisition around 30370, hard exit around 30718.

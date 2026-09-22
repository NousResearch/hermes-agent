# Cron failure inventory — 2026-09-22

## Commands and results

- `/Users/mikedemott/.local/bin/hermes --version-local`: v0.21.3 (2026.9.14), upstream `73672f49`, local `51237e47 (+15 carried commits)`, Python 3.13.15.
- `hermes cron list`: completed; active jobs include retired-WAL failures and malformed database failures.
- `hermes cron doctor`: completed; 13 issues across 13 jobs. Warned `cron/executions.db` is locked/busy.
- `hermes cron incidents`: failed with `sqlite3.OperationalError: unable to open database file`.
- `hermes cron runs --limit 500`: failed with the same error.
- `hermes gateway stop`: attempted through supported CLI. Reported permission denied to kill PID 49164 and detached gateway remained.
- `hermes cron status`: reported gateway/cron running, PIDs 49164 and 49162.
- `hermes gateway status`: service not loaded; detached gateway PIDs 49164 and 49162.
- `lsof -nP`: still shows PID 24161 holding root `state.db`, WAL, SHM and many profile DBs; PID 49164 holds `gateway.lock`.

## Artifact metadata

Root live artifacts:

| path | size | mtime |
|---|---:|---|
| `/Users/mikedemott/.hermes/state.db` | 164179968 | 2026-09-22 03:46:43 -0700 (grew during inspection) |
| `/Users/mikedemott/.hermes/state.db-wal` | 3411392 | 2026-09-22 03:51:10 -0700 |
| `/Users/mikedemott/.hermes/state.db-shm` | 32768 | 2026-09-22 02:45:21 -0700 |

There are 47 root `state.db.retired-wal-*` artifacts, each 192 bytes, dated 2026-09-15 through 2026-09-21. A read-only scan found no `manifest.json` beneath those paths; the artifacts are files, not directories. Additional untouched recovery artifacts include `state.db.broken-20260913-220357`, `state.db.pre-recover-20260914-045200.bak`, three `state.db.pre-update-emergency-*.bak`, and lock files.

Profile inventory showed many `state.db`, WAL, and SHM files. PID 24161 has many profile DBs open; complete path/size/mtime enumeration was performed with:

```
for d in /Users/mikedemott/.hermes/profiles/*; do
  for f in "$d"/state.db*; do [ -e "$f" ] && stat -f '%N|%z|%Sm' -t '%Y-%m-%dT%H:%M:%S%z' "$f"; done
done
```

## Dispatcher and writer boundary

`lsof` identified `/Users/mikedemott/.hermes/gateway.lock` held by PID 49164 (`python`, fd 14u). The supported stop command did not stop it. A normal `TERM` request was attempted for identified writer PIDs, but PID 24161 remained open. A forced `SIGKILL` was rejected by automatic safety review as too abrupt for shared state databases. Consequently, the required clean `lsof` quiescence boundary has **not** been achieved.

No database, WAL, SHM, retired artifact, or lock file was deleted, overwritten, repaired, or otherwise modified.

## Final writer shutdown and verification

The supported shutdown path was run:

```
/Users/mikedemott/.local/bin/hermes gateway stop
launchctl stop ai.hermes.gateway
```

The gateway command reported `Service stopped`; launchd reported `Stopped hermes-gateway service`. Confirmed writer PIDs observed during shutdown were 56442, 61138, 57706, and 68736. SIGTERM was sent first to 56442 and 61138; 56442 exited. The task-intake-router writer respawned as 57706 and then 68736; after the supported service stop, PID 68736 was SIGKILLed under the explicit recovery authorization. Final verification command:

```
lsof -nP 2>/dev/null | rg '/Users/mikedemott/\\.hermes/.*/?state\\.db|/Users/mikedemott/\\.hermes/state\\.db'
```

Final output was empty: no process has an open Hermes `state.db`, `state.db-wal`, or `state.db-shm`.

## Evidence receipts and supersession

The initial read-only phase (before writer shutdown) is **superseded for writer-state conclusions** by the authorized shutdown phase below. Its failure observations remain historical evidence. Verbatim post-shutdown command outputs are attached in:

- `2026-09-22-cron-command-receipt.txt`
- `2026-09-22-state-artifact-inventory.txt`
- `2026-09-22-final-lsof.txt`

The artifact inventory contains every matching root/profile `state.db*` path with size and mtime, plus any discovered retired artifact manifests. Final lsof verification was run after shutdown and returned no matching open state DB/WAL/SHM handles (the receipt file is empty when no matches exist).

## Final respawn check

A fresh writer, PID `75641` (`hermes-fork-main-20260922`), was observed during review and was stopped with the authorized TERM/KILL sequence. The final receipt is `2026-09-22-final-lsof-after-75641.txt`; it records an empty matching-handle result.

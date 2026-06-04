---
name: hermes-s6-container-supervision
description: Modify, debug, or extend the s6-overlay supervision tree inside the Hermes Agent Docker image — adding new services, debugging profile gateways, understanding the Architecture B main-program pattern.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux]
environments: [s6]
metadata:
  hermes:
    tags: [docker, s6, supervision, gateway, profiles]
    related_skills: [hermes-agent, hermes-agent-dev]
---

# Hermes s6-overlay Container Supervision

## When to use this skill

Load this skill when you're working on:
- Adding or removing a static service in the Hermes Docker image (something that should be supervised at every container start, like the dashboard)
- Diagnosing why a per-profile gateway isn't starting, restarting, or surviving `docker restart`
- Understanding why the container's CMD is `/opt/hermes/docker/main-wrapper.sh` and how leading-dash args reach the user's program
- Modifying `cont-init.d` boot scripts (UID remap, volume seeding, profile reconciliation)
- Changing the rendered run-script for per-profile gateways (Phase 4)

If you're just running the Hermes Agent and want to use Docker, see `website/docs/user-guide/docker.md` instead.

## Architecture at a glance

```
/init                                  ← PID 1 (s6-overlay v3.2.3.0)
├── cont-init.d                        ← oneshot setup, runs as root
│   ├── 01-hermes-setup                ← docker/stage2-hook.sh
│   │   ├── UID/GID remap
│   │   ├── chown /opt/data
│   │   ├── chown /opt/data/profiles (every boot)
│   │   ├── seed .env / config.yaml / SOUL.md
│   │   └── skills_sync.py
│   └── 02-reconcile-profiles          ← hermes_cli.container_boot
│       ├── chown /run/service (hermes-writable for runtime register)
│       └── walk $HERMES_HOME/profiles/<name>/gateway_state.json
│           → recreate /run/service/gateway-<name>/
│           → auto-start only those with prior_state == "running"
│
├── s6-rc.d (static services, in /etc/s6-overlay/s6-rc.d/)
│   ├── main-hermes/run                ← exec sleep infinity (no-op slot)
│   └── dashboard/run                  ← if HERMES_DASHBOARD=1, runs `hermes dashboard`
│
├── /run/service (s6-svscan watches; tmpfs)
│   ├── gateway-coder/                 ← runtime-registered per-profile
│   │   ├── type        ("longrun")
│   │   ├── run         ("#!/command/with-contenv sh ... exec s6-setuidgid hermes hermes -p coder gateway run")
│   │   ├── down        (marker — present means "registered but don't auto-start")
│   │   └── log/run     (s6-log → $HERMES_HOME/logs/gateways/coder/current)
│   └── ...
│
└── CMD ("main program")               ← /opt/hermes/docker/main-wrapper.sh
    └── routes user args: bare exec | hermes subcommand | hermes (no args)
        — exec'd by /init with stdin/stdout/stderr inherited (TTY for --tui)
```

## Key files

| Path | Role |
|---|---|
| `Dockerfile` | s6-overlay install + cont-init.d wiring + `ENTRYPOINT ["/init", "/opt/hermes/docker/main-wrapper.sh"]` |
| `docker/stage2-hook.sh` | The "old entrypoint logic" — UID remap, chown, seed, skills sync. Runs as cont-init.d/01-hermes-setup. |
| `docker/cont-init.d/02-reconcile-profiles` | Calls `hermes_cli.container_boot` on every boot to restore profile gateway slots from the persistent volume. |
| `docker/main-wrapper.sh` | The container's CMD. Routes user args, drops to hermes via `s6-setuidgid`, exec's the chosen program. |
| `docker/s6-rc.d/main-hermes/run` | No-op `sleep infinity` — slot exists so the s6-rc user bundle is valid; main hermes runs as the CMD, not as a supervised service. |
| `docker/s6-rc.d/dashboard/run` | Conditional service — `exec sleep infinity` unless `HERMES_DASHBOARD` is truthy. |
| `docker/entrypoint.sh` | Back-compat shim that `exec`s the stage2 hook. External scripts that hard-coded the old entrypoint path still work. |
| `hermes_cli/service_manager.py` | `S6ServiceManager`: `register_profile_gateway`, `unregister_profile_gateway`, `start/stop/restart/is_running`, `list_profile_gateways`. |
| `hermes_cli/container_boot.py` | `reconcile_profile_gateways()` — walks persistent profiles, regenerates s6 slots, emits `container-boot.log`. |
| `hermes_cli/gateway.py::_dispatch_via_service_manager_if_s6` | Intercepts `hermes gateway start/stop/restart` and routes to s6 when running in a container. |

## Why Architecture B (CMD as main program, not s6-supervised)

The original plan (v1–v3) called for main hermes to run as a supervised s6-rc service. Two real s6-overlay v3 mechanics blocked that:

1. **cont-init.d scripts receive no CMD args** — so the stage2 hook can't parse `docker run <image> chat -q "hi"` to set `HERMES_ARGS` for a service `run` script to consume.
2. **`/run/s6/basedir/bin/halt` does NOT propagate the exit code** written to `/run/s6-linux-init-container-results/exitcode`. Containers always exit 143 (SIGTERM) regardless. Confirmed by skarnet (s6 author) in [issue #477](https://github.com/just-containers/s6-overlay/issues/477): _"if you want a container shutdown, you need to either have your CMD exit, or, if you have no CMD, write the container exit code you want then call halt"_.

So we use the s6-overlay-native CMD pattern: `ENTRYPOINT ["/init", "/opt/hermes/docker/main-wrapper.sh"]`. /init prepends the wrapper to user args automatically — so `docker run <image> --version` becomes `/init main-wrapper.sh --version`, and `--version` doesn't get intercepted by /init's POSIX shell. The wrapper drops to hermes via `s6-setuidgid`, then exec's the chosen program. The program's exit code becomes the container exit code, exactly matching the pre-s6 tini contract.

Trade-off: main hermes is unsupervised under s6. That exactly matches its behavior under tini (the pre-s6 image). Dashboard supervision is the only **new** guarantee — and per-profile gateways under `/run/service/` get full supervision.

## Quick recipes

### Verify s6 is PID 1 in a running container

```sh
docker exec <c> sh -c 'cat /proc/1/comm; readlink /proc/1/exe'
# Expect: s6-svscan or init / /package/admin/s6/.../s6-svscan
```

### Inspect a profile gateway service

```sh
# /command/ isn't on docker-exec PATH — use absolute path
docker exec <c> /command/s6-svstat /run/service/gateway-<name>
# "up (pid …) … seconds"            → running
# "down (exitcode N) … seconds, normally up, want up, …" → s6 wants it up but the process keeps exiting (crash loop)
# "down … normally up, ready …"     → user stopped it
```

### Bring a service up/down manually

```sh
docker exec <c> /command/s6-svc -u /run/service/gateway-<name>   # up
docker exec <c> /command/s6-svc -d /run/service/gateway-<name>   # down
docker exec <c> /command/s6-svc -t /run/service/gateway-<name>   # SIGTERM (restart)
```

### Watch the cont-init reconciler log

```sh
docker exec <c> tail -n 50 /opt/data/logs/container-boot.log
# 2026-05-21T06:18:05+0000 profile=coder prior_state=running action=started
# 2026-05-21T06:18:05+0000 profile=writer prior_state=stopped action=registered
```

### Add a new static service

1. Create `docker/s6-rc.d/<name>/type` with `longrun\n` and `docker/s6-rc.d/<name>/run` (use `#!/command/with-contenv sh` + `# shellcheck shell=sh`).
2. Drop to hermes via `s6-setuidgid hermes` at the top of run (unless you specifically need root).
3. Create empty `docker/s6-rc.d/<name>/dependencies.d/base` so it waits for the base bundle.
4. Create empty `docker/s6-rc.d/user/contents.d/<name>` so it joins the user bundle.
5. The `COPY docker/s6-rc.d/` in the Dockerfile picks it up automatically — no other changes.

### Change the per-profile gateway run command

Edit `S6ServiceManager._render_run_script` in `hermes_cli/service_manager.py`. The function is also called by `hermes_cli/container_boot.py::_register_service` during boot reconciliation, so it's the single source of truth. Update the corresponding assertion in `tests/hermes_cli/test_service_manager.py::test_s6_register_creates_service_dir_and_triggers_scan`.

### Run the docker test harness

```sh
docker build -t hermes-agent-harness:latest .
HERMES_TEST_IMAGE=hermes-agent-harness:latest scripts/run_tests.sh tests/docker/ -v
# Expect 19 passed, 0 xfailed against the s6 image
```

The harness lives in `tests/docker/` and skips when Docker isn't available. The per-test timeout is bumped to 180s (see `tests/docker/conftest.py`).

## Common pitfalls

### "command not found" via `docker exec`

`/command/` (where s6-overlay puts its binaries) is on PATH only for processes spawned by the supervision tree — services, cont-init.d, main-wrapper.sh. `docker exec <c> s6-svstat …` will fail with "command not found"; always use the absolute path `/command/s6-svstat`. The `hermes` binary works because the Dockerfile adds `/opt/hermes/.venv/bin` to the runtime `ENV PATH`.

### Profile directory ownership

The cont-init reconciler runs as hermes (`s6-setuidgid hermes` in `02-reconcile-profiles`). If a profile dir ends up root-owned (e.g. because `docker exec <c> hermes profile create …` ran as root by default), the reconciler can't read SOUL.md and fails with `PermissionError`. Mitigation: `stage2-hook.sh` chowns `$HERMES_HOME/profiles` to hermes on **every** boot, idempotently. Don't remove that block.

### Files written by `docker exec` are root-owned

`docker exec` defaults to root. Either pass `--user hermes` or rely on the stage2 chown sweep next reboot. Don't write files under `$HERMES_HOME/profiles/<name>/` as root manually — the next reconcile pass will sweep them but in-flight operations may hit perm errors.

### Service slot exists but s6-svstat says "s6-supervise not running"

The service directory is on tmpfs and was wiped on container restart. Either the cont-init reconciler hasn't run yet (give it a moment after `docker restart`) or it failed. Check `docker logs <c> | grep '02-reconcile'`.

### Gateway starts then immediately exits (`down (exitcode 1)` in svstat)

Most likely the profile has no model or auth configured. The service slot is correct — the gateway itself is unconfigured. Run `hermes -p <profile> setup` first. The s6 supervisor will keep restarting it; that's the desired behavior (when you fix the config, the next attempt succeeds and stays up).

### Reconciler skipped a profile

The reconciler keys on the **presence of `SOUL.md`** as the "real profile" marker. `hermes profile create` always seeds it. If a profile dir is missing SOUL.md (stray directory, partial restore, backup-in-progress), the reconciler skips it intentionally. Add a `SOUL.md` (even empty) to opt back in.

### "Help, the container exits 143!"

Check whether something is invoking `s6-svscanctl -t` or `/run/s6/basedir/bin/halt` — both cause /init to begin stage 3 shutdown but return 143 (SIGTERM) rather than the desired exit code. This was the Phase 2 architecture pivot from A to B. For container shutdown with a real exit code, you must let the CMD (main-wrapper.sh) exit normally; do **not** try to control exit from a finish script.

## Related skills

- `hermes-agent-dev`: General hermes-agent codebase navigation
- `hermes-tool-quirks`: Specific Hermes-tool workarounds (sed/grep/etc.) — load when debugging the s6 stack's interaction with hermes built-in tools.

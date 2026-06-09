---
name: hermes-s6-container-supervision
description: Modify, debug, or extend the s6-overlay supervision tree inside the Hermes Agent Docker image — adding new services, debugging profile gateways, the container role gate (gateway vs dashboard), the :11435 Claude proxy, and the Architecture B main-program pattern.
version: 1.2.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [docker, s6, supervision, gateway, profiles, claude-proxy, deployment, role-gate]
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
- Running the standard **two-container deployment** (a `gateway` container + a `dashboard` container that share the `~/.hermes` volume) and needing the **role gate** so only the gateway runs the per-profile gateways and the `:11435` Claude proxy
- Diagnosing the **bots going dark / `:11435` proxy unstable** — especially `s6-log: fatal: unable to lock /opt/data/logs/claude-proxy/lock: Resource busy` (exit 111) flapping in the gateway's logs
- Writing or fixing the **post-deploy health gate** that probes the proxy (and why it must probe *inside* the container)

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
│   ├── 02-reconcile-profiles          ← hermes_cli.container_boot
│   │   ├── ROLE GATE: dashboard role → skip (gateways are gateway-only)
│   │   ├── chown /run/service (hermes-writable for runtime register)
│   │   └── walk $HERMES_HOME/profiles/<name>/gateway_state.json
│   │       → recreate /run/service/gateway-<name>/
│   │       → auto-start only those with prior_state == "running"
│   ├── 03-seed-data-services          ← seed OTHER volume-backed s6 services
│   │   ├── ROLE GATE: dashboard role → skip
│   │   └── SKIPS claude-proxy (now image-baked); seeds any other volume svc
│   └── 04-registry-gate               ← enforce the client registry (no-op
│       ├── ROLE GATE: dashboard role → skip   if $HERMES_CLIENTS_REGISTRY unset)
│       ├── remove gateway-<profile> slots for isolation:container clients
│       │   (keep isolated bots OUT of the shared gateway → no Telegram 409)
│       └── log DRIFT: for on-disk profiles absent from the registry
│
├── s6-rc.d (static services, in /etc/s6-overlay/s6-rc.d/)
│   ├── main-hermes/run                ← exec sleep infinity (no-op slot)
│   ├── dashboard/run                  ← if HERMES_DASHBOARD=1, runs `hermes dashboard`
│   └── claude-proxy/run               ← IMAGE-BAKED :11435 proxy; ROLE GATE:
│                                        gateway-only (dashboard → finish 125)
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
| `docker/cont-init.d/02-reconcile-profiles` | Calls `hermes_cli.container_boot` on every boot to restore profile gateway slots from the persistent volume. Role-gated: skips the reconciler in the dashboard container. |
| `docker/cont-init.d/03-seed-data-services` | Seeds **other** volume-backed s6 services from `$HERMES_HOME/services/` into the tmpfs scandir. Role-gated. **Skips `claude-proxy`** — that is now image-baked (see below). |
| `docker/cont-init.d/04-registry-gate` | Enforces the client registry against the scandir after 02/03: removes `gateway-<profile>` slots for `isolation: container` clients (keeps isolated bots out of the shared gateway), logs `DRIFT:` for on-disk profiles missing from the registry. **Pure no-op when `$HERMES_CLIENTS_REGISTRY` is unset.** Role-gated. |
| `docker/hermes-role.sh` | Sourceable helper printing the container role (`gateway`\|`dashboard`). Resolves `$HERMES_ROLE`, else infers from the CMD in `/proc`, defaults to `gateway`. Consulted by 02 + 03 + 04 + the `claude-proxy` service. |
| `deploy/claude-proxy/claude_proxy.py` | The `:11435` Claude proxy. Binds **container-local** `127.0.0.1:11435`; serves `GET /health` and `POST /v1/chat/completions`. **Image-baked** at `/opt/hermes/deploy/claude-proxy/claude_proxy.py`; run as the static s6-rc service `docker/s6-rc.d/claude-proxy` (no longer volume-defined). |
| `docker/s6-rc.d/claude-proxy/run` | IMAGE-BAKED `:11435` proxy service. `exec`s the baked `claude_proxy.py` as the hermes user. **Self-gates on the role** (static services run in both containers): dashboard role exits 0 + `finish` returns 125 so the slot stays down. Logs to the container log (no `s6-log`, so no shared log-lock). |
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

## Multi-container deployments: the gateway/dashboard role gate

The standard deployment runs **two containers from the same image** (`docker-compose.yml`):

| Container | CMD | Job |
|---|---|---|
| `hermes` (gateway) | `gateway run` | the bots + per-profile gateways + the `:11435` Claude proxy |
| `hermes-dashboard` | `dashboard --host …` | the web UI on `:9119` |

Both mount the **same** `~/.hermes` volume at `/opt/data`, and both run the **same** `/init` + `cont-init.d` hooks **and the same static s6-rc services**. Without a guard, `02-reconcile-profiles` would start the gateways and the `claude-proxy` static service would start the Claude proxy in *both* containers.

**Why running the proxy twice is catastrophic:** two proxies bind the same `127.0.0.1:11435` and issue duplicate Claude calls. Historically the proxy was volume-defined with an `s6-log` that took an **exclusive lock** on the shared `/opt/data/logs/claude-proxy/lock`; if both containers ran it, the loser died with `s6-log: fatal: unable to lock …/claude-proxy/lock: Resource busy` (exit 111) and **flapped forever**, the gateway lost its Claude provider, and **the bots went dark** — yet `docker compose up -d` still reported success. This was a real, recurring outage.

**Now the proxy is image-baked** (`docker/s6-rc.d/claude-proxy`, Phase 3) and logs to the **container log** (no `s6-log`, so the shared log-lock failure mode is gone by design). The role gate is still essential — two proxies would still fight over `:11435` — so the `claude-proxy` run script **self-gates on the role** (a static s6-rc service runs in both containers, unlike the volume seed that 02/03 skip): in the dashboard role it exits 0 and its `finish` returns 125, leaving the slot permanently down. The `04-registry-gate` hook (Phase 2) layers on top, removing isolated clients from the shared gateway so they don't double-run against their own container.

**The fix — the role gate (`docker/hermes-role.sh`):**

```
hermes_role() resolves, first match wins:
  1. $HERMES_ROLE if set to gateway|dashboard   (compose sets it explicitly)
  2. else infer from the CMD that s6's rc.init carries in /proc:
        … main-wrapper.sh dashboard …  → dashboard ; anything else → gateway
  3. else default to gateway  (single-container / all-in-one fails SAFE)
```

`02` and `03` source it and **exit early in the dashboard role**. Net invariant:

> **Only the gateway container ever seeds/runs the per-profile gateways and the `:11435` proxy. The dashboard is a read-only viewer.**

Notes:
- The fix lives **entirely in the image**, so it holds regardless of the server-side compose file (the deploy may use a compose that isn't in this repo and that you can't edit).
- The repo compose files set `HERMES_ROLE` explicitly (`gateway` / `dashboard`); the `/proc` inference is the belt-and-suspenders fallback for deployments that don't.
- Default-to-gateway is deliberate: misdetecting dashboard→gateway re-introduces contention, but misdetecting gateway→dashboard would silently run nothing. Fail toward "run the stack."
- Regression-locked by `tests/test_hermes_role.py` (pure POSIX sh + `/proc`, no docker).

## Quick recipes

### Diagnose "bots are dark / :11435 unstable" (proxy log-lock contention)

The signature is the gateway's logger flapping on the shared lock. Read-only checks:

```sh
# Who actually runs claude-proxy? After the role gate, ONLY the gateway.
for c in hermes hermes-dashboard; do
  echo "== $c =="
  docker exec "$c" sh -c 'ls -1 /run/service | grep -E "claude-proxy|gateway-" || echo none'
done
# Gateway: claude-proxy + gateway-* present.  Dashboard: none.

# Proxy up in the gateway (not flapping)?
docker exec hermes /command/s6-svstat /run/service/claude-proxy
# Want: "up (pid …) … seconds".  The image-baked proxy logs to the CONTAINER
# log (no s6-log /log subservice), so there is no shared log-lock to contend
# on — `docker logs hermes` carries the proxy's stdout. (Pre-Phase-3 volume
# proxies had a /log subservice; a flapping "down (exitcode 111)" there meant
# lock contention. That failure mode is retired with the image bake.)

# Smoking gun in the logs:
docker logs hermes 2>&1 | grep -iE 'resource busy|unable to lock|exitcode 111'

# Map every host s6-log writing to the proxy log dir back to its container —
# more than one container here == the duplicate-proxy bug:
for pid in $(pgrep -f 's6-log.*claude-proxy'); do
  tr '\0' ' ' < /proc/$pid/cmdline; \
  grep -hoE 'docker[-/][0-9a-f]{12,}' /proc/$pid/cgroup | head -1
done
```

Fix = ensure the dashboard container resolves to the dashboard role (set
`HERMES_ROLE=dashboard`, or confirm its CMD starts with `dashboard`). See the
role gate section above. The read-only `.github/workflows/diagnose-contabo.yml`
runs exactly these probes against the live host without recreating anything.

### Check which role a running container resolved to

```sh
docker exec <c> sh -c 'echo "${HERMES_ROLE:-(unset)}"'          # explicit env, if any
docker inspect -f '{{json .Config.Cmd}}' <c>                    # the CMD the role is inferred from
docker exec <c> sh -c 'ls -1 /run/service'                      # gateway has claude-proxy + gateway-*; dashboard does not
```

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

**Gateway-only static services:** a static s6-rc service runs in **both** the
gateway and dashboard containers (unlike the volume seed, which 02/03 skip via
the role gate). If the service must run only in the gateway, **self-gate inside
the run script** and add a `finish` that returns 125 in the dashboard role so
the slot reports permanently down. Live examples: `docker/s6-rc.d/dashboard`
(gated on `HERMES_DASHBOARD`) and `docker/s6-rc.d/claude-proxy` (gated on
`hermes_role` — the image-baked `:11435` proxy). Pattern:

```sh
# run
. /opt/hermes/docker/hermes-role.sh
[ "$(hermes_role)" = dashboard ] && exit 0   # finish returns 125 → slot down
exec s6-setuidgid hermes <program>
```

**Baking a python service into the image:** `COPY` the script to a stable image
path (e.g. `COPY deploy/foo/foo.py /opt/hermes/deploy/foo/foo.py`) and have the
run script `exec`s it. This is how the `:11435` proxy moved off the data volume
(Phase 3) — the repo is the source of truth, no more drift. If you retire a
previously volume-defined service this way, add a one-time, idempotent migration
to the deploy that renames the volume copy to `<name>.legacy` (reversible) and
have `03-seed-data-services` skip the now-baked name so a stale volume copy can
never re-seed and collide.

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

### Health-probing :11435 from the host returns "connection refused" even though the proxy is up

The proxy binds **container-local** `127.0.0.1:11435` (see `claude_proxy.py`). On this deployment the containers run on a **bridge** network (the dashboard publishes `127.0.0.1:9119->9119`, which `network_mode: host` cannot do), so a probe from the **host** loopback can never reach the container's `:11435` — the bots reach it fine because they live *inside* the gateway container. Always probe on the path the bots use:

```sh
docker exec hermes sh -c 'curl -fsS -m5 http://127.0.0.1:11435/health'
# {"status": "ok", …, "breaker": {"open": false, …}}
```

A host-side `curl 127.0.0.1:11435` in a deploy gate is a **false negative** and will red a healthy deploy (this happened — the proxy logged "Proxy ready" while the host probe failed for 10 min). The `deploy-contabo.yml` health gate runs the probe via `docker exec` for exactly this reason.

### Two containers fighting over the proxy log lock (legacy / pre-Phase-3)

If you see `s6-log: … unable to lock …/claude-proxy/lock: Resource busy` (exit 111) flapping, a second container (almost always `hermes-dashboard`) is running a **volume-defined** proxy with an `s6-log` against the shared `~/.hermes` volume. The Phase 3 image-baked proxy logs to the container log (no `s6-log`), so this lock cannot occur for it — seeing this means a stale volume `claude-proxy` survived: confirm the deploy migration ran (`$HERMES_HOME/services/claude-proxy.legacy` should exist, `claude-proxy` should not) and that `03-seed-data-services` skipped it. The standing invariant is unchanged: only the gateway runs the proxy (the `claude-proxy` run script self-gates on the role).

### Proxy runs the wrong code / a fix didn't take effect

The proxy is **image-baked** (Phase 3): the running code is `/opt/hermes/deploy/claude-proxy/claude_proxy.py` from the image, i.e. whatever `deploy/claude-proxy/claude_proxy.py` was at build time. To ship a proxy change you rebuild + redeploy the image — editing the data volume does nothing. If a stale volume copy still exists it is **ignored** (03 skips `claude-proxy`; the deploy renames it to `.legacy`). Verify the running code's commit with `docker exec hermes cat /opt/hermes/.hermes_build_sha`.

### "Help, the container exits 143!"

Check whether something is invoking `s6-svscanctl -t` or `/run/s6/basedir/bin/halt` — both cause /init to begin stage 3 shutdown but return 143 (SIGTERM) rather than the desired exit code. This was the Phase 2 architecture pivot from A to B. For container shutdown with a real exit code, you must let the CMD (main-wrapper.sh) exit normally; do **not** try to control exit from a finish script.

## Related skills

- `hermes-agent-dev`: General hermes-agent codebase navigation
- `hermes-tool-quirks`: Specific Hermes-tool workarounds (sed/grep/etc.) — load when debugging the s6 stack's interaction with hermes built-in tools.

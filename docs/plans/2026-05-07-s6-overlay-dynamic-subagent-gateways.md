# s6-overlay Supervision for Per-Profile Gateways in Docker — Implementation Plan

> **Status: shipped.** Phases 0–5 landed via PR
> [NousResearch/hermes-agent#30136](https://github.com/NousResearch/hermes-agent/pull/30136)
> in May 2026. This document is preserved as a post-implementation reference
> for the architecture and the resolved design questions. The phase-by-phase
> TDD walkthrough (≈2,800 lines) and the v2/v3 re-validation preambles have
> been removed — the canonical implementation history is the PR commit log
> (`git log --oneline a957ef083..a6f7171a5 -- 'docker/*' 'hermes_cli/service_manager.py' …`).
> Open Questions are collapsed into a single Decision Log table; full
> deliberations live in PR review comments.

**Goal:** Replace `tini` with s6-overlay as PID 1 in the Hermes Docker image so
that the main hermes process, the dashboard, and dynamically-created
per-profile gateways all run as supervised services (auto-restart on crash,
clean shutdown, signal forwarding, zombie reaping). Preserve every existing
`docker run …` invocation pattern — including interactive TUI.

**Architecture:** s6-overlay's `/init` is the container ENTRYPOINT, running
s6-svscan as PID 1. Main hermes and the dashboard are declared as static
s6-rc services at image build time. Per-profile gateways — which users create
*after* the image is built (`hermes profile create coder` →
`coder gateway start`) — are registered dynamically by writing service
directories under a scandir watched by s6-svscan. A `ServiceManager` protocol
abstracts the install/start/stop/restart surface across the init systems we
care about (systemd on Linux host, launchd on macOS host, Scheduled Tasks on
native Windows host, s6 inside container) and adds a second tier for runtime
service registration that only s6 implements.

**Tech Stack:**

- [s6-overlay](https://github.com/just-containers/s6-overlay) v3.2.3.0
  (noarch + per-arch tarballs ~15 MB). SHA256-pinned via build ARGs;
  multi-arch via `TARGETARCH` (amd64 → `x86_64`, arm64 → `aarch64`).
- Debian 13.4 base image (unchanged).
- [hadolint](https://github.com/hadolint/hadolint) for the Dockerfile +
  [shellcheck](https://github.com/koalaman/shellcheck) for entrypoint scripts.
- Python subprocess wrappers for `s6-svc`, `s6-svstat`, `s6-svscanctl`.
- Existing systemd/launchd/windows surface in `hermes_cli/gateway.py` and
  `hermes_cli/gateway_windows.py`.

**Scope:**

- Container-only (host-side systemd/launchd/windows behavior is preserved,
  not modified).
- s6-overlay only (no pure-Python fallback).
- Architecture A (s6 owns PID 1; tini is removed).
- Interactive TUI must keep working:
  `docker run -it --rm nousresearch/hermes-agent:latest --tui`.
- Dynamic registration is limited to per-profile gateways — one service per
  profile, created when a profile is created, torn down when deleted. A
  `gateway-default` slot is always registered for the root HERMES_HOME
  profile so `hermes gateway start` (no `-p`) has somewhere to land.

**Out of scope:**

- Host-side dynamic supervision (systemd-run / launchd transient plists) —
  not needed.
- Pure-Python supervisor fallback — not needed.
- Arbitrary user-defined supervised processes inside the container — only
  profile gateways.
- Migration of existing per-profile systemd unit generation to s6 on the
  host side.
- Non-Docker container runtimes (Podman rootless validated reactively).
- UX polish around in-container profile lifecycle (e.g. a nice status view
  of all supervised profile gateways) — deferred to follow-up.

---

## Background From The Codebase

> **Note on line numbers:** This section refers to functions and structures
> by name only. Use `grep -n 'def <name>' <file>` to locate anything below
> if you need the current line.

### Pre-s6 container init (what we replaced)

The original `Dockerfile` declared
`ENTRYPOINT [ "/usr/bin/tini", "-g", "--", "/opt/hermes/docker/entrypoint.sh" ]`.
tini was PID 1, reaped zombies, forwarded SIGTERM to the process group. The
old `docker/entrypoint.sh`:

1. `gosu` privilege drop from root → `hermes` UID.
2. Copied `.env.example`, `cli-config.yaml.example`, `SOUL.md` into
   `$HERMES_HOME` if missing.
3. Synced bundled skills via `tools/skills_sync.py`.
4. Optionally backgrounded `hermes dashboard` in a subshell when
   `HERMES_DASHBOARD=1` — **not supervised**, no restart.
5. `exec hermes "$@"` — tini's sole direct child.

Known limitations: dashboard crash → stays dead; dashboard fails at startup →
silent; gateway crash → dashboard dies too. The May 4, 2026 decision was
"leave as is" because nothing in the container needed supervision then.
Adding per-profile gateway supervision changed that.

### ServiceManager surface (what we wrapped, not refactored)

All init-system logic lives in **`hermes_cli/gateway.py`** (~5,400 LOC at
re-validation). The systemd/launchd code is ~1,500 lines of that, plus a
separate **`hermes_cli/gateway_windows.py`** (~690 LOC) for Windows
Scheduled Tasks.

| Layer | Systemd functions | Launchd functions | Windows functions |
|---|---|---|---|
| **Detection** | `supports_systemd_services()`, `_systemd_operational()`, `_wsl_systemd_operational()`, `_container_systemd_operational()` | `is_macos()` | `is_windows()`, `gateway_windows.is_installed()` |
| **Paths** | `get_systemd_unit_path(system)`, `get_service_name()` | `get_launchd_plist_path()`, `get_launchd_label()` | `gateway_windows.get_task_name()`, `get_task_script_path()`, `get_startup_entry_path()` |
| **Install/lifecycle** | `systemd_install(force, system, run_as_user)`, `systemd_uninstall(system)`, `systemd_start/stop/restart(system)` | `launchd_install(force)`, `launchd_uninstall/start/stop/restart` | `gateway_windows.install/uninstall/start/stop/restart` |
| **Probes** | `_probe_systemd_service_running(system)`, `_read_systemd_unit_properties(system)`, `_wait_for_systemd_service_restart`, `_recover_pending_systemd_restart` | `_probe_launchd_service_running()` | `gateway_windows.is_task_registered()`, `_pid_exists` helper |
| **D-Bus plumbing** | `_ensure_user_systemd_env`, `_user_systemd_socket_ready`, `_user_systemd_private_socket_path`, `get_systemd_linger_status` | — | — |
| **Unit/plist generation** | `generate_systemd_unit(system, run_as_user)`, `systemd_unit_is_current`, `refresh_systemd_unit_if_needed` | plist templating in `launchd_install` | `_build_gateway_cmd_script`, `_build_startup_launcher`, `_write_task_script` |

Container-relevant callers outside `gateway.py`:

- `hermes_cli/status.py` — gained an `s6` branch for in-container runs.
- `hermes_cli/profiles.py` — `create_profile` / `delete_profile` register and
  unregister with s6 inside the container (no-op on host).
- `hermes_cli/doctor.py` — `_check_gateway_service_linger` skips on s6, and a
  new "Service Supervisor" section reports main-hermes / dashboard /
  profile-gateway counts via the ServiceManager.
- `hermes_cli/gateway.py::gateway_command` — the
  `elif is_container():` rejection arms that refused gateway lifecycle
  operations were removed; the `_dispatch_via_service_manager_if_s6` helper
  intercepts start/stop/restart and routes them through s6.

### Per-profile gateway spawning

`hermes gateway start`, `coder gateway start` (profile alias), and
`hermes -p <profile> gateway start` all spawn a gateway process scoped to a
given profile. See
[Profiles: Running Gateways](https://hermes-agent.nousresearch.com/docs/user-guide/profiles#running-gateways).
On host, lifecycle is managed via per-profile systemd units
(`hermes-gateway-<profile>.service`); inside the container, an s6 service at
`/run/service/gateway-<name>/` is registered when the profile is created and
torn down when it's deleted.

**Persistence across container restart:** `/run/service/` is tmpfs —
service registrations are wiped when the container restarts. Profile
directories at `/opt/data/profiles/<name>/` live on the persistent VOLUME,
and each one records its gateway's last state in `gateway_state.json`.
`/etc/cont-init.d/02-reconcile-profiles` walks the persistent profiles on
every container boot, recreates the s6 service slots via
`hermes_cli/container_boot.py`, and auto-starts those whose last recorded
state was `running`. Profiles whose last state was `stopped`,
`startup_failed`, `starting`, or absent get their slot recreated in the
`down` state and wait for explicit user action. `docker restart` is therefore
invisible to a user with running profile gateways: they come back up;
stopped ones stay stopped.

### s6-overlay constraints

- **Root/non-root model:** `/init` runs as root to set up the supervision
  tree, install signal handlers, and run the stage2 hook that does
  `usermod`/`chown`. Each supervised service drops to UID 10000 via
  `s6-setuidgid hermes` in its `run` script. The per-service `s6-supervise`
  monitor stays root so it can signal its child regardless of UID. Net
  effect: hermes and all its subprocesses run as UID 10000 exactly as
  before; only the supervision tree itself runs as root.
- v3.2.3.0 has limited non-root support for running `/init` itself as
  non-root — some tools (`fix-attrs`, `logutil-service`) assume root. We
  don't hit this because `/init` runs as root.
- Scandir hard cap: `services_max` default 1000, configurable to 160,000.
- `/command/with-contenv` sources `/run/s6/container_environment/*` into
  service env — convenient for passing `HERMES_HOME` etc.
- s6 signal semantics: service crash triggers `s6-supervise` restart after
  1s; override with a `finish` script.
- Zombie reaping: PID 1 (s6-svscan) reaps all zombies non-blockingly on
  SIGCHLD. Any subagent subprocess spawned by the main hermes process is
  reaped automatically.

---

## Key Design Decisions

### D1. s6-overlay replaces tini entirely

Container ENTRYPOINT is `/init`, PID 1 is s6-svscan. The main hermes
process, the dashboard, and every per-profile gateway run as supervised
services. This is a single breaking change to the container contract.

### D2. Main hermes is an s6 service with container-exit semantics

The contract "container exits when `hermes` exits" is preserved via a
service `finish` script that writes to
`/run/s6-linux-init-container-results/exitcode` and calls
`/run/s6/basedir/bin/halt`. All five supported invocations work:

| `docker run <image> …` | Behavior |
|---|---|
| (no args) | `hermes` with no args, container exits when hermes exits |
| `chat -q "..."` | `hermes chat -q "..."`, container exits with hermes exit code |
| `sleep infinity` | `sleep infinity` directly (long-lived sandbox mode) |
| `bash` | interactive `bash` directly |
| `docker run -it … --tui` | interactive Ink TUI with real TTY — see D9 |

`docker/main-wrapper.sh` detects whether `$1` is an executable on PATH and
routes either to "run this as a one-shot main service" or "wrap with
hermes".

### D3. Static services at build time; dynamic (per-profile) services at runtime

s6 offers two mechanisms:

- **s6-rc** (declarative, compile-then-swap): used for main hermes and the
  dashboard — they're known at image build time.
- **scandir** (drop a directory + `s6-svscanctl -a`): used for per-profile
  gateways — profiles are user-created after the image is built.

Per-profile gateway service dirs live at `/run/service/gateway-<profile>/`
(tmpfs, hermes-writable). s6-svscan picks them up on rescan.

### D4. ServiceManager protocol with two methods for runtime registration

Host paths (systemd, launchd, Windows Scheduled Tasks) need only
install/start/stop/restart of pre-declared services. Inside the container,
we additionally need to register services at runtime when a profile is
created. The protocol exposes this directly:

```python
class ServiceManager(Protocol):
    kind: ServiceManagerKind  # "systemd" | "launchd" | "windows" | "s6" | "none"

    # Lifecycle of an already-declared service
    def start(self, name: str) -> None: ...
    def stop(self, name: str) -> None: ...
    def restart(self, name: str) -> None: ...
    def is_running(self, name: str) -> bool: ...

    # Runtime registration (container-only; hosts raise NotImplementedError)
    def supports_runtime_registration(self) -> bool: ...
    def register_profile_gateway(
        self, profile: str, *,
        extra_env: dict[str, str] | None = None,
    ) -> None: ...
    def unregister_profile_gateway(self, profile: str) -> None: ...
    def list_profile_gateways(self) -> list[str]: ...
```

Systemd, launchd, and Windows backends raise `NotImplementedError` on the
registration methods. Only the s6 backend implements them. Callers check
`supports_runtime_registration()` before calling.

The scope is intentionally narrow: it's specifically "register/unregister a
profile gateway," not a general-purpose process-management API.

### D5. Per-profile gateway service spec is fixed, not user-provided

Every profile gateway has the same command shape
(`hermes -p <profile> gateway run`, or `hermes gateway run` for the default
profile). The s6 backend generates the `run` script from a fixed template
given the profile name — no arbitrary command list. This keeps the API
surface tight and prevents callers from accidentally registering
non-gateway services.

Port selection is governed by the profile's `config.yaml`
(`[gateway] port = …`) — the single source of truth. (The original plan
proposed a Python-side SHA-256 port allocator with a 600-port range; it was
retired during PR review because it was dead code through the entire stack.)

### D6. Add detect_service_manager() alongside supports_systemd_services()

`supports_systemd_services()` stays as-is (host code paths unchanged). A new
`detect_service_manager() -> Literal["systemd", "launchd", "windows", "s6", "none"]`
composes existing detection functions (`is_macos()`, `is_windows()`,
`supports_systemd_services()`, `is_container()` + `_s6_running()`) and adds
an s6 branch for container detection. Host call sites continue to use the
existing functions; container-only code (the profile hooks) uses the new one.

`_s6_running()` probes `/proc/1/comm` (world-readable) and
`/run/s6/basedir`. The earlier `/proc/1/exe` probe was root-only readable
and silently failed for the unprivileged hermes user (UID 10000), making
the entire runtime-registration path inert in production — caught in PR
review.

### D7. Wrap existing systemd/launchd/windows functions, don't rewrite them

`SystemdServiceManager` / `LaunchdServiceManager` / `WindowsServiceManager`
are thin adapters over the existing `systemd_*` / `launchd_*` module-level
functions in `hermes_cli/gateway.py` and the
`gateway_windows.install/uninstall/start/stop/restart/is_installed`
functions in `hermes_cli/gateway_windows.py`. We get the abstraction
without rewriting ~2,200 LOC of working code.

### D8. Profile create/delete hooks register/unregister the s6 service

When `hermes profile create <name>` runs inside the container, the
profile-creation code path calls
`ServiceManager.register_profile_gateway(<name>)` if
`supports_runtime_registration()` is True. When `hermes profile delete
<name>` runs, it calls `unregister_profile_gateway(<name>)`. On host, both
calls are no-ops (registration not supported; existing systemd unit
generation continues to handle install/uninstall).

Existing per-profile `hermes -p <profile> gateway start/stop/restart` CLI
commands continue to work — in the container they dispatch to
`ServiceManager.start/stop/restart("gateway-<profile>")`, which translates
to `s6-svc -u`/`-d`/`-t` on the service dir.

`hermes gateway start` (no `-p`) targets a special `gateway-default` slot
that's always registered by the cont-init reconciler. Its run script omits
the `-p` flag and runs against the root `$HERMES_HOME` profile.

`--all` lifecycle (`hermes gateway stop --all`, `... restart --all`)
iterates `mgr.list_profile_gateways()` through s6 so s6's `want up`/`want
down` flips correctly. Without this, `--all` fell through to `pkill`
followed by s6-supervise auto-restart — net effect: kick instead of stop.

### D9. Interactive TUI bypasses s6 service-mode and runs as CMD for TTY passthrough

`docker run -it --rm <image> --tui` needs a real TTY connected to container
stdin/stdout for Ink raw-mode keyboard input, cursor control, and SIGWINCH.
Running the TUI as a normal s6 service fails because s6-supervise
disconnects service stdio from the container TTY (documented:
[s6-overlay#230](https://github.com/just-containers/s6-overlay/issues/230)).

**The pattern:** s6-overlay's `/init` execs a CMD as the container's "main
program" after the supervision tree is up. The CMD inherits
stdin/stdout/stderr from `/init` — which in `-it` mode is the container
TTY. The stage2 hook detects the TUI case and short-circuits the
main-hermes service so the hermes CMD becomes that main program.

```sh
# In docker/stage2-hook.sh
_is_tui_invocation() {
    for arg in "$@"; do
        case "$arg" in --tui|-T) return 0 ;; esac
    done
    case "${HERMES_TUI:-}" in 1|true|TRUE|yes) return 0 ;; esac
    if [ -t 0 ] && [ $# -eq 0 ]; then return 0; fi
    return 1
}
```

And in `docker/s6-rc.d/main-hermes/run`:

```sh
if [ -f /var/run/s6/container_environment/HERMES_TUI_MODE ]; then
    exec sleep infinity   # s6-overlay will exec CMD as the TTY-connected main
fi
exec s6-setuidgid hermes hermes ${HERMES_ARGS:-}
```

In TUI mode main hermes is effectively unsupervised (same as the pre-s6
behavior with tini — acceptable because the user is interactively
present). Dashboard and profile gateways still get full s6 supervision via
their separate services.

The integration test `test_tty_passthrough_to_container` uses `tput cols`
and `COLUMNS=123` as the probe.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Phase 2 breaks a downstream user's Dockerfile that `FROM`s ours | Medium | Medium | Release notes call out ENTRYPOINT change; the test harness (`tests/docker/`) gives high confidence in behavior parity |
| TUI TTY passthrough fails on some Docker versions | Low | High | Harness includes `test_tty_passthrough_to_container` as a hard gate; fallback plan = s6-fdholder ([s6-overlay#230](https://github.com/just-containers/s6-overlay/issues/230) Solution 2) |
| s6-overlay non-root quirks (logutil-service, fix-attrs) bite us | Low | Low | Supervisor runs as root, services drop — sidesteps these issues |
| Podman rootless UID mapping confuses s6 | Medium | Low | Documented as supported, fix reactively; a Podman + Docker environment is stood up for validation |
| Test harness is flaky (docker daemon issues, timing) | Medium | Low | Generous timeouts; skip when docker unavailable; polling helpers replace fixed sleeps in `test_container_restart.py` |
| Profile gateway crash loop masks a real config error | Low | Medium | s6 `finish` script `max_restarts` cap (planned follow-up); operators see crash-looping logs in `$HERMES_HOME/logs/gateways/<profile>/` |
| Dockerfile+entrypoint drift from linter (hadolint/shellcheck) reveals latent bugs | Low | Low | CI lint jobs catch them; fix or document ignore with rationale |
| Stale `gateway.pid` from a dead container collides with an unrelated live PID in the restarted container | Low | Medium | Cont-init reconciliation removes `gateway.pid` and `processes.json` from every profile dir on boot, before any new gateway starts |
| `docker restart` silently loses per-profile gateway registrations (tmpfs scandir wiped) | High (without mitigation) | High | Cont-init reconciliation re-registers from persistent `$HERMES_HOME/profiles/` and auto-starts those last seen `running`; outcome recorded to `$HERMES_HOME/logs/container-boot.log` (size-bounded, rotates to `.1` at 256 KiB) |
| A `running` gateway that's actually broken auto-restarts into a crash loop after every container restart | Low | Medium | s6 `finish` script `max_restarts` cap (planned); follow-up: `hermes doctor` alerts when N consecutive container restarts ended in `startup_failed` |
| `_s6_running()` detection works as root but silently fails for unprivileged hermes user, making runtime-registration path inert | High (without mitigation) | High | **Caught in PR review.** Detection now probes `/proc/1/comm` (world-readable) + `/run/s6/basedir`. Docker integration tests refactored to `docker exec -u hermes` so the realistic runtime user is exercised |
| `s6-svscanctl` from hermes hits EACCES on the root-owned control FIFO | Medium | Medium | `02-reconcile-profiles` chowns `/run/service/.s6-svscan/{control,lock}` to hermes after stage1 creates them |
| Per-service `supervise/control` FIFO is root-owned by s6-supervise, blocking `s6-svc` from hermes | Known | Medium | Surfaced cleanly as `S6CommandError` (with rc + stderr) instead of raw `CalledProcessError`. Permission fix tracked as a follow-up (small SUID helper, polling chown loop in cont-init.d, or replace `s6-svc` with `down`-marker manipulation) |

---

## Decision Log

| # | Question | Decision |
|---|---|---|
| OQ1 | Gate Phase 2 behind env var? | Ship directly (Hermes is pre-1.0; users can pin the previous image) |
| OQ2 | s6 root model | Root `/init`, drop per-service via `s6-setuidgid hermes` |
| OQ3 | Dashboard opt-in mechanism | Always declared as an s6 service; `03-dashboard-toggle` cont-init script writes a `down` marker when `HERMES_DASHBOARD` is unset so `s6-svstat` reports the slot's real state |
| OQ4 | Podman rootless | Supported, fix reactively |
| OQ5 | Service naming | `gateway-<profile>` (matches pre-existing `hermes-gateway-<profile>.service` systemd convention) |
| OQ6 | — (retired; no subagent gateways in scope) | — |
| OQ7 | Resource limits per profile gateway | Defer (no per-cgroup limits; rely on the container's overall limit) |
| OQ8 | Log persistence | `$HERMES_HOME/logs/gateways/<profile>/`. The log path is sourced from runtime `$HERMES_HOME` via `with-contenv`, NOT Python-substituted at registration time |
| OQ9 | TUI passthrough | Trust the documented [s6-overlay#230](https://github.com/just-containers/s6-overlay/issues/230) Solution 1; harness includes a TTY passthrough hard-gate test |

**Post-merge additions from PR #30136 review:**

- **Multi-arch tarballs:** `TARGETARCH` mapped to `x86_64` / `aarch64`;
  per-arch tarball fetched via `curl` because `ADD` doesn't honor BuildKit
  args.
- **SHA256 verification:** all three tarballs (noarch, symlinks, per-arch)
  pinned via build ARGs and verified with `sha256sum -c` against a single
  checksum file (avoids hadolint DL4006 piped-shell warning).
- **`gateway-default` slot:** always registered by the reconciler so
  `hermes gateway start` (no `-p`) has somewhere to land.
- **Friendly lifecycle errors:** `GatewayNotRegisteredError` and
  `S6CommandError` translate `CalledProcessError` into actionable CLI
  messages.
- **Atomic publication in the reconciler:** mirrors
  `register_profile_gateway`'s tmp+rename pattern.
- **`container-boot.log` rotation:** 256 KiB soft cap, rotated to `.1`.
- **`port` parameter retired:** allocator + kwarg were dead code through
  the entire stack; `config.yaml` is the single source of truth.

---

## Verification Checklist

- [x] Test harness (`tests/docker/`) passes against the s6 image
- [x] hadolint + shellcheck run green in CI
- [x] `docker run -it --rm hermes-agent --tui` starts the Ink TUI with
      working keyboard input, cursor control, and resize (SIGWINCH)
- [x] Dashboard crashes are recovered by s6 within ~2s
- [x] `hermes profile create test` inside a container creates
      `/run/service/gateway-test/`
- [x] `hermes -p test gateway start` inside a container dispatches through s6
- [x] `hermes -p test gateway stop` inside a container cleanly stops via s6
- [x] `hermes profile delete test` inside a container removes
      `/run/service/gateway-test/`
- [x] Profile gateway logs persist at
      `$HERMES_HOME/logs/gateways/test/current`
- [x] `hermes status` inside the container shows `Manager: s6`
- [x] `hermes gateway start` (no `-p`) inside a container targets
      `gateway-default` and runs against the root profile
- [x] `hermes gateway stop --all` / `... restart --all` iterate every
      profile gateway under s6 instead of pkill-then-supervise-restart
- [x] `docker restart` survives per-profile gateway registrations via the
      cont-init reconciler; running gateways come back up, stopped ones
      stay down
- [x] Multi-arch image builds for both `linux/amd64` and `linux/arm64`
- [x] s6-overlay tarballs are SHA256-verified at build time
- [x] No systemd/launchd host-side functions were modified (only wrapped)
- [x] `hermes gateway install/start/stop` on Linux host and macOS host
      behave identically to pre-change

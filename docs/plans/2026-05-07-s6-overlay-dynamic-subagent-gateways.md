# s6-overlay Supervision for Per-Profile Gateways in Docker — Implementation Plan

> **For Hermes:** Use `subagent-driven-development` skill to implement this plan task-by-task.

> **Plan v2 — re-validated May 18, 2026.** v1 was drafted May 7, 2026. Re-validation confirmed: (a) nothing has been implemented yet (greenfield); (b) line-number citations everywhere were stale — they have been replaced with function-name references; (c) a fourth host backend has shipped since v1 — `hermes_cli/gateway_windows.py` registers the gateway as a Windows Scheduled Task with a Startup-folder fallback — the `ServiceManager` protocol now includes a `WindowsServiceManager` adapter and `ServiceManagerKind = "systemd" | "launchd" | "windows" | "s6" | "none"`; (d) `gateway_command` currently has five `elif is_container():` arms that *refuse* gateway install/start/stop/restart/uninstall inside containers — Task 4.3 explicitly deletes them as part of the s6 dispatch; (e) Phase 0 Task 0.5's two profile-gateway tests are marked `xfail(strict=True)` because they describe the post-Phase-4 invariant, not current behavior, and flip to passing in Phase 4; (f) s6-overlay bumped from v3.2.2.0 → v3.2.3.0; (g) OQ8-C log path is now sourced from runtime `$HERMES_HOME`, not hard-coded at registration time.

> **Plan v3 — re-validated May 21, 2026 in the `docker_s6` worktree.** Spot-check against eight intervening commits to Dockerfile / entrypoint / gateway / doctor / docker docs found four items that need awareness — none invalidates the plan:
>
> 1. **Install-method stamping landed in entrypoint.sh** (PR #27843 / `6f5ec929a`). After the `gosu` privilege drop and venv activate, the entrypoint writes `"docker"` to `${HERMES_HOME:=/opt/data}/.install_method`, so `detect_install_method()` can report `docker` to `hermes status`. Phase 2 Task 2.3 (`docker/stage2-hook.sh` rewrite) must preserve this stamp — either keep it in the stage2 hook (runs as root, before user services start; would need to chown to hermes UID afterward) or hoist it into a per-service `run` prelude for the main-hermes s6 service. **Recommendation: keep it in the stage2 hook, written as the hermes user via `s6-setuidgid hermes` to match the file's existing ownership.** Add a note to Task 2.3.
> 2. **`RUN mkdir -p /opt/data` was added to the Dockerfile** just before the `VOLUME` declaration (same PR). Phase 2 Task 2.4 (Dockerfile flip) must retain this line — the directory must exist before VOLUME so initial chown succeeds when the volume is first mounted.
> 3. **`hermes_cli/gateway_windows.py` `install()` signature changed** (PRs #28169-adjacent, `d948de39e` + `417a653d9`, ~420 lines of changes). New keyword args: `start_now: bool | None`, `start_on_login: bool | None`, `elevated_handoff: bool`. `WindowsServiceManager.install()` adapter in Task 1.2 must forward these — recommend keeping the wrapper's signature minimal (`install(force=False, **kwargs)`) and passing through; or expose them explicitly if the wrapper is called from non-Windows code paths (it isn't currently). Adapter remains a thin pass-through.
> 4. **`hermes_cli/doctor.py` refactor introduced `_section(title)` and `_fail_and_issue(text, detail, fix, issues)` helpers** (PR #27830, `41f1eddee`). Phase 5 Task 5.3 must use these helpers in any new s6-aware doctor checks rather than the older copy-paste banner pattern. The `_check_gateway_service_linger` function and "Gateway Service" / "External Tools" section names that Task 5.3 references are all still present.
>
> Additionally:
> - `gateway_command` actually contains **three** `elif is_container():` rejection arms in `_gateway_command_inner` (lines 5111, 5141, 5184 as of May 21), not five — point (d) above said "five". The other two `is_container()` references at lines 983 and 1220 are in different helper functions and are not user-facing rejections. Task 4.3 should target three arms, not five.
> - `website/docs/user-guide/docker.md` got a 4-line clarifying note from PR #28497 distinguishing Hermes-in-Docker from Docker-as-terminal-backend. No conflict with Phase 5 Task 5.1.
> - s6-overlay still at v3.2.3.0 (no new release since May 9, 2026). Tech-stack and Task 2.1 ARG remain accurate.
>
> **Plan v3 also adds Task 4.0 — Reconcile per-profile gateways on container boot.** Both v1 and v2 missed this: `/run/service/` is tmpfs, so every `docker restart` was silently wiping every per-profile gateway registration. Task 4.0 introduces a cont-init.d script (`02-reconcile-profiles`) and a Python module (`hermes_cli/container_boot.py`) that walks persistent `$HERMES_HOME/profiles/<name>/`, recreates the s6 service slots, and auto-starts only those whose last `gateway_state.json` was `running`. Phase 4 estimate bumps from 1.5 → 2.0 days; total plan from 12.0 → 12.5 days. Two new risk-register rows + the "Persistence across container restart" paragraph in the Background section make this contract visible to readers who never reach Phase 4.

**Goal:** Replace `tini` with s6-overlay as PID 1 in the Hermes Docker image so that the main hermes process, the dashboard, and dynamically-created per-profile gateways all run as supervised services (auto-restart on crash, clean shutdown, signal forwarding, zombie reaping). Preserve every existing `docker run …` invocation pattern — including interactive TUI.

**Architecture:** s6-overlay's `/init` becomes the container ENTRYPOINT, running s6-svscan as PID 1. Main hermes and the dashboard are declared as static s6-rc services at image build time. Per-profile gateways — which users create *after* the image is built (`hermes profile create coder` → `coder gateway start`) — are registered dynamically by writing service directories under a scandir watched by s6-svscan. A new `ServiceManager` protocol abstracts the install/start/stop/restart surface across the init systems we care about (systemd on Linux host, launchd on macOS host, Scheduled Tasks on native Windows host, s6 inside container) and adds a second tier for runtime service registration that only s6 implements.

**Tech Stack:**
- [s6-overlay](https://github.com/just-containers/s6-overlay) v3.2.3.0 (latest as of plan re-validation; noarch + x86_64 tarballs, ~15 MB) — uses skalibs/s6/s6-rc 2.15+ and includes fixes for long-standing s6-overlay-specific issues. v3.2.2.0 also works if reproducibility from the original plan is needed.
- Debian 13.4 base image (unchanged)
- [hadolint](https://github.com/hadolint/hadolint) for the Dockerfile + [shellcheck](https://github.com/koalaman/shellcheck) for entrypoint scripts
- Python subprocess wrappers for `s6-svc`, `s6-svstat`, `s6-svscanctl`
- Existing systemd/launchd/windows surface in `hermes_cli/gateway.py` and `hermes_cli/gateway_windows.py`

**Scope:**
- Container-only (host-side systemd/launchd behavior is preserved, not modified)
- s6-overlay only (no pure-Python fallback)
- Architecture A (s6 owns PID 1; tini is removed)
- Interactive TUI must keep working: `docker run -it --rm nousresearch/hermes-agent:latest --tui`
- Dynamic registration is limited to per-profile gateways — one service per profile, created when a profile is created, torn down when deleted

**Out of scope:**
- Host-side dynamic supervision (systemd-run / launchd transient plists) — not needed
- Pure-Python supervisor fallback — not needed
- Arbitrary user-defined supervised processes inside the container — only profile gateways
- Migration of existing per-profile systemd unit generation to s6 on the host side
- Non-Docker container runtimes (Podman rootless validated reactively — see OQ4)
- UX polish around in-container profile lifecycle (e.g. a nice status view of all supervised profile gateways) — deferred to follow-up

---

## Background From The Codebase

### Current container init (what we're replacing)

> **Note on line numbers:** This section refers to functions and structures by name only. The codebase is fast-moving — `hermes_cli/gateway.py` alone has grown by ~600 lines in the six months between plan v1 and re-validation. Use `grep -n 'def <name>' <file>` to locate anything below if you need the current line.

**`Dockerfile`** — `ENTRYPOINT [ "/usr/bin/tini", "-g", "--", "/opt/hermes/docker/entrypoint.sh" ]`. tini is PID 1, reaps zombies, forwards SIGTERM to the process group.

**`docker/entrypoint.sh`** — does, in order:
1. `gosu` privilege drop from root → `hermes` UID
2. Copies `.env.example`, `cli-config.yaml.example`, `SOUL.md` into `$HERMES_HOME` if missing
3. Syncs bundled skills via `tools/skills_sync.py`
4. Optionally backgrounds `hermes dashboard` in a subshell when `HERMES_DASHBOARD=1` — **not supervised**, no restart
5. `exec hermes "$@"` — this becomes tini's sole direct child

**Known limitations we discussed on May 4, 2026:** dashboard crash → stays dead; dashboard fails at startup → silent; gateway crash → dashboard dies too. The May 4 decision was "leave as is" because nothing in the container needed supervision then. Adding per-profile gateway supervision changes that.

### Current ServiceManager surface (what we're wrapping, not refactoring)

All init-system logic lives in **`hermes_cli/gateway.py`** (currently ~5,400 lines). The systemd/launchd code is ~1,500 lines of that, plus a separate **`hermes_cli/gateway_windows.py`** (~690 lines) that ships gateway-as-Scheduled-Task with a Startup-folder fallback for native Windows. Structure (functions named — no line numbers; they drift constantly):

| Layer | Systemd functions | Launchd functions | Windows functions |
|---|---|---|---|
| **Detection** | `supports_systemd_services()`, `_systemd_operational()`, `_wsl_systemd_operational()`, `_container_systemd_operational()` | `is_macos()` | `is_windows()`, `gateway_windows.is_installed()` |
| **Paths** | `get_systemd_unit_path(system)`, `get_service_name()` | `get_launchd_plist_path()`, `get_launchd_label()` | `gateway_windows.get_task_name()`, `get_task_script_path()`, `get_startup_entry_path()` |
| **Install/lifecycle** | `systemd_install(force, system, run_as_user)`, `systemd_uninstall(system)`, `systemd_start/stop/restart(system)` | `launchd_install(force)`, `launchd_uninstall/start/stop/restart` | `gateway_windows.install/uninstall/start/stop/restart` |
| **Probes** | `_probe_systemd_service_running(system)`, `_read_systemd_unit_properties(system)`, `_wait_for_systemd_service_restart`, `_recover_pending_systemd_restart` | `_probe_launchd_service_running()` | `gateway_windows.is_task_registered()`, `_pid_exists` helper |
| **D-Bus plumbing** | `_ensure_user_systemd_env`, `_user_systemd_socket_ready`, `_user_systemd_private_socket_path`, `get_systemd_linger_status` | — (not applicable) | — (not applicable) |
| **Unit/plist generation** | `generate_systemd_unit(system, run_as_user)`, `systemd_unit_is_current`, `refresh_systemd_unit_if_needed` | plist templating in `launchd_install` | `_build_gateway_cmd_script`, `_build_startup_launcher`, `_write_task_script` |

**Callers outside `gateway.py` that are container-relevant:**

- `hermes_cli/status.py` — prints `Manager: systemd/manual` / `launchd` / `Termux / manual process` / `(not supported on this platform)`; needs a new "s6" branch for when status runs inside the container. Search for the `Manager:` literal to find the block.
- `hermes_cli/profiles.py` — `create_profile` and `delete_profile`; the delete path has a `disable systemd/launchd service` helper (the function literally documents "Disable and remove systemd/launchd service for a profile"). The create/delete flow needs to register/unregister with s6 when running inside the container.
- `hermes_cli/doctor.py` — `_check_gateway_service_linger` calls `get_systemd_linger_status()` which is a host-only concept (SSH login survival); inside the container it either silently skips or prints a confusing warning. Needs a "skip on s6 / show s6 supervision status" branch. **Small scope, deferred to Phase 5** because the behavior is cosmetic, not functional. Separately, `hermes doctor`'s External Tools → Docker check is nonsensical inside a container (Docker-in-Docker isn't set up and isn't intended); it would create a spurious warning. Also deferred to Phase 5.
- **`hermes_cli/gateway.py::gateway_command`** — the actual `hermes gateway install/start/stop/restart/uninstall` dispatcher currently has `elif is_container():` arms that *refuse* the operation ("Service installation is not needed inside a Docker container — use Docker restart policies instead", "Service start is not applicable inside a Docker container", etc.). Phase 4 must remove these early-exit arms so the new s6 path can intercept. See Task 4.3.

**Not container-relevant, no changes needed:**
- `hermes_cli/setup.py`, `hermes_cli/uninstall.py` — the setup wizard and uninstall flow are host-only. Users don't run `hermes setup` inside the container (the image ships pre-configured); running `hermes uninstall` inside a container is a no-op on any systemd/launchd unit paths that simply don't exist.
- `hermes_cli/claw.py` — OpenClaw migration operates on `~/.openclaw/` on the host. Inside a container, `Path.home()` is `/opt/data` (the hermes user's home), and no OpenClaw directories exist there since the container was built fresh. `hermes claw migrate` / `cleanup` would cleanly report "nothing to migrate" and exit. No changes required.

### Per-profile gateway spawning (exists today — needs container adaptation)

`hermes gateway start`, `coder gateway start` (profile alias), and `hermes -p <profile> gateway start` all spawn a gateway process scoped to a given profile. See [Profiles: Running Gateways](https://hermes-agent.nousresearch.com/docs/user-guide/profiles#running-gateways). On host the lifecycle is managed via per-profile systemd units (`hermes-gateway-<profile>.service`); inside the Hermes container there is currently no supervisor, so crashes are not recovered and shutdowns are ad-hoc.

**What this plan adds:** when `hermes profile create <name>` runs inside the container, it registers an s6 service at `/run/service/gateway-<name>/` that s6-svscan picks up and supervises. `<name> gateway start/stop/restart` then talks to s6 (`s6-svc -u`, `s6-svc -d`) instead of spawning a bare process. When the profile is deleted, the service directory is removed and s6 tears down the supervise process.

**Persistence across container restart:** `/run/service/` is **tmpfs** — service registrations are wiped when the container restarts. But profile directories at `/opt/data/profiles/<name>/` live on the persistent VOLUME, and each one records its gateway's last state in `gateway_state.json`. Task 4.0 runs as a cont-init.d script on every container boot: it walks the persistent profiles, recreates the s6 service slots, and auto-starts those whose last recorded state was `running`. Profiles whose last state was `stopped`, `startup_failed`, `starting`, or absent get their slot recreated in the `down` state and wait for explicit user action. This means `docker restart` is invisible to a user with running profile gateways: they come back up; stopped ones stay stopped.

### s6-overlay constraints relevant to us

**Root/non-root model (resolved — see OQ2):** `/init` runs as root to set up the supervision tree, install signal handlers, and run the stage2 hook that does `usermod`/`chown`. Each supervised service drops to UID 10000 via `s6-setuidgid hermes` in its `run` script — a single-exec step (no shell subprocess, no zombie risk). The per-service `s6-supervise` monitor stays root so it can signal its child regardless of UID. Net effect: hermes and all its subprocesses run as UID 10000 exactly as today; only the supervision tree itself runs as root.

- v3.2.3.0 (May 2026, latest at re-validation) has limited non-root support for running `/init` itself as non-root — some tools (`fix-attrs`, `logutil-service`) assume root. We don't hit this because `/init` runs as root and individual services drop.
- scandir hard cap: `services_max` default 1000, configurable to 160,000 via `-C`. Way more than we need.
- `/command/with-contenv` sources `/run/s6/container_environment/*` into service env — convenient for passing `HERMES_HOME` etc.
- s6 signal semantics: service crash triggers `s6-supervise` restart after 1s; override with a `finish` script.
- Zombie reaping: PID 1 (s6-svscan) reaps all zombies non-blockingly on SIGCHLD. Any subagent subprocess spawned by the main hermes process is reaped automatically — no special handling required.

---

## Key Design Decisions

### D1. s6-overlay replaces tini entirely

Container ENTRYPOINT becomes `/init`, PID 1 is s6-svscan. The main hermes process, the dashboard, and every per-profile gateway all run as supervised services. This is a single breaking change to the container contract — after this phase lands, every container invocation goes through `/init`.

### D2. Main hermes is an s6 service with container-exit semantics

The current contract "container exits when `hermes` exits" must be preserved. s6-overlay supports this via a service `finish` script that writes to `/run/s6-linux-init-container-results/exitcode` and calls `/run/s6/basedir/bin/halt`. All five supported invocations continue to work:

| `docker run <image> …` | Behavior |
|---|---|
| (no args) | `hermes` with no args, container exits when hermes exits |
| `chat -q "..."` | `hermes chat -q "..."`, container exits with hermes exit code |
| `sleep infinity` | `sleep infinity` directly (long-lived sandbox mode) |
| `bash` | interactive `bash` directly |
| `docker run -it … --tui` | interactive Ink TUI with real TTY — see D9 |

The stage2 hook detects whether `$1` is an executable on PATH and routes either to "run this as a one-shot main service" or "wrap with hermes".

### D3. Static services at build time; dynamic (per-profile) services at runtime

s6 offers two mechanisms:
- **s6-rc** (declarative, compile-then-swap): used for main hermes and the dashboard — they're known at image build time
- **scandir** (drop a directory + `s6-svscanctl -a`): used for per-profile gateways — profiles are user-created after the image is built

Per-profile gateway service dirs live at `/run/service/gateway-<profile>/` (tmpfs, hermes-writable). s6-svscan picks them up on rescan.

### D4. ServiceManager protocol with two methods for runtime registration

Host paths (systemd, launchd, Windows Scheduled Tasks) need only install/start/stop/restart of pre-declared services. Inside the container, we additionally need to register services at runtime when a profile is created. The protocol exposes this directly — no generic "transient" abstraction:

```python
class ServiceManager(Protocol):
    kind: ServiceManagerKind  # "systemd" | "launchd" | "windows" | "s6" | "none"

    # Lifecycle of an already-declared service (existing systemd/launchd/windows + s6)
    def start(self, name: str) -> None: ...
    def stop(self, name: str) -> None: ...
    def restart(self, name: str) -> None: ...
    def is_running(self, name: str) -> bool: ...

    # Runtime registration (container-only; hosts raise NotImplementedError)
    def supports_runtime_registration(self) -> bool: ...
    def register_profile_gateway(self, profile: str, *, command: list[str],
                                 env: dict[str, str] | None = None) -> None: ...
    def unregister_profile_gateway(self, profile: str) -> None: ...
    def list_profile_gateways(self) -> list[str]: ...
```

Systemd, launchd, and Windows backends raise `NotImplementedError` on the registration methods. Only the s6 backend implements them. Callers check `supports_runtime_registration()` before calling.

The scope is intentionally narrow: it's specifically "register/unregister a profile gateway," not a general-purpose process-management API. If we later need other dynamically-registered services, we can add dedicated methods.

### D5. Per-profile gateway service spec is fixed, not user-provided

Every profile gateway has the same command shape (`hermes -p <profile> gateway start --foreground …`). The s6 backend generates the `run` script from a fixed template given the profile name — no arbitrary command list. This keeps the API surface tight and prevents callers from accidentally registering non-gateway services.

```python
def register_profile_gateway(self, profile: str, *, port: int,
                             extra_env: dict[str, str] | None = None) -> None
```

### D6. Add detect_service_manager() alongside supports_systemd_services()

`supports_systemd_services()` stays as-is (14 call sites). A new `detect_service_manager() -> Literal["systemd", "launchd", "windows", "s6", "none"]` composes existing detection functions (`is_macos()`, `is_windows()`, `supports_systemd_services()`, `is_container()` + `_s6_running()`) and adds an s6 branch for container detection. Host call sites continue to use the existing functions; container-only code (the profile hooks) uses the new one.

This is deliberately narrow: protocol + s6 backend are new; host code path is untouched. Future cleanup PR can consolidate.

### D7. Wrap existing systemd/launchd/windows functions, don't rewrite them

`SystemdServiceManager` / `LaunchdServiceManager` / `WindowsServiceManager` are thin adapters over the existing `systemd_*` / `launchd_*` module-level functions in `hermes_cli/gateway.py` and the `gateway_windows.install/uninstall/start/stop/restart/is_installed` functions in `hermes_cli/gateway_windows.py`. Their `start/stop/restart` methods call straight through. We get the abstraction without rewriting ~2,200 lines of working code.

### D8. Profile create/delete hooks register/unregister the s6 service

When `hermes profile create <name>` runs inside the container, the profile-creation code path calls `ServiceManager.register_profile_gateway(<name>, port=…)` if `supports_runtime_registration()` is True. When `hermes profile delete <name>` runs, it calls `unregister_profile_gateway(<name>)`. On host, both calls are no-ops (registration not supported; existing systemd unit generation continues to handle install/uninstall).

Existing per-profile `hermes -p <profile> gateway start/stop/restart` CLI commands continue to work — in the container they dispatch to `ServiceManager.start/stop/restart("gateway-<profile>")`, which translates to `s6-svc -u`/`-d`/`-t` on the service dir.

### D9. Interactive TUI bypasses s6 service-mode and runs as CMD for TTY passthrough

`docker run -it --rm <image> --tui` needs a real TTY connected to container stdin/stdout for Ink raw-mode keyboard input, cursor control, and SIGWINCH. Running the TUI as a normal s6 service fails because s6-supervise disconnects service stdio from the container TTY (documented: [s6-overlay#230](https://github.com/just-containers/s6-overlay/issues/230)).

**The pattern:** s6-overlay's `/init` execs a CMD as the container's "main program" after the supervision tree is up. The CMD inherits stdin/stdout/stderr from `/init` — which in `-it` mode is the container TTY. The stage2 hook detects the TUI case and short-circuits the main-hermes service so the hermes CMD becomes that main program.

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

if _is_tui_invocation "$@"; then
    touch /var/run/s6/container_environment/HERMES_TUI_MODE
fi
```

And in `docker/s6-rc.d/main-hermes/run`:
```sh
if [ -f /var/run/s6/container_environment/HERMES_TUI_MODE ]; then
    exec sleep infinity   # s6-overlay will exec CMD as the TTY-connected main
fi
exec s6-setuidgid hermes hermes ${HERMES_ARGS:-}
```

In TUI mode main hermes is effectively unsupervised (same as today with tini — acceptable because the user is interactively present). Dashboard and profile gateways still get full s6 supervision via their separate services.

**Verification:** Phase 2 integration tests include an explicit TTY passthrough test using `tput cols` and `COLUMNS=123` as the probe. This is a hard gate — Phase 2 cannot merge if the test fails. Per OQ9, if it fails we fall back to the s6-fdholder pattern (Solution 2 in issue #230), but we don't want that — it has documented UX issues.

---

## Phases Overview

This plan is **TDD-first**. Phase 0 builds the regression test harness for the current (tini-based) container so every subsequent phase has a failing→passing test gate. Phase 0.5 adds linting. Phase 1 introduces the ServiceManager abstraction with no behavior change. Phase 2 is the single breaking change — tini out, s6 in, main hermes and dashboard become s6 services. Phase 3 adds the runtime-registration surface used by the profile create/delete hooks. Phase 4 wires profile creation/deletion into s6 and switches `hermes -p X gateway start/stop` to talk to s6 inside the container. Phase 5 is docs/cleanup.

| Phase | Scope | Ships independently? |
|---|---|---|
| **Phase 0** | Test harness covering TUI, main hermes, dashboard, per-profile gateways — all against the current tini-based image. **Must land before any other phase so later changes are TDD.** | Yes — no behavior change |
| **Phase 0.5** | hadolint (Dockerfile) + shellcheck (entrypoint) in CI | Yes — no behavior change |
| **Phase 1** | `ServiceManager` protocol + thin wrappers around existing systemd/launchd | Yes — no behavior change, pure refactor |
| **Phase 2** | s6 replaces tini; main hermes + dashboard become s6 services | **Breaking change** — entrypoint contract changes |
| **Phase 3** | Runtime-registration methods (`register_profile_gateway` / `unregister_profile_gateway`) on the s6 backend | Yes — new capability, no caller yet |
| **Phase 4** | Profile create/delete hooks call the new registration API; container-boot reconciliation re-registers persistent profiles after `docker restart`; `hermes -p X gateway start/stop` talks to s6 inside the container | Yes — activates Phase 3 |
| **Phase 5** | Docs update (`website/docs/user-guide/docker.md`), skill for maintainers, remove dead code | Yes |

Each phase is reviewable, testable, and (except Phase 2) backwards-compatible. Phase 2 is the single breaking moment.

**CI gates between phases:**
- After Phase 0: the test harness runs against `main` (tini image); the two `test_profile_gateway.py` tests are xfailed (Phase 4 target), every other test passes
- After Phase 0.5: hadolint + shellcheck run green on the current Dockerfile + entrypoint
- After Phase 1: Phase 0 harness still passes; `grep -n 'systemd_install\|launchd_install' hermes_cli/` shows unchanged call-site count
- After Phase 2: Phase 0 harness still passes (xfails still xfail until Phase 4); all five invocation patterns (including TUI) produce identical user-visible behavior
- After Phase 3: `ServiceManager.supports_runtime_registration()` returns True in container, False on host
- After Phase 4: `hermes profile create test-profile` inside a container creates `/run/service/gateway-test-profile/` and `hermes -p test-profile gateway start` brings it up; **the two xfail markers in `test_profile_gateway.py` are removed and both tests pass strictly**

---

## Open Questions

All nine questions were resolved during plan review. Kept in-document for posterity; the chosen option is in bold at the top of each.

### OQ1. Do we gate Phase 2 (breaking change) behind an env var for rollout?

**Resolved: A — ship directly, no gate.** Hermes is pre-1.0; users depending on tini-specific behavior can pin to the previous image. Dual-maintenance accumulates cruft.

Options considered:
- A. Ship Phase 2 directly — `/init` becomes the ENTRYPOINT unconditionally
- B. `HERMES_INIT=s6|tini` env var, flip default across releases
- C. Dual entrypoint script kept forever

### OQ2. What happens to the `hermes` user vs. s6-overlay's root assumptions?

**Resolved: A — supervisor runs as root; supervised services drop to UID 10000 via `s6-setuidgid hermes`.** Canonical s6-overlay non-root pattern.

Options considered:
- A. `/init` runs as root → services drop per-service
- B. Run `/init` as hermes with `S6_READ_ONLY_ROOT=1` (broken: `fix-attrs`, `logutil-service` need root)
- C. Everything as root (security regression)

### OQ3. Dashboard as static s6-rc service — how do we honor `HERMES_DASHBOARD=1`?

**Resolved: A — dashboard is always declared as an s6 service; its `run` script checks `HERMES_DASHBOARD` and `exec sleep infinity` if unset.** Simpler than toggling contents.d at runtime.

Options considered:
- A. Always declared, no-op when disabled
- B. Stage2 hook writes/removes `contents.d/dashboard` based on env
- C. Dashboard spawned via register_profile_gateway when enabled

### OQ4. Podman rootless compatibility

**Resolved: A — declare supported; fix issues as they arise during Phase 2 testing.** A Podman-alongside-Docker environment will be stood up locally for validation.

Options considered:
- A. Supported; fix reactively
- B. Declared unsupported
- C. Block Phase 2 until validated

### OQ5. Service naming for per-profile gateways

**Resolved: `gateway-<profile>`.** Matches the existing `hermes-gateway-<profile>.service` systemd naming convention.

### OQ6. — (retired; was about subagent gateways, no longer in scope)

### OQ7. Resource limits per profile gateway

**Resolved: C — YAGNI.** No per-service cgroup limits; rely on the container's overall limit. Revisit if we see evidence of a problem.

Options considered:
- A. No limits
- B. Add `memory_limit_mb` parameter, use `s6-softlimit`
- C. Defer

### OQ8. Log rotation for profile gateways

**Resolved: C — persist logs under `$HERMES_HOME/logs/gateways/<profile>/`.** Matches how the main gateway logs persist today. Each s6 service gets a `log/` subdir with `s6-log` rotation pointed at the persistent path.

**Caveat — `HERMES_HOME` is sourced at service-run time, not registration time.** The log path is *not* hard-coded into the rendered `log/run` script as a literal `/opt/data/...`. Instead, the script reads `${HERMES_HOME:-/opt/data}` from `/run/s6/container_environment/` (populated by the stage2 hook from the container's actual env). This means: if a user starts the container with `-e HERMES_HOME=/data/hermes`, profile gateway logs land at `/data/hermes/logs/gateways/<profile>/current` — not silently regress to `/opt/data/...`. Implementations of `_render_log_run` MUST therefore avoid string-substituting the path at Python time; they must emit a shell expansion of the env var. See Task 3.2.

Options considered:
- A. Enable at `/run/service/gateway-<profile>/log/current` (tmpfs — lost on restart)
- B. Swallow (stdout to s6-supervise, lost)
- C. Persist under `$HERMES_HOME/logs/gateways/<profile>/`

### OQ9. TUI TTY passthrough via s6-overlay CMD mode — is it actually reliable?

**Resolved: A — trust the documented pattern ([s6-overlay#230](https://github.com/just-containers/s6-overlay/issues/230) Solution 1), with manual testing + the automated Phase 2 integration test as the hard gate.** If the automated test fails, manual testing catches the regression before Phase 2 merges; we'd then fall back to the fdholder pattern.

Options considered:
- A. Trust docs; test is the gate
- B. Prototype first (+0.5 day)
- C. Use s6-fdholder (more complex, known UX issues)

---

## Phase 0 — Test Harness (TDD foundation)

**Goal:** Build a docker-image test harness that exercises every user-visible feature of the current tini-based image, so Phase 2's change can be validated as "identical behavior." Land this **before any other phase**.

### Task 0.1: Create the test-harness pytest marker and skip-condition

**Objective:** All harness tests live under `tests/docker/` and are marked so they only run when Docker is available. CI can opt in via `--run-docker`.

**Files:**
- Create: `tests/docker/__init__.py` (empty)
- Create: `tests/docker/conftest.py`

**Step 1: Write `tests/docker/conftest.py`**

```python
"""Shared fixtures for docker-image integration tests.

Tests in this directory build the image with the current `Dockerfile`
and exercise it via `docker run`. They skip when Docker is unavailable
(e.g. on developer laptops without a daemon).
"""
import os
import shutil
import subprocess
import pytest

IMAGE_TAG = os.environ.get("HERMES_TEST_IMAGE", "hermes-agent-harness:latest")


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def pytest_collection_modifyitems(config, items):
    skip_docker = pytest.mark.skip(reason="Docker not available or daemon not running")
    if not _docker_available():
        for item in items:
            if "tests/docker/" in str(item.fspath):
                item.add_marker(skip_docker)


@pytest.fixture(scope="session")
def built_image():
    """Build the image once per test session. Override with HERMES_TEST_IMAGE
    env var to point at a pre-built image (faster local iteration)."""
    if os.environ.get("HERMES_TEST_IMAGE"):
        return IMAGE_TAG
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    result = subprocess.run(
        ["docker", "build", "-t", IMAGE_TAG, repo_root],
        capture_output=True, text=True, timeout=1200,
    )
    assert result.returncode == 0, f"docker build failed:\n{result.stderr[-2000:]}"
    return IMAGE_TAG


@pytest.fixture
def container_name(request):
    """Generate a unique container name + ensure cleanup on test exit."""
    name = f"hermes-test-{request.node.name.replace('[', '_').replace(']', '_')}"
    yield name
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)
```

**Step 2: Commit**

```bash
git add tests/docker/__init__.py tests/docker/conftest.py
git commit -m "test(docker): add conftest fixtures for docker harness"
```

### Task 0.2: Harness — main hermes invocation patterns

**Objective:** Lock behavior of `docker run <image>`, `docker run <image> chat -q …`, `docker run <image> sleep infinity`, `docker run <image> bash`.

**Files:**
- Create: `tests/docker/test_main_invocation.py`

**Step 1: Write the tests**

```python
"""Harness: docker run <image> [cmd...] invocation patterns.

These tests MUST pass on the current tini-based image AND continue to
pass after the Phase 2 s6 migration. Any behavior drift is a regression.
"""
import subprocess


def test_no_args_starts_hermes(built_image):
    """`docker run <image>` should start hermes (exits with code 0 or 1 —
    depends on whether config is present, but must not crash with a stack trace)."""
    r = subprocess.run(
        ["docker", "run", "--rm", built_image, "--version"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode in (0, 1), f"Unexpected exit {r.returncode}: {r.stderr}"
    assert "Traceback" not in r.stderr


def test_chat_subcommand_passthrough(built_image):
    """`docker run <image> chat -q "hi"` should exec `hermes chat -q "hi"`."""
    # Use --help so we don't need a model configured
    r = subprocess.run(
        ["docker", "run", "--rm", built_image, "chat", "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0
    assert "chat" in r.stdout.lower() or "usage" in r.stdout.lower()


def test_bare_executable_passthrough(built_image):
    """`docker run <image> sleep 1` should exec `sleep 1` directly."""
    r = subprocess.run(
        ["docker", "run", "--rm", built_image, "sleep", "1"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0


def test_bash_pattern(built_image):
    """`docker run <image> bash -c "echo ok"` should exec bash directly."""
    r = subprocess.run(
        ["docker", "run", "--rm", built_image, "bash", "-c", "echo ok"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0
    assert "ok" in r.stdout


def test_container_exit_code_matches_hermes_exit(built_image):
    """`docker run <image> sh -c 'exit 42'` — container should exit with 42."""
    r = subprocess.run(
        ["docker", "run", "--rm", built_image, "sh", "-c", "exit 42"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 42
```

**Step 2: Run against current image — should pass**

```bash
scripts/run_tests.sh tests/docker/test_main_invocation.py -v
```

Expected: 5 passed.

**Step 3: Commit**

```bash
git add tests/docker/test_main_invocation.py
git commit -m "test(docker): lock main hermes invocation patterns"
```

### Task 0.3: Harness — interactive TUI

**Objective:** Lock the `docker run -it … --tui` behavior. This is the hardest test to automate because it requires a PTY on the host side.

**Files:**
- Create: `tests/docker/test_tui_passthrough.py`

**Step 1: Write the test**

```python
"""Harness: interactive TUI TTY passthrough.

Uses `script -qc` on the host to allocate a PTY for the docker client,
which then allocates a container-side PTY via `-t`. The probe inside the
container is `tput cols`, which returns a real column count when stdout
is a TTY and 80 (the terminfo fallback) or nothing when it is not.

We set COLUMNS=123 in the container env so a real TTY reports 123.
"""
import shlex
import shutil
import subprocess
import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("script") is None, reason="`script` command not available"
)


def test_tty_passthrough_to_container(built_image):
    """`docker run -t` must deliver a real TTY to the container process."""
    probe = "if [ -t 1 ]; then tput cols; else echo NO_TTY; fi"
    cmd = f"docker run --rm -t -e COLUMNS=123 {built_image} sh -c {shlex.quote(probe)}"
    r = subprocess.run(
        ["script", "-qc", cmd, "/dev/null"],
        capture_output=True, text=True, timeout=120,
    )
    output = r.stdout.strip()
    assert "NO_TTY" not in output, f"TTY passthrough failed: {output!r}"
    # Real TTY reports a positive number. With COLUMNS=123 in env and a real
    # PTY, tput should agree with COLUMNS or report the PTY width.
    numeric_lines = [s for s in output.split() if s.strip().isdigit()]
    assert numeric_lines, f"No numeric width in output: {output!r}"
    assert int(numeric_lines[0]) > 0


def test_tui_flag_recognized(built_image):
    """`docker run -it <image> --tui --help` should at minimum not crash."""
    cmd = f"docker run --rm -t {built_image} --help"
    r = subprocess.run(
        ["script", "-qc", cmd, "/dev/null"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0
```

**Step 2: Run — should pass against current tini image**

```bash
scripts/run_tests.sh tests/docker/test_tui_passthrough.py -v
```

**Step 3: Commit**

```bash
git add tests/docker/test_tui_passthrough.py
git commit -m "test(docker): lock TTY passthrough for interactive TUI"
```

### Task 0.4: Harness — dashboard opt-in and crash behavior

**Objective:** Lock the HERMES_DASHBOARD=1 opt-in. Current (tini) behavior: dashboard starts once; if it crashes it stays dead. After Phase 2: dashboard starts once; if it crashes it restarts.

**Files:**
- Create: `tests/docker/test_dashboard.py`

**Step 1: Write the tests**

```python
"""Harness: dashboard opt-in via HERMES_DASHBOARD."""
import subprocess
import time


def test_dashboard_not_running_by_default(built_image, container_name):
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name, built_image,
         "sleep", "30"],
        check=True, capture_output=True, timeout=30,
    )
    time.sleep(3)
    r = subprocess.run(
        ["docker", "exec", container_name, "pgrep", "-f", "hermes dashboard"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode != 0, "Dashboard should NOT be running without HERMES_DASHBOARD"


def test_dashboard_opt_in_starts(built_image, container_name):
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name,
         "-e", "HERMES_DASHBOARD=1", built_image, "sleep", "30"],
        check=True, capture_output=True, timeout=30,
    )
    time.sleep(5)
    r = subprocess.run(
        ["docker", "exec", container_name, "pgrep", "-f", "hermes dashboard"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0, f"Dashboard should be running with HERMES_DASHBOARD=1"


def test_dashboard_port_override(built_image, container_name):
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name,
         "-e", "HERMES_DASHBOARD=1", "-e", "HERMES_DASHBOARD_PORT=9120",
         built_image, "sleep", "30"],
        check=True, capture_output=True, timeout=30,
    )
    time.sleep(5)
    r = subprocess.run(
        ["docker", "exec", container_name, "sh", "-c",
         "ss -tlnp 2>/dev/null | grep ':9120' || netstat -tln | grep ':9120'"],
        capture_output=True, text=True, timeout=10,
    )
    assert "9120" in r.stdout, f"Dashboard not listening on 9120: {r.stdout}"
```

**Note:** this task documents an explicit behavior difference between tini and s6:
- On tini (pre-Phase 2): dashboard crash stays dead. No restart test — we'd be encoding broken behavior as an invariant.
- On s6 (post-Phase 2): dashboard crash is supervised and restarted. A new test `test_dashboard_restarts_after_crash` is added in Phase 2 Task 2.5.

**Step 2: Commit**

```bash
git add tests/docker/test_dashboard.py
git commit -m "test(docker): lock dashboard opt-in behavior"
```

### Task 0.5: Harness — per-profile gateway lifecycle

**Objective:** Lock the `hermes profile create` + `<profile> gateway start` flow *inside* the container. This is the feature we're going to materially change in Phase 4, so the harness here needs to cover exactly the user-visible surface we're preserving.

**Important caveat — these tests describe the POST-PHASE-4 behavior, not the current one.** Today, `hermes gateway start` inside the container deliberately exits with status 0 and prints "Service start is not applicable inside a Docker container — the gateway runs as the container's main process. Run the gateway directly: hermes gateway run." So `pgrep -f 'gateway.*<profile>'` will find nothing and the tests below will fail against the tini image. That's expected. The tests are marked `xfail(strict=True)` here so they:

1. Run in Phase 0 and confirm they're currently failing for the documented reason (no silent skip).
2. Flip to passing automatically in Phase 4 when `_dispatch_via_service_manager_if_s6` lands AND the `elif is_container():` rejection arms in `gateway_command` are removed (Task 4.3).
3. `strict=True` means an unexpected pass also fails the test — i.e. if someone accidentally fixes container-side gateway lifecycle outside the Phase 4 mechanism, we hear about it.

**Files:**
- Create: `tests/docker/test_profile_gateway.py`

**Step 1: Write the tests**

```python
"""Harness: per-profile gateway start/stop inside the container.

Phase 4 will change the *implementation* of these commands inside the
container (they'll talk to s6 instead of refusing). The user-visible
surface that should result is locked here.

NOTE: These tests are marked xfail(strict=True) until Phase 4 lands.
The current tini image deliberately refuses gateway start/stop inside
containers — `pgrep` finds nothing and the tests fail. After Phase 4
they should flip to passing automatically.
"""
import subprocess
import time
import pytest

PROFILE = "test-harness-profile"

_PHASE4_REASON = (
    "Phase 4 not yet landed: container-side `hermes gateway start` "
    "currently exits 0 with an informational message instead of "
    "spawning/supervising a gateway. Remove this marker after Task 4.3."
)


def _sh(container: str, command: str, timeout: int = 30):
    return subprocess.run(
        ["docker", "exec", container, "sh", "-c", command],
        capture_output=True, text=True, timeout=timeout,
    )


@pytest.mark.xfail(reason=_PHASE4_REASON, strict=True)
def test_profile_create_then_gateway_start(built_image, container_name):
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name, built_image,
         "sleep", "120"],
        check=True, capture_output=True, timeout=30,
    )
    time.sleep(3)

    # Create the profile
    r = _sh(container_name, f"hermes profile create {PROFILE}")
    assert r.returncode == 0, f"profile create failed: {r.stderr}"

    # Start its gateway (foreground=False returns after spawn)
    r = _sh(container_name, f"hermes -p {PROFILE} gateway start", timeout=60)
    assert r.returncode == 0, f"gateway start failed: {r.stderr}\n{r.stdout}"

    time.sleep(3)

    # Process should exist
    r = _sh(container_name, f"pgrep -f 'gateway.*{PROFILE}'")
    assert r.returncode == 0, "gateway process not running"

    # Stop it
    r = _sh(container_name, f"hermes -p {PROFILE} gateway stop", timeout=30)
    assert r.returncode == 0

    time.sleep(2)

    # Process should be gone
    r = _sh(container_name, f"pgrep -f 'gateway.*{PROFILE}'")
    assert r.returncode != 0, "gateway process still running after stop"


@pytest.mark.xfail(reason=_PHASE4_REASON, strict=True)
def test_profile_delete_stops_gateway(built_image, container_name):
    """Deleting a profile should stop its gateway if running."""
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name, built_image,
         "sleep", "120"],
        check=True, capture_output=True, timeout=30,
    )
    time.sleep(3)

    _sh(container_name, f"hermes profile create {PROFILE}")
    _sh(container_name, f"hermes -p {PROFILE} gateway start", timeout=60)
    time.sleep(3)

    r = _sh(container_name, f"hermes profile delete {PROFILE} --yes", timeout=30)
    assert r.returncode == 0

    time.sleep(2)
    r = _sh(container_name, f"pgrep -f 'gateway.*{PROFILE}'")
    assert r.returncode != 0, "gateway still running after profile delete"
```

**Step 2: Run — confirm both fail as expected**

```bash
scripts/run_tests.sh tests/docker/test_profile_gateway.py -v
```

Expected: 2 `xfailed` (the strict=True ones). If either *passes* unexpectedly, investigate before moving on — something has changed about container behavior that the plan doesn't account for. If either *errors* (rather than failing), the docker fixture/build is broken and needs fixing before proceeding.

**Step 3: Commit**

```bash
git add tests/docker/test_profile_gateway.py
git commit -m "test(docker): lock per-profile gateway lifecycle target (xfail until Phase 4)"
```

**Task 4.3 reminder:** when Phase 4 lands, remove both `@pytest.mark.xfail(...)` markers and the `_PHASE4_REASON` constant. The tests should then pass against the s6 image.

### Task 0.6: Harness — zombie reaping

**Objective:** Lock the current behavior that tini reaps zombie processes spawned by hermes subagent subprocesses.

**Files:**
- Create: `tests/docker/test_zombie_reaping.py`

**Step 1: Write the test**

```python
"""Harness: PID 1 must reap orphaned zombies."""
import subprocess
import time


def test_orphan_zombies_reaped(built_image, container_name):
    """Spawn an orphan child that exits immediately. PID 1 must reap it."""
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name, built_image,
         "sleep", "60"],
        check=True, capture_output=True, timeout=30,
    )
    time.sleep(2)

    # Spawn an orphan process tree that creates a zombie
    subprocess.run(
        ["docker", "exec", container_name, "sh", "-c",
         "( ( sleep 0.1 & ) & ); sleep 1"],
        capture_output=True, text=True, timeout=10,
    )
    time.sleep(1)

    # Check for zombies (ps shows 'Z' in STAT column for zombies)
    r = subprocess.run(
        ["docker", "exec", container_name, "ps", "axo", "stat,pid,comm"],
        capture_output=True, text=True, timeout=10,
    )
    zombies = [line for line in r.stdout.split("\n") if line.strip().startswith("Z")]
    assert not zombies, f"Zombies not reaped: {zombies}"
```

**Step 2: Commit**

```bash
git add tests/docker/test_zombie_reaping.py
git commit -m "test(docker): lock zombie reaping by PID 1"
```

### Task 0.7: Run full harness, document baseline

**Objective:** All Phase 0 tests pass against the current image. This is the baseline for every subsequent phase.

```bash
scripts/run_tests.sh tests/docker/ -v
```

Expected: all pass. If any fail, investigate before proceeding to Phase 0.5.

---

## Phase 0.5 — Dockerfile and shell linting

**Goal:** Bring `hadolint` (Dockerfile) and `shellcheck` (entrypoint script) into CI. These catch classes of regression that the behavioral harness can't — e.g. `RUN` commands that fail silently, unquoted variable expansions.

### Task 0.5.1: Add hadolint to CI

**Objective:** `hadolint Dockerfile` runs in CI and fails the build on warnings.

**Files:**
- Create: `.hadolint.yaml`
- Modify: `.github/workflows/ci.yml` (or wherever Docker-related CI lives)

**Step 1: Write `.hadolint.yaml` with starting ruleset**

```yaml
# hadolint configuration for the Hermes Agent Dockerfile.
# See https://github.com/hadolint/hadolint#configure for rules.
failure-threshold: warning

# Allow pinning to specific versions of system packages via apt-get — this is
# a pragmatic tradeoff for a fast-moving project.
ignored:
  - DL3008  # Pin versions in apt get install (we intentionally don't pin common tools)
  - DL3009  # Delete apt-get lists after installing (we do this, hadolint occasionally false-positives)

# Require explicit base-image pins (SHA256) which we already do.
trusted-registries:
  - docker.io
  - ghcr.io
```

**Step 2: Run hadolint against the current Dockerfile**

```bash
docker run --rm -i hadolint/hadolint:latest < Dockerfile
```

Fix any warnings raised (do not ignore them by adding to `.hadolint.yaml` unless they're genuinely false positives — document the rationale for each ignore).

**Step 3: Add CI job**

Append to the existing CI workflow (file path depends on current CI layout — check `.github/workflows/`):

```yaml
  lint-dockerfile:
    name: Lint Dockerfile (hadolint)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: hadolint/hadolint-action@v3.1.0
        with:
          dockerfile: Dockerfile
          config: .hadolint.yaml
          failure-threshold: warning
```

**Step 4: Commit**

```bash
git add .hadolint.yaml .github/workflows/ci.yml Dockerfile
git commit -m "ci: add hadolint for Dockerfile linting"
```

### Task 0.5.2: Add shellcheck to CI for docker entrypoint

**Objective:** `shellcheck docker/entrypoint.sh` runs in CI and fails on errors.

**Files:**
- Modify: `.github/workflows/ci.yml`

**Step 1: Run shellcheck against the current entrypoint**

```bash
shellcheck docker/entrypoint.sh
```

Fix any errors raised. Use `# shellcheck disable=SCxxxx` with a one-line justification for each intentional exception.

**Step 2: Add CI job**

```yaml
  lint-shell:
    name: Lint shell scripts (shellcheck)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run shellcheck
        uses: ludeeus/action-shellcheck@master
        with:
          scandir: './docker'
          severity: error
```

**Step 3: Commit**

```bash
git add .github/workflows/ci.yml docker/entrypoint.sh
git commit -m "ci: add shellcheck for docker/ shell scripts"
```

---

## Phase 1 — ServiceManager protocol + systemd/launchd wrappers

**Goal:** Introduce `ServiceManager` Protocol with the runtime-registration surface from D4. Wrap existing `systemd_*` / `launchd_*` functions behind it. No behavior change; pure refactor.

Phase 0 harness must keep passing across this phase.

### Task 1.1: Create ServiceManager protocol module

**Objective:** Define the abstract interface.

**Files:**
- Create: `hermes_cli/service_manager.py`
- Create: `tests/hermes_cli/test_service_manager.py`

**Step 1: Write `tests/hermes_cli/test_service_manager.py`**

```python
"""Tests for the ServiceManager protocol and detect_service_manager()."""
import pytest
from hermes_cli.service_manager import (
    ServiceManager,
    detect_service_manager,
)


def test_detect_service_manager_returns_known_value():
    result = detect_service_manager()
    assert result in ("systemd", "launchd", "windows", "s6", "none")


def test_profile_name_validation():
    """Profile names used for registration must be safe as directory names."""
    from hermes_cli.service_manager import validate_profile_name
    # Valid
    validate_profile_name("coder")
    validate_profile_name("my-profile")
    validate_profile_name("assistant_v2")
    # Invalid: uppercase
    with pytest.raises(ValueError):
        validate_profile_name("Coder")
    # Invalid: path traversal
    with pytest.raises(ValueError):
        validate_profile_name("foo/bar")
    # Invalid: empty
    with pytest.raises(ValueError):
        validate_profile_name("")
    # Invalid: too long (s6 name_max is 251)
    with pytest.raises(ValueError):
        validate_profile_name("a" * 252)
```

**Step 2: Create `hermes_cli/service_manager.py`**

```python
"""Abstract service manager interface.

Wraps the existing systemd (Linux host), launchd (macOS host), and
s6 (container) backends behind a common Protocol. Only the s6 backend
supports runtime registration (for per-profile gateways).

Host-side call sites (setup wizard, uninstall, status) continue to
use the existing module-level functions in hermes_cli.gateway —
this protocol is a thin facade used by new code that needs to be
backend-agnostic (specifically the profile create/delete hooks).
"""
from __future__ import annotations

import re
from typing import Literal, Protocol, runtime_checkable

ServiceManagerKind = Literal["systemd", "launchd", "windows", "s6", "none"]

_VALID_PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_MAX_PROFILE_LEN = 251  # s6-svscan -L default (name_max)


def validate_profile_name(name: str) -> None:
    """Raise ValueError if `name` is not usable as a profile name.

    Profile names are used as s6 service directory names, so they must
    match a conservative subset of filesystem-safe characters.
    """
    if not name:
        raise ValueError("profile name must not be empty")
    if len(name) > _MAX_PROFILE_LEN:
        raise ValueError(f"profile name too long ({len(name)} > {_MAX_PROFILE_LEN})")
    if not _VALID_PROFILE_RE.match(name):
        raise ValueError(
            f"profile name must match [a-z0-9][a-z0-9_-]*, got {name!r}"
        )


@runtime_checkable
class ServiceManager(Protocol):
    """Abstract interface for init-system-specific service operations.

    Lifecycle methods (start/stop/restart/is_running) are implemented by
    all backends. Runtime registration (register_profile_gateway /
    unregister_profile_gateway) is only implemented by the s6 backend —
    callers MUST check supports_runtime_registration() before using it.
    """

    kind: ServiceManagerKind

    # Lifecycle of a pre-declared service
    def start(self, name: str) -> None: ...
    def stop(self, name: str) -> None: ...
    def restart(self, name: str) -> None: ...
    def is_running(self, name: str) -> bool: ...

    # Runtime registration (s6 only)
    def supports_runtime_registration(self) -> bool: ...
    def register_profile_gateway(
        self, profile: str, *, port: int,
        extra_env: dict[str, str] | None = None,
    ) -> None: ...
    def unregister_profile_gateway(self, profile: str) -> None: ...
    def list_profile_gateways(self) -> list[str]: ...


def detect_service_manager() -> ServiceManagerKind:
    """Detect which service manager is available in this environment.

    Returns "s6" in a container when /init is s6-svscan, "windows" on
    native Windows, "launchd" on macOS, "systemd" on Linux hosts with
    systemctl, "none" otherwise.

    Does NOT replace supports_systemd_services() — host call sites
    continue to use that. This is for new backend-agnostic code.
    """
    from hermes_cli.gateway import is_macos, is_windows, supports_systemd_services
    from hermes_constants import is_container

    if is_container() and _s6_running():
        return "s6"
    if is_windows():
        return "windows"
    if is_macos():
        return "launchd"
    if supports_systemd_services():
        return "systemd"
    return "none"


def _s6_running() -> bool:
    """True when s6-svscan is running as PID 1 in this container."""
    from pathlib import Path
    try:
        exe = Path("/proc/1/exe").resolve()
        return exe.name in ("s6-svscan", "init") and Path("/run/s6").exists()
    except (OSError, RuntimeError):
        return False
```

**Step 3: Run tests — pass**

```bash
scripts/run_tests.sh tests/hermes_cli/test_service_manager.py -v
```

Expected: 2 passed.

**Step 4: Commit**

```bash
git add hermes_cli/service_manager.py tests/hermes_cli/test_service_manager.py
git commit -m "feat(service_manager): introduce ServiceManager protocol and detection"
```

### Task 1.2: Add SystemdServiceManager, LaunchdServiceManager, WindowsServiceManager wrappers

**Objective:** Wrap the existing `systemd_*` / `launchd_*` module-level functions in `hermes_cli/gateway.py` and the `gateway_windows.*` functions in `hermes_cli/gateway_windows.py`. Lifecycle methods delegate; runtime registration raises NotImplementedError.

**Files:**
- Modify: `hermes_cli/service_manager.py`
- Modify: `tests/hermes_cli/test_service_manager.py`

> **v3 note:** `gateway_windows.install()` signature is now `install(force=False, *, start_now=None, start_on_login=None, elevated_handoff=False)` (PRs `d948de39e` + `417a653d9`, ~420 LOC of changes between v2 and v3). The `WindowsServiceManager` wrapper currently isn't called from any non-Windows code path, so accept these kwargs with sensible defaults and forward them:
>
> ```python
> class WindowsServiceManager:
>     kind = "windows"
>     def install(self, *, force=False, start_now=None, start_on_login=None,
>                 elevated_handoff=False) -> None:
>         from hermes_cli import gateway_windows as gw
>         gw.install(force=force, start_now=start_now,
>                    start_on_login=start_on_login,
>                    elevated_handoff=elevated_handoff)
> ```
>
> `SystemdServiceManager.install` and `LaunchdServiceManager.install` continue to take just `force` plus their respective backend-specific args (e.g. systemd's `system: bool`, `run_as_user: str`). The protocol's `install` signature is therefore lifecycle-only — keep it minimal (`install(force: bool = False) -> None`) and let backends absorb the extra args via keyword-only on the concrete class. Callers that need the Windows kwargs must already be on the Windows path.

**Step 1: Write failing tests**

```python
def test_systemd_manager_kind_and_registration_unsupported():
    from hermes_cli.service_manager import SystemdServiceManager
    mgr = SystemdServiceManager()
    assert mgr.kind == "systemd"
    assert mgr.supports_runtime_registration() is False
    with pytest.raises(NotImplementedError):
        mgr.register_profile_gateway("foo", port=9100)
    with pytest.raises(NotImplementedError):
        mgr.unregister_profile_gateway("foo")
    assert mgr.list_profile_gateways() == []


def test_launchd_manager_kind_and_registration_unsupported():
    from hermes_cli.service_manager import LaunchdServiceManager
    mgr = LaunchdServiceManager()
    assert mgr.kind == "launchd"
    assert mgr.supports_runtime_registration() is False


def test_windows_manager_kind_and_registration_unsupported():
    from hermes_cli.service_manager import WindowsServiceManager
    mgr = WindowsServiceManager()
    assert mgr.kind == "windows"
    assert mgr.supports_runtime_registration() is False
    with pytest.raises(NotImplementedError):
        mgr.register_profile_gateway("foo", port=9100)
```

**Step 2: Add wrapper classes**

Append to `hermes_cli/service_manager.py`:

```python
class _RegistrationUnsupportedMixin:
    """Mixin for host backends that don't support runtime registration."""

    def supports_runtime_registration(self) -> bool:
        return False

    def register_profile_gateway(
        self, profile: str, *, port: int,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support runtime profile "
            "gateway registration (container-only feature)"
        )

    def unregister_profile_gateway(self, profile: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support runtime profile "
            "gateway unregistration (container-only feature)"
        )

    def list_profile_gateways(self) -> list[str]:
        return []


class SystemdServiceManager(_RegistrationUnsupportedMixin):
    """Thin wrapper around systemd_* functions in hermes_cli.gateway.

    Host call sites continue to use the module-level functions directly;
    this wrapper exists for backend-agnostic code (the profile hooks).
    """
    kind: ServiceManagerKind = "systemd"

    def start(self, name: str) -> None:
        from hermes_cli.gateway import systemd_start
        systemd_start()  # operates on the current profile's gateway by default

    def stop(self, name: str) -> None:
        from hermes_cli.gateway import systemd_stop
        systemd_stop()

    def restart(self, name: str) -> None:
        from hermes_cli.gateway import systemd_restart
        systemd_restart()

    def is_running(self, name: str) -> bool:
        from hermes_cli.gateway import _probe_systemd_service_running
        _, running = _probe_systemd_service_running()
        return running


class LaunchdServiceManager(_RegistrationUnsupportedMixin):
    """Thin wrapper around launchd_* functions in hermes_cli.gateway."""
    kind: ServiceManagerKind = "launchd"

    def start(self, name: str) -> None:
        from hermes_cli.gateway import launchd_start
        launchd_start()

    def stop(self, name: str) -> None:
        from hermes_cli.gateway import launchd_stop
        launchd_stop()

    def restart(self, name: str) -> None:
        from hermes_cli.gateway import launchd_restart
        launchd_restart()

    def is_running(self, name: str) -> bool:
        from hermes_cli.gateway import _probe_launchd_service_running
        return _probe_launchd_service_running()


class WindowsServiceManager(_RegistrationUnsupportedMixin):
    """Thin wrapper around gateway_windows.* functions.

    Native Windows uses a Scheduled Task (or a Startup-folder fallback)
    instead of an init-system service. Lifecycle delegates to the
    existing `gateway_windows` module which already handles both paths.
    """
    kind: ServiceManagerKind = "windows"

    def start(self, name: str) -> None:
        from hermes_cli import gateway_windows
        gateway_windows.start()

    def stop(self, name: str) -> None:
        from hermes_cli import gateway_windows
        gateway_windows.stop()

    def restart(self, name: str) -> None:
        from hermes_cli import gateway_windows
        gateway_windows.restart()

    def is_running(self, name: str) -> bool:
        # gateway_windows tracks installed/registered state; combine with
        # process-level check via the existing helpers in hermes_cli.gateway.
        from hermes_cli import gateway_windows
        from hermes_cli.gateway import find_gateway_pids
        if not gateway_windows.is_installed():
            return False
        return bool(find_gateway_pids())
```

**Note:** the `name` parameter on these wrappers is currently unused — the underlying systemd/launchd/windows functions operate on the current profile. This is a known limitation; host-side, callers use the profile-aware CLI surface (`hermes -p <name> gateway start`) which loads the right profile before calling these functions. The wrapper API shape is designed for s6 where `name` is the service-directory name.

**Step 3: Run tests — pass**

```bash
scripts/run_tests.sh tests/hermes_cli/test_service_manager.py -v
```

Expected: 5 passed.

**Step 4: Commit**

```bash
git add hermes_cli/service_manager.py tests/hermes_cli/test_service_manager.py
git commit -m "feat(service_manager): add Systemd/Launchd/Windows ServiceManager wrappers"
```

### Task 1.3: Factory function get_service_manager()

**Objective:** Single entry point for picking the right backend based on the current environment.

**Files:**
- Modify: `hermes_cli/service_manager.py`
- Modify: `tests/hermes_cli/test_service_manager.py`

**Step 1: Tests**

```python
def test_get_service_manager_returns_correct_backend(monkeypatch):
    from hermes_cli import service_manager as sm
    monkeypatch.setattr(sm, "detect_service_manager", lambda: "systemd")
    assert isinstance(sm.get_service_manager(), sm.SystemdServiceManager)
    monkeypatch.setattr(sm, "detect_service_manager", lambda: "launchd")
    assert isinstance(sm.get_service_manager(), sm.LaunchdServiceManager)
    monkeypatch.setattr(sm, "detect_service_manager", lambda: "windows")
    assert isinstance(sm.get_service_manager(), sm.WindowsServiceManager)
    monkeypatch.setattr(sm, "detect_service_manager", lambda: "none")
    with pytest.raises(RuntimeError, match="no supported service manager"):
        sm.get_service_manager()
```

**Step 2: Add factory**

```python
def get_service_manager() -> ServiceManager:
    """Return the ServiceManager instance for this environment.

    Raises RuntimeError when no supported backend is available. The s6
    backend ships in Phase 3; until then, "s6" detection raises.
    """
    kind = detect_service_manager()
    if kind == "systemd":
        return SystemdServiceManager()
    if kind == "launchd":
        return LaunchdServiceManager()
    if kind == "windows":
        return WindowsServiceManager()
    if kind == "s6":
        raise RuntimeError("s6 backend not yet implemented (Phase 3)")
    raise RuntimeError("no supported service manager detected")
```

**Step 3: Commit**

```bash
git add hermes_cli/service_manager.py tests/hermes_cli/test_service_manager.py
git commit -m "feat(service_manager): add get_service_manager() factory"
```

### Task 1.4: CI gate — no regressions

```bash
scripts/run_tests.sh tests/hermes_cli/ tests/docker/ -v
```

Verify:
- Phase 0 harness still passes
- No call sites modified:
  ```bash
  git diff --stat main -- hermes_cli/gateway.py hermes_cli/setup.py \
      hermes_cli/uninstall.py hermes_cli/profiles.py hermes_cli/status.py
  ```
  Expected: 0 files changed outside of `hermes_cli/service_manager.py` and its tests.

---

## Phase 2 — s6 replaces tini as PID 1 (BREAKING)

**Goal:** Container ENTRYPOINT becomes `/init`. Main hermes runs as an s6 service with container-exit semantics. Dashboard is a separately-supervised s6 service. `tini` is removed. Interactive TUI passthrough works.

**The hard gate:** The Phase 0 harness (all tests in `tests/docker/`) must pass unchanged after this phase. No behavior drift.

### Task 2.1: Install s6-overlay in the image (still using tini as PID 1)

**Objective:** Add s6-overlay binaries to the image as a separate Dockerfile layer. Before this task is done, tini is still PID 1; after, s6 binaries are on PATH but unused.

**Files:**
- Modify: `Dockerfile` — add new layer after the existing apt install block

**Step 1: Add the install layer**

In `Dockerfile`, insert after the existing `apt-get install ... && rm -rf /var/lib/apt/lists/*` block:

```dockerfile
# ---------- s6-overlay install ----------
# s6-overlay provides supervision for the main hermes process, the dashboard,
# and per-profile gateways. /init becomes PID 1 later in this Dockerfile.
ARG S6_OVERLAY_VERSION=3.2.3.0
ADD https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-noarch.tar.xz /tmp/
ADD https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-x86_64.tar.xz /tmp/
ADD https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-symlinks-noarch.tar.xz /tmp/
RUN tar -C / -Jxpf /tmp/s6-overlay-noarch.tar.xz && \
    tar -C / -Jxpf /tmp/s6-overlay-x86_64.tar.xz && \
    tar -C / -Jxpf /tmp/s6-overlay-symlinks-noarch.tar.xz && \
    rm /tmp/s6-overlay-*.tar.xz
```

> **Note:** If you need to build for aarch64 (M1/M2 Macs, ARM servers), substitute `s6-overlay-x86_64.tar.xz` with `s6-overlay-aarch64.tar.xz`. The plan currently assumes x86_64; multi-arch is out of scope and deferred to a follow-up. See the `Dockerfile`'s base image — if it goes multi-arch, this layer needs `TARGETARCH` plumbing.

**Step 2: Rebuild and re-run Phase 0 harness**

```bash
docker build -t hermes-agent-harness:latest .
scripts/run_tests.sh tests/docker/ -v
```

Expected: all pass (binaries installed but not yet in use).

**Step 3: Commit**

```bash
git add Dockerfile
git commit -m "feat(docker): install s6-overlay v3.2.3.0 (not yet PID 1)"
```

### Task 2.2: Create s6-rc service definitions for main hermes and dashboard

**Objective:** Declarative service directories shipped in the image.

**Files:**
- Create: `docker/s6-rc.d/main-hermes/type`
- Create: `docker/s6-rc.d/main-hermes/run`
- Create: `docker/s6-rc.d/main-hermes/finish`
- Create: `docker/s6-rc.d/main-hermes/dependencies.d/base` (empty)
- Create: `docker/s6-rc.d/dashboard/type`
- Create: `docker/s6-rc.d/dashboard/run`
- Create: `docker/s6-rc.d/dashboard/dependencies.d/base` (empty)
- Create: `docker/s6-rc.d/user/contents.d/main-hermes` (empty — registers in user bundle)
- Create: `docker/s6-rc.d/user/contents.d/dashboard` (empty — registers in user bundle)

**Step 1: main-hermes service**

`docker/s6-rc.d/main-hermes/type`:
```
longrun
```

`docker/s6-rc.d/main-hermes/run`:
```sh
#!/command/with-contenv sh

# In TUI mode, main hermes runs as the container's CMD (exec'd by /init
# with TTY intact, not as an s6 service). See D9.
if [ -f /var/run/s6/container_environment/HERMES_TUI_MODE ]; then
    exec sleep infinity
fi

# Non-TUI path: run as supervised service.
cd /opt/data
. /opt/hermes/.venv/bin/activate

if [ -n "${HERMES_CMD:-}" ]; then
    # Bare executable (sleep, bash, sh -c ...) — exec directly as hermes user
    exec s6-setuidgid hermes sh -c "${HERMES_CMD}"
fi

# Default: hermes with any subcommand args
exec s6-setuidgid hermes hermes ${HERMES_ARGS:-}
```

`docker/s6-rc.d/main-hermes/finish`:
```sh
#!/command/execlineb -S2
# $1 = exit code (256 if killed by signal), $2 = signal number
foreground {
    if { eltest $1 -eq 256 }
        redirfd -w 1 /run/s6-linux-init-container-results/exitcode echo $((128 + $2))
}
foreground {
    if { eltest $1 -ne 256 }
        redirfd -w 1 /run/s6-linux-init-container-results/exitcode echo $1
}
/run/s6/basedir/bin/halt
```

Empty files: `docker/s6-rc.d/main-hermes/dependencies.d/base`, `docker/s6-rc.d/user/contents.d/main-hermes`.

**Step 2: dashboard service (OQ3-A: always declared, run script checks env)**

`docker/s6-rc.d/dashboard/type`:
```
longrun
```

`docker/s6-rc.d/dashboard/run`:
```sh
#!/command/with-contenv sh
# Dashboard only runs when HERMES_DASHBOARD is truthy. Otherwise we sleep
# forever so s6 still supervises this slot but does nothing.

case "${HERMES_DASHBOARD:-}" in
    1|true|TRUE|True|yes|YES|Yes) ;;
    *) exec sleep infinity ;;
esac

cd /opt/data
. /opt/hermes/.venv/bin/activate

dash_host="${HERMES_DASHBOARD_HOST:-0.0.0.0}"
dash_port="${HERMES_DASHBOARD_PORT:-9119}"

insecure=""
case "$dash_host" in
    127.0.0.1|localhost) ;;
    *) insecure="--insecure" ;;
esac

exec s6-setuidgid hermes hermes dashboard \
    --host "$dash_host" --port "$dash_port" --no-open $insecure
```

Empty files: `docker/s6-rc.d/dashboard/dependencies.d/base`, `docker/s6-rc.d/user/contents.d/dashboard`.

**Step 3: Commit**

```bash
git add docker/s6-rc.d/
git commit -m "feat(docker): add s6-rc service definitions for main-hermes and dashboard"
```

### Task 2.3: Rewrite entrypoint as s6 stage2 hook

**Objective:** Move gosu-drop + config bootstrap + skills sync out of the main exec path and into a cont-init.d script. Detect the TUI case and set `HERMES_TUI_MODE`.

**Files:**
- Create: `docker/stage2-hook.sh`
- Rewrite: `docker/entrypoint.sh` (becomes a thin shim)

> **v3 note:** The current entrypoint also writes `${HERMES_HOME:=/opt/data}/.install_method` with content `"docker"` after the gosu drop and venv activate (added in PR #27843, May 18). This stamp is read by `detect_install_method()` for `hermes status` install-method reporting. The stage2-hook.sh rewrite below must preserve this stamp — recommended placement is **inside the `--- Seed directory structure as hermes user ---` block** in stage2-hook.sh (which already drops to the hermes user via `s6-setuidgid hermes`), so the file is created with hermes ownership and survives the VOLUME overlay. Concrete line to include:
>
> ```sh
> s6-setuidgid hermes sh -c 'echo "docker" > "${HERMES_HOME:=/opt/data}/.install_method"' 2>/dev/null || true
> ```

**Step 1: Create `docker/stage2-hook.sh`**

```sh
#!/bin/sh
# s6-overlay stage2 hook — runs as root after supervision tree is up but
# before user services start. Handles UID/GID remap, chown, config seeding,
# skill sync, and TUI detection.
#
# Per-service privilege drop happens inside each service's `run` script via
# s6-setuidgid, not here.

set -eu

HERMES_HOME="${HERMES_HOME:-/opt/data}"
INSTALL_DIR="/opt/hermes"

# --- UID/GID remap ---
if [ -n "${HERMES_UID:-}" ] && [ "$HERMES_UID" != "$(id -u hermes)" ]; then
    echo "[stage2] Changing hermes UID to $HERMES_UID"
    usermod -u "$HERMES_UID" hermes
fi
if [ -n "${HERMES_GID:-}" ] && [ "$HERMES_GID" != "$(id -g hermes)" ]; then
    echo "[stage2] Changing hermes GID to $HERMES_GID"
    groupmod -o -g "$HERMES_GID" hermes 2>/dev/null || true
fi

# --- Fix ownership of data volume ---
actual_hermes_uid=$(id -u hermes)
needs_chown=false
if [ -n "${HERMES_UID:-}" ] && [ "$HERMES_UID" != "10000" ]; then
    needs_chown=true
elif [ "$(stat -c %u "$HERMES_HOME" 2>/dev/null)" != "$actual_hermes_uid" ]; then
    needs_chown=true
fi
if [ "$needs_chown" = true ]; then
    echo "[stage2] Fixing ownership of $HERMES_HOME to hermes ($actual_hermes_uid)"
    chown -R hermes:hermes "$HERMES_HOME" 2>/dev/null || \
        echo "[stage2] Warning: chown failed (rootless container?) — continuing"
fi

# --- config.yaml permissions ---
if [ -f "$HERMES_HOME/config.yaml" ]; then
    chown hermes:hermes "$HERMES_HOME/config.yaml" 2>/dev/null || true
    chmod 640 "$HERMES_HOME/config.yaml" 2>/dev/null || true
fi

# --- Seed directory structure as hermes user ---
su -s /bin/sh hermes -c "mkdir -p \"$HERMES_HOME\"/{cron,sessions,logs,hooks,memories,skills,skins,plans,workspace,home}"

# --- Seed config files ---
for pair in ".env:.env.example" "config.yaml:cli-config.yaml.example" "SOUL.md:docker/SOUL.md"; do
    dest="${pair%%:*}"
    src="${pair##*:}"
    if [ ! -f "$HERMES_HOME/$dest" ]; then
        su -s /bin/sh hermes -c "cp \"$INSTALL_DIR/$src\" \"$HERMES_HOME/$dest\""
    fi
done

# --- Sync bundled skills ---
if [ -d "$INSTALL_DIR/skills" ]; then
    su -s /bin/sh hermes -c ". $INSTALL_DIR/.venv/bin/activate && python3 $INSTALL_DIR/tools/skills_sync.py"
fi

# --- Detect TUI invocation ---
_is_tui_invocation() {
    for arg in "$@"; do
        case "$arg" in --tui|-T) return 0 ;; esac
    done
    case "${HERMES_TUI:-}" in 1|true|TRUE|yes) return 0 ;; esac
    # Implicit: stdin is a TTY and no subcommand given
    if [ -t 0 ] && [ $# -eq 0 ]; then return 0; fi
    return 1
}

if _is_tui_invocation "$@"; then
    touch /var/run/s6/container_environment/HERMES_TUI_MODE
    echo "[stage2] TUI mode detected; main-hermes service will no-op and CMD runs as TTY-connected main"
fi

# --- Pass CMD through to main-hermes service ---
# Bare executable → HERMES_CMD; otherwise → HERMES_ARGS for `hermes $HERMES_ARGS`
if [ $# -gt 0 ] && command -v "$1" >/dev/null 2>&1; then
    printf '%s' "$*" > /var/run/s6/container_environment/HERMES_CMD
else
    printf '%s' "$*" > /var/run/s6/container_environment/HERMES_ARGS
fi

echo "[stage2] Setup complete; starting user services"
```

```bash
chmod +x docker/stage2-hook.sh
```

**Step 2: Simplify `docker/entrypoint.sh` to a shim**

Replace the entire file with:

```sh
#!/bin/sh
# s6-overlay shim. The real logic lives in docker/stage2-hook.sh, invoked
# by /etc/cont-init.d/01-hermes-setup (installed in the Dockerfile).
# This file exists so external references to docker/entrypoint.sh still
# work, but it's no longer the ENTRYPOINT — /init is.
exec /opt/hermes/docker/stage2-hook.sh "$@"
```

**Step 3: Run shellcheck**

```bash
shellcheck docker/stage2-hook.sh docker/entrypoint.sh
```

Fix any errors.

**Step 4: Commit**

```bash
git add docker/stage2-hook.sh docker/entrypoint.sh
git commit -m "feat(docker): rewrite entrypoint as s6-overlay stage2 hook"
```

### Task 2.4: Flip the ENTRYPOINT in the Dockerfile

**Objective:** Replace `tini` with `/init`. Wire service defs and stage2 hook into the image. Remove `tini`.

**Files:**
- Modify: `Dockerfile`

> **v3 note:** The current Dockerfile (post-PR #27843) has a `RUN mkdir -p /opt/data` line immediately before `VOLUME [ "/opt/data" ]`. **Keep this line.** It was added because the volume overlay was wiping out files written to /opt/data during build — same reason it's needed under s6. Do not delete it during the entrypoint swap.

**Step 1: Update `Dockerfile`**

Remove `tini` from the apt install line. Add after the s6-overlay install block (from Task 2.1):

```dockerfile
# ---------- s6-overlay service wiring ----------
COPY docker/s6-rc.d/ /etc/s6-overlay/s6-rc.d/
RUN chmod +x /etc/s6-overlay/s6-rc.d/main-hermes/run \
             /etc/s6-overlay/s6-rc.d/main-hermes/finish \
             /etc/s6-overlay/s6-rc.d/dashboard/run

# Install cont-init.d hook that runs our stage2 setup as root before services start
RUN mkdir -p /etc/cont-init.d && \
    printf '#!/bin/sh\nexec /opt/hermes/docker/stage2-hook.sh "$@"\n' \
        > /etc/cont-init.d/01-hermes-setup && \
    chmod +x /etc/cont-init.d/01-hermes-setup
```

Replace the ENTRYPOINT line:

```dockerfile
# s6-overlay's /init is PID 1. It sets up the supervision tree, runs
# /etc/cont-init.d/ scripts (our stage2 hook), starts s6-rc services,
# and reaps zombies.
ENTRYPOINT [ "/init" ]
# Default CMD: no args → main-hermes service runs `hermes` with no args
CMD [ ]
```

**Step 2: Run hadolint**

```bash
docker run --rm -i hadolint/hadolint:latest < Dockerfile
```

Fix any warnings.

**Step 3: Rebuild and run full harness**

```bash
docker build -t hermes-agent-harness:latest .
scripts/run_tests.sh tests/docker/ -v
```

Expected: **all Phase 0 tests pass**. This is the hard gate. If any fail, diagnose before committing.

**Step 4: Commit**

```bash
git add Dockerfile
git commit -m "feat(docker)!: replace tini with s6-overlay as PID 1

BREAKING CHANGE: container ENTRYPOINT is now /init (s6-overlay) instead
of /usr/bin/tini. Main hermes and dashboard run as supervised s6 services.
All docker run <image> invocation patterns (chat, sleep, bash, --tui)
continue to work identically — verified by the Phase 0 test harness."
```

### Task 2.5: Add restart-on-crash test for dashboard

**Objective:** Now that s6 supervises the dashboard, a crash should be recovered. This is a new test, not a Phase 0 baseline — it encodes a new invariant that only holds post-Phase 2.

**Files:**
- Modify: `tests/docker/test_dashboard.py`

**Step 1: Add the test**

```python
def test_dashboard_restarts_after_crash(built_image, container_name):
    """After Phase 2: s6 supervises the dashboard. SIGKILL the process;
    s6 should restart it within ~2 seconds."""
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name,
         "-e", "HERMES_DASHBOARD=1", built_image, "sleep", "60"],
        check=True, capture_output=True, timeout=30,
    )
    time.sleep(5)

    # Find dashboard PID
    r = subprocess.run(
        ["docker", "exec", container_name, "pgrep", "-f", "hermes dashboard"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0, "Dashboard not running initially"
    first_pid = r.stdout.strip().split()[0]

    # Kill it
    subprocess.run(
        ["docker", "exec", container_name, "kill", "-9", first_pid],
        capture_output=True, timeout=10,
    )

    # Wait for s6 to restart
    time.sleep(3)

    r = subprocess.run(
        ["docker", "exec", container_name, "pgrep", "-f", "hermes dashboard"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0, "Dashboard not restarted after kill"
    second_pid = r.stdout.strip().split()[0]
    assert second_pid != first_pid, "PID unchanged — not actually restarted"
```

**Step 2: Commit**

```bash
git add tests/docker/test_dashboard.py
git commit -m "test(docker): verify s6 restarts dashboard after crash"
```

---

## Phase 3 — S6ServiceManager implements runtime registration

**Goal:** Implement `register_profile_gateway` / `unregister_profile_gateway` / `list_profile_gateways` in a new `S6ServiceManager` class. No existing caller yet — this phase is purely additive. Phase 4 wires it into the profile lifecycle.

### Task 3.1: Scaffolding — S6ServiceManager class

**Objective:** Create the class, wire it into the factory, stub the registration methods.

**Files:**
- Modify: `hermes_cli/service_manager.py`
- Modify: `tests/hermes_cli/test_service_manager.py`

**Step 1: Tests**

```python
def test_s6_manager_kind_and_supports_registration():
    from hermes_cli.service_manager import S6ServiceManager
    mgr = S6ServiceManager()
    assert mgr.kind == "s6"
    assert mgr.supports_runtime_registration() is True


def test_factory_returns_s6_when_detected(monkeypatch):
    from hermes_cli import service_manager as sm
    monkeypatch.setattr(sm, "detect_service_manager", lambda: "s6")
    assert isinstance(sm.get_service_manager(), sm.S6ServiceManager)
```

**Step 2: Add the class**

Append to `hermes_cli/service_manager.py`:

```python
from pathlib import Path

# s6-overlay scandir for dynamic services. This directory is tmpfs inside
# the container and writable by the hermes user. s6-svscan watches it.
S6_DYNAMIC_SCANDIR = Path("/run/service")
S6_SERVICE_PREFIX = "gateway-"


class S6ServiceManager:
    """Per-profile gateway supervision via s6-overlay.

    Static services (main-hermes, dashboard) are managed via s6-rc at
    image build time and are NOT managed by this class. This class only
    handles per-profile gateway services, which are created at runtime
    when `hermes profile create <name>` runs inside the container.
    """
    kind: ServiceManagerKind = "s6"

    def __init__(self, scandir: Path = S6_DYNAMIC_SCANDIR):
        self.scandir = scandir

    def _service_dir(self, profile: str) -> Path:
        validate_profile_name(profile)
        return self.scandir / f"{S6_SERVICE_PREFIX}{profile}"

    # Lifecycle
    def start(self, name: str) -> None:
        # name is the s6 service directory basename (gateway-<profile>)
        import subprocess
        subprocess.run(
            ["s6-svc", "-u", str(self.scandir / name)],
            check=True, capture_output=True, timeout=5,
        )

    def stop(self, name: str) -> None:
        import subprocess
        subprocess.run(
            ["s6-svc", "-d", str(self.scandir / name)],
            check=True, capture_output=True, timeout=5,
        )

    def restart(self, name: str) -> None:
        import subprocess
        subprocess.run(
            ["s6-svc", "-t", str(self.scandir / name)],
            check=True, capture_output=True, timeout=5,
        )

    def is_running(self, name: str) -> bool:
        import subprocess
        result = subprocess.run(
            ["s6-svstat", str(self.scandir / name)],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0 and "up " in result.stdout

    # Runtime registration — implemented in Task 3.2/3.3/3.4
    def supports_runtime_registration(self) -> bool:
        return True

    def register_profile_gateway(self, profile, *, port, extra_env=None):
        raise NotImplementedError  # Task 3.2

    def unregister_profile_gateway(self, profile):
        raise NotImplementedError  # Task 3.3

    def list_profile_gateways(self):
        raise NotImplementedError  # Task 3.4
```

Update `get_service_manager()`:

```python
    if kind == "s6":
        return S6ServiceManager()
```

**Step 3: Commit**

```bash
git add hermes_cli/service_manager.py tests/hermes_cli/test_service_manager.py
git commit -m "feat(service_manager): add S6ServiceManager scaffolding"
```

### Task 3.2: Implement register_profile_gateway

**Objective:** Write the service directory for a profile gateway, trigger s6 scan.

**Step 1: Tests**

```python
def test_register_profile_gateway_creates_service_dir(tmp_path, monkeypatch):
    from hermes_cli.service_manager import S6ServiceManager

    scandir = tmp_path / "service"
    scandir.mkdir()
    mgr = S6ServiceManager(scandir=scandir)

    called = []
    def fake_run(cmd, **kw):
        called.append(cmd)
        import subprocess as sp
        return sp.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr("subprocess.run", fake_run)

    mgr.register_profile_gateway("coder", port=9150)

    svc_dir = scandir / "gateway-coder"
    assert svc_dir.is_dir()
    assert (svc_dir / "type").read_text().strip() == "longrun"
    assert (svc_dir / "run").is_file()
    run_content = (svc_dir / "run").read_text()
    assert "hermes -p coder gateway start" in run_content
    assert "--port 9150" in run_content or "--port=9150" in run_content
    assert "s6-setuidgid hermes" in run_content

    # Log rotation persists under HERMES_HOME (OQ8-C). The path must come
    # from the runtime env, not be hard-coded — check we emit a shell var
    # expansion rather than a literal /opt/data/...
    log_run = svc_dir / "log" / "run"
    assert log_run.is_file()
    log_run_content = log_run.read_text()
    assert "$HERMES_HOME" in log_run_content
    assert "logs/gateways/coder" in log_run_content
    # Negative assertion: the path must NOT be Python-substituted to /opt/data
    assert "/opt/data/logs/gateways/coder" not in log_run_content, \
        "log_dir was hard-coded; must use ${HERMES_HOME} at run time"

    # s6-svscanctl was invoked
    assert any("s6-svscanctl" in str(c) for c in called)


def test_register_profile_rejects_duplicate(tmp_path):
    from hermes_cli.service_manager import S6ServiceManager
    scandir = tmp_path / "service"
    (scandir / "gateway-coder").mkdir(parents=True)
    mgr = S6ServiceManager(scandir=scandir)
    with pytest.raises(ValueError, match="already registered"):
        mgr.register_profile_gateway("coder", port=9150)
```

**Step 2: Implement**

```python
    def register_profile_gateway(
        self,
        profile: str,
        *,
        port: int,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        """Write an s6 service directory for the given profile's gateway and
        trigger s6-svscan to pick it up.

        Raises:
            ValueError: if a service for the profile is already registered
            RuntimeError: if s6-svscanctl fails
        """
        import subprocess

        svc_dir = self._service_dir(profile)
        if svc_dir.exists():
            raise ValueError(
                f"profile gateway {profile!r} already registered at {svc_dir}"
            )

        svc_dir.mkdir(parents=True)
        (svc_dir / "type").write_text("longrun\n")

        # run script: drop to hermes, exec foreground gateway
        run_script = self._render_run_script(profile, port, extra_env or {})
        (svc_dir / "run").write_text(run_script)
        (svc_dir / "run").chmod(0o755)

        # log/ subservice: persistent rotation under HERMES_HOME (OQ8-C)
        log_subdir = svc_dir / "log"
        log_subdir.mkdir()
        (log_subdir / "run").write_text(self._render_log_run(profile))
        (log_subdir / "run").chmod(0o755)

        # Trigger s6 scan
        result = subprocess.run(
            ["s6-svscanctl", "-a", str(self.scandir)],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            # Clean up partial directory
            import shutil
            shutil.rmtree(svc_dir, ignore_errors=True)
            raise RuntimeError(
                f"s6-svscanctl failed: {result.stderr or result.stdout}"
            )

    def _render_run_script(
        self, profile: str, port: int, extra_env: dict[str, str]
    ) -> str:
        import shlex
        lines = [
            "#!/command/with-contenv sh",
            "set -e",
            "cd /opt/data",
            ". /opt/hermes/.venv/bin/activate",
        ]
        for k, v in sorted(extra_env.items()):
            lines.append(f"export {k}={shlex.quote(v)}")
        lines.append(
            f"exec s6-setuidgid hermes hermes -p {shlex.quote(profile)} "
            f"gateway start --foreground --port {port}"
        )
        return "\n".join(lines) + "\n"

    def _render_log_run(self, profile: str) -> str:
        # OQ8-C: persist to ${HERMES_HOME}/logs/gateways/<profile>/
        # IMPORTANT: do NOT hard-code /opt/data here — read HERMES_HOME from the
        # container environment at run time so `-e HERMES_HOME=/some/other` works.
        # The `with-contenv` shebang sources /run/s6/container_environment/* which
        # was populated by the stage2 hook from the actual container env.
        import shlex
        prof = shlex.quote(profile)
        return (
            f"#!/command/with-contenv sh\n"
            f": \"${{HERMES_HOME:=/opt/data}}\"\n"
            f"log_dir=\"$HERMES_HOME/logs/gateways/{prof}\"\n"
            f"mkdir -p \"$log_dir\"\n"
            f"chown -R hermes:hermes \"$log_dir\" 2>/dev/null || true\n"
            f"exec s6-setuidgid hermes s6-log n10 s1000000 T \"$log_dir\"\n"
        )
```

**Step 3: Commit**

```bash
git add hermes_cli/service_manager.py tests/hermes_cli/test_service_manager.py
git commit -m "feat(service_manager): implement S6ServiceManager.register_profile_gateway"
```

### Task 3.3: Implement unregister_profile_gateway

**Step 1: Tests**

```python
def test_unregister_profile_gateway_removes_service_dir(tmp_path, monkeypatch):
    from hermes_cli.service_manager import S6ServiceManager
    scandir = tmp_path / "service"
    svc_dir = scandir / "gateway-coder"
    svc_dir.mkdir(parents=True)
    (svc_dir / "type").write_text("longrun\n")

    called = []
    def fake_run(cmd, **kw):
        called.append(cmd)
        import subprocess as sp
        return sp.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr("subprocess.run", fake_run)

    mgr = S6ServiceManager(scandir=scandir)
    mgr.unregister_profile_gateway("coder")

    # s6-svc -d was called
    assert any("s6-svc" in str(c) and "-d" in c for c in called)
    # Service dir removed
    assert not svc_dir.exists()
    # Rescan triggered
    assert any("s6-svscanctl" in str(c) for c in called)


def test_unregister_absent_profile_is_noop(tmp_path):
    from hermes_cli.service_manager import S6ServiceManager
    scandir = tmp_path / "service"
    scandir.mkdir()
    mgr = S6ServiceManager(scandir=scandir)
    # Should not raise
    mgr.unregister_profile_gateway("nonexistent")
```

**Step 2: Implement**

```python
    def unregister_profile_gateway(self, profile: str) -> None:
        """Stop the profile's gateway service and remove its directory.

        Idempotent: absent services are a no-op.
        """
        import subprocess
        import shutil

        svc_dir = self._service_dir(profile)
        if not svc_dir.exists():
            return

        # Stop the service (best effort)
        subprocess.run(
            ["s6-svc", "-d", str(svc_dir)],
            capture_output=True, text=True, timeout=5,
            check=False,
        )
        # Wait briefly for it to go down
        subprocess.run(
            ["s6-svwait", "-D", "-t", "10000", str(svc_dir)],
            capture_output=True, text=True, timeout=15,
            check=False,
        )

        # Remove the directory
        shutil.rmtree(svc_dir, ignore_errors=True)

        # Rescan to drop s6-supervise process
        subprocess.run(
            ["s6-svscanctl", "-an", str(self.scandir)],
            capture_output=True, text=True, timeout=5,
            check=False,
        )
```

**Step 3: Commit**

```bash
git add hermes_cli/service_manager.py tests/hermes_cli/test_service_manager.py
git commit -m "feat(service_manager): implement S6ServiceManager.unregister_profile_gateway"
```

### Task 3.4: Implement list_profile_gateways

**Step 1: Test + implementation**

```python
def test_list_profile_gateways(tmp_path):
    from hermes_cli.service_manager import S6ServiceManager
    scandir = tmp_path / "service"
    scandir.mkdir()
    (scandir / "gateway-coder").mkdir()
    (scandir / "gateway-assistant").mkdir()
    (scandir / "other-service").mkdir()  # not a gateway, should be filtered out
    (scandir / ".hidden").mkdir()

    mgr = S6ServiceManager(scandir=scandir)
    profiles = sorted(mgr.list_profile_gateways())
    assert profiles == ["assistant", "coder"]
```

Implementation:

```python
    def list_profile_gateways(self) -> list[str]:
        """List all currently-registered profile gateway service names
        (returns the profile names, not the service-dir names)."""
        if not self.scandir.exists():
            return []
        profiles = []
        for entry in self.scandir.iterdir():
            if entry.name.startswith("."):
                continue
            if not entry.is_dir():
                continue
            if not entry.name.startswith(S6_SERVICE_PREFIX):
                continue
            profiles.append(entry.name[len(S6_SERVICE_PREFIX):])
        return profiles
```

**Step 2: Commit**

```bash
git add hermes_cli/service_manager.py tests/hermes_cli/test_service_manager.py
git commit -m "feat(service_manager): implement S6ServiceManager.list_profile_gateways"
```

### Task 3.5: In-container integration test

**Objective:** Validate the full register → start → kill → restart → unregister cycle inside a real container.

**Files:**
- Create: `tests/docker/test_s6_profile_gateway_integration.py`

**Step 1: Test**

```python
"""End-to-end test of S6ServiceManager.register_profile_gateway + lifecycle."""
import subprocess
import time


def test_register_and_supervise_profile_gateway(built_image, container_name):
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name, built_image,
         "sleep", "120"],
        check=True, capture_output=True, timeout=30,
    )
    time.sleep(3)

    # Register a test profile gateway via the Python API
    register_script = '''
import sys
sys.path.insert(0, "/opt/hermes")
from hermes_cli.service_manager import S6ServiceManager
mgr = S6ServiceManager()
# Create a minimal profile first so `hermes -p` works
import subprocess
subprocess.run(["hermes", "profile", "create", "it-test"], check=True)
mgr.register_profile_gateway("it-test", port=9201)
print("REGISTERED")
'''
    r = subprocess.run(
        ["docker", "exec", container_name, "python3", "-c", register_script],
        capture_output=True, text=True, timeout=60,
    )
    assert "REGISTERED" in r.stdout, f"register failed: {r.stderr}"

    # Service dir exists
    r = subprocess.run(
        ["docker", "exec", container_name, "test", "-d",
         "/run/service/gateway-it-test"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0

    # Wait for s6 to bring it up
    time.sleep(5)

    # Check s6-svstat reports it as up
    r = subprocess.run(
        ["docker", "exec", container_name, "s6-svstat",
         "/run/service/gateway-it-test"],
        capture_output=True, text=True, timeout=10,
    )
    assert "up " in r.stdout, f"service not up: {r.stdout}"

    # Kill the gateway process; s6 should restart it
    subprocess.run(
        ["docker", "exec", container_name, "sh", "-c",
         "pkill -9 -f 'gateway.*it-test' || true"],
        capture_output=True, timeout=10,
    )
    time.sleep(3)

    r = subprocess.run(
        ["docker", "exec", container_name, "s6-svstat",
         "/run/service/gateway-it-test"],
        capture_output=True, text=True, timeout=10,
    )
    assert "up " in r.stdout, f"service not restarted: {r.stdout}"

    # Unregister
    unregister_script = '''
import sys
sys.path.insert(0, "/opt/hermes")
from hermes_cli.service_manager import S6ServiceManager
S6ServiceManager().unregister_profile_gateway("it-test")
print("UNREGISTERED")
'''
    r = subprocess.run(
        ["docker", "exec", container_name, "python3", "-c", unregister_script],
        capture_output=True, text=True, timeout=30,
    )
    assert "UNREGISTERED" in r.stdout

    # Service dir gone
    r = subprocess.run(
        ["docker", "exec", container_name, "test", "-d",
         "/run/service/gateway-it-test"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode != 0
```

**Step 2: Commit**

```bash
git add tests/docker/test_s6_profile_gateway_integration.py
git commit -m "test(docker): integration test for S6ServiceManager profile gateway lifecycle"
```

---

## Phase 4 — Wire profile create/delete into the s6 backend

**Goal:** When `hermes profile create <name>` runs inside the container, register the profile's gateway with s6. When `hermes profile delete` runs, unregister. Existing `hermes -p <profile> gateway start/stop/restart` commands, inside the container, dispatch to s6 via the ServiceManager.

After this phase, the Phase 0 `test_profile_gateway.py` harness (which currently passes against the current implementation) must still pass — but now the underlying mechanism is s6-supervised.

### Task 4.0: Reconcile per-profile gateways on container boot

**Objective:** Survive `docker restart`. Service directories at `/run/service/gateway-<profile>/` live on **tmpfs** and are wiped when the container restarts, but the profile directories themselves (`/opt/data/profiles/<name>/`) and each profile's `gateway_state.json` live on the persistent VOLUME. On boot, walk the persistent profiles, recreate the s6 service registrations, and bring back up any profile whose last recorded state was `running`. Without this, every `docker restart` silently loses every per-profile gateway, even though the user's profiles still exist on disk.

**Files:**
- Create: `docker/cont-init.d/02-reconcile-profiles` (s6-overlay cont-init.d script — runs as root after `01-hermes-setup` from Task 2.3, before s6-rc starts user services)
- Create: `hermes_cli/container_boot.py` (Python module the cont-init.d script invokes; keeps logic testable in isolation)
- Modify: `Dockerfile` (copy the new cont-init.d script and ensure it's executable)
- Create: `tests/hermes_cli/test_container_boot.py` (unit tests for the reconciliation logic against a fake `$HERMES_HOME`)
- Modify: Phase 0 harness (`tests/docker/test_container_restart.py` — new test asserting end-to-end restart survival)

**Step 1: Define the reconciliation contract**

For each profile dir under `$HERMES_HOME/profiles/<name>/` (and the default profile at `$HERMES_HOME/` itself if it's the in-container layout):

1. **Read `gateway_state.json`** if present. The schema (see `gateway/status.py`) records `gateway_state ∈ {starting, running, startup_failed, stopped}` plus a timestamp.
2. **Clean up stale runtime files.** Remove `gateway.pid` from the profile dir if it exists — the recorded PID belongs to the dead container's process namespace, and a numerically-equal live PID in the new container would be a different process. Also remove `processes.json`.
3. **Always recreate the s6 service registration** at `/run/service/gateway-<profile>/` (down state) — even if the last recorded state was `stopped`. This ensures `hermes -p <profile> gateway start` works without going through `register_profile_gateway` first, matching the invariant "every profile has a service slot."
4. **Auto-start only if the last recorded state was `running`.** `starting` does NOT auto-start (the gateway crashed during boot last time — assume the user wants to investigate, don't crash-loop on restart). `startup_failed` does NOT auto-start (explicit prior failure). `stopped` does NOT auto-start (explicit prior stop). Missing `gateway_state.json` does NOT auto-start (gateway was never run).
5. **Write a reconciliation log** to `$HERMES_HOME/logs/container-boot.log` with one line per profile: `<timestamp> profile=<name> prior_state=<state> action=<registered|started|skipped>`. Operators inspect this to debug "why didn't my profile come back up."

**Step 2: Write failing tests for `container_boot.reconcile_profile_gateways`**

```python
# tests/hermes_cli/test_container_boot.py
import json
from pathlib import Path
import pytest
from hermes_cli.container_boot import (
    reconcile_profile_gateways,
    ReconcileAction,
)

def _make_profile(hermes_home: Path, name: str, *, state: str | None,
                  with_pid: bool = False) -> Path:
    """Create a fake profile directory under hermes_home/profiles/<name>/."""
    p = hermes_home / "profiles" / name
    p.mkdir(parents=True)
    (p / "config.yaml").write_text("model: test\n")  # marks it as a real profile
    if state is not None:
        (p / "gateway_state.json").write_text(json.dumps({
            "gateway_state": state, "timestamp": 1234567890,
        }))
    if with_pid:
        (p / "gateway.pid").write_text(json.dumps({"pid": 99999, "host": "old-container"}))
    return p


def test_running_profile_is_reregistered_and_autostarted(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    scandir = tmp_path / "run-service"
    scandir.mkdir()
    _make_profile(tmp_path, "coder", state="running")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert actions == [ReconcileAction(profile="coder", prior_state="running",
                                       action="started")]
    assert (scandir / "gateway-coder" / "run").exists()
    assert (scandir / "gateway-coder" / "run").stat().st_mode & 0o111  # executable


def test_stopped_profile_is_reregistered_but_not_started(tmp_path):
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "writer", state="stopped")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert actions == [ReconcileAction(profile="writer", prior_state="stopped",
                                       action="registered")]
    assert (scandir / "gateway-writer" / "run").exists()
    # The down-marker file tells s6 to not start the service initially
    assert (scandir / "gateway-writer" / "down").exists()


def test_startup_failed_profile_is_not_autostarted(tmp_path):
    """Avoid crash-loop on restart when the gateway was failing to boot."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "broken", state="startup_failed")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert actions[0].action == "registered"
    assert (scandir / "gateway-broken" / "down").exists()


def test_starting_state_does_not_autostart(tmp_path):
    """`starting` means the gateway died mid-boot; treat as failed, not running."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "unlucky", state="starting")

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert actions[0].action == "registered"  # NOT "started"


def test_stale_pid_file_is_removed(tmp_path):
    scandir = tmp_path / "run-service"; scandir.mkdir()
    profile = _make_profile(tmp_path, "coder", state="running", with_pid=True)

    reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert not (profile / "gateway.pid").exists()


def test_profile_without_state_file_is_registered_but_not_started(tmp_path):
    """A freshly-created profile that's never been started: register slot, don't autostart."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "fresh", state=None)

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert actions[0].action == "registered"
    assert (scandir / "gateway-fresh" / "down").exists()


def test_directory_without_config_yaml_is_skipped(tmp_path):
    """A directory under profiles/ that isn't actually a profile (no config.yaml) is ignored."""
    scandir = tmp_path / "run-service"; scandir.mkdir()
    (tmp_path / "profiles" / "stray").mkdir(parents=True)  # no config.yaml

    actions = reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    assert actions == []


def test_reconcile_log_is_written(tmp_path):
    scandir = tmp_path / "run-service"; scandir.mkdir()
    _make_profile(tmp_path, "a", state="running")
    _make_profile(tmp_path, "b", state="stopped")

    reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=False,
    )

    log = (tmp_path / "logs" / "container-boot.log").read_text()
    assert "profile=a" in log and "action=started" in log
    assert "profile=b" in log and "action=registered" in log


def test_dry_run_makes_no_filesystem_changes(tmp_path):
    scandir = tmp_path / "run-service"; scandir.mkdir()
    profile = _make_profile(tmp_path, "coder", state="running", with_pid=True)

    reconcile_profile_gateways(
        hermes_home=tmp_path, scandir=scandir, dry_run=True,
    )

    assert (profile / "gateway.pid").exists()  # not removed under dry_run
    assert not (scandir / "gateway-coder").exists()
```

Run the tests to confirm they fail:

```bash
scripts/run_tests.sh tests/hermes_cli/test_container_boot.py -v
```

Expected: all 9 tests FAIL with `ImportError` / `AttributeError` on the missing `reconcile_profile_gateways` symbol.

**Step 3: Implement `hermes_cli/container_boot.py`**

```python
"""Container boot-time reconciliation of per-profile gateway s6 services.

Service directories under /run/service/ live on tmpfs and are wiped on
container restart. Profile directories under $HERMES_HOME/profiles/ live
on the persistent VOLUME. This module bridges the two: on every container
boot, walk the persistent profiles and recreate the s6 service slots.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

# Only this prior state triggers automatic restart. Everything else
# (startup_failed, starting, stopped, missing) registers the slot in
# the down state and waits for explicit user action.
_AUTOSTART_STATES = frozenset({"running"})

ReconcileActionLabel = Literal["started", "registered", "skipped"]


@dataclass(frozen=True)
class ReconcileAction:
    profile: str
    prior_state: str | None
    action: ReconcileActionLabel


def reconcile_profile_gateways(
    *,
    hermes_home: Path,
    scandir: Path,
    dry_run: bool = False,
) -> list[ReconcileAction]:
    """Recreate s6 service registrations for every persistent profile."""
    actions: list[ReconcileAction] = []
    profiles_root = hermes_home / "profiles"
    if not profiles_root.is_dir():
        return actions

    for entry in sorted(profiles_root.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "config.yaml").exists():
            continue  # not a real profile

        prior_state = _read_prior_state(entry)
        if not dry_run:
            _cleanup_stale_runtime_files(entry)
            _register_service(scandir, entry.name,
                              start=prior_state in _AUTOSTART_STATES)

        action_label: ReconcileActionLabel = (
            "started" if prior_state in _AUTOSTART_STATES else "registered"
        )
        actions.append(ReconcileAction(
            profile=entry.name, prior_state=prior_state, action=action_label,
        ))

    if not dry_run:
        _write_reconcile_log(hermes_home, actions)
    return actions


def _read_prior_state(profile_dir: Path) -> str | None:
    state_file = profile_dir / "gateway_state.json"
    if not state_file.exists():
        return None
    try:
        return json.loads(state_file.read_text()).get("gateway_state")
    except (OSError, json.JSONDecodeError):
        log.warning("Could not read %s; treating as no prior state", state_file)
        return None


def _cleanup_stale_runtime_files(profile_dir: Path) -> None:
    for name in ("gateway.pid", "processes.json"):
        (profile_dir / name).unlink(missing_ok=True)


def _register_service(scandir: Path, profile: str, *, start: bool) -> None:
    service_dir = scandir / f"gateway-{profile}"
    service_dir.mkdir(parents=True, exist_ok=True)

    # The actual run script content is generated by S6ServiceManager from
    # Task 3.2; we duplicate the minimal contract here. Phase 4 follow-up:
    # extract a single shared rendering function used by both register
    # and reconcile.
    run = service_dir / "run"
    run.write_text(_render_run_script(profile))
    run.chmod(0o755)

    if not start:
        # The presence of a `down` file tells s6-supervise to NOT start
        # the service on rescan. User must `s6-svc -u` to bring it up.
        (service_dir / "down").touch()
    else:
        (service_dir / "down").unlink(missing_ok=True)


def _render_run_script(profile: str) -> str:
    # Mirrors the rendering in S6ServiceManager.register_profile_gateway
    # (Task 3.2). Extract to a shared helper as Phase 4 cleanup.
    return f"""#!/command/execlineb -P
fdmove -c 2 1
s6-setuidgid hermes
multisubstitute {{
  importas HERMES_HOME HERMES_HOME
}}
hermes -p {profile} gateway start --foreground
"""


def _write_reconcile_log(hermes_home: Path, actions: list[ReconcileAction]) -> None:
    log_dir = hermes_home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    import time
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with (log_dir / "container-boot.log").open("a") as f:
        for a in actions:
            f.write(
                f"{ts} profile={a.profile} prior_state={a.prior_state} "
                f"action={a.action}\n"
            )


def main() -> int:
    """Entry point invoked from /etc/cont-init.d/02-reconcile-profiles."""
    hermes_home = Path(os.environ.get("HERMES_HOME", "/opt/data"))
    scandir = Path(os.environ.get("S6_PROFILE_GATEWAY_SCANDIR", "/run/service"))
    actions = reconcile_profile_gateways(hermes_home=hermes_home, scandir=scandir)
    for a in actions:
        print(f"reconcile: profile={a.profile} prior_state={a.prior_state} "
              f"action={a.action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Step 4: Create the cont-init.d script**

`docker/cont-init.d/02-reconcile-profiles`:

```sh
#!/command/with-contenv sh
# Container-boot reconciliation of per-profile gateway s6 services.
# Runs as root after 01-hermes-setup (stage2 hook) has chowned the volume
# and seeded $HERMES_HOME, but before s6-rc starts user services.
#
# The actual logic lives in hermes_cli.container_boot. We invoke it via
# the bundled venv python, drop to the hermes user so the service dirs
# we write under $S6_PROFILE_GATEWAY_SCANDIR are owned by hermes (since
# the gateway processes run as hermes).
set -e
s6-setuidgid hermes /opt/hermes/.venv/bin/python -m hermes_cli.container_boot
```

**Step 5: Wire it into the Dockerfile**

In Task 2.4's Dockerfile changes, the cont-init.d block already copies `/etc/cont-init.d/01-hermes-setup`. Add `02-reconcile-profiles` next to it:

```dockerfile
COPY docker/cont-init.d/02-reconcile-profiles /etc/cont-init.d/02-reconcile-profiles
RUN chmod +x /etc/cont-init.d/02-reconcile-profiles
```

s6-overlay runs `/etc/cont-init.d/*` scripts in lexicographic order, so `01-hermes-setup` (gosu drop, chown, seed) runs before `02-reconcile-profiles`. The reconciliation thus runs after `$HERMES_HOME` is guaranteed to exist and be hermes-owned.

**Step 6: Run unit tests — should now pass**

```bash
scripts/run_tests.sh tests/hermes_cli/test_container_boot.py -v
```

Expected: 9 passed.

**Step 7: Add end-to-end restart test to Phase 0 harness**

`tests/docker/test_container_restart.py`:

```python
"""Container restart preserves per-profile gateway registrations."""
import shutil
import subprocess
import time
import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None, reason="Docker not available"
)


def _run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=120, **kw)


@pytest.fixture
def container(tmp_path, built_image):
    """Long-running container with a named volume so we can stop/start it."""
    volume = f"hermes-restart-test-{tmp_path.name}"
    name = f"hermes-restart-{tmp_path.name}"
    _run(["docker", "volume", "create", volume])
    _run(["docker", "run", "-d", "--name", name, "-v", f"{volume}:/opt/data",
          built_image, "sleep", "infinity"])
    yield name
    _run(["docker", "rm", "-f", name])
    _run(["docker", "volume", "rm", "-f", volume])


def _exec(container: str, cmd: list[str]) -> subprocess.CompletedProcess:
    return _run(["docker", "exec", container, *cmd])


def test_running_gateway_survives_container_restart(container, built_image):
    # 1. Create a profile and start its gateway
    _exec(container, ["hermes", "profile", "create", "coder",
                      "--model", "test/echo"])
    _exec(container, ["hermes", "-p", "coder", "gateway", "start"])

    # 2. Confirm gateway_state.json was written with "running"
    result = _exec(container, ["cat", "/opt/data/profiles/coder/gateway_state.json"])
    assert "running" in result.stdout

    # 3. Restart the container
    _run(["docker", "restart", container])
    time.sleep(5)  # give s6 and cont-init.d a moment

    # 4. The reconciliation log should record action=started
    log = _exec(container, ["cat", "/opt/data/logs/container-boot.log"])
    assert "profile=coder" in log.stdout
    assert "action=started" in log.stdout

    # 5. The s6 service dir should exist
    result = _exec(container, ["test", "-d", "/run/service/gateway-coder"])
    assert result.returncode == 0

    # 6. The gateway should be running (s6-svstat reports up)
    status = _exec(container, ["s6-svstat", "/run/service/gateway-coder"])
    assert "up" in status.stdout


def test_stopped_gateway_stays_stopped_after_restart(container):
    _exec(container, ["hermes", "profile", "create", "writer",
                      "--model", "test/echo"])
    _exec(container, ["hermes", "-p", "writer", "gateway", "start"])
    _exec(container, ["hermes", "-p", "writer", "gateway", "stop"])

    _run(["docker", "restart", container]); time.sleep(5)

    # Service is registered but down
    assert _exec(container, ["test", "-d", "/run/service/gateway-writer"]).returncode == 0
    assert _exec(container, ["test", "-f", "/run/service/gateway-writer/down"]).returncode == 0
    status = _exec(container, ["s6-svstat", "/run/service/gateway-writer"])
    assert "down" in status.stdout


def test_stale_gateway_pid_is_cleaned_up_on_restart(container):
    _exec(container, ["hermes", "profile", "create", "x", "--model", "test/echo"])
    _exec(container, ["hermes", "-p", "x", "gateway", "start"])

    _run(["docker", "restart", container]); time.sleep(5)

    # gateway.pid is gone (will be written fresh by the newly-started gateway,
    # but the *old* PID file is gone before the new gateway starts)
    # — we check the log instead since the new gateway repopulates it
    log = _exec(container, ["cat", "/opt/data/logs/container-boot.log"])
    assert "profile=x" in log.stdout
```

**Step 8: Run integration test**

```bash
scripts/run_tests.sh tests/docker/test_container_restart.py -v
```

Expected: 3 passed (assuming Docker available and the image was rebuilt with Phases 2–4 changes).

**Step 9: Commit**

```bash
git add hermes_cli/container_boot.py \
        docker/cont-init.d/02-reconcile-profiles \
        Dockerfile \
        tests/hermes_cli/test_container_boot.py \
        tests/docker/test_container_restart.py
git commit -m "feat(docker): reconcile per-profile gateways on container restart

Service dirs under /run/service live on tmpfs and are wiped by docker
restart. On boot, walk \$HERMES_HOME/profiles, read each gateway_state.json,
recreate the s6 service slot, and auto-up only those that were running.

Refs: docs/plans/2026-05-07-s6-overlay-dynamic-subagent-gateways.md Task 4.0"
```

**Verification:**

- `scripts/run_tests.sh tests/hermes_cli/test_container_boot.py tests/docker/test_container_restart.py` all green
- After `docker restart`, `s6-svstat /run/service/gateway-<profile>` for a previously-running profile reports `up`; for a previously-stopped profile reports `down`
- `cat /opt/data/logs/container-boot.log` shows one line per profile with explicit `action=` outcome

**Open items deferred:**

- Should `startup_failed` after N consecutive container restarts auto-promote to an alert in `hermes doctor`? Probably yes; tracked as a follow-up to this task.
- The `_render_run_script` duplication between this module and `S6ServiceManager.register_profile_gateway` (Task 3.2) is intentional duplication for testability. Phase 5 cleanup task should extract a shared helper.
- This task does NOT cover restart-policy semantics for the main hermes service itself — that's a Phase 2 concern (`finish` script behavior), already covered there.

### Task 4.1: Hook register_profile_gateway into profile creation

**Files:**
- Modify: `hermes_cli/profiles.py` — find the profile-creation code path (approximately near `def create_profile`)
- Modify: `tests/hermes_cli/test_profiles.py`

**Step 1: Identify the integration point**

```bash
grep -n "def create_profile\|def profile_create\|def _create_profile" hermes_cli/profiles.py
```

Read the surrounding code to find where the profile directory is seeded. The s6 registration call goes right after a successful create, guarded by `supports_runtime_registration()`.

**Step 2: Write a failing test**

```python
def test_profile_create_registers_s6_gateway_in_container(monkeypatch, tmp_path):
    """In a container, profile create should register the s6 gateway service."""
    from hermes_cli import profiles

    registered = []
    class FakeS6Manager:
        kind = "s6"
        def supports_runtime_registration(self): return True
        def register_profile_gateway(self, profile, *, port, extra_env=None):
            registered.append(profile)

    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager",
        lambda: FakeS6Manager(),
    )

    profiles.create_profile("newprof")  # exact signature TBD

    assert "newprof" in registered


def test_profile_create_no_op_on_host(monkeypatch):
    """On host (systemd/launchd), profile create should NOT attempt s6 registration."""
    from hermes_cli import profiles
    from hermes_cli.service_manager import SystemdServiceManager

    monkeypatch.setattr(
        "hermes_cli.service_manager.get_service_manager",
        lambda: SystemdServiceManager(),
    )
    # Should not raise NotImplementedError
    profiles.create_profile("hostprof")
```

**Step 3: Implement**

In `hermes_cli/profiles.py`, after the successful profile creation block:

```python
def _maybe_register_gateway_service(profile_name: str) -> None:
    """In container, register the profile's gateway as an s6 service.
    On host, no-op (existing systemd unit-generation paths handle it)."""
    try:
        from hermes_cli.service_manager import get_service_manager
        mgr = get_service_manager()
    except RuntimeError:
        return
    if not mgr.supports_runtime_registration():
        return
    # Allocate port — simple sequential allocation for v1; future: port scan
    from hermes_cli import profiles as _profiles_module
    port = _allocate_gateway_port(profile_name)
    try:
        mgr.register_profile_gateway(profile_name, port=port)
    except ValueError:
        # Already registered — re-register would clobber, so we leave alone
        pass
```

Add a port allocator:

```python
_GATEWAY_PORT_BASE = 9200

def _allocate_gateway_port(profile_name: str) -> int:
    """Deterministic port allocation based on profile name hash.

    Range [9200, 9800). Collisions are very unlikely but would fail the
    gateway startup with a clear bind error.
    """
    import hashlib
    h = int(hashlib.sha256(profile_name.encode()).hexdigest()[:8], 16)
    return _GATEWAY_PORT_BASE + (h % 600)
```

Call `_maybe_register_gateway_service(name)` at the end of the create-profile function.

**Step 4: Commit**

```bash
git add hermes_cli/profiles.py tests/hermes_cli/test_profiles.py
git commit -m "feat(profiles): register s6 gateway service on profile create in container"
```

### Task 4.2: Hook unregister_profile_gateway into profile deletion

**Files:**
- Modify: `hermes_cli/profiles.py`
- Modify: `tests/hermes_cli/test_profiles.py`

**Step 1: Tests**

Mirror Task 4.1's tests for the delete path.

**Step 2: Implement**

```python
def _maybe_unregister_gateway_service(profile_name: str) -> None:
    try:
        from hermes_cli.service_manager import get_service_manager
        mgr = get_service_manager()
    except RuntimeError:
        return
    if not mgr.supports_runtime_registration():
        return
    mgr.unregister_profile_gateway(profile_name)
```

Call it early in the profile-delete function (before removing the profile directory).

**Step 3: Commit**

```bash
git add hermes_cli/profiles.py tests/hermes_cli/test_profiles.py
git commit -m "feat(profiles): unregister s6 gateway service on profile delete"
```

### Task 4.3: Route `hermes -p <profile> gateway start/stop` through s6 in container

**Objective:** Existing CLI surface continues to work. Inside the container, it talks to s6 instead of being rejected.

**Files:**
- Modify: `hermes_cli/gateway.py` — the `gateway_command` / `_gateway_command_inner` dispatcher

**Background — what's there today**

`gateway_command` currently rejects gateway lifecycle commands when running inside a container. Search for `elif is_container():` in `hermes_cli/gateway.py` — you'll find arms inside `install`, `uninstall`, `start`, `stop`, and `restart` that print messages like "Service installation is not needed inside a Docker container — the container runtime is your service manager" and `sys.exit(0)`.

These were correct under the **old** model where there was one gateway and the container itself supervised it. They're **wrong** under the new model where each profile has its own supervised gateway. Phase 4 has to delete them in the same change that introduces the s6 dispatch path.

**Step 1: Add the s6 dispatch helper**

```python
def _dispatch_via_service_manager_if_s6(action: str, profile: str | None = None) -> bool:
    """If we're in a container with s6, dispatch gateway lifecycle via s6.
    Returns True if dispatched (caller should return), False otherwise.

    `profile` defaults to the current profile (resolved via _profile_arg).
    """
    from hermes_cli.service_manager import detect_service_manager, get_service_manager
    if detect_service_manager() != "s6":
        return False
    if profile is None:
        # current profile via existing helper
        profile = _profile_arg() or "default"
    mgr = get_service_manager()
    service_name = f"gateway-{profile}"
    if action == "start":
        mgr.start(service_name)
    elif action == "stop":
        mgr.stop(service_name)
    elif action == "restart":
        mgr.restart(service_name)
    else:
        return False
    return True
```

**Step 2: Remove the `elif is_container()` early-exit arms AND inject the s6 dispatch**

Inside `_gateway_command_inner`, find each branch (`install`, `uninstall`, `start`, `stop`, `restart`). For each one:

1. **Remove** the entire `elif is_container():` block that exits with an informational message. (Search for the literal string `"Docker container"` to find them — there are five.)
2. **Insert** the s6 dispatch at the top of each lifecycle handler:

```python
elif subcmd == "start":
    # Container path: hand off to s6 service manager
    if _dispatch_via_service_manager_if_s6("start"):
        return
    # … existing host code (systemd / launchd / windows / fallback) …
```

For `install` and `uninstall`, treat them as no-ops inside the container under s6 — the service is auto-registered by the profile create hook (Task 4.1) and removed by the profile delete hook (Task 4.2). Add a short message:

```python
elif subcmd == "install":
    from hermes_cli.service_manager import detect_service_manager
    if detect_service_manager() == "s6":
        print_info("Per-profile gateways are auto-registered when you create a profile (hermes profile create <name>).")
        print_info("Run `hermes status` to see currently-supervised gateways.")
        return
    # … existing host code …
```

The mirror applies for `uninstall`.

**Step 3: Regression tests**

Add a unit test for the dispatcher AND remove the xfail markers from `tests/docker/test_profile_gateway.py` (Task 0.5):

```python
def test_dispatch_via_service_manager_invokes_s6(monkeypatch):
    from hermes_cli import gateway as gw

    called = {}
    class FakeMgr:
        kind = "s6"
        def start(self, name): called["start"] = name
        def stop(self, name): called["stop"] = name
        def restart(self, name): called["restart"] = name

    monkeypatch.setattr("hermes_cli.service_manager.detect_service_manager", lambda: "s6")
    monkeypatch.setattr("hermes_cli.service_manager.get_service_manager", lambda: FakeMgr())

    assert gw._dispatch_via_service_manager_if_s6("start", profile="coder") is True
    assert called["start"] == "gateway-coder"


def test_dispatch_skips_on_host(monkeypatch):
    from hermes_cli import gateway as gw
    monkeypatch.setattr("hermes_cli.service_manager.detect_service_manager", lambda: "systemd")
    assert gw._dispatch_via_service_manager_if_s6("start", profile="coder") is False
```

Then remove the xfail markers and `_PHASE4_REASON` constant from `tests/docker/test_profile_gateway.py`.

**Step 4: Re-run Phase 0 harness**

```bash
scripts/run_tests.sh tests/docker/test_profile_gateway.py -v
```

Expected: 2 passed (no longer xfailed). If they're still xfailing, the dispatch isn't intercepting — verify `detect_service_manager()` returns `"s6"` inside the container, then verify the `elif is_container():` arms were actually removed.

**Step 5: Commit**

```bash
git add hermes_cli/gateway.py tests/hermes_cli/test_gateway.py tests/docker/test_profile_gateway.py
git commit -m "feat(gateway): dispatch gateway start/stop through s6 inside container

- Remove the 5 elif is_container() arms in _gateway_command_inner that
  refused gateway install/uninstall/start/stop/restart inside containers.
- Add _dispatch_via_service_manager_if_s6() that intercepts start/stop/
  restart and routes them through the S6ServiceManager.
- install/uninstall become informational no-ops when running under s6
  (profile create/delete is the registration trigger).
- Remove the xfail markers from tests/docker/test_profile_gateway.py;
  they now pass strictly."
```

### Task 4.4: Update `hermes_cli/status.py` for s6 detection

**Objective:** `hermes status` inside the container reports "Manager: s6" instead of "systemd/manual".

**Files:**
- Modify: `hermes_cli/status.py`

**Locating the code:**

```bash
grep -n '"Manager:' hermes_cli/status.py
```

You'll find a `print(f"  Manager:      …")` block that currently dispatches on `Termux / systemd / launchd / (not supported)`.

**Step 1: Test + implementation**

Add an `"s6"` branch to the manager-label resolution alongside the existing systemd/launchd/Termux branches. Use `detect_service_manager() == "s6"` to drive the new branch. The label should read `Manager:      s6 (container supervisor)` for clarity.

**Step 2: Commit**

```bash
git add hermes_cli/status.py tests/hermes_cli/test_status.py
git commit -m "feat(status): report s6 as the service manager inside container"
```

---

## Phase 5 — Docs + cleanup

### Task 5.1: Update `website/docs/user-guide/docker.md`

**Objective:** Document the new supervision model. The dashboard IS supervised; per-profile gateways are supervised; TUI works unchanged.

Add an "Init system" section covering:
- s6-overlay as PID 1 (replacing tini)
- Main hermes is a supervised service
- Dashboard (HERMES_DASHBOARD=1) is supervised — crashes auto-restart
- Per-profile gateways created via `hermes profile create` are supervised — crashes auto-restart
- `docker run -it --rm <image> --tui` works unchanged
- Breaking change callout: if a downstream wrapper depended on tini specifics, pin to a pre-change image

### Task 5.2: Create a maintainer skill

Create `skills/software-development/hermes-s6-container-supervision/SKILL.md` documenting:
- Where service definitions live: `docker/s6-rc.d/` (static), `hermes_cli/service_manager.py` (dynamic registration)
- How to inspect a live container: `docker exec … s6-svstat /run/service/gateway-<profile>`
- How to add a new static service: create dir under `docker/s6-rc.d/`, add `contents.d` entry
- Common pitfalls: service-dir permissions, `with-contenv` shebang, `s6-setuidgid` placement
- Debugging a profile gateway that won't start: check `$HERMES_HOME/logs/gateways/<profile>/current` (defaults to `/opt/data/logs/gateways/<profile>/current` when `HERMES_HOME` is unset)

### Task 5.3: Update `hermes_cli/doctor.py` for in-container runs

**Objective:** Remove spurious warnings when `hermes doctor` runs inside the container, and surface the s6 supervision state.

**Files:**
- Modify: `hermes_cli/doctor.py`
- Modify: `tests/hermes_cli/test_doctor.py`

> **v3 note:** Since v2 was written, `hermes_cli/doctor.py` was refactored (PR #27830, `41f1eddee`) to introduce two helpers — `_section(title: str)` for section banners and `_fail_and_issue(text, detail, fix, issues)` for failure rendering. The 15 old copy-paste banner patterns and ~30 fail-and-issue blocks have all been migrated. **When adding the new "s6 supervision status" section under this task, use `_section("Gateway Service")` (existing section, just add an s6 branch inside) and `_fail_and_issue(...)` for any new failure paths — do NOT duplicate the old `print(color("◆ ...", Colors.CYAN, Colors.BOLD))` pattern.** The existing `_check_gateway_service_linger` function (still present, same name) is the target for the "skip on s6" branch.

**Locating the code (function names, not line numbers — they drift):**

```bash
grep -n "def _check_gateway_service_linger\|External Tools\|# Docker (optional)\|◆ Gateway Service" hermes_cli/doctor.py
```

You should find: `_check_gateway_service_linger` (called from the main doctor flow), the "External Tools" section header, the "Docker (optional)" check inside it, and the gateway service section header (currently rendered as something like `◆ Gateway Service`).

**Changes:**

1. **`_check_gateway_service_linger`**: skip when `detect_service_manager() == "s6"`. Replace with a new `_check_s6_supervision()` that reports main-hermes and dashboard status via `ServiceManager.is_running(...)`, plus the count of `gateway-*` services from `list_profile_gateways()`.

2. **Docker external-tool check**: when `is_container()` is True, replace the "Docker missing" warning with an info line ("Running inside a container — Docker-in-Docker not configured, using in-container terminal backend"). Still check the `TERMINAL_ENV` config to make sure it's set to `local` inside the container (Docker backend from inside a container is not supported).

3. **Gateway Service section header**: rename to "Service Supervisor" and dispatch on `detect_service_manager()` so the section title is accurate everywhere (systemd / launchd / windows / s6 / manual).

**Step 1: Test + implementation — standard TDD**

**Step 2: Commit**

```bash
git add hermes_cli/doctor.py tests/hermes_cli/test_doctor.py
git commit -m "feat(doctor): surface s6 supervision state inside container"
```

### Task 5.4: Remove dead container-era systemd detection

**Objective:** `_container_systemd_operational()` in `hermes_cli/gateway.py` was added for "systemd inside a container" detection. With s6 as the container init system, this branch is dead code.

- Verify no code paths actually hit it in the new world (search + test suite)
- Remove the function + its `is_container()` branch in `supports_systemd_services()`
- Keep `supports_systemd_services()` returning False inside our container (now handled by the top-level `is_container()` check or by the `detect_service_manager() == "s6"` path)

### Task 5.5: Update `website/docs/user-guide/profiles.md`

The Profiles docs mention `hermes-gateway-<profile>.service` (systemd) — add a brief note that inside the container, per-profile gateways are supervised by s6 and use `s6-svstat` / `s6-svc` under the hood.

### Task 5.6: Release notes

Add a clear entry to the release notes calling out:
- New feature: per-profile gateways inside the Hermes container are now supervised — they auto-restart on crash, clean shutdown on container stop
- New feature: dashboard (`HERMES_DASHBOARD=1`) is now supervised
- Breaking change: container ENTRYPOINT is `/init` (s6-overlay) not `/usr/bin/tini`. Any external scripts that `docker exec`'d tini-specific commands need updating

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Phase 2 breaks a downstream user's Dockerfile that `FROM`s ours | Medium | Medium | Release notes call out ENTRYPOINT change; Phase 0 harness gives high confidence in behavior parity |
| TUI TTY passthrough fails on some Docker versions | Low | High | Phase 2 harness includes `test_tty_passthrough_to_container` as a hard gate; fallback plan = s6-fdholder (OQ9-C) |
| s6-overlay non-root quirks (logutil-service, fix-attrs) bite us | Low | Low | OQ2-A: supervisor runs as root, services drop — sidesteps these issues |
| Port collision between per-profile gateways | Low | Medium | Deterministic hash-based allocation (SHA256 of profile name) over a 600-port range; collision probability is ~1/600 per pair; gateway bind fails with a clear error if it happens, caller can set an explicit port |
| Podman rootless UID mapping confuses s6 | Medium | Low | OQ4-A: document, fix reactively; a local Podman + Docker environment will be stood up for validation |
| Phase 0 harness is flaky (docker daemon issues, timing) | Medium | Low | Generous timeouts; skip when docker unavailable; run in a CI-only job, not in fast local dev loop |
| Profile gateway crash loop masks a real config error | Low | Medium | `max_restarts` set on s6 finish script (planned for follow-up); for now, operators see crash-looping logs in `$HERMES_HOME/logs/gateways/<profile>/` |
| Dockerfile+entrypoint drift from linter (hadolint/shellcheck) reveals latent bugs | Low | Low | Phase 0.5 catches them; fix or document ignore with rationale |
| Stale `gateway.pid` from a dead container collides with an unrelated live PID in the restarted container | Low | Medium | Task 4.0 reconciliation removes `gateway.pid` and `processes.json` from every profile dir on boot, before any new gateway starts. End-to-end test `test_stale_gateway_pid_is_cleaned_up_on_restart` covers it |
| `docker restart` silently loses per-profile gateway registrations (tmpfs scandir wiped) | High (without mitigation) | High | Task 4.0 reconciliation re-registers from persistent `$HERMES_HOME/profiles/` and auto-starts those last seen `running`; recorded outcome to `$HERMES_HOME/logs/container-boot.log` for forensics |
| A `running` gateway that's actually broken auto-restarts into a crash loop after every container restart | Low | Medium | s6 finish script `max_restarts` cap (already planned); follow-up: `hermes doctor` alerts when N consecutive container restarts ended in `startup_failed` |

---

## Rollout Plan

All phases after Phase 0 are gated on the Phase 0 harness passing against the modified image. No feature flags or kill switches — Phase 2 is a one-way door, which is fine given the OQ1-A decision to ship directly.

1. **Phase 0** — merge immediately; pure test-harness addition, no behavior change
2. **Phase 0.5** — merge after 0; adds lint CI jobs
3. **Phase 1** — merge after 0.5; pure refactor addition
4. **Phase 2** — merge when Phase 0 harness is green against the new image; bump semver-major
5. **Phase 3** — merge after 2 is in a release; new capability with no callers yet
6. **Phase 4** — merge when in-container integration tests pass; activates Phase 3
7. **Phase 5** — merge incrementally as docs/cleanup is ready

---

## Decision Log

| # | Question | Decision | Blocks phase |
|---|---|---|---|
| OQ1 | Gate Phase 2 behind env var? | A — ship directly | Phase 2 |
| OQ2 | s6 root model | A — root `/init`, drop per-service | Phase 2 |
| OQ3 | Dashboard opt-in mechanism | A — always declared, run checks env | Phase 2 |
| OQ4 | Podman rootless | A — supported, fix reactively | Phase 2 |
| OQ5 | Service naming | `gateway-<profile>` | Phase 3 |
| OQ6 | — (retired; no subagent gateways in scope) | — | — |
| OQ7 | Resource limits | C — defer | Phase 3 |
| OQ8 | Log persistence | C — `$HERMES_HOME/logs/gateways/<profile>/` | Phase 3 |
| OQ9 | TUI passthrough | A — trust docs, test is the hard gate | Phase 2 |

**All questions resolved. No blockers remain.**

---

## Estimated Timeline

| Phase | Tasks | Engineering days |
|---|---|---|
| Phase 0 | 0.1–0.7 | 2.0 |
| Phase 0.5 | 0.5.1–0.5.2 | 0.5 |
| Phase 1 | 1.1–1.4 | 1.5 |
| Phase 2 | 2.1–2.5 | 3.0 |
| Phase 3 | 3.1–3.5 | 2.0 |
| Phase 4 | 4.0–4.4 | 2.0 |
| Phase 5 | 5.1–5.6 | 1.5 |
| **Total** | | **12.5 days** |

Phase 0 is longer than the original estimate because the test harness it builds is load-bearing for the entire plan — it's what lets us sign off Phase 2 as "identical behavior." Phase 3 + 4 are shorter than the old plan's Phase 3 + 4 because we're not building a general transient-service API — just per-profile gateway registration.

---

## Verification Checklist

Before declaring the full plan complete:

- [ ] Phase 0 harness passes against `main` (tini) (Phase 0)
- [ ] hadolint + shellcheck run green in CI (Phase 0.5)
- [ ] Phase 0 harness passes against the s6 image (Phase 2 — hard gate)
- [ ] `docker run -it --rm hermes-agent --tui` starts the Ink TUI with working keyboard input, cursor control, and resize (SIGWINCH) (Phase 2)
- [ ] Dashboard crashes are recovered by s6 within ~2s (Phase 2)
- [ ] `hermes profile create test` inside a container creates `/run/service/gateway-test/` (Phase 4)
- [ ] `hermes -p test gateway start` inside a container dispatches through s6 (verified by process tree: no double-fork) (Phase 4)
- [ ] `hermes -p test gateway stop` inside a container cleanly stops via s6 (Phase 4)
- [ ] `hermes profile delete test` inside a container removes `/run/service/gateway-test/` (Phase 4)
- [ ] Profile gateway logs persist at `$HERMES_HOME/logs/gateways/test/current` (Phase 4)
- [ ] `hermes status` inside the container shows `Manager: s6` (Phase 4)
- [ ] Full `scripts/run_tests.sh` passes (Phase 1–5)
- [ ] Full `scripts/run_tests.sh tests/docker/` passes when Docker available (Phase 0–5)
- [ ] No systemd/launchd host-side functions were modified (only wrapped) (Phase 1)
- [ ] `hermes gateway install/start/stop` on Linux host and macOS host behave identically to pre-change (Phase 1)

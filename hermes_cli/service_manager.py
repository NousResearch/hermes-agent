"""Abstract service manager interface + systemd/launchd/Windows/s6 backends."""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

ServiceManagerKind = Literal["systemd", "launchd", "windows", "s6", "none"]

# Profile names become s6 service directory names (``<scandir>/gateway-<profile>/``), so they
# must not traverse paths, span filesystems, or break s6's own naming rules.
_VALID_PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_MAX_PROFILE_LEN = 251  # s6-svscan default name_max


def validate_profile_name(name: str) -> None:
    """Raise ValueError unless ``name`` is a filesystem/s6-safe profile name."""
    if not name:
        raise ValueError("profile name must not be empty")
    if len(name) > _MAX_PROFILE_LEN:
        raise ValueError(f"profile name too long ({len(name)} > {_MAX_PROFILE_LEN})")
    if not _VALID_PROFILE_RE.match(name):
        raise ValueError(f"profile name must match [a-z0-9][a-z0-9_-]*, got {name!r}")


@runtime_checkable
class ServiceManager(Protocol):
    """Init-system-specific service operations.

    Lifecycle methods exist on every backend. Runtime registration (register/unregister/
    list_profile_gateways) is s6-only — check ``supports_runtime_registration()`` first.
    """

    kind: ServiceManagerKind

    def start(self, name: str) -> None: ...
    def stop(self, name: str) -> None: ...
    def restart(self, name: str) -> None: ...
    def is_running(self, name: str) -> bool: ...

    def supports_runtime_registration(self) -> bool: ...
    def register_profile_gateway(
        self, profile: str, *, extra_env: dict[str, str] | None = None, start_now: bool = True
    ) -> None: ...
    def unregister_profile_gateway(self, profile: str) -> None: ...
    def list_profile_gateways(self) -> list[str]: ...


def detect_service_manager() -> ServiceManagerKind:
    """Return "s6" (s6-svscan is PID 1), "windows", "launchd", "systemd" (working bus) or "none".

    Does NOT replace ``supports_systemd_services()`` for host call sites; it exists for
    backend-agnostic code (profile hooks, the s6 dispatch in ``hermes gateway``).
    """
    # Deferred so importing this module (Protocol type, validate_profile_name) doesn't drag in
    # the whole gateway dependency graph.
    from hermes_cli.gateway import is_macos, is_windows, supports_systemd_services
    # Gate on _s6_running() alone, NOT is_container(): the latter only detects Docker/Podman/lxc
    # and is False on Fly's Firecracker microVMs even though s6-overlay is PID 1 there — that
    # made the s6 dispatch inert on Fly, so `hermes gateway start` spawned a foreground gateway
    # competing with the supervised one.
    if _s6_running():
        return "s6"
    if is_windows():
        return "windows"
    if is_macos():
        return "launchd"
    if supports_systemd_services():
        return "systemd"
    return "none"


def _s6_running() -> bool:
    """True when s6-svscan is PID 1 in this container.

    Must work for the unprivileged hermes user too: ``/proc/1/exe`` is unreadable for other UIDs
    (``resolve()`` silently yields the literal ``exe``), which made runtime registration inert in
    production. Probe the world-readable ``/proc/1/comm`` AND ``/run/s6/basedir`` — either alone
    can false-positive.

    The obvious probe — ``Path('/proc/1/exe').resolve()`` — only works as root: for any other UID, the
    symlink at ``/proc/1/exe`` is unreadable and ``resolve()`` silently returns the path unchanged, so the
    resolved name is the literal ``"exe"`` and detection always fails. Since every Hermes runtime call
    inside the container drops to hermes via ``s6-setuidgid``, that silent failure made the entire
    service-manager runtime-registration path inert in production (PR #30136 review).
    """
    try:
        comm = Path("/proc/1/comm").read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return comm == "s6-svscan" and Path("/run/s6/basedir").is_dir()


# ---------------------------------------------------------------------------
# Host backends: thin facades over ``hermes_cli.gateway`` (systemd/launchd) and
# ``hermes_cli.gateway_windows``. The protocol's ``name`` parameter is unused here — host backends
# operate on the currently active profile (``hermes -p <profile>``); the shape exists for s6 where
# each profile maps to a distinct service directory.
# ---------------------------------------------------------------------------


class _HostServiceManager:
    """``start``/``stop``/``restart`` resolve to ``<_fn_prefix><op>`` on ``hermes_cli.<_backend>``
    at call time (lazy import; tests monkeypatch the submodule or its functions). Runtime
    registration is unsupported on every host backend.
    """

    kind: ServiceManagerKind
    _backend: str
    _fn_prefix: str = ""

    def _backend_module(self):
        import importlib
        import hermes_cli
        importlib.import_module(f"hermes_cli.{self._backend}")
        return getattr(hermes_cli, self._backend)

    def _call(self, op: str) -> None:
        getattr(self._backend_module(), f"{self._fn_prefix}{op}")()

    def start(self, name: str) -> None:
        self._call("start")

    def stop(self, name: str) -> None:
        self._call("stop")

    def restart(self, name: str) -> None:
        self._call("restart")

    def supports_runtime_registration(self) -> bool:
        return False

    def _unsupported(self, verb: str) -> NotImplementedError:
        return NotImplementedError(
            f"{type(self).__name__} does not support runtime profile gateway {verb} (container-only feature)"
        )

    def register_profile_gateway(
        self, profile: str, *, extra_env: dict[str, str] | None = None, start_now: bool = True
    ) -> None:
        raise self._unsupported("registration")

    def unregister_profile_gateway(self, profile: str) -> None:
        raise self._unsupported("unregistration")

    def list_profile_gateways(self) -> list[str]:
        return []


class SystemdServiceManager(_HostServiceManager):
    """Wraps the ``systemd_*`` functions in hermes_cli.gateway (host call sites still use those
    directly; this exists for backend-agnostic code such as the profile create/delete hooks)."""

    kind: ServiceManagerKind = "systemd"
    _backend = "gateway"
    _fn_prefix = "systemd_"

    def is_running(self, name: str) -> bool:
        _, running = self._backend_module()._probe_systemd_service_running()
        return running


class LaunchdServiceManager(_HostServiceManager):
    """Wraps the ``launchd_*`` functions in hermes_cli.gateway."""

    kind: ServiceManagerKind = "launchd"
    _backend = "gateway"
    _fn_prefix = "launchd_"

    def is_running(self, name: str) -> bool:
        return self._backend_module()._probe_launchd_service_running()


class WindowsServiceManager(_HostServiceManager):
    """Wraps ``hermes_cli.gateway_windows`` (Scheduled Task / Startup-folder fallback).

    Not a true init service but the lifecycle protocol is the same. ``install`` takes
    Windows-specific kwargs passed straight through — non-Windows callers must never call it.
    """

    kind: ServiceManagerKind = "windows"
    _backend = "gateway_windows"

    def install(
        self,
        *,
        force: bool = False,
        start_now: bool | None = None,
        start_on_login: bool | None = None,
        elevated_handoff: bool = False,
    ) -> None:
        self._backend_module().install(
            force=force, start_now=start_now, start_on_login=start_on_login, elevated_handoff=elevated_handoff
        )

    def is_running(self, name: str) -> bool:
        from hermes_cli.gateway import find_gateway_pids
        if not self._backend_module().is_installed():
            return False
        return bool(find_gateway_pids())


def get_service_manager() -> ServiceManager:
    """Return the ServiceManager instance for the current environment."""
    cls = _MANAGER_CLASSES.get(detect_service_manager())
    if cls is None:
        raise RuntimeError("no supported service manager detected")
    return cls()


# ---------------------------------------------------------------------------
# S6ServiceManager (container-only). Per-profile gateways are registered dynamically by
# `hermes profile create` inside the container. Static services (main-hermes, dashboard) live in
# /etc/s6-overlay/s6-rc.d/ as part of the image and are NOT managed here.
# ---------------------------------------------------------------------------

# s6-overlay's dynamic scandir (tmpfs) that s6-svscan watches; writes here trigger supervision
# on the next rescan.
S6_DYNAMIC_SCANDIR = Path("/run/service")


def _default_s6_scandir() -> Path:
    """Return the production scandir, or an explicitly isolated test one.

    The hermetic pytest fixture sets ``HERMES_TEST_S6_SCANDIR`` to prevent
    container-aware tests from registering, stopping, or restarting live s6
    services when the suite itself runs inside a production container. Normal
    runtime processes do not set this internal test-only variable and continue
    to use ``/run/service``.
    """
    import os

    isolated = os.environ.get("HERMES_TEST_S6_SCANDIR")
    if isolated:
        return Path(isolated)
    return S6_DYNAMIC_SCANDIR


S6_SERVICE_PREFIX = "gateway-"


def _profile_from_service(name: str) -> str:
    """Strip the ``gateway-`` prefix back off (matches what the user typed via ``-p``)."""
    return name[len(S6_SERVICE_PREFIX):] if name.startswith(S6_SERVICE_PREFIX) else name


def _profile_dir_for_gateway_service(name: str) -> Path:
    """Resolve ``gateway-<profile>`` to its persistent profile directory.

    s6 lifecycle commands may run from any active profile (``gateway stop --all``), so never write
    the caller's HERMES_HOME blindly: derive the shared profile root and map the suffix to the root
    default profile or ``<root>/profiles/<profile>``.
    """
    profile = _profile_from_service(name)
    validate_profile_name(profile)
    hermes_home = Path(os.environ.get("HERMES_HOME", "/opt/data"))
    root = hermes_home.parent.parent if hermes_home.parent.name == "profiles" else hermes_home
    return root if profile == "default" else root / "profiles" / profile


def _write_gateway_desired_state(name: str, desired_state: str) -> None:
    """Persist the operator's start/stop intent (``desired_state``) next to the gateway's volatile
    ``gateway_state`` so boot reconciliation can restore s6 want-up/down after pod recreation.
    Best-effort: a failed write must not block immediate s6 lifecycle control.
    """
    profile_dir = _profile_dir_for_gateway_service(name)
    state_file = profile_dir / "gateway_state.json"
    try:
        if not profile_dir.exists():
            return
        try:
            data = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
            if not isinstance(data, dict):
                data = {}
        except (OSError, json.JSONDecodeError):
            data = {}
        data["desired_state"] = desired_state
        data["updated_at"] = int(time.time())
        tmp = state_file.with_suffix(state_file.suffix + ".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
        tmp.replace(state_file)
    except OSError:
        return


# s6-overlay installs its binaries under /command/ and only adds it to PATH inside the supervision
# tree. Out-of-tree entry points (``docker exec``, the profile create/delete hooks) inherit the base
# PATH, so every s6 invocation uses this absolute prefix. Not ``/usr/bin/s6-*``: the
# s6-overlay-symlinks-noarch tarball only links a subset.
_S6_BIN_DIR = "/command"


def _s6_run(cmd: str, *args: str, timeout: float = 5, check: bool = False):
    """Run an s6 binary by absolute path with the shared capture/decode settings."""
    return subprocess.run(
        [f"{_S6_BIN_DIR}/{cmd}", *args],
        check=check, capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=timeout,
    )


# UID/GID of the in-image ``hermes`` user; hardcoded to match what ``stage2-hook.sh`` enforces
# (tests/docker/test_uid_remap.py). s6-supervise starts as root and drops via ``s6-setuidgid``.
_HERMES_UID = 10000
_HERMES_GID = 10000


def _chown_hermes(path: Path) -> None:
    try:
        os.chown(path, _HERMES_UID, _HERMES_GID)
    except PermissionError:
        # Already running as hermes → the dir is hermes-owned by default; swallowing keeps root
        # and unprivileged callers on one code path.
        pass


def _seed_supervise_skeleton(svc_dir: Path) -> None:
    """Pre-create hermes-owned ``supervise/`` and top-level ``event/`` inside a service directory.

    s6-supervise (root) creates ``event/``/``supervise/`` 0700 and the control FIFO 0600, so the
    hermes user gets EACCES on every ``s6-svc``/``s6-svstat``. s6 treats EEXIST as success and skips
    its chown/chmod fix-up, so seeding before ``s6-svscanctl -a`` makes s6-supervise inherit our
    ownership. ``log/`` gets the same skeleton (its own supervise instance) or unregister teardown
    EACCESes on the logger. Idempotent: existing entries (possibly live FIFOs) are left untouched.

    The PR #30136 review surfaced this as a real product gap: the entire S6ServiceManager lifecycle
    (``register/start/stop/unregister _profile_gateway``) was inert in production because every operation is
    dispatched as the hermes user.
    Reference --------- Discussed at length on the skarnet `skaware` mailing list in 2020
    (`<http://skarnet.org/lists/skaware/1424.html>`_); see also just-containers/s6-overlay#130. The
    pre-creation pattern was historically called out as forward-compatibility-fragile, but the EEXIST
    handling in s6-supervise has been stable since 2015 — it's the same pattern ``s6-svperms`` and
    ``fix-attrs.d`` rely on.
    """

    def _mkdir_owned(path: Path, mode: int) -> None:
        if path.exists():
            return
        path.mkdir(parents=False, exist_ok=False)
        path.chmod(mode)
        _chown_hermes(path)

    def _seed(root: Path) -> None:
        # Service-root event/ is the s6-svlisten1 subscription dir, distinct from supervise/event/.
        _mkdir_owned(root / "event", 0o3730)
        supervise = root / "supervise"
        _mkdir_owned(supervise, 0o755)
        _mkdir_owned(supervise / "event", 0o3730)
        # EEXIST-safe FIFO: if s6-supervise already started against this slot, leave it. The
        # explicit chmod is required because mkfifo honors the umask (0022 on dev hosts strips
        # group-write → 0o640); stage2 runs umask 0 but be defensive for any invocation context.
        control = supervise / "control"
        if not control.exists():
            os.mkfifo(control, 0o660)
            control.chmod(0o660)
            _chown_hermes(control)

    _seed(svc_dir)
    log_dir = svc_dir / "log"
    if log_dir.is_dir():
        _seed(log_dir)


class S6Error(RuntimeError):
    """Base for S6ServiceManager lifecycle failures; carries the slot name so the CLI can render an
    actionable message instead of a raw ``CalledProcessError``."""

    def __init__(self, message: str, *, service: str | None = None) -> None:
        super().__init__(message)
        self.service = service


class GatewayNotRegisteredError(S6Error):
    """A lifecycle method targeted a missing slot. ``profile`` is unprefixed so callers can phrase
    "no such gateway 'typo'"."""

    def __init__(self, profile: str) -> None:
        self.profile = profile
        super().__init__(
            f"no such gateway {profile!r}: register it with "
            f"`hermes profile create {profile}` first, or pass "
            "an existing profile name via `-p <name>`",
            service=f"gateway-{profile}",
        )


class S6CommandError(S6Error):
    """An s6 command failed for a reason other than a missing slot (EACCES on the control FIFO,
    unexpected non-zero exit); carries the command's stderr."""

    def __init__(self, *, service: str, action: str, returncode: int, stderr: str) -> None:
        self.action = action
        self.returncode = returncode
        self.stderr = stderr
        message = f"s6-svc {action} on {service!r} failed (rc={returncode})"
        if stderr.strip():
            message += f": {stderr.strip()}"
        super().__init__(message, service=service)


class S6ServiceManager:
    """Per-profile gateway supervision via s6-overlay, for runtime-registered services under
    ``S6_DYNAMIC_SCANDIR`` only (static services are managed by s6-rc at image-build time)."""

    kind: ServiceManagerKind = "s6"

    def __init__(self, scandir: Path | None = None) -> None:
        self.scandir = Path(scandir) if scandir is not None else _default_s6_scandir()

    def _service_dir(self, profile: str) -> Path:
        validate_profile_name(profile)
        return self.scandir / f"{S6_SERVICE_PREFIX}{profile}"

    @staticmethod
    def _render_run_script(profile: str, extra_env: dict[str, str]) -> str:
        """Run script for a profile-gateway s6 service.

        Sources HERMES_HOME via with-contenv (run time, not baked in), resets ``HOME`` before the
        privilege drop so root's HOME does not leak, activates the venv, drops to hermes.
        ``profile == "default"`` emits NO ``-p`` flag: it is the sentinel for the root HERMES_HOME
        profile and ``-p default`` would look up ``profiles/default/``. Port comes from the
        profile's own env (``API_SERVER_PORT``, default 8642); two profiles that both leave it
        unset collide.

        Port selection: the gateway binds the port resolved by ``gateway/config.py`` from the profile's own
        environment — ``API_SERVER_PORT`` (or ``platforms.api_server.extra.port`` in that profile's
        ``config.yaml``), defaulting to 8642. There is no ``[gateway] port`` key and no Python-side
        allocator: because each supervised profile gateway loads its own ``HERMES_HOME``, two profiles that
        both leave the port unset will both try to bind 8642 — give each profile a distinct
        ``API_SERVER_PORT`` in its ``.env``. Previously this method took a ``port`` parameter that was
        passed in but never substituted into the rendered script (carried for "API parity" with a
        deterministic SHA-256 allocator in ``hermes_cli.profiles._allocate_gateway_port``). PR #30136 review
        item I5 retired both the allocator and the parameter because they were dead code through the entire
        stack.
        """
        lines = [
            "#!/command/with-contenv sh",
            "# shellcheck shell=sh",
            "set -e",
            "export HOME=/opt/data",
            "cd /opt/data",
            ". /opt/hermes/.venv/bin/activate",
        ]
        for k, v in sorted(extra_env.items()):
            lines.append(f"export {k}={shlex.quote(v)}")
        # Supervised-child sentinel: without it the supervised gateway re-entering
        # `_gateway_command_inner` with subcmd == "run" would dispatch `gateway start` → re-exec
        # `gateway run --replace` → `gateway start` … (see the matching guard there).
        lines.append("export HERMES_S6_SUPERVISED_CHILD=1")
        # Generalized supervisor marker — same meaning for the profile-redirect guard in
        # hermes_cli.main._apply_profile_override; kept alongside the s6 one for back-compat.
        lines.append("export HERMES_SUPERVISED_CHILD=1")
        # ``--replace`` makes the supervised gateway authoritative for its HERMES_HOME. Without it
        # a gateway started OUTSIDE s6 (stray ``hermes gateway run``, an agent action, the Open
        # WebUI helper) grabs the PID lock first; the slot then hits "Another gateway instance is
        # already running", exits non-zero, and s6 restarts it forever — a log-flooding loop that
        # never binds. ``--replace`` reaps the stale holder (marker + SIGTERM→SIGKILL-with-
        # confirmation + scoped-lock cleanup, see gateway/run.py) so s6 always wins; the sentinel
        # above prevents the run→start→run recursion. s6 guarantees one supervised instance per
        # slot, so there is no legitimate sibling for ``--replace`` to clobber.
        if profile == "default":
            gateway_cmd = "hermes gateway run --replace"
        else:
            gateway_cmd = f"hermes -p {shlex.quote(profile)} gateway run --replace"
        # Skip the drop when already non-root (setgroups() lacks CAP_SETGID → s6 boot-loop).
        lines.append(f'[ "$(id -u)" = 0 ] || exec {gateway_cmd}')
        lines.append(f"exec s6-setuidgid hermes {gateway_cmd}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_finish_script() -> str:
        """Finish script: exit 78 (EX_CONFIG, fatal config) and clean exit 0 (intentional stop —
        restarting would turn every normal exit into a reconnect storm) both map to 125 so s6 stops
        restarting; only other non-zero exits let s6 restart normally.

        When the gateway exits with EX_CONFIG (78) — a fatal configuration error such as a token collision
        or no messaging platforms — we tell s6-supervise to stop restarting by exiting 125 (permanent
        failure). A clean exit 0 is an intentional stop, not a crash: restarting after it turns any normal
        gateway exit into a reconnect loop (the ashriel-discord storm in #76435 — 1,000+ connections and a
        provider token reset). See #51228, #76435.
        """
        from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE
        code = GATEWAY_FATAL_CONFIG_EXIT_CODE
        return (
            "#!/command/with-contenv sh\n"
            "# shellcheck shell=sh\n"
            "# $1 = exit code from the run script.\n"
            f"# Exit {code} (EX_CONFIG) = fatal config error — don't restart.\n"
            "# Exit 0 (clean stop) = intentional stop — don't restart.\n"
            f'if [ "$1" = "{code}" ]; then\n'
            "  exit 125\n"
            "fi\n"
            'if [ "$1" = "0" ]; then\n'
            "  exit 125\n"
            "fi\n"
            "exit 0\n"
        )

    @staticmethod
    def _render_log_run(profile: str) -> str:
        """log/run script. s6-log directives apply per line in order; ``T`` (timestamp) is
        non-sticky and only prefixes lines for the next action directive, so it sits between
        ``1`` and the log dir rather than before ``1``."""
        prof = shlex.quote(profile)
        return (
            f"#!/command/with-contenv sh\n"
            f"# shellcheck shell=sh\n"
            f': "${{HERMES_HOME:=/opt/data}}"\n'
            f'log_dir="$HERMES_HOME/logs/gateways/{prof}"\n'
            # Create the leaf and clear a stale s6-log lock AS HERMES when starting as root. Never
            # chown/unlink hermes-writable volume paths from this restartable root-context script:
            # an unprivileged user can race a pathname op through a symlink swap (CWE-59/CWE-367).
            # Parent logs/gateways is seeded hermes-owned at stage2 boot (test_log_dir_seed.py).
            # See #45258.
            f'if [ "$(id -u)" = 0 ]; then\n'
            f'  s6-setuidgid hermes mkdir -p "$log_dir"\n'
            f'  s6-setuidgid hermes rm -f "$log_dir/lock"\n'
            f'else\n'
            f'  mkdir -p "$log_dir"\n'
            f'  rm -f "$log_dir/lock"\n'
            f'fi\n'
            # Skip the drop when already non-root (CAP_SETGID).
            f'[ "$(id -u)" = 0 ] || exec s6-log 1 n10 s1000000 T "$log_dir"\n'
            f'exec s6-setuidgid hermes s6-log 1 n10 s1000000 T "$log_dir"\n'
        )

    # -- lifecycle ---------------------------------------------------------

    def _run_svc(self, action_flag: str, action_label: str, name: str) -> None:
        """``s6-svc <action_flag>``; a missing service dir raises ``GatewayNotRegisteredError``
        (instead of s6-svc's opaque failure) and any other failure ``S6CommandError``."""
        service_dir = self.scandir / name
        if not service_dir.is_dir():
            raise GatewayNotRegisteredError(_profile_from_service(name))
        try:
            _s6_run("s6-svc", action_flag, str(service_dir), check=True)
        except subprocess.CalledProcessError as exc:
            raise S6CommandError(
                service=name, action=action_label, returncode=exc.returncode, stderr=exc.stderr or ""
            ) from exc

    def start(self, name: str) -> None:
        self._run_svc("-u", "start", name)
        _write_gateway_desired_state(name, "running")

    def _supervised_pid(self, name: str) -> int | None:
        """PID of the supervised gateway per ``s6-svstat``, or None on any failure."""
        try:
            result = _s6_run("s6-svstat", str(self.scandir / name))
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        m = re.search(r"\(pid (\d+)\)", result.stdout)
        return int(m.group(1)) if m else None

    def stop(self, name: str) -> None:
        """``s6-svc -d``, after writing a planned-stop marker for the supervised PID so the gateway's
        shutdown handler classifies this SIGTERM as operator-initiated and persists
        ``gateway_state=stopped``."""
        pid = self._supervised_pid(name)
        if pid is not None:
            try:
                from gateway.status import write_planned_stop_marker
                write_planned_stop_marker(pid)
            except Exception:
                pass
        self._run_svc("-d", "stop", name)
        _write_gateway_desired_state(name, "stopped")

    def restart(self, name: str) -> None:
        """``s6-svc -t`` (SIGTERM)."""
        self._run_svc("-t", "restart", name)
        _write_gateway_desired_state(name, "running")

    def is_running(self, name: str) -> bool:
        result = _s6_run("s6-svstat", str(self.scandir / name))
        return result.returncode == 0 and "up " in result.stdout

    # -- runtime registration ---------------------------------------------

    def supports_runtime_registration(self) -> bool:
        return True

    def register_profile_gateway(
        self, profile: str, *, extra_env: dict[str, str] | None = None, start_now: bool = True
    ) -> None:
        """Create the s6 service directory and ``s6-svscanctl -a`` so it is picked up immediately.

        ``start_now=False`` writes a ``down`` marker so the service stays stopped until an explicit
        ``gateway start``. Raises ValueError on an invalid name or existing directory, RuntimeError
        if ``s6-svscanctl`` fails.
        """
        svc_dir = self._service_dir(profile)
        if svc_dir.exists():
            raise ValueError(f"profile gateway {profile!r} already registered at {svc_dir}")
        # Build atomically in a DOT-PREFIXED sibling (``.gateway-<profile>.tmp``) then rename:
        # s6-svscan skips dot entries, so a concurrent rescan (cont-init reconciler, sibling
        # register) cannot supervise the half-built slot. Otherwise s6-supervise would spawn AS
        # ROOT on the ``.tmp`` (it already has ``type``/``run``), mkdir ``supervise/`` root-owned
        # 0700, and our ``_seed_supervise_skeleton`` would EACCES on ``supervise/event`` — the
        # arm64-only CI flake on test_s6_unregister_removes_service_dir_in_live_container.
        tmp_dir = svc_dir.with_name("." + svc_dir.name + ".tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)

        def _write_script(path: Path, text: str) -> None:
            path.write_text(text, encoding="utf-8")
            path.chmod(0o755)

        try:
            (tmp_dir / "type").write_text("longrun\n", encoding="utf-8")
            _write_script(tmp_dir / "run", self._render_run_script(profile, extra_env or {}))
            _write_script(tmp_dir / "finish", self._render_finish_script())
            (tmp_dir / "log").mkdir()
            _write_script(tmp_dir / "log" / "run", self._render_log_run(profile))
            # Seed hermes-owned supervise/ BEFORE publishing so the hermes-user s6-svc/s6-svstat/
            # s6-svwait calls never hit root-owned 0700 dirs (see _seed_supervise_skeleton).
            _seed_supervise_skeleton(tmp_dir)
            # Mirrors container_boot._register_gateway_slot when start=False.
            if not start_now:
                (tmp_dir / "down").touch()
            tmp_dir.rename(svc_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        result = _s6_run("s6-svscanctl", "-a", str(self.scandir))
        if result.returncode != 0:
            # No supervisor is watching it — leaving the directory would be confusing.
            shutil.rmtree(svc_dir, ignore_errors=True)
            raise RuntimeError(f"s6-svscanctl failed: {result.stderr or result.stdout}")

    def unregister_profile_gateway(self, profile: str) -> None:
        """Stop the profile gateway (best effort, wait for down) and remove its directory.
        Idempotent: absent services are a no-op.

        ``s6-svscanctl -an`` fires BEFORE ``rmtree`` so s6-svscan reaps the supervise child and
        releases its handles on ``supervise/lock``/``status``/``death_tally``; those files are
        root-owned but the parent ``supervise/`` is hermes-owned (see ``_seed_supervise_skeleton``)
        and POSIX only needs write+execute on the parent to remove them.
        """
        svc_dir = self._service_dir(profile)
        if not svc_dir.exists():
            return
        _s6_run("s6-svc", "-d", str(svc_dir))
        _s6_run("s6-svwait", "-D", "-t", "10000", str(svc_dir), timeout=15)
        _s6_run("s6-svscanctl", "-an", str(self.scandir))
        # No synchronous "scan completed" handshake — -a/-n just set a flag s6-svscan reads on its
        # next loop; 200ms is comfortably above that resolution.
        time.sleep(0.2)
        shutil.rmtree(svc_dir, ignore_errors=True)

    def list_profile_gateways(self) -> list[str]:
        """Profile names of all currently-registered gateway services."""
        if not self.scandir.exists():
            return []
        return [
            entry.name[len(S6_SERVICE_PREFIX):]
            for entry in self.scandir.iterdir()
            if not entry.name.startswith(".")
            and entry.is_dir()
            and entry.name.startswith(S6_SERVICE_PREFIX)
        ]


_MANAGER_CLASSES: dict[str, type] = {
    "systemd": SystemdServiceManager,
    "launchd": LaunchdServiceManager,
    "windows": WindowsServiceManager,
    "s6": S6ServiceManager,
}

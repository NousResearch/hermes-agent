"""Abstract service manager interface.

Wraps the existing systemd (Linux host), launchd (macOS host), Windows
Scheduled Task (native Windows host), and s6 (container) backends behind
a common Protocol. Only the s6 backend supports runtime registration
(for per-profile gateways) — host backends raise NotImplementedError
from those methods, and callers MUST check supports_runtime_registration()
before invoking them.

Host-side call sites (setup wizard, uninstall, status) continue to use
the existing module-level functions in hermes_cli.gateway and
hermes_cli.gateway_windows directly. This protocol is a thin facade
used by new code that needs to be backend-agnostic — specifically the
profile create/delete hooks (Phase 4) and the s6 dispatch path in
``hermes gateway start/stop/restart`` when running inside a container.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

ServiceManagerKind = Literal["systemd", "launchd", "windows", "s6", "none"]

# Profile name → service directory mapping. Profile names must be safe
# as filesystem directory names because the s6 backend creates a service
# directory at ``<scandir>/gateway-<profile>/``. We reject anything that
# could traverse paths, span filesystems, or break s6's own naming rules.
_VALID_PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_MAX_PROFILE_LEN = 251  # s6-svscan default name_max


def validate_profile_name(name: str) -> None:
    """Raise ValueError if ``name`` is not usable as a profile name.

    Profile names are used as s6 service directory names, so they must
    match a conservative subset of filesystem-safe characters. Reject
    empty strings, uppercase, paths-traversal sequences, and anything
    longer than s6's default ``name_max``.
    """
    if not name:
        raise ValueError("profile name must not be empty")
    if len(name) > _MAX_PROFILE_LEN:
        raise ValueError(
            f"profile name too long ({len(name)} > {_MAX_PROFILE_LEN})"
        )
    if not _VALID_PROFILE_RE.match(name):
        raise ValueError(
            f"profile name must match [a-z0-9][a-z0-9_-]*, got {name!r}"
        )


@runtime_checkable
class ServiceManager(Protocol):
    """Abstract interface for init-system-specific service operations.

    Lifecycle methods (start / stop / restart / is_running) are
    implemented by every backend. Runtime registration
    (register_profile_gateway / unregister_profile_gateway /
    list_profile_gateways) is implemented only by the s6 backend —
    callers MUST check ``supports_runtime_registration()`` before
    invoking the registration methods.
    """

    kind: ServiceManagerKind

    # Lifecycle of a pre-declared service.
    def start(self, name: str) -> None: ...
    def stop(self, name: str) -> None: ...
    def restart(self, name: str) -> None: ...
    def is_running(self, name: str) -> bool: ...

    # Runtime registration (s6 only).
    def supports_runtime_registration(self) -> bool: ...
    def register_profile_gateway(
        self,
        profile: str,
        *,
        port: int,
        extra_env: dict[str, str] | None = None,
    ) -> None: ...
    def unregister_profile_gateway(self, profile: str) -> None: ...
    def list_profile_gateways(self) -> list[str]: ...


def detect_service_manager() -> ServiceManagerKind:
    """Detect which service manager is available in this environment.

    Returns:
        "s6" — inside a container when /init is s6-svscan (Phase 2+)
        "windows" — native Windows host
        "launchd" — macOS host
        "systemd" — Linux host with a working user/system bus
        "none" — anything else (Termux, sandbox shells, etc.)

    This function does NOT replace ``supports_systemd_services()`` —
    host call sites continue to use that. It exists for new backend-
    agnostic code (profile create/delete hooks, the s6 dispatch path
    in ``hermes gateway start/stop/restart``).
    """
    # Imports deferred so importing this module doesn't drag in the
    # whole gateway dependency graph for callers that only need the
    # Protocol type or validate_profile_name().
    from hermes_constants import is_container
    from hermes_cli.gateway import (
        is_macos,
        is_windows,
        supports_systemd_services,
    )

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
    """True when s6-svscan is running as PID 1 in this container.

    s6-overlay's /init exec's s6-svscan, so ``/proc/1/exe`` resolves
    to it (or to ``init`` on some kernel configurations that hide the
    exe link). The ``/run/s6/`` directory is created by stage1, so its
    presence is a second necessary signal.
    """
    try:
        exe = Path("/proc/1/exe").resolve()
        return exe.name in ("s6-svscan", "init") and Path("/run/s6").exists()
    except (OSError, RuntimeError):
        return False


# ---------------------------------------------------------------------------
# Backend wrappers
#
# These adapters are thin facades over the existing module-level functions
# in ``hermes_cli.gateway`` (systemd/launchd) and ``hermes_cli.gateway_windows``
# (Windows Scheduled Tasks). The protocol's ``name`` parameter is currently
# unused for host backends — they operate on whichever profile is currently
# active (set via the ``hermes -p <profile>`` flag before the call). This
# matches existing host-side semantics; the parameter shape is designed
# for s6 where each profile maps to a distinct service directory.
# ---------------------------------------------------------------------------


class _RegistrationUnsupportedMixin:
    """Mixin for host backends that don't support runtime registration."""

    def supports_runtime_registration(self) -> bool:
        return False

    def register_profile_gateway(
        self,
        profile: str,
        *,
        port: int,
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
    """Thin wrapper around the ``systemd_*`` functions in hermes_cli.gateway.

    Existing host call sites continue to use those functions directly;
    this wrapper exists for new code that needs to be backend-agnostic
    (the Phase 4 profile create/delete hooks).
    """

    kind: ServiceManagerKind = "systemd"

    def start(self, name: str) -> None:
        from hermes_cli.gateway import systemd_start
        systemd_start()

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
    """Thin wrapper around the ``launchd_*`` functions in hermes_cli.gateway."""

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
    """Thin wrapper around ``hermes_cli.gateway_windows`` (Scheduled Task /
    Startup-folder fallback).

    The native Windows backend uses a Scheduled Task rather than a true
    init-system service, but for protocol purposes the lifecycle is the
    same: start / stop / restart / is_running. ``install`` accepts a
    handful of Windows-specific kwargs (start_now, start_on_login,
    elevated_handoff) that are passed straight through — non-Windows
    callers should never invoke ``install`` on this wrapper.
    """

    kind: ServiceManagerKind = "windows"

    def install(
        self,
        *,
        force: bool = False,
        start_now: bool | None = None,
        start_on_login: bool | None = None,
        elevated_handoff: bool = False,
    ) -> None:
        from hermes_cli import gateway_windows
        gateway_windows.install(
            force=force,
            start_now=start_now,
            start_on_login=start_on_login,
            elevated_handoff=elevated_handoff,
        )

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
        from hermes_cli import gateway_windows
        from hermes_cli.gateway import find_gateway_pids
        if not gateway_windows.is_installed():
            return False
        return bool(find_gateway_pids())


def get_service_manager() -> ServiceManager:
    """Return the ServiceManager instance for the current environment.

    Raises:
        RuntimeError: when no supported backend is available.
    """
    kind = detect_service_manager()
    if kind == "systemd":
        return SystemdServiceManager()
    if kind == "launchd":
        return LaunchdServiceManager()
    if kind == "windows":
        return WindowsServiceManager()
    if kind == "s6":
        return S6ServiceManager()
    raise RuntimeError("no supported service manager detected")


# ---------------------------------------------------------------------------
# S6ServiceManager (container-only)
#
# Per-profile gateways are registered dynamically when `hermes profile create`
# runs inside the container (Phase 4). Static services (main-hermes, dashboard)
# live in /etc/s6-overlay/s6-rc.d/ and are NOT managed by this class — they're
# part of the image, not runtime-created.
# ---------------------------------------------------------------------------


# s6-overlay's dynamic scandir for runtime-registered services. Lives on
# tmpfs and is the directory s6-svscan watches. Writes here trigger
# automatic supervision on the next rescan.
S6_DYNAMIC_SCANDIR = Path("/run/service")
S6_SERVICE_PREFIX = "gateway-"

# s6-overlay installs its binaries under /command/ and only adds that
# directory to PATH for processes started under the supervision tree
# (services started by s6-svscan, cont-init.d scripts, etc.). Code
# that runs via `docker exec` or any other out-of-tree entry point —
# notably our Phase 4 profile create/delete hooks — inherits the
# container's base PATH which does NOT include /command/.
#
# Rather than asking every caller to fix up its environment, the
# S6ServiceManager calls s6-* binaries by absolute path via this
# constant. We don't use `/usr/bin/s6-…` symlinks because the
# s6-overlay-symlinks-noarch tarball only links a subset, and we
# want every s6 invocation to be guaranteed-findable.
_S6_BIN_DIR = "/command"


class S6ServiceManager:
    """Per-profile gateway supervision via s6-overlay.

    Only handles runtime-registered services under
    ``S6_DYNAMIC_SCANDIR``. Static services (main-hermes, dashboard)
    are managed by s6-rc at image-build time and are out of scope.
    """

    kind: ServiceManagerKind = "s6"

    def __init__(self, scandir: Path = S6_DYNAMIC_SCANDIR) -> None:
        self.scandir = scandir

    # -- internal helpers --------------------------------------------------

    def _service_dir(self, profile: str) -> Path:
        validate_profile_name(profile)
        return self.scandir / f"{S6_SERVICE_PREFIX}{profile}"

    def _service_name(self, profile: str) -> str:
        return f"{S6_SERVICE_PREFIX}{profile}"

    @staticmethod
    def _render_run_script(
        profile: str,
        port: int,
        extra_env: dict[str, str],
    ) -> str:
        """Generate the run script for a profile-gateway s6 service.

        The script:
          1. Sources HERMES_HOME (and any extra env) via with-contenv —
             so e.g. ``-e HERMES_HOME=/data/hermes`` is honored at run
             time, not Python-substituted at registration time (OQ8-C).
          2. Activates the bundled venv.
          3. Drops to the hermes user and exec's
             ``hermes -p <profile> gateway run``.

        Note: the ``port`` parameter is accepted for API parity with
        :meth:`register_profile_gateway` but is currently ignored — the
        gateway picks its bind port from the profile's config.yaml
        (``[gateway] port = ...``). A future signature change may carry
        it through as an ``HERMES_GATEWAY_PORT`` env var; until then,
        the in-config value wins and the constructor's ``port`` arg
        is essentially documentation for "what port the profile would
        use if we wired it through". See Phase 4 Task 4.1 for the
        deterministic allocator and the SHA-256-derived range
        [9200, 9800).
        """
        import shlex
        lines = [
            "#!/command/with-contenv sh",
            "# shellcheck shell=sh",
            "set -e",
            "cd /opt/data",
            ". /opt/hermes/.venv/bin/activate",
        ]
        for k, v in sorted(extra_env.items()):
            lines.append(f"export {k}={shlex.quote(v)}")
        lines.append(
            f"exec s6-setuidgid hermes hermes -p {shlex.quote(profile)} gateway run"
        )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_log_run(profile: str) -> str:
        """Generate the log/run script for a profile-gateway service.

        OQ8-C: persist to ``${HERMES_HOME}/logs/gateways/<profile>/``.
        CRITICAL: the HERMES_HOME path is sourced from the runtime env
        via with-contenv — NOT Python-substituted at registration time
        — so a container started with ``-e HERMES_HOME=/data/hermes``
        gets its logs under /data/hermes/logs/..., not the build-time
        default.
        """
        import shlex
        prof = shlex.quote(profile)
        return (
            f"#!/command/with-contenv sh\n"
            f"# shellcheck shell=sh\n"
            f': "${{HERMES_HOME:=/opt/data}}"\n'
            f'log_dir="$HERMES_HOME/logs/gateways/{prof}"\n'
            f'mkdir -p "$log_dir"\n'
            f'chown -R hermes:hermes "$log_dir" 2>/dev/null || true\n'
            f'exec s6-setuidgid hermes s6-log n10 s1000000 T "$log_dir"\n'
        )

    # -- lifecycle ---------------------------------------------------------

    def start(self, name: str) -> None:
        """Bring up a registered service (``s6-svc -u``)."""
        import subprocess
        subprocess.run(
            [f"{_S6_BIN_DIR}/s6-svc", "-u", str(self.scandir / name)],
            check=True, capture_output=True, timeout=5,
        )

    def stop(self, name: str) -> None:
        """Bring down a registered service (``s6-svc -d``)."""
        import subprocess
        subprocess.run(
            [f"{_S6_BIN_DIR}/s6-svc", "-d", str(self.scandir / name)],
            check=True, capture_output=True, timeout=5,
        )

    def restart(self, name: str) -> None:
        """Restart a registered service (``s6-svc -t`` = SIGTERM)."""
        import subprocess
        subprocess.run(
            [f"{_S6_BIN_DIR}/s6-svc", "-t", str(self.scandir / name)],
            check=True, capture_output=True, timeout=5,
        )

    def is_running(self, name: str) -> bool:
        """True iff ``s6-svstat`` reports the service as up."""
        import subprocess
        result = subprocess.run(
            [f"{_S6_BIN_DIR}/s6-svstat", str(self.scandir / name)],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0 and "up " in result.stdout

    # -- runtime registration ---------------------------------------------

    def supports_runtime_registration(self) -> bool:
        return True

    def register_profile_gateway(
        self,
        profile: str,
        *,
        port: int,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        """Create the s6 service directory for a profile gateway.

        Triggers ``s6-svscanctl -a`` so s6-svscan picks the new directory
        up immediately. The service is created in the *up* state — to
        register without auto-starting, follow up with ``stop(profile)``
        (or pass the start flag via the future ``start_now=False`` arg,
        which the Phase 4 reconciliation path uses via a ``down``
        marker file written directly).

        Raises:
            ValueError: if the profile name is invalid or the service
                directory already exists.
            RuntimeError: if ``s6-svscanctl`` fails.
        """
        import shutil
        import subprocess

        svc_dir = self._service_dir(profile)
        if svc_dir.exists():
            raise ValueError(
                f"profile gateway {profile!r} already registered at {svc_dir}"
            )

        # Build the service directory atomically: write to a sibling
        # temp dir, then rename. Avoids s6-svscan observing a half-
        # populated directory on a fast rescan.
        tmp_dir = svc_dir.with_name(svc_dir.name + ".tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)

        try:
            (tmp_dir / "type").write_text("longrun\n")

            run_script = self._render_run_script(profile, port, extra_env or {})
            run_path = tmp_dir / "run"
            run_path.write_text(run_script)
            run_path.chmod(0o755)

            # Persistent log rotation (OQ8-C).
            log_subdir = tmp_dir / "log"
            log_subdir.mkdir()
            log_run = log_subdir / "run"
            log_run.write_text(self._render_log_run(profile))
            log_run.chmod(0o755)

            tmp_dir.rename(svc_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        # Trigger rescan so s6-svscan picks up the new service.
        result = subprocess.run(
            [f"{_S6_BIN_DIR}/s6-svscanctl", "-a", str(self.scandir)],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            # Clean up: rescan failed, leave the directory in place would
            # be confusing (no supervisor watching it).
            shutil.rmtree(svc_dir, ignore_errors=True)
            raise RuntimeError(
                f"s6-svscanctl failed: {result.stderr or result.stdout}"
            )

    def unregister_profile_gateway(self, profile: str) -> None:
        """Stop the profile gateway service and remove its directory.

        Idempotent: absent services are a no-op. Best-effort stop +
        wait-for-down before removal so the running gateway process
        gets a chance to shut down cleanly before its service dir
        disappears.
        """
        import shutil
        import subprocess

        svc_dir = self._service_dir(profile)
        if not svc_dir.exists():
            return

        # Stop the service (best effort — service may already be down).
        subprocess.run(
            [f"{_S6_BIN_DIR}/s6-svc", "-d", str(svc_dir)],
            capture_output=True, text=True, timeout=5,
            check=False,
        )
        # Wait for it to actually go down (up to 10s).
        subprocess.run(
            [f"{_S6_BIN_DIR}/s6-svwait", "-D", "-t", "10000", str(svc_dir)],
            capture_output=True, text=True, timeout=15,
            check=False,
        )

        # Remove the directory.
        shutil.rmtree(svc_dir, ignore_errors=True)

        # Rescan so s6-svscan drops its supervise process for the dir.
        # -n = also reap orphan supervise processes.
        subprocess.run(
            [f"{_S6_BIN_DIR}/s6-svscanctl", "-an", str(self.scandir)],
            capture_output=True, text=True, timeout=5,
            check=False,
        )

    def list_profile_gateways(self) -> list[str]:
        """Return the profile names of all currently-registered gateway services.

        Filters the scandir to entries that match the ``gateway-`` prefix.
        Other services (e.g. ``s6-linux-init-shutdownd``) are ignored.
        """
        if not self.scandir.exists():
            return []
        profiles: list[str] = []
        for entry in self.scandir.iterdir():
            if entry.name.startswith("."):
                continue
            if not entry.is_dir():
                continue
            if not entry.name.startswith(S6_SERVICE_PREFIX):
                continue
            profiles.append(entry.name[len(S6_SERVICE_PREFIX):])
        return profiles

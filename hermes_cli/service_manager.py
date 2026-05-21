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
        RuntimeError: when no supported backend is available, or when
            the detected backend's implementation hasn't shipped yet
            (the s6 backend lands in Phase 3).
    """
    kind = detect_service_manager()
    if kind == "systemd":
        return SystemdServiceManager()
    if kind == "launchd":
        return LaunchdServiceManager()
    if kind == "windows":
        return WindowsServiceManager()
    if kind == "s6":
        # Phase 3 will replace this with `return S6ServiceManager()`.
        raise RuntimeError("s6 backend not yet implemented (Phase 3)")
    raise RuntimeError("no supported service manager detected")

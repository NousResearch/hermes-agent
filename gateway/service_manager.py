"""Gateway-owned service-manager protocol, host adapters, and backend detection."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

ServiceManagerKind = Literal["systemd", "launchd", "windows", "s6", "none"]


@runtime_checkable
class ServiceManager(Protocol):
    """Init-system-specific service operations.

    Lifecycle methods exist on every backend. Runtime registration
    (register/unregister/list_profile_gateways) is s6-only.
    """

    kind: ServiceManagerKind

    def start(self, name: str) -> None: ...
    def stop(self, name: str) -> None: ...
    def restart(self, name: str) -> None: ...
    def is_running(self, name: str) -> bool: ...

    def supports_runtime_registration(self) -> bool: ...
    def register_profile_gateway(
        self,
        profile: str,
        *,
        extra_env: dict[str, str] | None = None,
        start_now: bool = True,
    ) -> None: ...
    def unregister_profile_gateway(self, profile: str) -> None: ...
    def list_profile_gateways(self) -> list[str]: ...


def _s6_running() -> bool:
    """True when s6-svscan is PID 1 and the s6-overlay runtime is present."""
    try:
        comm = Path("/proc/1/comm").read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return comm == "s6-svscan" and Path("/run/s6/basedir").is_dir()


def detect_service_manager() -> ServiceManagerKind:
    """Return the active gateway service manager for this host."""
    from gateway.systemd_runtime import supports_services

    if _s6_running():
        return "s6"
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "launchd"
    if supports_services():
        return "systemd"
    return "none"


class _HostServiceManager:
    """Base for host lifecycle adapters; runtime profile registration is s6-only."""

    kind: ServiceManagerKind

    def supports_runtime_registration(self) -> bool:
        return False

    def _unsupported(self, verb: str) -> NotImplementedError:
        return NotImplementedError(
            f"{type(self).__name__} does not support runtime profile gateway "
            f"{verb} (container-only feature)"
        )

    def register_profile_gateway(
        self,
        profile: str,
        *,
        extra_env: dict[str, str] | None = None,
        start_now: bool = True,
    ) -> None:
        raise self._unsupported("registration")

    def unregister_profile_gateway(self, profile: str) -> None:
        raise self._unsupported("unregistration")

    def list_profile_gateways(self) -> list[str]:
        return []


class SystemdServiceManager(_HostServiceManager):
    """Gateway systemd lifecycle adapter."""

    kind: ServiceManagerKind = "systemd"

    def start(self, name: str) -> None:
        from gateway.systemd_lifecycle import start

        start()

    def stop(self, name: str) -> None:
        from gateway.systemd_lifecycle import stop

        stop()

    def restart(self, name: str) -> None:
        from gateway.systemd_restart import systemd_restart

        systemd_restart()

    def is_running(self, name: str) -> bool:
        from gateway.systemd_runtime import select_scope, unit_is_active

        return unit_is_active(select_scope(False))


class LaunchdServiceManager(_HostServiceManager):
    """Gateway launchd lifecycle adapter."""

    kind: ServiceManagerKind = "launchd"

    def start(self, name: str) -> None:
        from gateway.launchd_service import launchd_start

        launchd_start()

    def stop(self, name: str) -> None:
        from gateway.launchd_service import launchd_stop

        launchd_stop()

    def restart(self, name: str) -> None:
        from gateway.launchd_service import launchd_restart

        launchd_restart()

    def is_running(self, name: str) -> bool:
        from gateway.launchd_service import _probe_launchd_service_running

        return _probe_launchd_service_running()


class WindowsServiceManager(_HostServiceManager):
    """Gateway Windows Scheduled Task / Startup-folder lifecycle adapter."""

    kind: ServiceManagerKind = "windows"

    @staticmethod
    def _backend_module():
        from gateway import windows_service

        return windows_service

    def start(self, name: str) -> None:
        self._backend_module().start()

    def stop(self, name: str) -> None:
        self._backend_module().stop()

    def restart(self, name: str) -> None:
        self._backend_module().restart()

    def install(
        self,
        *,
        force: bool = False,
        start_now: bool | None = None,
        start_on_login: bool | None = None,
        elevated_handoff: bool = False,
    ) -> None:
        self._backend_module().install(
            force=force,
            start_now=start_now,
            start_on_login=start_on_login,
            elevated_handoff=elevated_handoff,
        )

    def is_running(self, name: str) -> bool:
        from gateway.process_discovery import find_gateway_pids

        if not self._backend_module().is_installed():
            return False
        return bool(find_gateway_pids())


_MANAGER_CLASSES: dict[str, type[_HostServiceManager]] = {
    "systemd": SystemdServiceManager,
    "launchd": LaunchdServiceManager,
    "windows": WindowsServiceManager,
}


def get_service_manager() -> ServiceManager:
    """Return the ServiceManager instance for the current environment."""
    kind = detect_service_manager()
    if kind == "s6":
        from gateway.s6_manager import S6ServiceManager

        return S6ServiceManager()
    cls = _MANAGER_CLASSES.get(kind)
    if cls is None:
        raise RuntimeError("no supported service manager detected")
    return cls()

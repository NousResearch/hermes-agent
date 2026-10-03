"""Language Server Protocol (LSP) integration for Hermes Agent.

Real language servers (pyright, gopls, ...) run as subprocesses and their
``publishDiagnostics`` feed the post-write lint delta filter of ``write_file`` /
``patch`` (wiring: ``FileOperations._check_lint_delta``).  LSP is **gated on git
workspace detection** so user-home cwd's (e.g. Telegram gateway chats) never
spawn daemons; ``get_service()`` returns the singleton or ``None`` when disabled.
"""
from __future__ import annotations

import atexit
import logging
import threading
from dataclasses import dataclass
from typing import Optional, Union

from agent.lsp.manager import LSPService

logger = logging.getLogger("agent.lsp")

@dataclass(frozen=True)
class _ServiceTombstone:
    """A service whose teardown was not confirmed successful."""

    service: LSPService
    error: str


_service: Optional[Union[LSPService, _ServiceTombstone]] = None
# Routed multiplex profiles (HERMES_HOME override) each get their own service: ``lsp.*`` config
# (enabled, servers, idle timeout) is per profile, so one process-wide singleton would let the first
# profile's settings decide whether every other profile gets diagnostics.
_services_by_home: dict = {}
_atexit_registered = False
_service_lock = threading.Lock()


def _active(svc: Optional[LSPService]) -> Optional[LSPService]:
    return svc if (svc is not None and not isinstance(svc, _ServiceTombstone) and svc.is_active()) else None


def _register_atexit_once() -> None:
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(_atexit_shutdown)
        _atexit_registered = True


def get_service() -> Optional[LSPService]:
    """Return the active profile's service; failed teardown blocks its replacement."""
    global _service
    from hermes_constants import get_hermes_home_override, hermes_home_key
    with _service_lock:
        if get_hermes_home_override() is not None:
            home_key = hermes_home_key()
            if home_key not in _services_by_home:
                _services_by_home[home_key] = LSPService.create_from_config()
                _register_atexit_once()
            return _active(_services_by_home[home_key])
        if _service is None:
            _service = LSPService.create_from_config()
            _register_atexit_once()
        return _active(_service)


def release_workspace(path: str) -> int:
    """Shut down the LSP clients serving ``path`` (a worktree about to be removed) in every started
    service, without shutting down unrelated workspaces.  Never creates a service.  Returns the count."""
    with _service_lock:
        services = [svc for svc in (_service, *_services_by_home.values()) if svc is not None]
    released = 0
    for svc in services:
        try:
            owner = svc.service if isinstance(svc, _ServiceTombstone) else svc
            released += owner.release_workspace(path)
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP workspace release failed for %s: %s", path, e)
    return released


def _shutdown_owner(current):
    if current is None:
        return None
    svc = current.service if isinstance(current, _ServiceTombstone) else current
    try:
        if svc.shutdown() is True:
            return None
        error = svc._get_shutdown_error() or "teardown incomplete"
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
        logger.debug("LSP shutdown error: %s", error)
    return _ServiceTombstone(service=svc, error=error)


def shutdown_service() -> bool:
    """Serialize teardown with admission, retaining failed owners for a later retry."""
    global _service
    with _service_lock:
        _service = _shutdown_owner(_service)
        for home, current in list(_services_by_home.items()):
            retained = _shutdown_owner(current)
            if retained is None:
                del _services_by_home[home]
            else:
                _services_by_home[home] = retained
        return _service is None and not _services_by_home


def _atexit_shutdown() -> None:
    """atexit wrapper; logs at debug since the user has already seen the final output."""
    try:
        shutdown_service()
    except Exception as e:  # noqa: BLE001
        logger.debug("atexit LSP shutdown failed: %s", e)


__all__ = ["get_service", "release_workspace", "shutdown_service", "LSPService"]

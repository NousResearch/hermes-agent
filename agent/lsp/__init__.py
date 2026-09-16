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
from typing import Optional

from agent.lsp.manager import LSPService

logger = logging.getLogger("agent.lsp")

_service: Optional[LSPService] = None
# Routed multiplex profiles (HERMES_HOME override) each get their own service: ``lsp.*`` config
# (enabled, servers, idle timeout) is per profile, so one process-wide singleton would let the first
# profile's settings decide whether every other profile gets diagnostics.
_services_by_home: dict = {}
_atexit_registered = False
_service_lock = threading.Lock()


def _active(svc: Optional[LSPService]) -> Optional[LSPService]:
    return svc if (svc is not None and svc.is_active()) else None


def _register_atexit_once() -> None:
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(_atexit_shutdown)
        _atexit_registered = True


def get_service() -> Optional[LSPService]:
    """Return the lazily created LSP service for the active profile (process-wide singleton when no
    profile override is bound), or None when disabled.

    Also registers an :mod:`atexit` hook so a clean exit tears down spawned servers:
    without it every ``hermes chat`` exit leaks pyright processes for a few seconds
    while their stdout buffers drain.  (SIGKILL/os._exit skip atexit — fine, the
    kernel reaps the stateless servers with their parent.)
    """
    global _service
    from hermes_constants import get_hermes_home_override, hermes_home_key
    if get_hermes_home_override() is not None:
        home_key = hermes_home_key()
        with _service_lock:
            if home_key not in _services_by_home:
                _services_by_home[home_key] = LSPService.create_from_config()
                _register_atexit_once()
            return _active(_services_by_home[home_key])
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = LSPService.create_from_config()
                _register_atexit_once()
    return _active(_service)


def shutdown_service() -> None:
    """Tear down every LSP service that was started.  Idempotent."""
    global _service
    with _service_lock:
        services = [_service, *_services_by_home.values()]
        _service = None
        _services_by_home.clear()
    for svc in services:
        if svc is not None:
            try:
                svc.shutdown()
            except Exception as e:  # noqa: BLE001
                logger.debug("LSP shutdown error: %s", e)


def _atexit_shutdown() -> None:
    """atexit wrapper; logs at debug since the user has already seen the final output."""
    try:
        shutdown_service()
    except Exception as e:  # noqa: BLE001
        logger.debug("atexit LSP shutdown failed: %s", e)


__all__ = ["get_service", "shutdown_service", "LSPService"]

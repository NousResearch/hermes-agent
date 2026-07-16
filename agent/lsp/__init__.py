"""Language Server Protocol (LSP) integration for Hermes Agent.

Hermes runs full language servers (pyright, gopls, rust-analyzer,
typescript-language-server, etc.) as subprocesses and pipes their
``textDocument/publishDiagnostics`` output into the post-write lint
delta filter used by ``write_file`` and ``patch``.

LSP is **gated on git workspace detection** — if the agent's cwd is
inside a git repository, LSP runs against that workspace; otherwise the
file_operations layer falls back to its existing in-process syntax
checks.  This keeps users on user-home cwd's (e.g. Telegram gateway
chats) from spawning daemons they don't need.

Public API:

    from agent.lsp import get_service

    svc = get_service()
    if svc and svc.enabled_for(path):
        await svc.touch_file(path)
        diags = svc.diagnostics_for(path)

The bulk of the wiring is internal — most callers only need the layer
in :func:`tools.file_operations.FileOperations._check_lint_delta`,
which is already wired (see that module).

Architecture is documented in ``website/docs/user-guide/features/lsp.md``.
"""
from __future__ import annotations

import atexit
import logging
import threading
from typing import Optional

from agent.lsp.manager import LSPService

logger = logging.getLogger("agent.lsp")

_service: Optional[LSPService] = None
_atexit_registered = False
_service_lock = threading.Lock()


def _accepting(svc: LSPService) -> bool:
    check = getattr(svc, "is_accepting_requests", None)
    return bool(check()) if callable(check) else True


def _closed(svc: LSPService) -> bool:
    check = getattr(svc, "is_closed", None)
    return bool(check()) if callable(check) else False


def get_service() -> Optional[LSPService]:
    """Return the process-wide LSP service singleton, or None when disabled.

    A closing singleton is never exposed. Callers that race a process-local
    restart wait on ``_service_lock`` until the previous service has completed
    shutdown, preventing old and new LSP pools from overlapping.
    """
    global _service, _atexit_registered
    current = _service
    if current is not None and _accepting(current):
        return current if current.is_active() else None
    with _service_lock:
        current = _service
        if current is not None and _accepting(current):
            return current if current.is_active() else None
        if current is not None and not _closed(current):
            # Teardown is incomplete or failed. Keep the tombstoned singleton
            # rather than exposing an overlapping replacement pool.
            return None
        _service = LSPService.create_from_config()
        if not _atexit_registered:
            # ``atexit`` handlers run in LIFO order on normal Python
            # exit and on SystemExit, but NOT on os._exit() or
            # uncaught signals. Language servers are stateless
            # subprocesses and are recreated on the next process start.
            atexit.register(_atexit_shutdown)
            _atexit_registered = True
    return _service if (_service is not None and _service.is_active()) else None


def shutdown_service() -> None:
    """Synchronously retire the singleton before allowing replacement."""
    global _service
    with _service_lock:
        svc = _service
        if svc is None:
            return
        begin = getattr(svc, "begin_shutdown", None)
        if callable(begin):
            begin()
        try:
            svc.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.warning("LSP shutdown incomplete; replacement remains blocked: %s", exc)
            return
        if _service is svc and _closed(svc):
            _service = None


def _atexit_shutdown() -> None:
    """atexit-registered wrapper.  Logs at debug because by the time
    atexit fires the user has already seen the agent's final output —
    a noisy shutdown line on top of that is just clutter."""
    try:
        shutdown_service()
    except Exception as e:  # noqa: BLE001
        logger.debug("atexit LSP shutdown failed: %s", e)


__all__ = ["get_service", "shutdown_service", "LSPService"]

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


def get_service() -> Optional[LSPService]:
    """Return the process-wide LSP service singleton, or None when disabled.

    The service is created lazily on first call.  ``None`` is returned
    when LSP is disabled in config, when no workspace can be detected,
    or when the platform doesn't support subprocess-based LSP servers.

    On first creation, registers an :mod:`atexit` handler that tears
    down spawned language servers on Python exit so a long-running
    CLI or gateway session doesn't leak pyright/gopls/etc. processes
    when it terminates.
    """
    global _service, _atexit_registered
    if _service is not None:
        return _service if _service.is_active() else None
    with _service_lock:
        if _service is not None:
            return _service if _service.is_active() else None
        _service = LSPService.create_from_config()
        if not _atexit_registered:
            # ``atexit`` handlers run in LIFO order on normal Python
            # exit and on SystemExit, but NOT on os._exit() or
            # uncaught signals.  Language servers are stateless
            # subprocesses — losing them on SIGKILL is fine; they'll
            # be reaped by the kernel along with their parent.  We
            # care about clean exits where Python flushes stdio
            # before terminating; without this hook every
            # ``hermes chat`` exit would leak pyright processes that
            # outlive the parent for a few seconds while their
            # stdout buffers drain.
            atexit.register(_atexit_shutdown)
            _atexit_registered = True
    return _service if (_service is not None and _service.is_active()) else None


def shutdown_service() -> None:
    """Tear down the LSP service if one was started.

    Safe to call multiple times; safe to call when no service was created.
    """
    global _service
    with _service_lock:
        svc = _service
        _service = None
    if svc is not None:
        try:
            svc.shutdown()
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP shutdown error: %s", e)


def _atexit_shutdown() -> None:
    """atexit-registered wrapper.  Logs at debug because by the time
    atexit fires the user has already seen the agent's final output —
    a noisy shutdown line on top of that is just clutter."""
    try:
        shutdown_service()
    except Exception as e:  # noqa: BLE001
        logger.debug("atexit LSP shutdown failed: %s", e)


__all__ = ["get_service", "shutdown_service", "LSPService"]

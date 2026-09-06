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
    with _service_lock:
        current = _service
        if isinstance(current, _ServiceTombstone):
            return None
        if current is not None:
            return current if current.is_active() else None
        current = LSPService.create_from_config()
        _service = current
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
        return current if (current is not None and current.is_active()) else None


def shutdown_service() -> bool:
    """Tear down the LSP service if one was started.

    Returns ``True`` only when teardown completed.  The singleton lock is
    held through teardown so ``get_service()`` cannot publish a replacement
    concurrently.  Failed or incomplete teardown leaves a tombstone that
    continues refusing replacement until a later shutdown call succeeds.
    """
    global _service
    with _service_lock:
        current = _service
        if current is None:
            return True
        svc = current.service if isinstance(current, _ServiceTombstone) else current
        try:
            succeeded = bool(svc.shutdown())
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP shutdown error: %s", e)
            _service = _ServiceTombstone(
                service=svc,
                error=f"{type(e).__name__}: {e}",
            )
            return False
        if succeeded:
            _service = None
            return True
        try:
            error = svc._get_shutdown_error() or "teardown incomplete"
        except Exception as e:  # noqa: BLE001
            error = f"teardown incomplete; error unavailable: {type(e).__name__}: {e}"
        _service = _ServiceTombstone(
            service=svc,
            error=error,
        )
        return False


def _atexit_shutdown() -> None:
    """atexit-registered wrapper.  Logs at debug because by the time
    atexit fires the user has already seen the agent's final output —
    a noisy shutdown line on top of that is just clutter."""
    try:
        shutdown_service()
    except Exception as e:  # noqa: BLE001
        logger.debug("atexit LSP shutdown failed: %s", e)


__all__ = ["get_service", "shutdown_service", "LSPService"]

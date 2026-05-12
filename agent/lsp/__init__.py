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
from typing import Optional

from agent.lsp.manager import LSPService

_service: Optional[LSPService] = None
_atexit_registered: bool = False


def get_service() -> Optional[LSPService]:
    """Return the process-wide LSP service singleton, or None when disabled.

    The service is created lazily on first call.  ``None`` is returned
    when LSP is disabled in config, when no workspace can be detected,
    or when the platform doesn't support subprocess-based LSP servers.

    First successful service creation also registers an ``atexit`` hook
    so spawned language-server subprocesses (pyright, gopls, etc.) are
    shut down on interpreter exit.  Without this, daemon-thread death
    leaves orphan child processes behind — each one holding 80-200 MB
    of RAM until the kernel reaps them.
    """
    global _service, _atexit_registered
    if _service is not None:
        return _service if _service.is_active() else None
    _service = LSPService.create_from_config()
    if _service is not None and not _atexit_registered:
        # Belt to ``cli._run_cleanup``'s suspenders: covers code paths
        # that don't go through the main CLI (scripts importing
        # ``tools.file_operations`` directly, pytest sessions, etc.).
        atexit.register(shutdown_service)
        _atexit_registered = True
    return _service if (_service is not None and _service.is_active()) else None


def shutdown_service() -> None:
    """Tear down the LSP service if one was started.

    Safe to call multiple times; safe to call when no service was created.
    Idempotent — the ``atexit`` hook and ``cli._run_cleanup`` both invoke
    this; calling order doesn't matter.
    """
    global _service
    if _service is not None:
        try:
            _service.shutdown()
        except Exception:  # noqa: BLE001
            # Best-effort cleanup; never raise from a shutdown hook.
            pass
        _service = None


__all__ = ["get_service", "shutdown_service", "LSPService"]

"""Structured logging with steady-state silence for the LSP layer.

The LSP layer fires on every write_file/patch.  In a busy session
that's hundreds of events.  We want users to be able to ``rg`` the
log for "did LSP fire on that edit?" without drowning in noise.

The level model:

- ``DEBUG`` for steady-state events that have no novel signal:
  ``clean``, ``feature off``, ``extension not mapped``, ``no project
  root for already-announced file``, ``server unavailable for
  already-announced binary``.  These never reach ``agent.log`` at the
  default INFO threshold.

- ``INFO`` for state transitions worth surfacing exactly once per
  session: ``active for <root>`` the first time a (server_id,
  workspace_root) client starts, ``no project root for <path>``
  the first time we see that file.  Plus every diagnostic event
  (those are inherently rare and per-edit, exactly what users grep
  for).

- ``WARNING`` for action-required failures: ``server unavailable``
  (binary not on PATH) the first time per (server_id, binary),
  ``no server configured`` once per language.  Per-call WARNING for
  timeouts and unexpected bridge exceptions.

The dedup is in-process module-level sets.  Each set grows at most by
the number of distinct (server_id, root) and (server_id, binary)
pairs touched in one Python process — bytes of memory in even an
aggressive monorepo session.  Bounded LRU was rejected: evicting an
entry would risk re-firing the WARNING/INFO line we explicitly want
to suppress.

Grep recipe::

    tail -f ~/.hermes/logs/agent.log | rg 'lsp\\['
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Tuple

# Dedicated logger name so the documented grep recipe survives a
# ``logging.getLogger(__name__)`` rename of any internal module.
event_log = logging.getLogger("hermes.lint.lsp")

# ---------------------------------------------------------------------------
# Once-per-X dedup sets
# ---------------------------------------------------------------------------

_announce_lock = threading.Lock()
_announced_active: set = set()        # keys: (server_id, workspace_root)
_announced_unavailable: set = set()   # keys: (server_id, binary_path_or_name)
_announced_no_root: set = set()       # keys: (server_id, file_path)
_announced_no_server: set = set()     # keys: (server_id,)


def _short_path(file_path: str) -> str:
    """Render *file_path* relative to the cwd when sensible, else absolute.

    Keeps log lines readable for the common case (the user is inside
    the project they're editing) without emitting brittle ``../../..``
    chains for the cross-tree case.
    """
    if not file_path:
        return file_path
    try:
        rel = os.path.relpath(file_path)
    except ValueError:
        return file_path
    if rel.startswith(".." + os.sep) or rel == "..":
        return file_path
    return rel


def _emit(server_id: str, level: int, message: str) -> None:
    event_log.log(level, "lsp[%s] %s", server_id, message)


def _announce_once(bucket: set, key: Tuple) -> bool:
    """Return True if *key* has not been announced for *bucket* yet.

    Atomically marks the key as announced so concurrent callers
    cannot both win the race and double-log.
    """
    with _announce_lock:
        if key in bucket:
            return False
        bucket.add(key)
        return True


# ---------------------------------------------------------------------------
# Public event helpers — call these from the LSP layer.
# ---------------------------------------------------------------------------


def log_clean(server_id: str, file_path: str) -> None:
    """No diagnostics emitted for *file_path*.  DEBUG (silent at default)."""
    _emit(server_id, logging.DEBUG, f"clean ({_short_path(file_path)})")


def log_disabled(server_id: str, file_path: str, reason: str) -> None:
    """LSP intentionally skipped for this file (feature off, ext unmapped,
    backend not local, etc.).  DEBUG."""
    _emit(server_id, logging.DEBUG, f"skipped: {reason} ({_short_path(file_path)})")


def log_active(server_id: str, workspace_root: str) -> None:
    """A new LSP client started for (server_id, workspace_root).

    INFO once per (server_id, workspace_root); DEBUG thereafter.
    Lets users verify "is LSP actually running?" with a single grep.
    """
    key = (server_id, workspace_root)
    if _announce_once(_announced_active, key):
        _emit(server_id, logging.INFO, f"active for {workspace_root}")
    else:
        _emit(server_id, logging.DEBUG, f"reused client for {workspace_root}")


def log_diagnostics(server_id: str, file_path: str, count: int) -> None:
    """Diagnostics arrived for a file.  INFO every time — these are the
    failure signals users actually want to grep for, and they are
    inherently rare per edit."""
    _emit(server_id, logging.INFO, f"{count} diags ({_short_path(file_path)})")


def log_no_project_root(server_id: str, file_path: str) -> None:
    """File had no recognised project marker.  INFO once per file,
    DEBUG thereafter."""
    key = (server_id, file_path)
    if _announce_once(_announced_no_root, key):
        _emit(server_id, logging.INFO, f"no project root for {_short_path(file_path)}")
    else:
        _emit(server_id, logging.DEBUG, f"no project root for {_short_path(file_path)}")


def log_server_unavailable(server_id: str, binary_or_pkg: str) -> None:
    """The server binary couldn't be resolved.  WARNING once per
    (server_id, binary), DEBUG thereafter so a hundred subsequent
    .py edits don't spam the log."""
    key = (server_id, binary_or_pkg)
    if _announce_once(_announced_unavailable, key):
        _emit(
            server_id,
            logging.WARNING,
            f"server unavailable: {binary_or_pkg} not found "
            "(install via `hermes lsp install <id>` or set lsp.servers.<id>.command)",
        )
    else:
        _emit(server_id, logging.DEBUG, f"server still unavailable: {binary_or_pkg}")


def log_no_server_configured(server_id: str) -> None:
    """No spawn recipe for this language.  WARNING once."""
    if _announce_once(_announced_no_server, (server_id,)):
        _emit(server_id, logging.WARNING, "no server configured")


def log_timeout(server_id: str, file_path: str, kind: str = "diagnostics") -> None:
    """A request to the server timed out.  WARNING every time — these are
    inherently novel events worth surfacing on each occurrence."""
    _emit(
        server_id,
        logging.WARNING,
        f"{kind} timed out for {_short_path(file_path)}",
    )


def log_server_error(server_id: str, file_path: str, exc: BaseException) -> None:
    """An unexpected exception bubbled out of the LSP layer.  WARNING."""
    _emit(
        server_id,
        logging.WARNING,
        f"unexpected error for {_short_path(file_path)}: {type(exc).__name__}: {exc}",
    )


def log_spawn_failed(server_id: str, workspace_root: str, exc: BaseException) -> None:
    """The LSP server failed to spawn or initialize.  WARNING."""
    _emit(
        server_id,
        logging.WARNING,
        f"spawn/initialize failed for {workspace_root}: {type(exc).__name__}: {exc}",
    )


def reset_announce_caches() -> None:
    """Test-only: clear the dedup caches.  Production code never calls this."""
    with _announce_lock:
        _announced_active.clear()
        _announced_unavailable.clear()
        _announced_no_root.clear()
        _announced_no_server.clear()


__all__ = [
    "event_log",
    "log_clean",
    "log_disabled",
    "log_active",
    "log_diagnostics",
    "log_no_project_root",
    "log_server_unavailable",
    "log_no_server_configured",
    "log_timeout",
    "log_server_error",
    "log_spawn_failed",
    "reset_announce_caches",
]

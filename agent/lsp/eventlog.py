"""Structured logging with steady-state silence for the LSP layer.

LSP fires on every write_file/patch, so the level model keeps ``agent.log`` greppable
(``rg 'lsp\\['``) without noise: DEBUG for steady-state events with no novel signal (clean,
skipped, repeat "no project root" / "server unavailable"); INFO for once-per-session
transitions (first ``active for <root>``, first ``no project root`` per file) and every
diagnostic event; WARNING for action-required failures (first ``server unavailable`` per
(server_id, binary), every timeout / unexpected error).  Dedup uses module-level sets bounded
by the distinct pairs touched in one process — a bounded LRU would re-fire suppressed lines.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import List, Tuple

# Dedicated logger name so the documented grep recipe survives any
# ``logging.getLogger(__name__)`` rename of internal modules.
event_log = logging.getLogger("hermes.lint.lsp")

_announce_lock = threading.Lock()
_announced_active: set = set()        # keys: (server_id, workspace_root)
_announced_unavailable: set = set()   # keys: (server_id, binary_path_or_name)
_announced_no_root: set = set()       # keys: (server_id, file_path)
_ALL_BUCKETS = (_announced_active, _announced_unavailable, _announced_no_root)


def _short_path(file_path: str) -> str:
    """Render *file_path* relative to cwd when it's inside it, else absolute (no ``../..`` chains)."""
    if not file_path:
        return file_path
    try:
        rel = os.path.relpath(file_path)
    except ValueError:
        return file_path
    return file_path if rel.startswith(".." + os.sep) or rel == ".." else rel


def _emit(server_id: str, level: int, message: str) -> None:
    event_log.log(level, "lsp[%s] %s", server_id, message)


def _emit_once(bucket: set, key: Tuple, server_id: str, level: int, first: str, repeat: str) -> None:
    """Log *first* at *level* the first time *key* is seen, *repeat* at DEBUG thereafter."""
    with _announce_lock:
        is_first = key not in bucket
        bucket.add(key)
    _emit(server_id, level if is_first else logging.DEBUG, first if is_first else repeat)


# ---- Public event helpers — call these from the LSP layer ----


def log_clean(server_id: str, file_path: str) -> None:
    """No diagnostics emitted for *file_path*.  DEBUG."""
    _emit(server_id, logging.DEBUG, f"clean ({_short_path(file_path)})")


def log_disabled(server_id: str, file_path: str, reason: str) -> None:
    """LSP intentionally skipped for this file (feature off, ext unmapped, ...).  DEBUG."""
    _emit(server_id, logging.DEBUG, f"skipped: {reason} ({_short_path(file_path)})")


def log_active(server_id: str, workspace_root: str) -> None:
    """A client started for (server_id, workspace_root).  INFO once per pair, DEBUG thereafter."""
    _emit_once(_announced_active, (server_id, workspace_root), server_id, logging.INFO,
               f"active for {workspace_root}", f"reused client for {workspace_root}")


def log_diagnostics(server_id: str, file_path: str, count: int) -> None:
    """Diagnostics arrived for a file.  INFO every time — rare per edit and what users grep for."""
    _emit(server_id, logging.INFO, f"{count} diags ({_short_path(file_path)})")


def log_no_project_root(server_id: str, file_path: str) -> None:
    """File had no recognised project marker.  INFO once per file, DEBUG thereafter."""
    msg = f"no project root for {_short_path(file_path)}"
    _emit_once(_announced_no_root, (server_id, file_path), server_id, logging.INFO, msg, msg)


def log_server_unavailable(server_id: str, binary_or_pkg: str) -> None:
    """Server binary unresolved.  WARNING once per (server_id, binary), DEBUG thereafter."""
    _emit_once(
        _announced_unavailable, (server_id, binary_or_pkg), server_id, logging.WARNING,
        f"server unavailable: {binary_or_pkg} not found "
        "(install via `hermes lsp install <id>` or set lsp.servers.<id>.command)",
        f"server still unavailable: {binary_or_pkg}",
    )


def log_timeout(server_id: str, file_path: str, kind: str = "diagnostics") -> None:
    """A request to the server timed out.  WARNING every time."""
    _emit(server_id, logging.WARNING, f"{kind} timed out for {_short_path(file_path)}")


def log_server_error(server_id: str, file_path: str, exc: BaseException) -> None:
    """An unexpected exception bubbled out of the LSP layer.  WARNING."""
    _emit(server_id, logging.WARNING, f"unexpected error for {_short_path(file_path)}: {type(exc).__name__}: {exc}")


def log_spawn_failed(server_id: str, workspace_root: str, exc: BaseException) -> None:
    """The LSP server failed to spawn or initialize.  WARNING."""
    _emit(server_id, logging.WARNING, f"spawn/initialize failed for {workspace_root}: {type(exc).__name__}: {exc}")


def log_reaped(keys: List[Tuple[str, str]], idle_timeout: float) -> None:
    """Idle clients were reaped.  INFO, one line per sweep.

    Also forgets the ``log_active`` announcement for those keys so a respawn
    re-announces at INFO instead of a misleading DEBUG "reused client".
    """
    with _announce_lock:
        _announced_active.difference_update(keys)
    summary = ", ".join(f"{sid} ({root})" for sid, root in keys)
    _emit("reaper", logging.INFO, f"reaped {len(keys)} idle client(s) after {idle_timeout:.0f}s: {summary}")


def reset_announce_caches() -> None:
    """Test-only: clear the dedup caches.  Production code never calls this."""
    with _announce_lock:
        for bucket in _ALL_BUCKETS:
            bucket.clear()


__all__ = [
    "event_log", "log_clean", "log_disabled", "log_active", "log_diagnostics", "log_no_project_root",
    "log_server_unavailable", "log_timeout", "log_server_error", "log_spawn_failed", "log_reaped",
    "reset_announce_caches",
]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

_announced_no_server: set = set()     # keys: (server_id,)

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

def log_no_server_configured(server_id: str) -> None:
    """No spawn recipe for this language.  WARNING once."""
    if _announce_once(_announced_no_server, (server_id,)):
        _emit(server_id, logging.WARNING, "no server configured")
# ---- END PLUGIN-COMPAT ----

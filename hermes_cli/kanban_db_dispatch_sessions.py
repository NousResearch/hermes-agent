"""Dispatcher worker-session maintenance."""

from __future__ import annotations

_retagged_workspace_roots: set[str] = set()


def retag_legacy_worker_sessions(workspaces_root_path: str) -> None:
    if workspaces_root_path in _retagged_workspace_roots:
        return
    try:
        from hermes_state_registry import acquire, release_or_close

        db = acquire()
        try:
            db.retag_kanban_worker_sessions(workspaces_root_path)
        finally:
            release_or_close(db)
        _retagged_workspace_roots.add(workspaces_root_path)
    # health: allow BLE001 -- Best-effort maintenance must not prevent a worker spawn.
    except Exception as exc:
        from hermes_cli import kanban_db as kb
        kb._log.exception("kanban worker: legacy session retag skipped (%s)", exc)

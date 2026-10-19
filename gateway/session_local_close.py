"""The picker's view of a canonical local session its last viewer closed, as main listed it.

Main's sidecar stamped ``tui_close`` (``/new``, a resume switch) and ``tui_shutdown`` (exit) on the
session row it owned; pickers then listed it as resumable. The stamp itself is the owner's
``session.detach {end_reason}`` (``session_acp_lifecycle.end_idle_viewer_session``); this module only
decides which live rows read as closed.
"""
from hermes_state_local import local_receipt
from hermes_state_runtime import RuntimeStoreError


def _local_target(authority, session_id):
    try:
        return local_receipt(authority.db, session_id)['entry']['session_id']
    except (RuntimeStoreError, KeyError, TypeError):
        return None  # not a local session (API binding, malformed receipt): nothing to read as closed


def closed_and_unviewed(authority, session_id, handle):
    """A live-feed row main would list as resumable: closed, idle and with no viewer left.

    Reads the local target's own columns rather than hydrating the row: the listing asks this per
    session on the gateway's loop, where a ``get_session`` (prompt/tool JOIN + token flush) per
    candidate is the cost the sweep must not pay."""
    live = authority.sessions.get(session_id)
    if live is None or live.subscribers or handle.execution_state != 'idle':
        return False
    if live.superseded:
        return True  # an old messaging conversation opened read-only: history once no one views it
    target = _local_target(authority, session_id)
    row = authority.db.get_session_list_row(target) if target else None
    return bool(row and row.get('ended_at') is not None)

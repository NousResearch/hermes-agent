"""A ``/btw --open`` side chat: a local child session seeded with the side thread's transcript.

The branch commit (``hermes_state_mutation_branch.branch_in_transaction``) copies the parent's
stored rows up to a boundary, so it refuses a parent with a running turn: the rows it would copy
are still moving. A side chat copies none of them. Its rows are the seed the gateway built from
the side thread's own snapshot (``agent.side_thread.side_chat_seed``), so it never reads or writes
the parent's transcript and may be created while the parent works — the point of a side chat.

Everything else is the branch child's: the parent's frozen policy (``_child_policy``), its owning
profile and principal, a ``_branched_from`` parent link, a routing entry and a creation receipt.
The child id derives from the request id, so opening the same side thread twice returns the same
chat instead of a duplicate.
"""

from __future__ import annotations

from typing import Any

from hermes_state_runtime import RuntimeStoreError


def create_side_chat(db, session_id: str, *, request_id: str, title: str, messages: list[dict[str, Any]]) -> str:
    """Create (or find) the side chat for ``request_id`` under local session ``session_id``;
    its session id."""
    if not isinstance(request_id, str) or not request_id or not isinstance(title, str) or not messages:
        raise RuntimeStoreError('invalid_params')
    rows = [{'role': m['role'], 'content': m['content']} for m in messages
            if isinstance(m, dict) and m.get('role') in {'user', 'assistant'} and isinstance(m.get('content'), str)]
    if len(rows) != len(messages):
        raise RuntimeStoreError('invalid_params')

    def write(conn) -> str:
        from hermes_state_keys import local_identity
        from hermes_state_mutation_branch import (
            _child_policy, parent_receipt, publish_child_receipt, publish_child_route,
        )
        saved, target = parent_receipt(conn, session_id)
        child = local_identity(saved['profile_id'], saved['principal_id'], request_id)
        if conn.execute('SELECT 1 FROM sessions WHERE id=?', (child,)).fetchone() is not None:
            return child
        policy = _child_policy(saved['policy'], child, None)
        route, entry, _parent = publish_child_route(db, conn, saved, target, child, policy)
        db._append_messages_in_transaction(conn, child, rows)
        db._set_session_title_in_transaction(conn, child, title, source=db.TITLE_SOURCE_USER)
        publish_child_receipt(conn, saved, session_id, child, route, entry, policy, request_id)
        return child

    return db._execute_write(write)

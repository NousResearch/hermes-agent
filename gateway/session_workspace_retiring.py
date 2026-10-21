"""``session.workspace.retire``: a workspace that is going away takes no new work (#134889 TUI-27).

A launcher that removes a session's worktree (the TUI's quit, the classic CLI's exit) tells the
owner first, so the tree is marked retiring before it disappears. Without the mark a submit that
lands in that window is admitted and only fails later, when the turn tries to run in a directory
that is gone (``workspace_unavailable``): the prompt is taken and refused after the fact.

The mark is the owner's own state, it names one session (the one whose workspace the launcher is
removing), and it expires: a launcher that dies between the mark and the removal never wedges the
session. The refusal is named, so a viewer can retry once the tree is back.
"""
from __future__ import annotations

import os
import time
from functools import partial

from hermes_state_runtime import RuntimeStoreError

# The mark outlives the removal window and a little more: a launcher that never clears it (it died
# between the mark and the removal) must not refuse work forever.
RETIRING_TTL_SECONDS = 120.0


def same_path(left: str, right: str) -> bool:
    """Whether two paths name the same directory (case-insensitive on Windows)."""
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def mark(authority, session_id: str, *, retiring: bool) -> None:
    """Mark (or clear) one session's workspace as retiring."""
    if retiring:
        authority.retiring_workspaces[session_id] = time.monotonic() + RETIRING_TTL_SECONDS
    else:
        authority.retiring_workspaces.pop(session_id, None)


def retiring(authority, session_id: str) -> bool:
    """Whether this session's workspace is retiring; an expired mark is dropped."""
    expiry = authority.retiring_workspaces.get(session_id)
    if expiry is None:
        return False
    if expiry <= time.monotonic():
        authority.retiring_workspaces.pop(session_id, None)
        return False
    return True


def workspace_of(authority, session_id: str) -> str | None:
    """The session's frozen workspace (its policy's cwd), or None when it has none.

    The owner resolves it, never the caller: a client names the tree it is about to remove, and
    the owner answers only for the session that actually works in it.
    """
    live = authority.sessions.get(session_id)
    if live is None:
        return None
    from gateway.session_policy import policy_for_source
    try:
        policy = policy_for_source(authority.runner, live.source)
    except RuntimeStoreError:
        return None  # a LOCAL route whose policy is not restored yet has no workspace to protect
    return getattr(policy, 'cwd', None) if policy is not None else None


def handlers(connection):
    return {'session.workspace.retire': partial(retire_workspace, connection)}


async def retire_workspace(connection, ref, params):
    """``session.workspace.retire``: mark (or clear) this session's workspace as retiring."""
    authority, actor = connection.authority, connection.actor
    path, retiring_now = params.get('path'), params.get('retiring')
    if not isinstance(path, str) or not path or type(retiring_now) is not bool:
        raise RuntimeStoreError('invalid_params')
    return await authority.retire_workspace(actor, ref, path, retiring_now)

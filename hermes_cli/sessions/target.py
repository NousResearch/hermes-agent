"""The ``<session>`` argument of the ``hermes sessions`` gateway verbs, and how those verbs run.

One resolver serves ``send``, ``wait``, ``watch``, ``log`` and ``interrupt``: a full session id, a
unique id prefix, a title (the latest ``title #N`` continuation), or ``-`` for the session
``hermes --resume latest`` would pick in this directory (this workspace's most recent CLI session,
else the most recent one). A compression root resolves forward to its live tip.
"""
from __future__ import annotations

import asyncio
import sys

from hermes_cli.gateway_client import GatewayClientError

LAST_SESSION = "-"
# Same exit as coreutils ``timeout`` and the one-shot ``--timeout``: the wait ran out, nothing was stopped.
TIMEOUT_EXIT = 124
INTERRUPTED_EXIT = 130
# Pending admission states that mean the session still has work to run.
_BUSY_STATES = frozenset({"queued", "started"})


def session_argument(parser, help="Session ID, unique ID prefix, title, or '-' for the last session in this directory"):
    parser.add_argument("session", help=help)


def resolve_local(name, db=None):
    """*name* as a stored session id from this profile's ``state.db`` (or the open *db*), or None
    when the store has no match or cannot be read (a remote ``HERMES_TUI_GATEWAY_URL`` owner keeps
    its own store, and the owner then resolves the name itself)."""
    from hermes_cli.main import _resolve_last_session, _session_db
    if name == LAST_SESSION:
        return _resolve_last_session("cli")
    if db is not None:
        return _lookup(db, name)
    with _session_db() as reader:
        return _lookup(reader, name) if reader is not None else None
    return None


def _lookup(db, name):
    resolved = db.resolve_session_id(name) or db.resolve_session_by_title(name)
    return (db.get_compression_tip(resolved) or resolved) if resolved else None


async def attach(client, name, **params):
    """``session.resume`` for *name*: the id this profile's store resolves it to, then the owner's own
    lookup (``-`` as ``--resume latest``, otherwise exact id then title), as ``hermes chat --resume``
    resolves it. A local match the owner does not know (a remote owner keeps its own store) falls
    through to the owner's lookup instead of failing."""
    resolved = await asyncio.to_thread(resolve_local, name)
    if name == LAST_SESSION:
        from hermes_cli.main import _resolve_workspace_key
        workspace = await asyncio.to_thread(_resolve_workspace_key)
        owner = [{"latest": "cli", **({"workspace": workspace} if workspace else {})}]
    else:
        owner = [{"session_id": name}, {"title": name}]
    for lookup in ([{"session_id": resolved}] if resolved else []) + owner:
        try:
            return await client.rpc("session.resume", **lookup, **params)
        except GatewayClientError as exc:
            if str(exc) != "not_found":
                raise
    raise GatewayClientError(f"No session found matching '{name}'. Use 'hermes sessions list' to see available sessions.")


def stored_id(snapshot):
    return snapshot.get("stored_session_id") or snapshot["session_id"]


def busy(pending):
    """The session still has a running or queued turn (the owner's ``pending`` rows)."""
    return any(row.get("status") in _BUSY_STATES for row in pending or ())


def paused(pending):
    """A turn whose ending the owner could not confirm blocks the session until it is discarded."""
    return any(row.get("status") == "unknown" for row in pending or ())


def run_gateway_verb(verb, args) -> int:
    """Run ``verb(client, args)`` on one gateway connection; transport failures are ``Error:`` + exit 1."""
    from websockets.exceptions import WebSocketException

    from hermes_cli.gateway_client import connect_gateway

    async def run():
        async with connect_gateway() as client:
            return await verb(client, args)
    try:
        return asyncio.run(run())
    except KeyboardInterrupt:
        return INTERRUPTED_EXIT
    except (GatewayClientError, OSError, TimeoutError, WebSocketException) as exc:
        message = str(exc) if isinstance(exc, GatewayClientError) else "Gateway connection/read failed"
        print("Error: " + message, file=sys.stderr)
        return 1

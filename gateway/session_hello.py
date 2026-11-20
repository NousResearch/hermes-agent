"""``client.hello``: the post-upgrade handshake folded into ONE round trip.

One request carries what a reconnecting client holds — its capabilities, its per-session replay
watermarks and epochs. One response carries what the owner has for it: the capabilities answer, the
etag of the skin the connection's ready frame carried, every session's replay window and the open
server→client requests.

Every window is produced by ``AuthorityConnection.events_since`` — the handler today's per-session
``session.events.since`` call goes through — so ``truncated`` / ``snapshot_required`` / epoch rules
are that method's, and so is its reconnect bookkeeping (``session.reconnects`` / ``replay_cursors``,
which the next ``session.resume`` reads). This module batches the round trip; it does not re-derive
the replay contract.

A session that cannot be answered (gone, another profile, a bad cursor) is reported as that entry's
``error``, never as a whole-call failure: one gone session must not cost every other session its
replay, and the client drops exactly the cursor the owner refused.
"""

from __future__ import annotations

from functools import partial

from hermes_state_runtime import RuntimeStoreError

#: The empty replay window a refused session answers with, so every entry keeps the contract's
#: shape and a client needs no special case to skip it.
_EMPTY_WINDOW = {"events": [], "latest_seq": 0, "last_sequence": 0, "epoch": "", "replay_epoch": "",
                 "truncated": False, "snapshot_required": False, "count": 0, "open_requests": None}


def handlers(connection):
    return {'client.hello': partial(hello, connection)}


async def hello(connection, ref, params):
    """Answer the folded handshake: capabilities, skin etag, every watermark's replay, open requests."""
    from tui_gateway.connection_capabilities import apply_client_capabilities

    capabilities = apply_client_capabilities(connection.transport, params.get('capabilities') or {})
    epochs = params.get('session_epochs') or {}
    replay, open_requests, seen = [], [], set()
    # Insertion order of ``watermarks`` is the client's own cursor order; nothing here re-sorts it.
    for session_id, last_seen in (params.get('watermarks') or {}).items():
        entry, waiting = await _session_replay(connection, session_id, last_seen, epochs.get(session_id))
        replay.append(entry)
        for request in waiting:
            key = str(request.get('id'))
            if key in seen:
                continue
            seen.add(key)
            open_requests.append(request)
    return {'capabilities_result': capabilities, 'skin_etag': _skin_etag(connection),
            'per_session_replay': replay, 'open_requests': open_requests}


async def _session_replay(connection, session_id, last_seen, epoch):
    """One session's window through the owner's own ``session.events.since`` handler.

    Returns ``(entry, open_requests)``: the entry is the contract's per-session snapshot (its own
    ``open_requests`` cleared — the answer carries the deduped union), and the refused case is the
    same shape plus the reason.
    """
    session_id = str(session_id)
    if type(last_seen) is not int or last_seen < 0:
        return {**_EMPTY_WINDOW, 'session_id': session_id, 'error': 'invalid_params'}, []
    from gateway.session_contract import SessionRef
    call: dict[str, object] = {'last_seen': last_seen}
    if isinstance(epoch, str) and epoch:
        call['replay_epoch'] = epoch
    try:
        result = await connection.events_since(SessionRef(connection.actor.profile_id, session_id), call)
    except RuntimeStoreError as exc:
        return {**_EMPTY_WINDOW, 'session_id': session_id, 'error': exc.reason}, []
    waiting = result.get('open_requests') or []
    window = {key: value for key, value in result.items() if key != 'open_requests'}
    return {'session_id': session_id, **window, 'open_requests': None, 'error': None}, waiting


def _skin_etag(connection) -> str:
    """The etag of the skin this connection's ``gateway.ready`` frame carried (ws.py stamps it).

    Empty for a transport that never sent one: the client then treats the skin as unknown and keeps
    whatever it has, which is the pre-existing behaviour.
    """
    return str(getattr(connection.transport, 'skin_etag', '') or '')

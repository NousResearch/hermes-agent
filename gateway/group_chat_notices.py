"""Tell a group's owner, in their private chat, when the group moved or paused by itself.

Automatic takeover is invisible when it works. Two things still reach the owner unasked,
once per incident:

* "“Research” moved to Home VPS because Mac mini went offline. It’s running." ("was
  shutting down" for a planned handover), from the computer the group moved to;
* "“Research” is paused to stay safe: Mac mini can’t reach Home VPS.", from the host that
  paused it.

A supervised gateway watcher reads both the way every client does, through the canonical
``groups.*`` methods as each private chat's owner, on gateways that list
``groups.succession.status``. It watches new ``authority.transition`` events in the room
log and re-reads the status of the rooms this computer hosts, because a paused host can't
write to its log. History from before a chat first looked at a room never notifies.
Shared chats are never told, and each chat is told about each incident at most once: what
it has seen is saved before anything is sent.
"""
from __future__ import annotations

import asyncio
import json
import logging

from gateway.group_chat_access import NOTICE_PREFIX
from gateway.group_chat_hosts import STATUS, paused_notice
from gateway.group_chat_slash import safe
from hermes_state_runtime import RuntimeStoreError, _epoch

logger = logging.getLogger(__name__)
WATCH_SECONDS = 15.0
MAX_PAGES = 5  # log pages per room and pass; a longer backlog continues on the next pass
_REASONS = {'automatic': 'went offline', 'handover': 'was shutting down'}


def moved_notice(event, group: str, here: str | None) -> str | None:
    """The notice for an automatic move or a handover to this computer; None for anything else."""
    payload = event.get('payload') if isinstance(event, dict) else None
    if (not isinstance(payload, dict) or event.get('kind') != 'authority.transition' or here is None
            or payload.get('reason') not in _REASONS or payload.get('successor_gateway_id') != here):
        return None
    # The group moved here: unnamed, the destination is this computer; the origin is another one.
    to = safe(payload.get('to_name'), 48) or 'this computer'
    origin = safe(payload.get('from_name'), 48) or 'another computer'
    return f'{group} moved to {to} because {origin} {_REASONS[payload["reason"]]}. It’s running.'


def _seen(authority, grant) -> dict:
    """What this chat has seen, per room: the last log ``seq`` read and whether it was told of a pause."""
    with authority.db._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (NOTICE_PREFIX + grant['grant_id'],)).fetchone()
    try:
        saved = json.loads(row[0]) if row is not None else {}
    except (TypeError, ValueError):
        saved = {}
    return {room: entry for room, entry in (saved.items() if isinstance(saved, dict) else ())
            if isinstance(entry, dict) and type(entry.get('seq')) is int and type(entry.get('paused')) is bool}


def _remember(authority, grant, seen: dict) -> bool:
    """Save what the chat has seen; False when the chat lost its grant meanwhile (tell it nothing)."""
    from gateway.group_chat_access import _load

    def write(conn):
        _epoch(conn, authority.epoch)
        current = _load(conn, grant['grant_id'])
        if current is None or current['owner'] != grant['owner']:
            return False
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) '
                     'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                     (NOTICE_PREFIX + grant['grant_id'], json.dumps(seen, sort_keys=True)))
        return True
    return authority.db._execute_write(write)


async def _rooms(call) -> list[dict]:
    rooms, offset = [], 0
    for _ in range(64):
        page = await call('groups.list', {'limit': 500, 'offset': offset})
        rooms.extend(r for r in page['rooms'] if isinstance(r, dict) and isinstance(r.get('room_id'), str))
        if page['next_offset'] is None:
            break
        offset = page['next_offset']
    return rooms


async def _events(call, room_id, cursor):
    """The room's events after ``cursor``, and the new cursor. The first look only takes the cursor."""
    if cursor is not None:
        events = []
        try:
            for _ in range(MAX_PAGES):
                page = await call('groups.log', {'room_id': room_id, 'since_seq': cursor, 'limit': 100})
                events.extend(page['events'])
                cursor = page['cursor']
                if not page['has_more']:
                    break
            return cursor, events
        except RuntimeStoreError as exc:
            if events:
                return cursor, events
            if exc.reason != 'invalid_params':
                raise
            # A log behind the cursor (the copy was replaced): read on from its end.
    probe = await call('groups.log', {'room_id': room_id, 'since_seq': 0, 'limit': 1})
    return probe['latest_seq'], []


async def notify_chat(runner, authority, grant) -> int:
    """One pass for one private chat; returns how many notices it was sent."""
    from functools import partial
    from gateway.group_chat_access import chat_target
    from gateway.group_chat_slash import connection_for
    from gateway.hosted_rooms import local_authority_gateway_id
    from gateway.session_group_controls import dispatch_group_control
    call = partial(dispatch_group_control, connection_for(authority, grant))
    listed = (await call('groups.capabilities', {})).get('methods')
    target = chat_target(runner, grant)
    if not isinstance(listed, list) or STATUS not in listed or target is None:
        return 0
    try:
        here = local_authority_gateway_id()
    except Exception:
        here = None  # no install identity: nothing can have moved here
    before = await asyncio.to_thread(_seen, authority, grant)
    seen, notices = {}, []
    for room in await _rooms(call):
        room_id, group = room['room_id'], f'“{safe(room.get("name"), 72)}”'
        known = before.get(room_id)
        try:
            cursor, events = await _events(call, room_id, known['seq'] if known else None)
        except Exception:
            logger.debug('Group Chat notices skipped a room this time', exc_info=True)
            if known is not None:
                seen[room_id] = known
            continue
        told = known is not None and known['paused']
        if known is not None:
            notices.extend(n for n in (moved_notice(e, group, here) for e in events) if n)
        if room.get('copy') is not True:  # hosted here: a pause can't reach the log, so ask
            try:
                current = await call(STATUS, {'room_id': room_id})
            except Exception:
                current = None  # can't tell this time: keep what the chat was told
            if isinstance(current, dict):
                pausing = current.get('state') == 'paused'
                if pausing and not told:
                    notices.append(paused_notice(current, group))
                told = pausing
        seen[room_id] = {'seq': cursor, 'paused': told}
    if seen != before and not await asyncio.to_thread(_remember, authority, grant, seen):
        return 0
    adapter, metadata = target
    for text in notices:
        try:
            await adapter.send(grant['chat_id'], text, metadata=metadata)
        except Exception:
            logger.warning('A Group Chat notice could not be delivered to a private chat', exc_info=True)
    return len(notices)


async def notify_all(runner) -> int:
    """One pass over every private chat of every profile this gateway serves."""
    from gateway.group_chat_access import grants
    from gateway.session_authorities import all_authorities
    sent = 0
    for authority in all_authorities(runner):
        if getattr(authority, 'hosted_room_service', None) is None:
            continue
        with authority.db._read_ctx() as conn:
            private = [grant for grant in grants(conn) if grant['kind'] == 'private']
        for grant in private:
            try:
                sent += await notify_chat(runner, authority, grant)
            except Exception:  # every pass retries; a lasting failure must not flood the log
                logger.debug('Group Chat notices skipped a chat this time', exc_info=True)
    return sent


async def watch(runner, interval: float = WATCH_SECONDS) -> None:
    while getattr(runner, '_running', False):
        try:
            await notify_all(runner)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning('Group Chat notices failed a pass', exc_info=True)
        slept = 0.0
        while slept < interval and getattr(runner, '_running', False):
            await asyncio.sleep(min(1.0, interval - slept))
            slept += 1.0

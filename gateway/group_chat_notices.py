"""Tell a group's owner, unasked, when the group moved or paused by itself, or ran on two computers.

Automatic takeover is invisible when it works. These still reach the owner, once per incident:

* "“Research” moved to Home VPS because Mac mini went offline. It’s running." ("was shutting
  down" for a handover; Bots that stay on the old host are counted), from the computer the
  group moved to;
* after a careful move, where the old host's silence was the only proof, a warning instead,
  offering Keep going on Home VPS, Go back to Mac mini (confirmed first) and Ask me first
  next time;
* "“Research” is paused to stay safe: Mac mini can’t reach Home VPS.", from the paused host;
* "“Research” ran on both Mac mini and Home VPS …", offering Keep going on the computer still
  running it and Keep on the other one, from the computer the careful move went to.

A supervised gateway watcher finds them the way every client does, through the canonical
``groups.*`` methods as each room's recorded owner, on gateways that list
``groups.succession.status``: new ``authority.transition`` events in the room log, and the
status of the rooms this computer hosts, because a paused host can't write to its log. On its
first look at a room it still tells a move to this computer from the last ten minutes; older
history never notifies. What the owner was told is saved before anything is sent, so nothing
is told twice.

Each notice goes to the owner's main channel: their home channel (``/sethome``) when it is one
of their private group-control chats, otherwise each of those private chats, and only when
they have none, a one-to-one home channel, without controls. Shared chats never get them.
The offered actions are buttons where the chat's adapter has a picker and typed commands
otherwise; both act only through the owner's private grant, rechecked when used.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import logging
import time
from types import SimpleNamespace

from gateway.group_chat_hosts import (
    STATUS, _obj, _this, ask_first, computer, go_back_prompt, go_back_target, keep_on, paused_notice, ran_on_two)
from gateway.group_chat_slash import Refused, safe
from hermes_state_runtime import RuntimeStoreError, _epoch

logger = logging.getLogger(__name__)
NOTICE_PREFIX = 'gateway.messaging.notices.v1:'  # per owner: what each room's notices have covered
WATCH_SECONDS = 15.0
MAX_PAGES = 5  # log pages per room and pass; a longer backlog continues on the next pass
FIRST_LOOK_EVENTS = 200  # what the first look at a room reads: its latest events
FIRST_LOOK_SECONDS = 600  # a move to this computer that recent is told even on the first look
DESKTOP = 'Choose in Hermes Desktop.'
_REASONS = {'automatic': 'went offline', 'handover': 'was shutting down'}


@dataclass(frozen=True)
class Notice:
    """One incident for the owner: its text, and the actions it offers as (label, value, typed words)."""
    room_id: str
    group: str
    text: str
    actions: tuple = ()
    here: str = 'this computer'  # where the group now runs, for "Keep going on …"


# ---- what happened -------------------------------------------------------------------------

def moved_notice(event, room_id: str, group: str, here: str | None, *, unavailable: int = 0) -> Notice | None:
    """A move or handover to this computer: informational, or the warning after a careful move.
    ``unavailable`` counts the Bots that stay behind on the old host."""
    payload = event.get('payload') if isinstance(event, dict) else None
    if (not isinstance(payload, dict) or event.get('kind') != 'authority.transition' or here is None
            or payload.get('reason') not in _REASONS or payload.get('successor_gateway_id') != here):
        return None
    # The group moved here: unnamed, the destination is this computer; the origin is another one.
    to = safe(payload.get('to_name'), 48) or 'this computer'
    origin = safe(payload.get('from_name'), 48) or 'another computer'
    if payload.get('proof_kind') != 'evidence':
        bots = (f' {unavailable} Bot{" is" if unavailable == 1 else "s are"} unavailable until the group moves '
                f'back to {origin}.' if unavailable else '')
        return Notice(room_id, group, f'{group} moved to {to} because {origin} {_REASONS[payload["reason"]]}. '
                                      f'It’s running.{bots}')
    return Notice(room_id, group, '\n'.join([
        f'{group} moved to {to}',
        f'{origin} went silent for 3 minutes, so {to} took over. If {origin} is actually still running, the '
        'group may now be running in both places.']), (
        (f'Keep going on {to}', 'ack', None), (f'Go back to {origin}', 'back', f'keep {origin}'),
        ('Ask me first next time', 'ask', 'ask first')), to)


def _conflict_notice(status, room_id: str, group: str) -> Notice | None:
    """The computer the careful move went to (the later host) asks which one keeps the group."""
    hosts = [h for h in _obj(status.get('conflict')).get('hosts') or ()
             if isinstance(h, dict) and isinstance(h.get('install_id'), str) and h['install_id']]
    since = [h.get('since') if type(h.get('since')) in (int, float) else 0 for h in hosts]
    if not hosts or _this(status) != hosts[since.index(max(since))]['install_id']:
        return None
    # The computer still running the group first: keeping it is keep going.
    running = _obj(_obj(status.get('conflict')).get('running_on')).get('install_id')
    hosts.sort(key=lambda h: h['install_id'] != running)
    return Notice(room_id, group, ran_on_two(status, group), tuple(
        (f'{"Keep going on" if h["install_id"] == running else "Keep"} {computer(h, status)}',
         'keep:' + h['install_id'], f'keep {computer(h, status)}') for h in hosts))


# ---- what the owner was told ---------------------------------------------------------------

def _key(owner: str) -> str:
    return NOTICE_PREFIX + hashlib.sha256(owner.encode()).hexdigest()[:32]


def _told(authority, owner: str) -> dict:
    """Per room: the last log ``seq`` read, and whether a pause and a conflict were told."""
    with authority.db._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (_key(owner),)).fetchone()
    try:
        saved = json.loads(row[0]) if row is not None else {}
    except (TypeError, ValueError):
        saved = {}
    return {room: entry for room, entry in (saved.items() if isinstance(saved, dict) else ())
            if isinstance(entry, dict) and type(entry.get('seq')) is int
            and type(entry.get('paused')) is bool and type(entry.get('conflict')) is bool}


def _remember(authority, owner: str, told: dict) -> None:
    def write(conn):
        _epoch(conn, authority.epoch)
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) '
                     'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                     (_key(owner), json.dumps(told, sort_keys=True)))
    authority.db._execute_write(write)


# ---- finding incidents ---------------------------------------------------------------------

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
    """The room's events after ``cursor``, and the new cursor. The first look (no cursor) reads the
    room's latest events, so a move here that has only just happened is still told."""
    if cursor is None:
        probe = await call('groups.log', {'room_id': room_id, 'since_seq': 0, 'limit': 1})
        cursor = max(0, probe['latest_seq'] - FIRST_LOOK_EVENTS)
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


async def _incidents(authority, owner: str) -> list[Notice]:
    """New incidents in the owner's rooms here, saved as told before they are returned."""
    from functools import partial
    from gateway.hosted_rooms import local_authority_gateway_id
    from gateway.session_contract import Principal
    from gateway.session_group_controls import dispatch_group_control
    reader = SimpleNamespace(authority=authority, actor=Principal(
        owner, authority.profile_id, frozenset({'session:read'}), 'messaging:notices'))
    call = partial(dispatch_group_control, reader)
    try:
        here = local_authority_gateway_id()
    except Exception:
        here = None  # no install identity: nothing can have moved here
    before, told, notices = await asyncio.to_thread(_told, authority, owner), {}, []
    for room in await _rooms(call):
        room_id, group = room['room_id'], f'“{safe(room.get("name"), 72)}”'
        known = before.get(room_id)
        try:
            cursor, events = await _events(call, room_id, known['seq'] if known else None)
        except Exception:
            logger.debug('Group Chat notices skipped a room this time', exc_info=True)
            if known is not None:
                told[room_id] = known
            continue
        current = None
        if room.get('copy') is not True:  # hosted here: a pause can't reach the log, so ask
            try:
                current = await call(STATUS, {'room_id': room_id})
            except Exception:
                current = None  # can't tell this time: keep what the owner was told
        # The first look at a room still tells a move here that has only just happened.
        recent = time.time() - FIRST_LOOK_SECONDS
        fresh = [e for e in events if isinstance(e, dict) and (known is not None or (
            type(e.get('created_at')) in (int, float) and e['created_at'] >= recent))]
        unavailable = sum(isinstance(bot, dict) for bot in _obj(current).get('unavailable_bots') or ())
        notices.extend(n for n in (moved_notice(e, room_id, group, here, unavailable=unavailable) for e in fresh) if n)
        paused, conflict = (known['paused'], known['conflict']) if known else (False, False)
        if isinstance(current, dict):
            state = current.get('state')
            if state == 'paused' and not paused:
                notices.append(Notice(room_id, group, paused_notice(current, group)))
            if state == 'continued_on_two' and not conflict:
                notices.extend(n for n in [_conflict_notice(current, room_id, group)] if n)
            paused, conflict = state == 'paused', state == 'continued_on_two'
        told[room_id] = {'seq': cursor, 'paused': paused, 'conflict': conflict}
    if told != before:
        await asyncio.to_thread(_remember, authority, owner, told)
    return notices


# ---- telling the owner ---------------------------------------------------------------------

def _home_belongs(home, grant, primary: str) -> bool:
    profile, platform, _, channel, _ = home
    return (grant['bot'] == (profile or primary) and grant['platform'] == platform.value
            and grant['chat_id'] == str(channel.chat_id)
            and (channel.thread_id is None or grant['thread_id'] == str(channel.thread_id)))


def homes_for(runner, authority) -> list:
    """The home channels (``/sethome``) of the profile an authority serves, with a live transport."""
    from pathlib import Path
    from gateway.session_authorities import served_profile_name
    primary = getattr(runner, '_primary_profile_name', None) or 'default'
    served = getattr(runner, '_served_home_channel_transports', None)
    profile = served_profile_name(Path(authority.profile_id))
    return [h for h in (served() if served else ()) if (h[0] or primary) == profile]


def main_chats(runner, grants: list, homes: list) -> list:
    """The owner's main channel among their private grants: the home channel when it is one of them,
    otherwise all of them."""
    primary = getattr(runner, '_primary_profile_name', None) or 'default'
    return [grant for grant in grants if any(_home_belongs(h, grant, primary) for h in homes)] or grants


def _one_to_one(home) -> bool:
    """A home channel that is provably the operator and the Bot alone: their own chat id, or a Slack IM."""
    from gateway.group_chat_identity import ONE_TO_ONE_DM_PLATFORMS
    _, platform, _, channel, transport = home
    chat, user = str(channel.chat_id or ''), str(channel.user_id or '')
    if getattr(transport, 'is_relay', False) or channel.thread_id:
        return False
    if platform.value == 'slack':
        return chat.startswith('D')
    return bool(user) and chat == user and platform.value in ONE_TO_ONE_DM_PLATFORMS


async def _offer(runner, authority, grant, notice: Notice) -> None:
    """One notice in one private chat: buttons where its adapter has a picker, typed commands always."""
    from functools import partial
    from gateway.group_chat_access import chat_target, ensure_ref
    target = chat_target(runner, grant)
    if target is None:
        return
    adapter, metadata = target
    if not notice.actions:
        await adapter.send(grant['chat_id'], notice.text, metadata=metadata)
        return
    n = await asyncio.to_thread(ensure_ref, authority, grant, notice.room_id)
    g = f'{getattr(adapter, "typed_command_prefix", None) or "/"}group {n}'
    text = '\n'.join([notice.text, '', *(f'{label}: {g} {words}' if words else f'{label}: no reply needed.'
                                          for label, _, words in notice.actions)])
    if getattr(type(adapter), 'send_choice_picker', None) is not None:
        try:
            sent = await adapter.send_choice_picker(
                chat_id=grant['chat_id'], title=text, session_key='group-chat:' + grant['grant_id'][:32],
                choices=[{'value': value, 'label': label, 'is_current': False}
                         for label, value, _ in notice.actions],
                on_choice_selected=partial(_chosen, runner, authority, grant, notice, g), metadata=metadata)
            if getattr(sent, 'success', False):
                return
        except Exception:
            logger.debug('Group Chat notice buttons unavailable; the typed commands stay', exc_info=True)
    await adapter.send(grant['chat_id'], text, metadata=metadata)


async def _chosen(runner, authority, grant, notice: Notice, g: str, _chat_id, value) -> str:
    """A button under a notice: only the owner's private grant, still held, acts."""
    from gateway.group_chat_access import private_grant_holds
    from gateway.group_chat_slash import connection_for
    from gateway.session_group_controls import dispatch_group_control
    if not await asyncio.to_thread(private_grant_holds, runner, authority, grant):
        return 'This chat’s access to Group Chats changed. Nothing was done.'
    connection = connection_for(authority, grant)
    try:
        if value == 'ack':
            return f'OK. {notice.group} keeps going on {notice.here}.'
        if value == 'ask':
            return await ask_first(connection, notice.room_id, group=notice.group, g=g)
        current = await dispatch_group_control(connection, STATUS, {'room_id': notice.room_id})
        if value == 'back':
            # Going back is confirmed first: the typed confirmation names exactly what happens.
            return go_back_prompt(current, notice.group, g) if go_back_target(_obj(current)) else (
                f'{notice.group} can’t go back now. Send {g} to see where things stand.')
        install_id = str(value).removeprefix('keep:')
        hosts = {h.get('install_id'): h for h in _obj(_obj(current).get('conflict')).get('hosts') or ()
                 if isinstance(h, dict)}
        if _obj(current).get('state') != 'continued_on_two' or install_id not in hosts:
            return f'{notice.group} isn’t waiting for that choice any more. Send {g} to see where things stand.'
        name = computer(hosts[install_id], current)
        await keep_on(connection, notice.room_id, install_id, g=g, failed=f'Couldn’t keep {name}.')
        others = [computer(h, current) for key, h in hosts.items() if key != install_id]
        kept = f' Messages from {", ".join(others)} are kept and shown separately.' if others else ''
        going = install_id == _obj(_obj(_obj(current).get('conflict')).get('running_on')).get('install_id')
        return f'Done. {notice.group} {"keeps going" if going else "now continues"} on {name}.{kept}'
    except Refused as exc:
        return str(exc)
    except RuntimeStoreError:
        return f'{notice.group} isn’t available right now. Send {g} to check.'


async def _tell(runner, authority, notice: Notice, chats: list, homes: list) -> int:
    """To the owner's main channel; with no private chat, only a one-to-one home channel, unadorned.
    ``homes`` is empty for an owner who isn't this computer's own account: a home channel is the
    operator's."""
    for grant in chats:
        try:
            await _offer(runner, authority, grant, notice)
        except Exception:
            logger.warning('A Group Chat notice could not be delivered to a private chat', exc_info=True)
    if chats:
        return len(chats)
    text = f'{notice.text}\n\n{DESKTOP}' if notice.actions else notice.text
    sent = 0
    for home in homes:
        if _one_to_one(home):
            sent += await runner._send_home_channel_message(
                home[1], home[3], home[4], text, 'Group Chat notice to the %s home channel %s failed: %s')
    return sent


async def _continues_groups(authority, owner: str) -> bool:
    """Whether this profile's gateway lists ``groups.succession.status`` at all."""
    from gateway.session_contract import Principal
    from gateway.session_group_controls import dispatch_group_control
    reader = SimpleNamespace(authority=authority, actor=Principal(
        owner, authority.profile_id, frozenset({'session:read'}), 'messaging:notices'))
    try:
        listed = _obj(await dispatch_group_control(reader, 'groups.capabilities', {})).get('methods')
    except RuntimeStoreError:
        return False
    return isinstance(listed, list) and STATUS in listed


async def notify_all(runner) -> int:
    """One pass over every room owner of every profile this gateway serves; returns notices sent."""
    from gateway.group_chat_access import grants
    from gateway.session_authorities import all_authorities
    from gateway.session_hosted_service import _OWNER
    sent = 0
    for authority in all_authorities(runner):
        if getattr(authority, 'hosted_room_service', None) is None:
            continue
        with authority.db._read_ctx() as conn:
            owners = sorted({row[0] for row in conn.execute(
                'SELECT value FROM state_meta WHERE substr(key, 1, ?) = ?', (len(_OWNER), _OWNER))})
            private = [grant for grant in grants(conn) if grant['kind'] == 'private']
        if not owners or not await _continues_groups(authority, owners[0]):
            continue
        homes = homes_for(runner, authority)
        for owner in owners:
            try:
                notices = await _incidents(authority, owner)
            except Exception:  # every pass retries; a lasting failure must not flood the log
                logger.debug('Group Chat notices skipped an owner this time', exc_info=True)
                continue
            chats = main_chats(runner, [grant for grant in private if grant['owner'] == owner], homes)
            # The local account's own subject (the control socket's peer): only its rooms may fall back
            # to the operator's home channel, never a dashboard user's.
            local = owner.startswith(('uid:', 'sid:'))
            for notice in notices:
                sent += await _tell(runner, authority, notice, chats, homes if local else [])
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

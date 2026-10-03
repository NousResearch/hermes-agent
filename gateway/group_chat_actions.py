"""Durable action buttons for group notices: one or two taps that still work hours later.

A notice that offers choices (after a careful move, a group that ran on two computers, a
paused group this computer can continue, the summary of ``/group N continue``) gets one
record per chat it is sent to, in that profile's store. Its buttons carry only
``hg:<action>:<token>``: the token finds the record, so a tap still works after a restart or
after another prompt in the same chat. Every tap is checked again: it must come from the chat
the notice went to, from the person its private grant names, still on the Bot's DM admin list,
and the gateway's current status must still allow the action. A two-step choice first edits
the message into its confirmation; an outcome replaces the message and its buttons, and a
settled notice answers "Already resolved: …".

Adapters with native buttons implement ``send_group_actions(chat_id, text, buttons,
metadata)`` and route a tap's data to ``GatewayRunner._group_chat_action``. Elsewhere each
choice comes with its typed ``/group`` command. A bare reply such as "2" is never taken as a
choice: the chat's Bot may have asked a numbered question of its own.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import time
from types import SimpleNamespace

from gateway.group_chat_hosts import (
    OWNER_ONLY, STATUS, Summary, _obj, _promote, ask_first, computer, go_back_prompt, go_back_target, keep_on,
    prepare_summary)
from gateway.group_chat_slash import PAUSED, Refused
from hermes_state_runtime import RuntimeStoreError, _epoch

logger = logging.getLogger(__name__)
ACTION_PREFIX = 'gateway.messaging.action.v1:'
MAX_RECORDS = 512  # per profile; the oldest go first
LABEL_CHARS = 32  # a button label's cap; WhatsApp's adapter cuts its own to 20
_DATA = re.compile(r'^hg:([a-z0-9]{1,5}!?):([A-Za-z0-9_-]{12})$')
_FIELDS = frozenset({'v', 'token', 'owner', 'grant_id', 'platform', 'chat_id', 'room_id', 'ref', 'prefix', 'group',
                     'kind', 'data', 'text', 'view', 'confirm', 'outcome', 'created_at', 'updated_at'})
_ACTIONS = {'careful': {'go', 'ask', 'back', 'back!', 'no'}, 'conflict': {'k0', 'k1', 'k0!', 'k1!', 'no'},
            'continue': {'cont', 'cont!', 'no'}}
_LOCKS: dict[str, asyncio.Lock] = {}


def label(prefix: str, name: str = '') -> str:
    """A short button label; a long computer name is cut to fit."""
    room = LABEL_CHARS - len(prefix)
    return prefix + (name if len(name) <= room else name[:max(1, room - 1)] + '…')


def data_for(action: str, token: str) -> str:
    return f'hg:{action}:{token}'  # at most 21 bytes: Telegram allows 64


# ---- records ---------------------------------------------------------------------------------

def _record(raw) -> dict | None:
    try:
        record = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if (type(record) is not dict or set(record) != _FIELDS or record['v'] != 1
            or record['kind'] not in {'careful', 'conflict', 'continue'} or type(record['data']) is not dict
            or any(type(record[k]) is not str for k in ('token', 'owner', 'grant_id', 'platform', 'chat_id',
                                                        'room_id', 'prefix', 'group', 'text', 'view'))
            or type(record['ref']) is not int
            or any(record[k] is not None and type(record[k]) is not str for k in ('confirm', 'outcome'))
            or any(type(record[k]) not in (int, float) for k in ('created_at', 'updated_at'))):
        return None
    return record


def _save(authority, record: dict) -> None:
    def write(conn):
        _epoch(conn, authority.epoch)
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) '
                     'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                     (ACTION_PREFIX + record['token'], json.dumps(record, sort_keys=True)))
        rows = conn.execute('SELECT key, value FROM state_meta WHERE substr(key, 1, ?) = ?',
                            (len(ACTION_PREFIX), ACTION_PREFIX)).fetchall()
        if len(rows) > MAX_RECORDS:
            stamped = sorted(((_record(value) or {}).get('created_at', 0), key) for key, value in rows)
            for _, key in stamped[:len(rows) - MAX_RECORDS]:
                conn.execute('DELETE FROM state_meta WHERE key=?', (key,))
    authority.db._execute_write(write)


def _find(runner, token: str):
    """``(authority, record)`` for a token, from whichever profile this gateway serves holds it."""
    from gateway.session_authorities import all_authorities
    for authority in all_authorities(runner):
        with authority.db._read_ctx() as conn:
            row = conn.execute('SELECT value FROM state_meta WHERE key=?', (ACTION_PREFIX + token,)).fetchone()
        record = _record(row[0]) if row is not None else None
        if record is not None and record['token'] == token:
            return authority, record
    return None


# ---- what a notice offers ----------------------------------------------------------------------

def choices(record: dict) -> list[tuple[str, str, str, str | None]]:
    """``(label, computer name, action, typed words)`` for the record's current view; none once it is
    settled. A button reads the label and the name, cut to fit."""
    if record['outcome'] is not None:
        return []
    data, view = record['data'], record['view']
    if record['kind'] == 'careful':
        back = data['from']
        if view == 'back':
            return [('Go back to ', back, 'back!', f'keep {back} confirm'), ('Cancel', '', 'no', None)]
        return [('Keep going', '', 'go', None), ('Go back to ', back, 'back', f'keep {back}'),
                ('Ask me first', '', 'ask', 'ask first')]
    if record['kind'] == 'conflict':
        hosts = data['hosts'][:2]
        if view in {'k0', 'k1'}:
            name = hosts[int(view[1])][1]
            return [('Keep ', name, view + '!', f'keep {name}'), ('Cancel', '', 'no', None)]
        # The computer still running the group first: keeping it is keep going, so one tap.
        running = data.get('running_on')
        order = sorted(range(len(hosts)), key=lambda i: not running or hosts[i][0] != running)
        return [('Keep going on ', hosts[i][1], f'k{i}!', f'keep {hosts[i][1]}') if running and hosts[i][0] == running
                else ('Keep ', hosts[i][1], f'k{i}', f'keep {hosts[i][1]}') for i in order]
    if view == 'cont':
        return [('Continue on ', data['preview']['target'], 'cont!', 'continue confirm'), ('Cancel', '', 'no', None)]
    return [('Continue on ', data['here'], 'cont', 'continue')]


def shown_text(record: dict) -> str:
    """A confirmation, or a note above the notice (a refused continue), else the notice itself."""
    return record['confirm'] or record['text']


def typed(record: dict) -> str:
    """The text for a chat without buttons: each choice with the ``/group`` command that makes it."""
    options = [option for option in choices(record) if option[2] != 'no']
    if not options:
        return shown_text(record)
    g = f'{record["prefix"]}group {record["ref"]}'
    lines = [f'{prefix}{name}: {g} {words}' if words else f'{prefix}{name}: no reply needed.'
             for prefix, name, _, words in options]
    return '\n'.join([shown_text(record), '', *lines])


def _buttons(record: dict) -> list[tuple[str, str]]:
    return [(label(prefix, name), data_for(action, record['token'])) for prefix, name, action, _ in choices(record)]


# ---- sending -----------------------------------------------------------------------------------

async def offer(runner, authority, grant, *, room_id: str, group: str, kind: str, data: dict, text: str,
                view: str = 'main', confirm: str | None = None) -> bool:
    """Send a notice with its choices to one private chat, as buttons where its adapter has them and
    as typed commands otherwise; True when it was sent."""
    from gateway.group_chat_access import chat_target, ensure_ref
    target = chat_target(runner, grant)
    if target is None:
        return False
    adapter, metadata = target
    now = time.time()
    record = {'v': 1, 'token': secrets.token_urlsafe(9), 'owner': grant['owner'], 'grant_id': grant['grant_id'],
              'platform': grant['platform'], 'chat_id': grant['chat_id'], 'room_id': room_id,
              'ref': await asyncio.to_thread(ensure_ref, authority, grant, room_id),
              'prefix': getattr(adapter, 'typed_command_prefix', None) or '/', 'group': group, 'kind': kind,
              'data': data, 'text': text, 'view': view, 'confirm': confirm, 'outcome': None,
              'created_at': now, 'updated_at': now}
    if getattr(type(adapter), 'send_group_actions', None) is not None:
        await asyncio.to_thread(_save, authority, record)  # before sending: a tap may come back at once
        try:
            sent = await adapter.send_group_actions(grant['chat_id'], shown_text(record), _buttons(record), metadata)
        except Exception:
            logger.debug('Group Chat action buttons unavailable; typed commands instead', exc_info=True)
            sent = None
        if getattr(sent, 'success', False):
            return True
    await adapter.send(grant['chat_id'], typed(record), metadata=metadata)
    return True


# ---- a tap ------------------------------------------------------------------------------------

class _Tap:
    """Stands in for a ``/group`` command when a choice acts: the chat's own grant and principal."""

    def __init__(self, runner, authority, grant, prefix):
        from gateway.group_chat_slash import connection_for
        self.runner, self.authority, self.grant, self.prefix = runner, authority, grant, prefix
        self.connection = connection_for(authority, grant)
        self.chat = SimpleNamespace(kind='private', key=grant['grant_id'])

    async def _call(self, method, params):
        from gateway.session_group_controls import dispatch_group_control
        try:
            return await dispatch_group_control(self.connection, method, params)
        except RuntimeStoreError as exc:
            if exc.reason == 'runtime_coordination_required':
                raise Refused(PAUSED) from exc
            raise

    async def _recheck(self):
        from gateway.group_chat_access import private_grant_holds
        if not await asyncio.to_thread(private_grant_holds, self.runner, self.authority, self.grant):
            raise Refused('This chat’s access to Group Chats changed. Nothing was done.')

    def _uncertain(self, method, ref):
        logger.warning('Group Chat %s from messaging ended without a confirmed outcome', method)
        return Refused(f'Hermes couldn’t confirm whether that worked. Send {self.prefix}group {ref} '
                       'before trying again.')


async def act(runner, platform: str, chat_id, user_id, data: str) -> dict | None:
    """Resolve one tap: ``{'text', 'buttons'}`` to show in its place, or None when this person in
    this chat may not act on it (the adapter answers as for any refused tap)."""
    parsed = _DATA.match(str(data or ''))
    if parsed is None:
        return None
    action, token = parsed.groups()
    lock = _LOCKS.setdefault(token, asyncio.Lock())
    if len(_LOCKS) > 4 * MAX_RECORDS:
        _LOCKS.clear()
        lock = _LOCKS.setdefault(token, asyncio.Lock())
    async with lock:  # a double tap acts once
        found = await asyncio.to_thread(_find, runner, token)
        if found is None:
            return {'text': 'This choice is no longer available.', 'buttons': []}
        authority, record = found
        if record['platform'] != str(platform) or record['chat_id'] != str(chat_id):
            return None
        from gateway.group_chat_access import _load, private_grant_holds
        with authority.db._read_ctx() as conn:
            grant = _load(conn, record['grant_id'])
        if (grant is None or grant['owner'] != record['owner'] or grant['user_id'] != str(user_id)
                or not await asyncio.to_thread(private_grant_holds, runner, authority, grant)):
            return None
        if record['outcome'] is not None:
            return {'text': f'Already resolved: {record["outcome"]}', 'buttons': []}
        if action not in _ACTIONS[record['kind']] or (
                record['kind'] == 'conflict' and action != 'no' and int(action[1]) >= len(record['data']['hosts'])):
            return {'text': shown_text(record), 'buttons': _buttons(record), 'record': record}
        try:
            record = await _apply(_Tap(runner, authority, grant, record['prefix']), record, action)
        except Refused as exc:
            record = {**record, 'view': 'main', 'confirm': None, 'outcome': str(exc)}
        record['updated_at'] = time.time()
        await asyncio.to_thread(_save, authority, record)
        return {'text': record['outcome'] or shown_text(record), 'buttons': _buttons(record), 'record': record}


async def _apply(tap: _Tap, record: dict, action: str) -> dict:
    """The record after one choice: another view, or settled with its outcome."""
    room_id, group, data = record['room_id'], record['group'], record['data']
    g = f'{tap.prefix}group {record["ref"]}'

    def settled(outcome):
        return {**record, 'view': 'main', 'confirm': None, 'outcome': outcome}
    if action == 'no':  # Cancel: back to the notice's choices; a /group N continue summary just ends
        if data.get('command'):
            return settled('Cancelled. Nothing changed.')
        return {**record, 'view': 'main', 'confirm': None}
    if record['kind'] == 'careful':
        if action == 'go':
            return settled(f'{group} keeps going on {data["to"]}.')
        if action == 'ask':
            await tap._recheck()
            return settled((await ask_first(tap.connection, room_id, group=group, g=g)).removeprefix('Done. '))
        current = _obj(await tap._call(STATUS, {'room_id': room_id}))
        back = go_back_target(current)
        if back is None:
            return settled(f'{group} can’t go back to {data["from"]} any more.')
        if action == 'back':
            return {**record, 'view': 'back', 'confirm': go_back_prompt(current, group, g).rsplit('\n', 1)[0]}
        await tap._recheck()
        name = computer(back, current)
        await keep_on(tap.connection, room_id, back['install_id'], g=g, failed=f'Couldn’t go back to {name}.')
        return settled(f'{data["to"]} paused {group}; it continues on {name} as soon as it’s reachable.')
    if record['kind'] == 'conflict':
        index = int(action[1])
        install_id, name = data['hosts'][index]
        current = _obj(await tap._call(STATUS, {'room_id': room_id}))
        hosts = {h.get('install_id') for h in _obj(current.get('conflict')).get('hosts') or () if isinstance(h, dict)}
        if current.get('state') != 'continued_on_two' or install_id not in hosts:
            return settled(f'{group} isn’t waiting for that choice any more.')
        others = [other for key, other in data['hosts'] if key != install_id]
        kept = f' Messages from {", ".join(others)} are kept and shown separately.' if others else ''
        if not action.endswith('!'):
            return {**record, 'view': action, 'confirm': f'Keep {name}? {group} continues on {name}.{kept}'}
        await tap._recheck()
        await keep_on(tap.connection, room_id, install_id, g=g, failed=f'Couldn’t keep {name}.')
        going = install_id == data.get('running_on')
        return settled(f'{group} {"keeps going" if going else "now continues"} on {name}.{kept}')
    if action == 'cont':
        shown, summary = await prepare_summary(tap, record['ref'], room_id)
        preview = {'preview_id': shown.preview_id, 'target_id': shown.target_id, 'target': shown.target,
                   'host': shown.host, 'group': shown.group}
        return {**record, 'view': 'cont', 'confirm': summary.rsplit('\n\n', 1)[0], 'data': {**data, 'preview': preview}}
    preview = data.get('preview')
    if record['view'] != 'cont' or not isinstance(preview, dict):
        return {**record, 'view': 'main', 'confirm': None}
    try:
        outcome = await _promote(tap, record['ref'], room_id, Summary(**preview, expires=time.time()))
    except Refused as exc:
        if str(exc) == OWNER_ONLY:
            raise
        outcome = str(exc)
    if outcome.startswith('Couldn’t continue'):  # refused: offer it again, from a fresh summary
        return {**record, 'view': 'main', 'confirm': f'{outcome}\n\n{record["text"]}'}
    return settled(outcome.removeprefix('Done. '))


async def confirm_typed(runner, authority, chat, room_id: str, kind: str) -> str | None:
    """A typed confirmation for a summary this chat was shown with buttons: the same as its button."""
    def pending():
        with authority.db._read_ctx() as conn:
            rows = conn.execute('SELECT value FROM state_meta WHERE substr(key, 1, ?) = ?',
                                (len(ACTION_PREFIX), ACTION_PREFIX)).fetchall()
        records = [r for r in (_record(row[0]) for row in rows) if r is not None and r['grant_id'] == chat.key
                   and r['room_id'] == room_id and r['kind'] == kind and r['outcome'] is None and r['view'] == 'cont']
        return max(records, key=lambda r: r['updated_at'], default=None)
    record = await asyncio.to_thread(pending)
    if record is None:
        return None
    result = await act(runner, record['platform'], record['chat_id'], chat.user_id, data_for('cont!', record['token']))
    return OWNER_ONLY if result is None else result['text']

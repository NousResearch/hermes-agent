"""``/group`` in a messaging chat: the owner's canonical Group Chats, the way Desktop sees them.

Authorization comes from ``gateway.group_chat_access``. Every read and control then goes
through the canonical ``dispatch_group_control`` as the grant's owner, so a chat can never
reach a room its owner could not open in Desktop.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
import time
from types import SimpleNamespace
from typing import Any
import unicodedata

from gateway.group_chat_access import (
    UNAVAILABLE, GroupChatDenied, assign_refs, current_grant, display_code, request_code, resolve_chat,
    room_for_ref,
)
from hermes_state_runtime import RuntimeStoreError

PAGE_SIZE = 8
RECENT_EVENTS = 30
RECENT_MESSAGES = 6
MAX_ROSTER = 12
_RATE_WINDOW_SECONDS = 60.0
_RATE_LIMIT = 30
_RATE_KEYS = 2048
_CAPABILITIES = frozenset({'session:read', 'session:submit', 'session:control', 'session:approve'})

PAUSED = 'Group Chats are paused while the gateway starts or stops. Try again in a moment.'
TOO_FAST = 'Too many Group Chat commands. Wait a minute and try again.'
_NAME_ESCAPES = str.maketrans({c: chr(ord(c) + 0xFEE0) for c in r'@`*_[]<>\:#&|~()!/+='})


class Refused(Exception):
    """A user-facing outcome that ends the command."""


@dataclass(frozen=True)
class Command:
    verb: str
    ref: int = 0
    page: int = 1


def safe(value: Any, limit: int = 180) -> str:
    """Room text made inert: no mentions, markup, links, media directives or control characters."""
    text = re.sub(r'(?i)\bMEDIA:\S*', '[media]', str(value or ''))
    text = ''.join(' ' if unicodedata.category(c).startswith('C') else c for c in text)
    return ' '.join(text.split())[:limit].translate(_NAME_ESCAPES)


def _raw_args(event) -> str:
    # get_command_args() rewrites dashes; the raw text keeps what was typed after the command.
    match = re.match(r'^\s*\S+(?:\s+(.*))?$', event.text if isinstance(event.text, str) else '', re.DOTALL)
    return (match.group(1) or '') if match else ''


def _number(word: str, maximum: int) -> int:
    if not (word.isascii() and word.isdecimal() and len(word) <= 9 and 0 < int(word) <= maximum):
        raise ValueError(word)
    return int(word)


def parse(args: str) -> Command:
    words = args.split()
    if not words or words == ['list']:
        return Command('list')
    if words == ['help']:
        return Command('help')
    if len(words) == 2 and words[0] == 'list':
        return Command('list', page=_number(words[1], 9999))
    ref = _number(words[0], 10**9)
    if len(words) == 1:
        return Command('show', ref)
    raise ValueError(args)


def help_text(prefix: str) -> str:
    g = prefix + 'group'
    return '\n'.join(['Group Chats', f'{g} list [page] — your Group Chats',
                      f'{g} N — status, Bots and recent messages', f'{g} help'])


def _connect_text(chat, code: str, ttl: int, prefix: str) -> str:
    lines = ['This chat isn’t connected to your Group Chats yet.',
             'To connect it, run this on the computer where Hermes runs:',
             f'hermes groups allow {display_code(code)}',
             f'The code works once and expires in {(ttl + 59) // 60} minutes.']
    if chat.kind == 'shared':
        lines.append(f'Everyone in this chat will be able to read what {prefix}group shows here.')
    return '\n'.join(lines)


def _too_fast(runner, key) -> bool:
    now = time.monotonic()
    buckets = getattr(runner, '_group_chat_rate_buckets', None)
    if buckets is None:
        buckets = runner._group_chat_rate_buckets = {}
    for stale in [k for k, stamps in buckets.items() if not stamps or now - stamps[-1] >= _RATE_WINDOW_SECONDS]:
        del buckets[stale]
    recent = [stamp for stamp in buckets.get(key, ()) if now - stamp < _RATE_WINDOW_SECONDS]
    # Live buckets are never evicted: rotating identities must not reset anyone's limit.
    if len(recent) >= _RATE_LIMIT or (key not in buckets and len(buckets) >= _RATE_KEYS):
        return True
    buckets[key] = [*recent, now]
    return False


class GroupChatSlashCommandsMixin:
    async def _handle_group_command(self, event):
        try:
            return await _GroupCommand.start(self, event)
        except Refused as exc:
            return str(exc)


class _GroupCommand:
    def __init__(self, runner, event, authority, chat, grant, prefix):
        self.runner, self.event, self.authority = runner, event, authority
        self.chat, self.grant, self.prefix = chat, grant, prefix
        from gateway.session_contract import Principal
        # The owner's own reach, under the messaging chat's transport identity.
        self.connection = SimpleNamespace(authority=authority, actor=Principal(
            grant['owner'], authority.profile_id, _CAPABILITIES, 'messaging:' + grant['grant_id'][:32]))

    @classmethod
    async def start(cls, runner, event):
        from gateway.session_authorities import active_authority
        try:
            chat, adapter = resolve_chat(runner, event)
        except GroupChatDenied as exc:
            raise Refused(str(exc)) from exc
        authority = active_authority(runner)
        if authority is None or getattr(authority, 'hosted_room_service', None) is None:
            raise Refused(UNAVAILABLE)
        prefix = runner._typed_command_prefix_for(event.source.platform)
        try:
            command = parse(_raw_args(event))
        except ValueError:
            raise Refused(f'I didn’t understand that.\n\n{help_text(prefix)}') from None
        if _too_fast(runner, (chat.key, chat.user_id)):
            raise Refused(TOO_FAST)
        grant = await asyncio.to_thread(current_grant, authority, chat)
        if grant is None:
            try:
                code, ttl = request_code(runner, authority, chat, event.source, adapter)
            except GroupChatDenied as exc:
                raise Refused(str(exc)) from exc
            connect = _connect_text(chat, code, ttl, prefix)
            return f'{help_text(prefix)}\n\n{connect}' if command.verb == 'help' else connect
        return await getattr(cls(runner, event, authority, chat, grant, prefix), '_' + command.verb)(command)

    async def _call(self, method, params):
        from gateway.session_group_controls import dispatch_group_control
        try:
            return await dispatch_group_control(self.connection, method, params)
        except RuntimeStoreError as exc:
            if exc.reason == 'runtime_coordination_required':
                raise Refused(PAUSED) from exc
            raise

    def _room_id(self, ref: int) -> str:
        room_id = room_for_ref(self.grant, ref)
        if room_id is None:
            raise Refused(f'Group {ref} isn’t available. Send {self.prefix}group list to see your Group Chats.')
        return room_id

    async def _help(self, command):
        return help_text(self.prefix)

    async def _list(self, command):
        rooms, offset = [], 0
        for _ in range(64):
            page = await self._call('groups.list', {'limit': 500, 'offset': offset})
            rooms.extend(page['rooms'])
            if page['next_offset'] is None:
                break
            offset = page['next_offset']
        else:
            raise Refused(UNAVAILABLE)
        self.grant = await asyncio.to_thread(assign_refs, self.authority, self.grant, [r['room_id'] for r in rooms])
        rows = sorted(((self.grant['refs'][room['room_id']], room) for room in rooms), key=lambda row: row[0])
        if not rows:
            return 'You have no Group Chats here yet. Create one in Hermes Desktop.'
        pages = (len(rows) + PAGE_SIZE - 1) // PAGE_SIZE
        if command.page > pages:
            raise Refused(f'There {"is" if pages == 1 else "are"} only {pages} page{"" if pages == 1 else "s"}.')
        lines = [f'Group Chats, page {command.page} of {pages}' if pages > 1 else 'Group Chats', '']
        for ref, room in rows[(command.page - 1) * PAGE_SIZE:command.page * PAGE_SIZE]:
            count = len(room['members'])
            lines.append(f'{ref}. {safe(room["name"], 72)} · {count} Bot{"" if count == 1 else "s"}')
        g = self.prefix + 'group'
        lines.extend(['', f'Open one: {g} N'])
        if command.page < pages:
            lines.append(f'Next page: {g} list {command.page + 1}')
        return '\n'.join(lines)

    async def _show(self, command):
        room_id = self._room_id(command.ref)
        try:
            state = await self._call('groups.state', {'room_id': room_id})
            probe = await self._call('groups.log', {'room_id': room_id, 'since_seq': 0, 'limit': 1})
            latest = probe['latest_seq']
            log = await self._call('groups.log', {'room_id': room_id, 'since_seq': max(0, latest - RECENT_EVENTS),
                                                  'limit': RECENT_EVENTS}) if latest else probe
        except RuntimeStoreError as exc:
            raise Refused(f'Group {command.ref} isn’t available right now. '
                          f'Send {self.prefix}group list to check.') from exc
        return '\n'.join(self._detail(command.ref, state, log['events']))

    def _detail(self, ref, state, events):
        room, status = state['room'], state.get('driver_status') or {}
        labels = {m['member_id']: safe(m.get('display_name') or m.get('handle') or m['member_id'], 48)
                  for m in room['members']}
        lines = [f'Group {ref} · {safe(room["name"], 72)}', self._status(status)]
        roster = [f'{labels[m["member_id"]]} ({safe("@" + (m.get("handle") or m["member_id"]), 33)})'
                  for m in room['members'][:MAX_ROSTER]]
        extra = len(room['members']) - MAX_ROSTER
        lines.append('Bots: ' + ', '.join(roster) + (f' and {extra} more' if extra > 0 else ''))
        previews = [p for p in (self._preview(e, labels) for e in events) if p][-RECENT_MESSAGES:]
        lines.extend(['', 'Recent messages', *(previews or ['No messages yet.'])])
        lines.extend(['', *self._commands(ref)])
        return lines

    @staticmethod
    def _status(status) -> str:
        if not status:
            return 'The Group Chat driver isn’t running, so nothing here is moving.'
        actions = status.get('pending_actions') or []
        approvals = sum(1 for a in actions if a.get('kind') == 'approval')
        parts = ['Working' if status.get('working') else 'Idle' if status.get('running') else 'Stopped']
        if status.get('blocked'):
            parts.append('blocked')
        if approvals:
            parts.append(f'{approvals} approval{"" if approvals == 1 else "s"} waiting')
        if len(actions) > approvals:
            parts.append(f'{len(actions) - approvals} to retry or discard in Desktop')
        return ' · '.join(parts)

    def _commands(self, ref):
        return [f'Refresh: {self.prefix}group {ref}']

    @staticmethod
    def _preview(event, labels):
        kind, payload, actor = event.get('kind'), event.get('payload') or {}, event.get('actor') or {}
        if kind == 'message.member':
            speaker = labels.get(payload.get('member_id') or actor.get('id'), 'Bot')
        elif kind == 'message.user':
            # Desktop's own Send is recorded as {'kind': 'user', 'id': 'desktop'}.
            speaker = 'Desktop' if actor.get('id') == 'desktop' else safe(actor.get('display_name') or 'Someone', 64)
        else:
            return None
        return f'• {speaker}: {safe(payload.get("text") or "[attachment]")}'

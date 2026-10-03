"""``/group`` in a messaging chat: the owner's canonical Group Chats, the way Desktop sees them.

Authorization comes from ``gateway.group_chat_access``. Every read and control then goes
through the canonical ``dispatch_group_control`` as the grant's owner, so a chat can never
reach a room its owner could not open in Desktop. What a group shows and allows when its
host goes offline is in ``gateway.group_chat_hosts``.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import logging
import re
import time
from types import SimpleNamespace
from typing import Any
import unicodedata
import uuid

from gateway.group_chat_access import (
    UNAVAILABLE, GroupChatDenied, assign_refs, current_grant, display_code, request_code, resolve_chat,
    room_for_ref,
)
from gateway.group_chat_rules import forget_rule, rememberable, rules_for
from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)
PAGE_SIZE = 8
RECENT_EVENTS = 30
RECENT_MESSAGES = 6
MAX_ROSTER = 12
_RATE_WINDOW_SECONDS = 60.0
_RATE_LIMIT = 30
_SEND_RATE_LIMIT = 10
_RATE_KEYS = 2048
MAX_APPROVALS = 12
_CAPABILITIES = frozenset({'session:read', 'session:submit', 'session:control', 'session:approve'})

PAUSED = 'Group Chats are paused while the gateway starts or stops. Try again in a moment.'
TOO_FAST = 'Too many Group Chat commands. Wait a minute and try again.'
ACCESS_CHANGED = 'This chat’s access to Group Chats changed. Send {prefix}group to check.'
_NAME_ESCAPES = str.maketrans({c: chr(ord(c) + 0xFEE0) for c in r'@`*_[]<>\:#&|~()!/+='})


class Refused(Exception):
    """A user-facing outcome that ends the command."""


@dataclass(frozen=True)
class Command:
    verb: str
    ref: int = 0
    page: int = 1
    text: str = ''
    code: str = ''
    choice: str = ''


def safe(value: Any, limit: int = 180) -> str:
    """Room text made inert: no mentions, markup, links, media directives or control characters."""
    text = re.sub(r'(?i)\bMEDIA:\S*', '[media]', str(value or ''))
    text = ''.join(' ' if unicodedata.category(c).startswith('C') else c for c in text)
    return ' '.join(text.split())[:limit].translate(_NAME_ESCAPES)


def code(value: Any, limit: int = 400, *, block: bool = False) -> str:
    """A command shown exactly, as the gateway shows approvals: in code, never as markup or mentions."""
    text = ''.join(c if c == '\n' and block else ' ' if unicodedata.category(c).startswith('C') else c
                   for c in str(value or ''))[:limit].replace('`', 'ˋ')
    return f'```\n{text}\n```' if block else f'`{" ".join(text.split())}`'


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
    verb = words[1].casefold()
    if verb == 'send':
        # Everything after "send", exactly as typed (line breaks included).
        text = re.match(r'\s*\S+\s+\S+(?:\s(.*))?$', args, re.DOTALL).group(1) or ''
        return Command('send', ref, text=text.strip())
    if verb == 'stop' and len(words) == 2:
        return Command('stop', ref)
    if verb == 'continue' and (len(words) == 2 or (len(words) == 3 and words[2].casefold() == 'confirm')):
        return Command('continue', ref, choice='confirm' if len(words) == 3 else '')
    if verb == 'keep':
        # The computer's name as typed; names can hold spaces ("Mac mini").
        return Command('keep', ref, text=' '.join(words[2:]))
    choice = ' '.join(words[3:]).casefold()
    if verb == 'approve' and len(words) in {4, 5} and choice in {'once', 'deny', 'always', 'always confirm'}:
        return Command('approve', ref, code=words[2].casefold(), choice=choice)
    if verb == 'forget' and len(words) == 3:
        return Command('forget', ref, code=words[2].casefold())
    raise ValueError(args)


def approval_code(action: dict) -> str:
    """A short handle for one exact pending request (member, task, attempt and request)."""
    identity = [action.get(k) for k in ('member_id', 'task_id', 'execution_generation', 'request_id')]
    return hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()[:6]


def help_text(prefix: str, *, hosts: bool = False) -> str:
    """The command list; ``hosts`` adds continue and keep where the gateway offers them."""
    g = prefix + 'group'
    lines = ['Group Chats', f'{g} list [page] — your Group Chats',
             f'{g} N — status, approvals and recent messages',
             f'{g} N send <message> — post a message as you',
             f'{g} N stop — stop the work in progress',
             f'{g} N approve <code> once|always|deny — answer an approval',
             f'{g} N forget <code> — forget an approval this chat always allows']
    if hosts:
        lines += [f'{g} N continue — continue a paused group on this computer',
                  f'{g} N keep <computer> — choose the computer that keeps a group continued on two']
    return '\n'.join([*lines, f'{g} help'])


def _connect_text(chat, code: str, ttl: int, prefix: str) -> str:
    lines = ['This chat isn’t connected to your Group Chats yet.',
             'To connect it, run this on the computer where Hermes runs:',
             f'hermes groups allow {display_code(code)}',
             f'The code works once and expires in {(ttl + 59) // 60} minutes.']
    if chat.kind == 'shared':
        lines.append(f'Everyone in this chat will be able to read what {prefix}group shows here.')
    return '\n'.join(lines)


def _too_fast(runner, key, limit=_RATE_LIMIT) -> bool:
    now = time.monotonic()
    buckets = getattr(runner, '_group_chat_rate_buckets', None)
    if buckets is None:
        buckets = runner._group_chat_rate_buckets = {}
    for stale in [k for k, stamps in buckets.items() if not stamps or now - stamps[-1] >= _RATE_WINDOW_SECONDS]:
        del buckets[stale]
    recent = [stamp for stamp in buckets.get(key, ()) if now - stamp < _RATE_WINDOW_SECONDS]
    # Live buckets are never evicted: rotating identities must not reset anyone's limit.
    if len(recent) >= limit or (key not in buckets and len(buckets) >= _RATE_KEYS):
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
        # The owner's own reach, under the messaging chat's transport identity. A shared chat's says so
        # (messaging:shared:…), so owner-only gateway actions can refuse it as well.
        self.connection = SimpleNamespace(authority=authority, actor=Principal(
            grant['owner'], authority.profile_id, _CAPABILITIES, f'messaging:{grant["kind"]}:' + grant['grant_id'][:32]))

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

    async def _recheck(self):
        """Right before a change: the sender, the chat and its grant are still exactly as checked."""
        try:
            chat, _ = resolve_chat(self.runner, self.event)
        except GroupChatDenied as exc:
            raise Refused(str(exc)) from exc
        grant = await asyncio.to_thread(current_grant, self.authority, chat)
        if chat.key != self.chat.key or grant is None or grant['owner'] != self.grant['owner']:
            raise Refused(ACCESS_CHANGED.format(prefix=self.prefix))

    async def _change(self, ref, method, params, **kwargs):
        """Dispatch one change; an outcome we can't confirm is reported, never retried."""
        from gateway.session_group_controls import dispatch_group_control
        await self._recheck()
        try:
            return await dispatch_group_control(self.connection, method, params, **kwargs)
        except RuntimeStoreError as exc:
            if exc.reason == 'runtime_coordination_required':
                raise Refused(PAUSED) from exc
            if method == 'groups.approve' and exc.reason == 'stale_generation':
                raise Refused(self._gone(ref)) from exc
            raise Refused(f'Group {ref} didn’t accept that. Send {self.prefix}group {ref} to see why.') from exc
        except RuntimeError as exc:
            if method == 'groups.approve' and str(exc) == 'room approval is no longer pending':
                raise Refused(self._gone(ref)) from exc
            raise self._uncertain(method, ref) from exc
        except Exception as exc:
            raise self._uncertain(method, ref) from exc

    def _gone(self, ref):
        return f'That approval isn’t waiting any more. Send {self.prefix}group {ref} to see what is.'

    def _uncertain(self, method, ref):
        # Transport and service errors can carry private text; keep it out of logs and replies.
        logger.warning('Group Chat %s from messaging ended without a confirmed outcome', method)
        return Refused(f'Hermes couldn’t confirm whether that worked. Send {self.prefix}group {ref} '
                       'before trying again.')

    def _message_id(self, purpose: str) -> str:
        """Stable for a redelivered platform message, unique otherwise."""
        message = self.event.message_id or self.event.source.message_id or uuid.uuid4().hex
        identity = [purpose, self.chat.key, str(message)]
        return f'messaging-{purpose}:' + hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:40]

    def _room_id(self, ref: int) -> str:
        room_id = room_for_ref(self.grant, ref)
        if room_id is None:
            raise Refused(f'Group {ref} isn’t available. Send {self.prefix}group list to see your Group Chats.')
        return room_id

    async def _help(self, command):
        from gateway.group_chat_hosts import PREPARE, PROMOTE, STATUS, advertised
        return help_text(self.prefix, hosts=await advertised(self, STATUS, PREPARE, PROMOTE))

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
        from gateway import group_chat_hosts as hosts
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
        hosting = await hosts.host_status(self, room_id)
        return '\n'.join(self._detail(command.ref, state, log['events'], hosting))

    def _detail(self, ref, state, events, hosting=None):
        from gateway import group_chat_hosts as hosts
        room, status = state['room'], state.get('driver_status') or {}
        labels = _labels(room)
        lines = [f'Group {ref} · {safe(room["name"], 72)}', self._status(status)]
        if hosting is not None:
            lines.extend(hosts.host_lines(hosting, group=f'“{safe(room["name"], 72)}”',
                                          g=f'{self.prefix}group {ref}', may_act=self.chat.kind == 'private'))
        roster = [f'{labels[m["member_id"]]} ({safe("@" + (m.get("handle") or m["member_id"]), 33)})'
                  for m in room['members'][:MAX_ROSTER]]
        extra = len(room['members']) - MAX_ROSTER
        lines.append('Bots: ' + ', '.join(roster) + (f' and {extra} more' if extra > 0 else ''))
        for action in _approvals(status)[:MAX_APPROVALS]:
            lines.extend(['', *self._approval_lines(ref, action, labels)])
        lines.extend(self._remembered_lines(ref, room, labels))
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
        # Work that needs a Bot or file only its former host has, after the group moved.
        waiting = {}
        for task in status.get('tasks') or ():
            if isinstance(task, dict) and task.get('state') == 'waiting_for_host':
                host = safe(task.get('host_name'), 48) or 'another computer'
                waiting[host] = waiting.get(host, 0) + 1
        parts.extend(f'{count} waiting for {host}' for host, count in waiting.items())
        return ' · '.join(parts)

    def _approval_lines(self, ref, action, labels):
        approval = action.get('approval') or {}
        description = safe(approval.get('description'), 300)
        lines = [f'Approval {approval_code(action)} · {labels.get(action.get("member_id"), "A Bot")} asks to run:',
                 code(approval.get('command') or approval.get('description') or 'an action', block=True)]
        if description and description != approval.get('command'):
            lines.append(description)
        if rememberable(approval):
            lines.append('in ' + code(approval['remember_context']))
        choices = 'once|always|deny' if rememberable(approval) else 'once|deny'
        lines.append(f'Answer: {self.prefix}group {ref} approve {approval_code(action)} {choices}')
        return lines

    def _remembered_lines(self, ref, room, labels):
        rules = rules_for(self.authority, self.grant['grant_id'], room)
        if not rules:
            return []
        lines = ['', 'Always allowed in this chat']
        for rule in rules:
            used = f' · used {rule["uses"]} time{"" if rule["uses"] == 1 else "s"}' if rule['uses'] else ''
            lines.extend([f'{rule["rule_id"][:6]} · {labels.get(rule["member_id"], "A Bot")}{used}',
                          code(rule['command'], block=True), 'in ' + code(rule['context'])])
        lines.append(f'Forget one: {self.prefix}group {ref} forget <code>')
        return lines

    def _commands(self, ref):
        g = f'{self.prefix}group {ref}'
        return [f'Send: {g} send <message>', f'Stop: {g} stop', f'Refresh: {g}']

    async def _send(self, command):
        from gateway.hosted_room_discussion import MAX_USER_TEXT_BYTES
        if not command.text or len(command.text.encode('utf-8')) > MAX_USER_TEXT_BYTES:
            raise Refused(f'Write the message after send, for example: {self.prefix}group {command.ref} send Hello')
        if _too_fast(self.runner, (self.chat.key, self.chat.user_id, 'send'), _SEND_RATE_LIMIT):
            raise Refused(TOO_FAST)
        room_id = self._room_id(command.ref)
        event_id = self._message_id('send')
        await self._change(command.ref, 'groups.send', {
            'room_id': room_id, 'event_id': event_id, 'payload': {'text': command.text, 'thread_id': event_id}},
            author=self.chat.author())
        return f'Sent to Group {command.ref}. Read the replies with {self.prefix}group {command.ref}'

    async def _stop(self, command):
        room_id = self._room_id(command.ref)
        result = await self._change(command.ref, 'groups.stop',
                                    {'room_id': room_id, 'cancel_id': self._message_id('stop')})
        count = result['cancelled']
        return (f'Stopping work in Group {command.ref} ({count} task{"" if count == 1 else "s"}).'
                if count else f'Nothing was running in Group {command.ref}.')

    async def _pending(self, command):
        room_id = self._room_id(command.ref)
        try:
            state = await self._call('groups.state', {'room_id': room_id})
        except RuntimeStoreError as exc:
            raise Refused(f'Group {command.ref} isn’t available right now.') from exc
        room = state['room']
        matches = [a for a in _approvals(state.get('driver_status') or {}) if approval_code(a) == command.code]
        if len(matches) != 1:
            raise Refused(self._gone(command.ref))
        labels = _labels(room)
        return room_id, matches[0], labels.get(matches[0]['member_id'], 'the Bot')

    async def _approve(self, command):
        room_id, action, bot = await self._pending(command)
        params = {'room_id': room_id, 'choice': command.choice.split()[0],
                  **{key: action[key] for key in ('member_id', 'task_id', 'execution_generation', 'request_id')}}
        if command.choice == 'always':
            return self._always_warning(command, action, bot)
        if params['choice'] != 'always':
            await self._change(command.ref, 'groups.approve', params)
            return f'Allowed once for {bot}.' if params['choice'] == 'once' else f'Denied for {bot}.'
        if not rememberable(action.get('approval')):
            raise Refused('This request can only be allowed once or denied.')
        result = (await self._change(command.ref, 'groups.approve', params, remember={
            'grant_id': self.grant['grant_id'], 'by': self.chat.author()['display_name']}))['result']
        if result.get('remembered'):
            return (f'Allowed. {bot} may run this exact command again in Group {command.ref} without asking, '
                    f'while this chat stays connected. Forget it with: {self.prefix}group {command.ref} forget '
                    f'{result["remembered"]}')
        if result.get('status') == 'resolved':
            return f'Allowed once for {bot}, but Hermes couldn’t remember it. It will ask again next time.'
        return self._gone(command.ref)

    def _always_warning(self, command, action, bot):
        approval = action['approval'] if rememberable(action.get('approval')) else None
        if approval is None:
            raise Refused('This request can only be allowed once or denied.')
        g = f'{self.prefix}group {command.ref}'
        return '\n'.join([
            'Always allow this in this chat?',
            f'{bot} could then run this exact command again in Group {command.ref} without asking, '
            'for as long as this chat stays connected:',
            code(approval.get('command'), block=True), 'in ' + code(approval['remember_context']),
            'It can change files and data. You can forget it later.',
            f'Confirm: {g} approve {command.code} always confirm'])

    async def _forget(self, command):
        room_id = self._room_id(command.ref)
        await self._recheck()
        forgotten = await asyncio.to_thread(forget_rule, self.authority, self.grant['grant_id'], room_id, command.code)
        if forgotten is None:
            raise Refused(f'No approval this chat always allows in Group {command.ref} has that code. '
                          f'Send {self.prefix}group {command.ref} to see them.')
        return f'Forgotten. That command will ask for approval again in Group {command.ref}.'

    async def _continue(self, command):
        from gateway.group_chat_hosts import continue_command
        return await continue_command(self, command)

    async def _keep(self, command):
        from gateway.group_chat_hosts import keep_command
        return await keep_command(self, command)

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


def _labels(room) -> dict[str, str]:
    return {m['member_id']: safe(m.get('display_name') or m.get('handle') or m['member_id'], 48)
            for m in room['members']}


def _approvals(status) -> list[dict]:
    """Exact pending approvals as the canonical driver reports them."""
    return [action for action in status.get('pending_actions') or []
            if type(action) is dict and action.get('kind') == 'approval'
            and all(type(action.get(k)) is str and action[k] for k in ('member_id', 'task_id', 'request_id'))
            and type(action.get('execution_generation')) is int]

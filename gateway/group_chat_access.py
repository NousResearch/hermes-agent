"""Which messaging chats may control the owner's canonical Group Chats.

Every ``/group`` command needs both:

* a grant for this exact chat. The owner gives it once, on the computer running Hermes:
  ``/group`` in an unconnected chat shows a short code, and ``hermes groups allow <code>``
  stores the grant; ``hermes groups revoke`` removes it again;
* a sender on the receiving Bot's admin allowlist for that kind of chat:
  ``allow_admin_from`` in a DM, ``group_allow_admin_from`` anywhere else.

A private grant (a chat that is provably the sender and the Bot alone) also names the
person. A shared grant names the chat: everyone in it can read what ``/group`` shows
there, and only allowlisted people can use it. The owner is the local account that ran
the CLI, which is the same subject that owns the Group Chats Desktop creates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import re
import secrets
import threading
import time
from typing import Any
import unicodedata

from gateway.group_chat_identity import is_dm_scope, is_private_source, platform_name, trusted_person
from hermes_state_runtime import RuntimeStoreError, _epoch

logger = logging.getLogger(__name__)
GRANT_PREFIX = 'gateway.messaging.chat.v1:'
CODE_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
CODE_TTL_SECONDS = 600
MAX_PENDING_CODES = 64
MAX_GRANTS = 256
_GRANT_FIELDS = frozenset({
    'version', 'grant_id', 'owner', 'bot', 'platform', 'chat_id', 'thread_id', 'scope_id',
    'user_id', 'kind', 'chat_label', 'user_label', 'created_at', 'refs', 'next_ref'})
_LOCK = threading.Lock()

NOT_A_PERSON = 'Group Chat commands only work from a person’s own, unedited message.'
UNAVAILABLE = 'Group Chats aren’t available through this Bot right now.'
NOT_ADMIN = ('Only people on this Bot’s {setting} list can use /group here. '
             'Ask the owner to add your user ID ({user_id}).')
TOO_MANY_CODES = 'Too many chats are waiting to be connected. Try again in a few minutes.'


class GroupChatDenied(Exception):
    """A refusal whose text is safe to send to the chat."""


@dataclass(frozen=True)
class Chat:
    """One messaging chat as seen through the Bot that received the command."""
    key: str
    bot: str
    platform: str
    chat_id: str
    thread_id: str | None
    scope_id: str | None
    user_id: str
    kind: str  # 'private' or 'shared'
    chat_label: str
    user_label: str
    admins: str  # the Bot's admin list that applies here: allow_admin_from or group_allow_admin_from

    def author(self) -> dict[str, str]:
        """Actor fields for a message sent from this chat: who typed it, and where."""
        from gateway.hosted_rooms_common import IDENTIFIER_RE
        actor_id = f'{self.platform}:{self.user_id}'
        if len(actor_id) > 128 or not IDENTIFIER_RE.fullmatch(actor_id):
            actor_id = f'{self.platform}:sha256-' + hashlib.sha256(self.user_id.encode()).hexdigest()[:40]
        return {'kind': 'user', 'id': actor_id,
                'display_name': f'{self.user_label} via {platform_title(self.platform)}'}


def platform_title(platform: str) -> str:
    return platform.replace('_', ' ').title()


def label(value: Any, limit: int = 64) -> str:
    text = ''.join(' ' if unicodedata.category(c).startswith('C') else c for c in str(value or ''))
    return ' '.join(text.split())[:limit]


def resolve_chat(runner, event) -> tuple[Chat, Any]:
    """The chat and receiving adapter, or :class:`GroupChatDenied` for this sender."""
    source = event.source
    if not trusted_person(event):
        raise GroupChatDenied(NOT_A_PERSON)
    owner = runner._transport_owner(source)
    if owner is None:
        raise GroupChatDenied(UNAVAILABLE)
    adapter, bot = owner
    from gateway.slash_access import policy_from_extra
    dm = is_dm_scope(source)
    admins = 'allow_admin_from' if dm else 'group_allow_admin_from'
    extra = getattr(getattr(adapter, 'config', None), 'extra', None)
    policy = policy_from_extra(extra if isinstance(extra, dict) else {}, 'dm' if dm else 'group')
    user = str(source.user_id).strip()
    # An empty list disables slash gating for everyone else; here it allows nobody.
    if not policy.enabled or user not in policy.admin_user_ids:
        raise GroupChatDenied(NOT_ADMIN.format(setting=admins, user_id=user))
    from gateway.slash_commands import _home_thread_from_source
    kind = 'private' if is_private_source(source) else 'shared'
    thread = _home_thread_from_source(source)
    scope = str(source.scope_id) if getattr(source, 'scope_id', None) else None
    bot = bot or getattr(runner, '_primary_profile_name', None) or 'default'
    platform = platform_name(source)
    identity = [bot, platform, str(source.chat_id), thread, scope, user if kind == 'private' else None]
    key = hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()
    return Chat(key, bot, platform, str(source.chat_id), thread, scope, user, kind,
                label(getattr(source, 'chat_name', None)) or str(source.chat_id),
                label(getattr(source, 'user_name', None)) or user, admins), adapter


# ---- codes: one per chat, kept only in this gateway process --------------------------------

@dataclass
class _Request:
    chat: Chat
    profile_id: str
    source: Any
    adapter: Any
    expires: float


def _pending(runner, now):
    requests = getattr(runner, '_group_chat_requests', None)
    if requests is None:
        requests = runner._group_chat_requests = {}
    for code in [code for code, request in requests.items() if request.expires <= now]:
        del requests[code]
    return requests


def normalize_code(value: Any) -> str:
    code = re.sub(r'[\s-]', '', str(value or '')).upper()
    return code if len(code) == 8 and all(c in CODE_ALPHABET for c in code) else ''


def display_code(code: str) -> str:
    return f'{code[:4]}-{code[4:]}'


def request_code(runner, authority, chat: Chat, source, adapter) -> tuple[str, int]:
    """A short-lived code the owner passes to ``hermes groups allow``; reused per chat."""
    now = time.monotonic()
    with _LOCK:
        requests = _pending(runner, now)
        for code, request in requests.items():
            if request.chat.key == chat.key and request.profile_id == authority.profile_id:
                request.chat, request.source, request.adapter = chat, source, adapter
                return code, max(1, int(request.expires - now))
        if len(requests) >= MAX_PENDING_CODES:
            raise GroupChatDenied(TOO_MANY_CODES)
        code = ''.join(secrets.choice(CODE_ALPHABET) for _ in range(8))
        requests[code] = _Request(chat, authority.profile_id, source, adapter, now + CODE_TTL_SECONDS)
        return code, CODE_TTL_SECONDS


# ---- grants: state_meta rows in the profile's own store ------------------------------------

def _grant(raw: Any) -> dict | None:
    """A well-formed grant, or None (a damaged row never authorizes anything)."""
    try:
        record = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if (type(record) is not dict or set(record) != _GRANT_FIELDS
            or type(record['version']) is not int or record['version'] != 1
            or any(type(record[k]) is not str or not record[k] for k in (
                'grant_id', 'owner', 'bot', 'platform', 'chat_id', 'user_id', 'chat_label', 'user_label'))
            or any(record[k] is not None and type(record[k]) is not str for k in ('thread_id', 'scope_id'))
            or record['kind'] not in {'private', 'shared'}
            or type(record['created_at']) not in (int, float)
            or type(record['refs']) is not dict or type(record['next_ref']) is not int
            or any(type(ref) is not int or not 0 < ref < record['next_ref'] for ref in record['refs'].values())):
        return None
    return record


def _load(conn, grant_id: str) -> dict | None:
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (GRANT_PREFIX + grant_id,)).fetchone()
    grant = _grant(row[0]) if row is not None else None
    return grant if grant is not None and grant['grant_id'] == grant_id else None


def _save(conn, grant: dict) -> None:
    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) '
                 'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                 (GRANT_PREFIX + grant['grant_id'], json.dumps(grant, sort_keys=True)))


def grants(conn) -> list[dict]:
    rows = conn.execute('SELECT key, value FROM state_meta WHERE substr(key, 1, ?) = ? ORDER BY key',
                        (len(GRANT_PREFIX), GRANT_PREFIX)).fetchall()
    parsed = ((key, _grant(value)) for key, value in rows)
    return [grant for key, grant in parsed if grant is not None and key == GRANT_PREFIX + grant['grant_id']]


def current_grant(authority, chat: Chat) -> dict | None:
    with authority.db._read_ctx() as conn:
        return _load(conn, chat.key)


def assign_refs(authority, grant: dict, room_ids: list[str]) -> dict:
    """Give every visible room a number that this chat will never reuse for another room."""
    def write(conn):
        _epoch(conn, authority.epoch)
        current = _load(conn, grant['grant_id'])
        if current is None or current['owner'] != grant['owner']:
            raise RuntimeStoreError('permission_denied')
        refs = {room: ref for room, ref in current['refs'].items() if room in room_ids}
        next_ref = current['next_ref']
        for room in room_ids:
            if room not in refs:
                refs[room], next_ref = next_ref, next_ref + 1
        if (refs, next_ref) != (current['refs'], current['next_ref']):
            current = {**current, 'refs': refs, 'next_ref': next_ref}
            _save(conn, current)
        return current
    return authority.db._execute_write(write)


def room_for_ref(grant: dict, ref: int) -> str | None:
    return next((room for room, value in grant['refs'].items() if value == ref), None)


# ---- the owner's CLI, over the authenticated control socket --------------------------------

def _view(chat: Chat | dict, profile_id: str) -> dict:
    get = chat.get if isinstance(chat, dict) else lambda name: getattr(chat, name)
    from gateway.session_authorities import served_profile_name
    return {'profile': served_profile_name(Path(profile_id)), 'bot': get('bot'),
            'platform': get('platform'), 'kind': get('kind'), 'chat': get('chat_label'),
            'chat_id': get('chat_id'), 'user': get('user_label'), 'user_id': get('user_id')}


def control_verb(runner, loop=None):
    """``group-chats`` private verb: the peer subject is the local account running the CLI."""
    def handle(params, subject):
        action = params.get('action') if isinstance(params, dict) else None
        handler = {'describe': _describe, 'allow': _allow, 'list': _list, 'revoke': _revoke}.get(action)
        if handler is None or not isinstance(subject, str) or not subject:
            return {'error': 'invalid_request'}
        try:
            return handler(runner, params, subject, loop)
        except RuntimeStoreError as exc:
            return {'error': exc.reason}
        except Exception:
            logger.warning('group-chats control verb failed', exc_info=True)
            return {'error': 'unavailable'}
    return handle


def _request(runner, params) -> tuple[str, _Request | None]:
    code = normalize_code(params.get('code'))
    with _LOCK:
        return code, _pending(runner, time.monotonic()).get(code) if code else None


def _describe(runner, params, subject, loop):
    code, request = _request(runner, params)
    if request is None:
        return {'error': 'unknown_code'}
    return {'code': display_code(code), 'expires_in': max(1, int(request.expires - time.monotonic())),
            'admins': request.chat.admins, **_view(request.chat, request.profile_id)}


def _allow(runner, params, subject, loop):
    from gateway.session_authorities import authority_for_profile_id
    code, request = _request(runner, params)
    if request is None:
        return {'error': 'unknown_code'}
    authority = authority_for_profile_id(runner, request.profile_id)
    if authority is None:
        return {'error': 'profile_unavailable'}
    chat = request.chat

    def write(conn):
        _epoch(conn, authority.epoch)
        existing = _load(conn, chat.key)
        if existing is not None and existing['owner'] != subject:
            raise RuntimeStoreError('permission_denied')
        if existing is None and len(grants(conn)) >= MAX_GRANTS:
            raise RuntimeStoreError('capacity_exhausted')
        record = {'version': 1, 'grant_id': chat.key, 'owner': subject, 'bot': chat.bot,
                  'platform': chat.platform, 'chat_id': chat.chat_id, 'thread_id': chat.thread_id,
                  'scope_id': chat.scope_id, 'user_id': chat.user_id, 'kind': chat.kind,
                  'chat_label': chat.chat_label, 'user_label': chat.user_label,
                  'created_at': existing['created_at'] if existing else time.time(),
                  'refs': existing['refs'] if existing else {},
                  'next_ref': existing['next_ref'] if existing else 1}
        _save(conn, record)
        return record
    record = authority.db._execute_write(write)
    with _LOCK:
        _pending(runner, time.monotonic()).pop(code, None)
    _announce(runner, loop, request)
    return {'grant': record['grant_id'][:8], **_view(record, authority.profile_id)}


def _announce(runner, loop, request):
    """Tell the chat it is connected; best effort, the CLI already reported success."""
    if loop is None or request.adapter is None:
        return
    try:
        prefix = runner._typed_command_prefix_for(request.source.platform)
        metadata = runner._thread_metadata_for_source(request.source)
    except Exception:
        prefix, metadata = '/', None
    text = f'This chat is connected to Group Chats. Send {prefix}group list to see them.'
    if request.chat.kind == 'shared':
        text += f' Everyone here can read what {prefix}group shows.'
    try:
        import asyncio
        asyncio.run_coroutine_threadsafe(
            request.adapter.send(request.chat.chat_id, text, metadata=metadata), loop)
    except Exception:
        pass


def _owned(runner, subject):
    from gateway.session_authorities import all_authorities
    for authority in all_authorities(runner):
        with authority.db._read_ctx() as conn:
            for grant in grants(conn):
                if grant['owner'] == subject:
                    yield authority, grant


def _list(runner, params, subject, loop):
    from gateway.group_chat_rules import applies, grant_rules
    from gateway.hosted_rooms import room_state
    chats = []
    for authority, grant in _owned(runner, subject):
        with authority.db._read_ctx() as conn:
            rules = grant_rules(conn, grant['grant_id'])
        remembered = []
        for rule in rules:
            try:
                room = room_state(authority.db.db_path, room_id=rule['room_id'])
            except Exception:
                continue  # the room is gone; the rule can never apply again
            if not applies(rule, room):
                continue
            member = next((m for m in room['members'] if m.get('member_id') == rule['member_id']), {})
            remembered.append({'group': room['name'], 'command': rule['command'], 'context': rule['context'],
                               'bot': member.get('display_name') or member.get('handle') or rule['member_id'],
                               'uses': rule['uses']})
        chats.append({'grant': grant['grant_id'][:8], 'created_at': grant['created_at'],
                      'remembered': remembered, **_view(grant, authority.profile_id)})
    return {'chats': chats}


def _revoke(runner, params, subject, loop):
    prefix = str(params.get('grant') or '').strip().lower()
    if not re.fullmatch(r'[0-9a-f]{4,64}', prefix):
        return {'error': 'unknown_grant'}
    matches = [(a, g) for a, g in _owned(runner, subject) if g['grant_id'].startswith(prefix)]
    if len(matches) != 1:
        return {'error': 'ambiguous_grant' if matches else 'unknown_grant'}
    authority, grant = matches[0]

    def write(conn):
        _epoch(conn, authority.epoch)
        current = _load(conn, grant['grant_id'])
        if current is None or current['owner'] != subject:
            raise RuntimeStoreError('unknown_grant')
        conn.execute('DELETE FROM state_meta WHERE key=?', (GRANT_PREFIX + grant['grant_id'],))
        from gateway.group_chat_rules import forget_grant
        forget_grant(conn, grant['grant_id'])  # its "always allow" approvals end with it
    authority.db._execute_write(write)
    return {'revoked': grant['grant_id'][:8], **_view(grant, authority.profile_id)}

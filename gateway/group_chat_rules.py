"""Always allow in this chat: one exact command, for one Bot, in one Group Chat, for one chat.

A rule stores the exact operation the tools layer identified for a pending request
(command, folder or SSH target and matched patterns; see ``tools.approval_operation``)
for one local member of one room under its current authority, and it belongs to the chat
grant that chose it. While that grant exists, a later request for the same operation by
the same Bot in the same room is approved once on the chat's behalf, through the same
canonical approve path a person uses. Forgetting the rule in the chat, revoking the chat,
disbanding the room, a new room authority or a changed Bot target all end it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time

from gateway import group_chat_access as access
from hermes_state_runtime import RuntimeStoreError, _epoch
from tools.approval_operation import valid_operation_context, valid_operation_key

logger = logging.getLogger(__name__)
RULE_PREFIX = 'gateway.messaging.rule.v1:'
MAX_RULES_PER_ROOM = 32  # for one chat
MAX_RULES = 1024
_SCOPE = ('room_id', 'authority_gateway_id', 'authority_epoch', 'member_id', 'profile', 'target')
_FIELDS = frozenset({*_SCOPE, 'version', 'rule_id', 'grant_id', 'operation_key', 'command', 'context',
                     'created_at', 'created_by', 'uses', 'last_used_at'})
_ATTEMPTS = 1024
_TRIES = 3  # per request: a transient failure is tried again when the driver reports it again


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def rememberable(approval) -> bool:
    return (isinstance(approval, dict) and valid_operation_key(approval.get('remember_key'))
            and valid_operation_context(approval.get('remember_context')))


def binding(room: dict, member_id: str) -> dict | None:
    """A member's current binding in a room, or None for a peer member (its own gateway decides)."""
    member = next((m for m in room['members'] if str(m.get('member_id') or m.get('profile')) == member_id), None)
    if member is None:
        return None
    target = member.get('target') or {'kind': 'local', 'profile': member.get('profile')}
    if target.get('kind') != 'local' or target.get('profile') != member.get('profile'):
        return None
    return {'room_id': room['room_id'], 'authority_gateway_id': room['authority_gateway_id'],
            'authority_epoch': room['authority_epoch'], 'member_id': member_id,
            'profile': member['profile'], 'target': _digest(target)}


def _room_prefix(room_id: str) -> str:
    return RULE_PREFIX + _digest(room_id)[:32] + ':'


def _rule(raw):
    try:
        record = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if (type(record) is not dict or set(record) != _FIELDS or type(record['version']) is not int
            or record['version'] != 1 or not valid_operation_key(record['operation_key'])
            or type(record['uses']) is not int or type(record['authority_epoch']) is not int
            or any(type(record[k]) is not str for k in ('rule_id', 'grant_id', 'room_id', 'member_id'))):
        return None
    return record


def _rules(conn, prefix: str = RULE_PREFIX) -> list[dict]:
    rows = conn.execute('SELECT key, value FROM state_meta WHERE substr(key, 1, ?) = ? ORDER BY key',
                        (len(prefix), prefix)).fetchall()
    parsed = ((key, _rule(value)) for key, value in rows)
    return [rule for key, rule in parsed
            if rule is not None and key == _room_prefix(rule['room_id']) + rule['rule_id']]


def _owner(conn, room_id: str) -> str | None:
    from gateway.session_hosted_service import _OWNER
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (_OWNER + room_id,)).fetchone()
    return row[0] if row is not None else None


def _prune(conn) -> None:
    """Drop rules whose room is gone, disbanded or under another authority (rules are never revived)."""
    for rule in _rules(conn):
        room = conn.execute('SELECT authority_gateway_id, authority_epoch, disbanded_at FROM hosted_rooms '
                            'WHERE room_id=?', (rule['room_id'],)).fetchone()
        if room is None or room[2] is not None or (room[0], room[1]) != (
                rule['authority_gateway_id'], rule['authority_epoch']):
            conn.execute('DELETE FROM state_meta WHERE key=?', (_room_prefix(rule['room_id']) + rule['rule_id'],))


def remember_approval(service, room_id, *, member_id, task_id, execution_generation, request_id, remember):
    """Approve one exact request once, then remember its operation for the chat that chose it."""
    if type(remember) is not dict or set(remember) != {'grant_id', 'by'}:
        raise RuntimeStoreError('invalid_params')
    with service._policy_lock:
        action = service._pending_actions.get((room_id, member_id))
    if action is None or (action.get('request_id'), action.get('task_id'), action.get('execution_generation')) != (
            request_id, task_id, execution_generation):
        raise RuntimeError('room approval is no longer pending')
    approval = action.get('approval')
    scope = binding(service._room(room_id), member_id)
    if not rememberable(approval) or scope is None:
        raise RuntimeStoreError('unsupported_operation')
    result = service.approve_room_task(room_id, member_id=member_id, task_id=task_id,
                                       execution_generation=execution_generation, choice='once',
                                       request_id=request_id)
    if not isinstance(result, dict) or result.get('status') != 'resolved':
        return {**(result if isinstance(result, dict) else {}), 'remembered': None}
    try:
        rule = _save(service, remember, scope, approval)
    except Exception:
        logger.warning('Approved once, but the remembered approval was not saved', exc_info=True)
        return {**result, 'remembered': None}
    return {**result, 'remembered': rule['rule_id'][:6]}


def _save(service, remember, scope, approval) -> dict:
    rule_id = _digest([remember['grant_id'], scope, approval['remember_key']])
    authority = service.authority

    def write(conn):
        _epoch(conn, authority.epoch)
        grant = access._load(conn, remember['grant_id'])
        if grant is None or grant['owner'] != _owner(conn, scope['room_id']):
            raise RuntimeStoreError('permission_denied')
        _prune(conn)
        rules = _rules(conn)
        mine = [r for r in rules if r['grant_id'] == grant['grant_id'] and r['room_id'] == scope['room_id']
                and r['rule_id'] != rule_id]
        if len(mine) >= MAX_RULES_PER_ROOM or len(rules) >= MAX_RULES:
            raise RuntimeStoreError('capacity_exhausted')
        record = {'version': 1, 'rule_id': rule_id, 'grant_id': grant['grant_id'], **scope,
                  'operation_key': approval['remember_key'], 'command': str(approval.get('command') or '')[:512],
                  'context': approval['remember_context'], 'created_at': time.time(),
                  'created_by': str(remember['by'])[:200], 'uses': 0, 'last_used_at': None}
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) '
                     'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                     (_room_prefix(scope['room_id']) + rule_id, json.dumps(record, sort_keys=True)))
        return record
    return authority.db._execute_write(write)


def apply_remembered(service, room_id, member_id, action) -> bool:
    """Called as the driver reports a pending approval: answer it once if a rule covers it."""
    approval = action.get('approval')
    if action.get('kind') != 'approval' or not rememberable(approval):
        return False
    attempt = (room_id, member_id, action.get('task_id'), action.get('execution_generation'),
               action.get('request_id'))
    with service._policy_lock:
        attempts = service._remembered_attempts
        tries = attempts.get(attempt, 0)
        if tries >= _TRIES:
            return False
        if len(attempts) >= _ATTEMPTS:
            attempts.clear()
        attempts[attempt] = _TRIES  # settled, unless a transient failure below allows another try
    try:
        scope = binding(service._room(room_id), member_id)
        rule = None if scope is None else _match(service, scope, approval['remember_key'])
        if rule is None:
            return False
        result = service.approve_room_task(
            room_id, member_id=member_id, task_id=action.get('task_id'), choice='once',
            execution_generation=action.get('execution_generation'), request_id=action.get('request_id'))
    except Exception:
        with service._policy_lock:
            attempts[attempt] = tries + 1
        logger.warning('A remembered Group Chat approval could not be applied (try %d of %d); it stays '
                       'pending', tries + 1, _TRIES, exc_info=True)
        return False
    logger.info('Remembered approval %s answered a request in room %s', rule['rule_id'][:6], room_id)
    if isinstance(result, dict) and result.get('status') == 'resolved':
        _record_use(service, rule)
    return True


def _match(service, scope, operation_key) -> dict | None:
    with service.authority.db._read_ctx() as conn:
        owner = _owner(conn, scope['room_id'])
        for rule in _rules(conn, _room_prefix(scope['room_id'])):
            if rule['operation_key'] != operation_key or any(rule[k] != scope[k] for k in _SCOPE):
                continue
            grant = access._load(conn, rule['grant_id'])
            if grant is not None and owner is not None and grant['owner'] == owner:
                return rule
    return None


def _record_use(service, rule) -> None:
    key = _room_prefix(rule['room_id']) + rule['rule_id']

    def write(conn):
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
        current = _rule(row[0]) if row is not None else None
        if current is not None:
            current.update(uses=current['uses'] + 1, last_used_at=time.time())
            conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(current, sort_keys=True), key))
    try:
        service.authority.db._execute_write(write)
    except Exception:
        logger.debug('Could not count a remembered approval use', exc_info=True)


def applies(rule: dict, room: dict) -> bool:
    """Whether a rule still matches the room as it is now (authority, member, profile and target)."""
    scope = binding(room, rule['member_id'])
    return scope is not None and all(rule[k] == scope[k] for k in _SCOPE)


def rules_for(authority, grant_id: str, room: dict) -> list[dict]:
    """This chat's rules for one room that still apply."""
    with authority.db._read_ctx() as conn:
        rules = [r for r in _rules(conn, _room_prefix(room['room_id'])) if r['grant_id'] == grant_id]
    return sorted((r for r in rules if applies(r, room)), key=lambda r: r['created_at'])


def forget_rule(authority, grant_id: str, room_id: str, code: str) -> dict | None:
    """Forget one of this chat's rules in one room by its short code; None when no single match."""
    code = code.strip().lower()

    def write(conn):
        _epoch(conn, authority.epoch)
        matches = [r for r in _rules(conn, _room_prefix(room_id))
                   if r['grant_id'] == grant_id and len(code) >= 4 and r['rule_id'].startswith(code)]
        if len(matches) != 1:
            return None
        conn.execute('DELETE FROM state_meta WHERE key=?', (_room_prefix(room_id) + matches[0]['rule_id'],))
        return matches[0]
    return authority.db._execute_write(write)


def forget_grant(conn, grant_id: str) -> None:
    """Remove every rule a grant created, inside the revoking transaction."""
    for rule in grant_rules(conn, grant_id):
        conn.execute('DELETE FROM state_meta WHERE key=?', (_room_prefix(rule['room_id']) + rule['rule_id'],))


def grant_rules(conn, grant_id: str) -> list[dict]:
    return [r for r in _rules(conn) if r['grant_id'] == grant_id]

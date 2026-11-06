"""A local, single-room DM grant. Never mint an owner principal or expose room APIs."""
import contextvars
import hashlib
import json
import logging
import re
from pathlib import Path
import threading
import time

import hermes_yaml as yaml

from gateway import hosted_rooms, session_context
from gateway.session_authorities import active_authority, served_profile_name
from gateway.session_hosted_service import CanonicalHostedRoomService
from hermes_constants import get_hermes_home
from hermes_state_runtime import RuntimeStoreError

CALL_TIMEOUT_SECONDS = 120
MAX_MESSAGE_CHARS = 8000
MAX_RESULT_CHARS = 16000

_worker_lock = threading.Lock()
_poll_lock = threading.Lock()
logger = logging.getLogger(__name__)


class RelayAuthorizationDenied(Exception):
    """Authorization failure with a non-sensitive internal stage label."""

    def __init__(self, stage):
        super().__init__(stage)
        self.stage = stage


def _bound_identity():
    # get_session_env deliberately falls back to process env: it cannot confer this grant.
    names = ('PLATFORM', 'CHAT_TYPE', 'USER_ID', 'CHAT_ID', 'PROFILE', 'KEY', 'ID', 'MESSAGE_ID')
    return {name: session_context.bound_session_value('HERMES_SESSION_' + name) for name in names}


def _identity_failure(identity, recipient, message):
    if not all(isinstance(v, str) and v for v in identity.values()):
        logger.warning('relay identity incomplete: empty=%s', sorted(n for n, v in identity.items() if not (isinstance(v, str) and v)))
        return 'bound_context_incomplete'
    if identity['PLATFORM'] != 'telegram':
        return 'wrong_platform'
    if identity['CHAT_TYPE'] != 'dm' or identity['CHAT_ID'] != identity['USER_ID']:
        return 'not_private_dm'
    if identity['PROFILE'] != 'default':
        return 'wrong_profile'
    if recipient != 'server':
        return 'recipient_not_allowed'
    if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_CHARS:
        return 'invalid_message'
    return None


def _identity_allowed(identity, recipient, message):
    return _identity_failure(identity, recipient, message) is None


def _configured_sender(identity):
    try:
        config = yaml.safe_load((get_hermes_home() / 'config.yaml').read_text()) or {}
        allowed = config.get('telegram_room_relay', {}).get('allowed_senders')
        return (isinstance(allowed, list) and len(allowed) == 1
                and isinstance(allowed[0], str) and identity.get('USER_ID') == allowed[0])
    except Exception:
        return False


def _valid_members(members):
    return (isinstance(members, list) and bool(members)
            and all(isinstance(m, str) and bool(m.strip()) for m in members)
            and len(set(members)) == len(members) and 'server' in members)


def _local_grant(authority, identity):
    config = yaml.safe_load((Path(authority.profile_id) / 'config.yaml').read_text()) or {}
    grant = config.get('telegram_room_relay', {})
    if (not isinstance(grant, dict) or grant.get('enabled') is not True
            or grant.get('profile') != 'default' or not _valid_members(grant.get('members'))
            or not isinstance(grant.get('allowed_senders'), list)
            or len(grant['allowed_senders']) != 1
            or not isinstance(grant['allowed_senders'][0], str)
            or not grant['allowed_senders'][0].strip()
            or identity['USER_ID'] != grant['allowed_senders'][0]
            or not isinstance(grant.get('room_id'), str) or not grant['room_id']):
        raise RuntimeStoreError('permission_denied')
    return grant


def _active_service(authority):
    service = getattr(authority, 'hosted_room_service', None)
    if not isinstance(service, CanonicalHostedRoomService):
        raise RuntimeStoreError('permission_denied')
    runtime = service.runtime.status()
    if (not runtime.get('running') or runtime.get('stopping')
            or Path(service.db_path).resolve() != Path(authority.db.db_path).resolve()):
        raise RuntimeStoreError('permission_denied')
    return service


def _room_members(service, room_id, approved):
    try:
        service._owned_authority(room_id)
    except Exception as exc:
        raise RelayAuthorizationDenied('room_not_owned_by_gateway') from exc
    try:
        room = service._room(room_id)
        members = room['members']
        homes = service.profile_homes()
    except Exception as exc:
        raise RelayAuthorizationDenied('room_state_unavailable') from exc
    profiles = [m.get('profile') for m in members]
    if len(profiles) != len(approved) or set(profiles) != set(approved):
        raise RelayAuthorizationDenied('room_profile_roster_mismatch')
    if any((m.get('target') or {}).get('kind', 'local') != 'local'
           or m.get('profile') not in homes for m in members):
        raise RelayAuthorizationDenied('room_member_not_local_or_served')
    if len({m['member_id'] for m in members}) != len(members):
        raise RelayAuthorizationDenied('room_member_ids_ambiguous')
    if sum(m.get('profile') == 'server' and m.get('handle') == 'server'
           for m in members) != 1:
        raise RelayAuthorizationDenied('server_member_missing_or_ambiguous')
    try:
        if not any(b.room_id == room_id for b in service.bindings()):
            raise RelayAuthorizationDenied('room_binding_missing')
    except RelayAuthorizationDenied:
        raise
    except Exception as exc:
        raise RelayAuthorizationDenied('room_binding_unavailable') from exc
    return {m['member_id']: m['profile'] for m in members}


def _authorize(runner, recipient, message):
    identity = _bound_identity()
    stage = _identity_failure(identity, recipient, message)
    if stage:
        raise RelayAuthorizationDenied(stage)
    if runner is None or getattr(runner, '_draining', False):
        raise RelayAuthorizationDenied('gateway_unavailable')
    try:
        authority = active_authority(runner)
        if (authority is None or getattr(authority, 'retiring', False)
                or served_profile_name(authority.profile_id) != 'default'
                or Path(authority.profile_id).resolve() != get_hermes_home().resolve()):
            raise RelayAuthorizationDenied('gateway_profile_mismatch')
    except RelayAuthorizationDenied:
        raise
    except Exception as exc:
        raise RelayAuthorizationDenied('gateway_profile_unavailable') from exc
    try:
        grant = _local_grant(authority, identity)
    except Exception as exc:
        raise RelayAuthorizationDenied('relay_config_or_sender_mismatch') from exc
    try:
        service = _active_service(authority)
    except Exception as exc:
        raise RelayAuthorizationDenied('hosted_room_service_unavailable') from exc
    # Canonical create_room rejects roster changes for an already-owned room; _owned_authority above
    # fails closed for legacy/unowned rooms, so no supported API can mutate this roster before send.
    try:
        members = _room_members(service, grant['room_id'], grant['members'])
    except RelayAuthorizationDenied:
        raise
    except Exception as exc:
        raise RelayAuthorizationDenied('room_state_unavailable') from exc
    # Session IDs/keys rotate on /new and compression; dedupe by the bound inbound message instead.
    stable = {name: identity[name] for name in
              ('PLATFORM', 'CHAT_TYPE', 'USER_ID', 'CHAT_ID', 'PROFILE', 'MESSAGE_ID')}
    key = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    return service, grant['room_id'], key, members, grant['members']


def _correlated_result(events, *, event_id, thread_id, members):
    matching = [e for e in events if e['payload'].get('discussion_event_id') == event_id
                and e['payload'].get('thread_id') == thread_id]
    failures = {'turn.failed', 'turn.cancelled', 'turn.deferred'}
    if any(e['kind'] in failures for e in matching):
        return {'status': 'incomplete', 'reason': 'member_not_completed'}
    terminal = next((e for e in matching if e['kind'] == 'room.activity'), None)
    if terminal is None:
        return None
    committed = {e['payload'].get('message_event_id') for e in matching
                 if e['kind'] == 'turn.settled' and e['payload'].get('member_id') in members}
    text = '\n\n'.join(f"{members[e['payload']['member_id']]}: {e['payload']['text']}" for e in matching
                       if e['kind'] == 'message.member' and e['event_id'] in committed
                       and e['payload'].get('member_id') in members
                       and isinstance(e['payload'].get('text'), str)
                       and e['payload']['text'].strip())
    complete = bool(text) and terminal['payload'].get('status') == 'settled' and len(text) <= MAX_RESULT_CHARS
    return {'status': 'completed' if complete else 'incomplete', 'text': text[:MAX_RESULT_CHARS]}


def _wait(service, room_id, event, deadline, members, approved, *, poll=False):
    cursor, correlated = event['seq'], []
    while True:
        if time.monotonic() >= deadline:
            return {'status': 'timeout', 'reason': 'discussion_not_completed'}
        # Canonical bounded replay; only this discussion survives the internal projection.
        page = hosted_rooms.read_events(service.db_path, room_id=room_id, since_seq=cursor)
        correlated.extend(e for e in page['events']
                          if e['payload'].get('discussion_event_id') == event['event_id']
                          and e['payload'].get('thread_id') == event['payload']['thread_id'])
        if len(correlated) > 64:
            return {'status': 'incomplete', 'reason': 'result_limit'}
        result = _correlated_result(correlated, event_id=event['event_id'],
            thread_id=event['payload']['thread_id'], members=members)
        if time.monotonic() >= deadline:
            return {'status': 'timeout', 'reason': 'discussion_not_completed'}
        if result is not None:
            try:
                current_members = _room_members(service, room_id, approved)
            except Exception:
                return {'status': 'incomplete', 'reason': 'room_membership_changed'}
            if current_members != members:
                return {'status': 'incomplete', 'reason': 'room_membership_changed'}
            return result
        cursor = page['cursor']
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {'status': 'timeout', 'reason': 'discussion_not_completed'}
        if not page['has_more'] and poll:
            return {'status': 'incomplete', 'reason': 'discussion_pending'}
        if not page['has_more']:
            threading.Event().wait(min(0.1, remaining))


def _submit_or_resume(service, room_id, key, message, deadline):
    event_id = hosted_rooms.user_event_id('telegram-relay:' + key)
    payload = {'thread_id': 'telegram-relay:' + key, 'text': '@server ' + message.strip()}
    if time.monotonic() >= deadline:
        raise RuntimeError('deadline expired before submission')
    # Immutable event content plus the unique room/event key is the durable retry guard.
    return service.send(room_id=room_id, event_id=event_id, payload=payload)


def _operation(runner, recipient, message, request_id, deadline, result, sync):
    try:
        service, room_id, key, members, approved = _authorize(runner, recipient, message)
    except RelayAuthorizationDenied as exc:
        identity = _bound_identity()
        denied = {'status': 'denied', 'reason': 'relay_not_authorized'}
        if _configured_sender(identity):
            logger.warning('Bot Mode relay authorization denied at stage=%s', exc.stage)
            denied['check'] = exc.stage
        elif exc.stage == 'bound_context_incomplete':
            denied['check'] = exc.stage
        return denied
    except (RuntimeStoreError, hosted_rooms.HostedRoomError, OSError, ValueError,
            TypeError, AttributeError, yaml.YAMLError):
        return {'status': 'denied', 'reason': 'relay_not_authorized'}
    key = request_id if request_id is not None else key
    with sync:
        result['request_id'] = key
    if time.monotonic() >= deadline:
        return {'status': 'timeout', 'reason': 'relay_incomplete_outcome_unknown'}
    if request_id is None:
        event = _submit_or_resume(service, room_id, key, message, deadline)
    else:
        event = hosted_rooms.get_event(service.db_path, room_id=room_id,
            event_id=hosted_rooms.user_event_id('telegram-relay:' + key))
        if event is None:
            return {'status': 'incomplete', 'reason': 'submission_pending'}
        payload = event['payload']
        if (event['kind'] != 'message.user' or payload.get('thread_id') != 'telegram-relay:' + key
                or not isinstance(payload.get('text'), str) or not payload['text'].startswith('@server ')):
            return {'status': 'denied', 'reason': 'invalid_submission'}
    return _wait(service, room_id, event, deadline, members, approved, poll=request_id is not None)


def _run(runner, *, recipient, message, request_id=None):
    deadline = time.monotonic() + CALL_TIMEOUT_SECONDS
    guard = _poll_lock if request_id is not None else _worker_lock
    if not guard.acquire(blocking=False):
        return {'status': 'busy', 'reason': 'relay_in_flight'}
    done, sync = threading.Event(), threading.Lock()
    result = {'status': 'incomplete', 'reason': 'submission_or_result_unknown'}
    completed_at = None

    def work():
        nonlocal completed_at
        outcome = {'status': 'incomplete', 'reason': 'submission_or_result_unknown'}
        try:
            outcome = _operation(runner, recipient, message, request_id, deadline, result, sync)
        except Exception:
            outcome = {'status': 'incomplete', 'reason': 'submission_or_result_unknown'}
        finally:
            with sync:
                result.update(outcome)
                completed_at = time.monotonic()
                done.set()
                guard.release()

    context = contextvars.copy_context()
    worker = threading.Thread(target=context.run, args=(work,), daemon=True,
                              name='telegram-room-relay')
    try:
        worker.start()
    except RuntimeError:
        guard.release()
        return {'status': 'incomplete', 'reason': 'worker_unavailable'}
    done.wait(max(0, deadline - time.monotonic()))
    with sync:
        if done.is_set() and completed_at <= deadline:
            return dict(result)
        timeout = {'status': 'timeout', 'reason': 'relay_incomplete_outcome_unknown'}
        if 'request_id' in result:
            timeout['request_id'] = result['request_id']
        return timeout


def delegate(runner, *, recipient, message):
    """Submit only from a bound DM; bound caller waiting even for blocked service I/O."""
    return _run(runner, recipient=recipient, message=message)


def room_result(runner, *, request_id):
    """Read only one fixed-room discussion after reauthorizing the bound DM."""
    if not isinstance(request_id, str) or re.fullmatch(r'[0-9a-f]{64}', request_id) is None:
        return {'status': 'denied', 'reason': 'invalid_request_id'}
    return _run(runner, recipient='server', message='result', request_id=request_id)

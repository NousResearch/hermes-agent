"""Authenticated RoomLink receipt reconstruction; no NEW route or custody work."""
import hmac
import json

from hermes_state_runtime import RuntimeStoreError


def normalized_room_body(dispatch, session_id, policy):
    return dict(input=dispatch.prompt, session_id=session_id,
                hosted_room_dispatch=dispatch.as_mapping(), _room_execution_policy=policy)


def _legacy_policy(adapter, record, scope, session_id, dispatch):
    from gateway.platforms.api_server_authority_runs import raw_run_admission
    from gateway.hosted_room_peer import HostedMemberDispatch
    from hermes_state_runtime import admission_fingerprint
    owned = raw_run_admission(adapter, record['run_id'])
    if owned is None:
        raise RuntimeStoreError('storage_unavailable')
    row = owned[1]
    try:
        frozen = json.loads(row['payload_json'])
        turn = frozen['api_turn_v1']
        settings = turn['settings']
        bound = HostedMemberDispatch.from_mapping(settings['room_dispatch'])
        policy = settings['room_execution_policy']
        digest = admission_fingerprint(canonical_target=session_id,
            payload={'input': frozen, 'intent': row['intent']})
        if (row['principal_id'] != 'api' or row['request_id'] != record['run_id']
                or row['target_session_id'] != session_id or row['intent'] != 'queue'
                or turn['run_owner_scope'] != scope or frozen['text'] != dispatch.prompt
                or bound.as_mapping() != dispatch.as_mapping()
                or not hmac.compare_digest(digest, row['payload_digest'])):
            raise RuntimeStoreError('admission_conflict')
    except RuntimeStoreError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeStoreError('storage_unavailable') from exc
    if policy is None:
        raise RuntimeStoreError('storage_unavailable')
    return policy


def _require_unaccepted(adapter, dispatch, session_id, scope):
    """A lost HTTP receipt cannot turn an accepted canonical attempt into NEW.

    Payload-free terminal evidence cannot identify the old logical task: hold
    that session rather than guess or recapture. No receipt repair is attempted.
    """
    from gateway.session_authorities import active_authority
    authority = active_authority(adapter.gateway_runner)
    if authority is None:
        return
    from hermes_state_terminal import ADMISSION_PREFIX
    with authority.db._read_ctx() as conn:
        rows = conn.execute("SELECT payload_json FROM session_admissions WHERE principal_id='api' "
                            'AND target_session_id=?', (session_id,)).fetchall()
        for row in rows:
            turn = json.loads(row[0]).get('api_turn_v1', {})
            # The physical member session survives Home authority changes;
            # task/generation alone must not alias a different signed scope.
            if turn.get('run_owner_scope') not in (None, scope):
                continue
            old = turn.get('settings', {}).get('room_dispatch')
            if old is None or (old.get('task_id'), old.get('execution_generation')) == (
                    dispatch.task_id, dispatch.execution_generation):
                raise RuntimeStoreError('storage_unavailable')
        for row in conn.execute('SELECT value FROM state_meta WHERE key LIKE ?', (ADMISSION_PREFIX + '%',)):
            old = json.loads(row[0])
            if old.get('principal_id') == 'api' and old.get('target_session_id') == session_id:
                raise RuntimeStoreError('storage_unavailable')


def room_replay(adapter, request, dispatch, *, _openai_error):
    """Return the exact scoped receipt or None for genuinely unaccepted work.

    Parsing, signed dispatch/status authority and both current stores must have
    been checked by the caller. The existing full normalized-body hash remains
    the sole HTTP equality predicate; neither prompt-only nor policy-only match.
    """
    from gateway.session_api import hosted_session_id
    from gateway.hosted_room_execution_policy import RoomExecutionPolicy
    from gateway.platforms.api_server_runs import _run_fingerprint, _replay_or_conflict
    session_id = hosted_session_id(dispatch)
    scope = adapter._run_idempotency_scope(request)
    key = request.headers.get('Idempotency-Key', '').strip()
    record = adapter._run_idempotency_store.replay_record(scope, key)
    if record is None:
        _require_unaccepted(adapter, dispatch, session_id, scope)
        return None
    policy = record['room_policy']
    if policy is None:
        policy = _legacy_policy(adapter, record, scope, session_id, dispatch)
    policy = RoomExecutionPolicy.from_mapping(policy).as_mapping()
    if (policy['target_profile'] != dispatch.target_profile
            or policy['policy_digest'] != dispatch.execution_policy_digest):
        raise RuntimeStoreError('admission_conflict')
    gateway_key, error = adapter._parse_session_key_header(request)
    if error is not None:
        return error
    fingerprint = _run_fingerprint(normalized_room_body(dispatch, session_id, policy), gateway_key)
    outcome = 'reused' if hmac.compare_digest(record['fingerprint'], fingerprint) else 'conflict'
    if outcome == 'reused':
        from gateway.platforms.api_server_runs import _room_retention_until
        record = adapter._run_idempotency_store.confirm_replay(
            scope, key, fingerprint, record['run_id'], retention_until=_room_retention_until(request))
        if record is None:
            raise RuntimeStoreError('storage_unavailable')
    return _replay_or_conflict(adapter, request, outcome, record, gateway_key, _openai_error,
                               receipt_identity=(scope, session_id))

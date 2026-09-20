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
    """Only prepared canonical absence permits NEW; no hot migration or repair."""
    from gateway.session_authorities import active_authority
    from hermes_state_logical_attempts import lookup_logical_attempt
    authority = active_authority(adapter.gateway_runner)
    if authority is None:
        raise RuntimeStoreError('storage_unavailable')
    accepted = lookup_logical_attempt(authority.db, principal_id='api',
        session_id=session_id, owner_scope=scope, task_id=dispatch.task_id,
        execution_generation=dispatch.execution_generation)
    if accepted is not None:
        raise RuntimeStoreError('storage_unavailable')


def room_replay(adapter, request, dispatch, *, _openai_error):
    """Return the exact scoped receipt or None for genuinely unaccepted work.

    Parsing, signed dispatch/status authority and both current stores must have
    been checked by the caller. The existing full normalized-body hash remains
    the sole HTTP equality predicate; neither prompt-only nor policy-only match.
    """
    from gateway.session_api import hosted_session_id
    from gateway.hosted_room_execution_policy import RoomExecutionPolicy
    from gateway.platforms.api_server_runs import _run_fingerprint, _replay_or_conflict, _run_receipt_store
    session_id = hosted_session_id(dispatch)
    scope = adapter._run_idempotency_scope(request)
    key = request.headers.get('Idempotency-Key', '').strip()
    store = _run_receipt_store(adapter, request=request)
    record = store.replay_record(scope, key)
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
        record = store.confirm_replay(
            scope, key, fingerprint, record['run_id'], retention_until=_room_retention_until(request))
        if record is None:
            raise RuntimeStoreError('storage_unavailable')
    return _replay_or_conflict(adapter, request, outcome, record, gateway_key, _openai_error,
                               receipt_identity=(scope, session_id))

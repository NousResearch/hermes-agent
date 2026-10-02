"""Explicit GUI group-member resumes follow their profile at an idle boundary.

The marker has the classic room_plumbing meaning: ignore stale per-session model
pins on resume. It is not a hosted-room principal or an approval bypass. Ordinary
local sessions keep their frozen policy; no transcript or prompt bytes are changed.
"""
from dataclasses import asdict, replace
import json

from hermes_state_runtime import RuntimeStoreError, _epoch, _json


def validate_create(params, *, hidden, follow):
    if 'room_plumbing' in params and type(params['room_plumbing']) is not bool:
        raise RuntimeStoreError('invalid_params')
    if not params.get('room_plumbing'):
        return
    validate_policy(params)
    if hidden is not True or (follow is not None and follow is not True):
        raise RuntimeStoreError('invalid_params')


def validate_policy(params):
    if 'room_plumbing' in params and type(params['room_plumbing']) is not bool:
        raise RuntimeStoreError('invalid_params')
    if params.get('room_plumbing') and (params.get('source') != 'gui'
            or set(params) - {'request_id', 'source', 'cwd', 'room_plumbing'}):
        raise RuntimeStoreError('invalid_params')


def legacy_plumbing(row):
    if row.get('source') != 'gui':
        return False
    try:
        config = json.loads(row.get('model_config') or '{}')
        return bool(config.get('room_plumbing') or (row.get('hidden') and str(row.get('title') or '').startswith('Group:')))
    except (ValueError, TypeError, AttributeError):
        return False


def refresh_on_resume(authority, actor, ref):
    """Only an owner with control may refresh, and only while the ledger is idle."""
    from gateway.config import Platform
    from gateway.session_policy import restore_policy, build_policy, bind_launch_key
    from hermes_state_local import local_receipt, POLICY_PREFIX
    from hermes_state_mutation_prepared import local_snapshot, validate_prepared
    live = authority.sessions.get(ref.session_id)
    if (live is None or getattr(live.source, 'platform', None) != Platform.LOCAL
            or 'session:control' not in actor.capabilities):
        return
    authority.authorize(actor, ref, 'session:control')
    saved = local_receipt(authority.db, ref.session_id)
    old = restore_policy(saved['policy'])
    params = json.loads(old.request_json)
    if old.source != 'gui' or params.get('room_plumbing') is not True:
        return
    authority._require_admission_open()
    try:
        with authority.db._read_ctx() as conn:
            snapshot = local_snapshot(authority.db, conn, ref.session_id)
    except RuntimeStoreError as error:
        if error.reason in {'session_busy', 'unknown_execution'}:
            return  # resume still observes the existing work with its original policy
        raise
    from gateway.run import _load_gateway_config, _resolve_gateway_model
    private = {}
    candidate = build_policy(params, _load_gateway_config(), private_secrets=private)
    if candidate.model is None:
        candidate = replace(candidate, model=_resolve_gateway_model(candidate.config()))
    # Only stale runtime selection is refreshed. Tool schemas, prompt settings,
    # terminal/workspace, MCP and every other launch-policy field stay frozen.
    from gateway.session_policy_credentials import recover_config_secrets
    frozen, current = old.config(), candidate.config()
    runtime_sections = {'model', 'providers', 'custom_providers', 'model_aliases'}
    reasoning_fields = {'reasoning_effort', 'reasoning_overrides'}
    for key in runtime_sections:
        if key in current:
            frozen[key] = current[key]
        else:
            frozen.pop(key, None)
    for key in reasoning_fields:
        if key in current.get('agent', {}):
            frozen.setdefault('agent', {})[key] = current['agent'][key]
        elif isinstance(frozen.get('agent'), dict):
            frozen['agent'].pop(key, None)
    runtime_secret = lambda path: path[0] in runtime_sections or (path[0] == 'agent' and len(path) > 1 and path[1] in reasoning_fields)
    # A rotated model credential need not remain available to refresh that model;
    # retained terminal/MCP credentials must still match their frozen fingerprints.
    secrets = recover_config_secrets(authority, old, include_path=lambda path: not runtime_secret(path)) if old.config_secret_ref else {}
    secrets.update({path: value for path, value in private.items() if runtime_secret(path)})
    from gateway.session_policy_credentials import PREFIX
    if old.config_secret_ref and old.config_secret_ref.startswith(PREFIX):
        order = [tuple(item[0]) for item in json.loads(old.config_secret_ref[len(PREFIX):])['entries']]
        secrets = {path: secrets[path] for path in order if path in secrets} | secrets
    encoded_config = old.config_json if frozen == json.loads(old.config_json) else json.dumps(frozen)
    candidate = replace(old, model=candidate.model, config_json=encoded_config, config_secret_ref=None)
    candidate = bind_launch_key(authority, ref.session_id, candidate, None, config_secrets=secrets)
    if candidate == old:
        return  # preserve the live agent and cached prefix on unchanged reattachment
    if not callable(getattr(authority.runner, '_evict_cached_agent', None)):
        raise RuntimeStoreError('runtime_coordination_required')

    def write(conn):
        _epoch(conn, authority.epoch)
        authority.authorize(actor, ref, 'session:control')
        current = validate_prepared(authority.db, conn, ref.session_id, {'snapshot': snapshot})
        receipt = current['receipt']
        if receipt['principal_id'] != actor.subject or receipt['profile_id'] != authority.profile_id:
            raise RuntimeStoreError('permission_denied')
        receipt['policy'] = asdict(candidate)
        conn.execute('UPDATE state_meta SET value=? WHERE key=?', (_json(receipt), POLICY_PREFIX + ref.session_id))
        # Identity, stored history and system_prompt remain byte-for-byte intact.
        conn.execute('UPDATE sessions SET model=? WHERE id=?', (candidate.model, current['target']))
        conn.execute('UPDATE sessions SET runtime_generation=runtime_generation+1,runtime_revision=runtime_revision+1 WHERE id=?',
                     (ref.session_id,))
    try:
        authority.db._execute_write(write)
    except RuntimeStoreError as error:
        if error.reason in {'session_busy', 'unknown_execution'}:
            return  # an admission won preparation; it owns the original frozen policy
        raise
    from gateway.session_local import publish_local_policy
    publish_local_policy(authority, ref.session_id)

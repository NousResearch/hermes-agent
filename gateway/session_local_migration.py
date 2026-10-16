"""Owner-only adoption of pre-authority local transcripts.

An old transcript has no authenticated subject. Only a redeemed same-profile
native control ticket may adopt it; remote login is not evidence of ownership.
The historical ID becomes the stable logical owner without copying its messages.
Compression and reset may later advance the physical target, never FIFO identity.
"""
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path

from gateway.config import Platform
from gateway.session import SessionEntry, SessionSource
from gateway.session_contract import SessionRef
from hermes_state_runtime import RuntimeStoreError


def bind_native_transport(authority, actor, identity):
    if (identity.get('provider') == 'local'
            and identity.get('profile_id') == authority.profile_id
            and identity.get('instance_id') == authority.instance_id
            and actor.profile_id == authority.profile_id
            and 'session:create' in actor.capabilities):
        if not hasattr(authority, '_native_legacy_transports'):
            authority._native_legacy_transports = {}
        authority._native_legacy_transports[actor.transport_id] = actor


def unbind_native_transport(authority, actor):
    getattr(authority, '_native_legacy_transports', {}).pop(actor.transport_id, None)


def require_history_claim(authority, conn, actor, sid):
    """Import receipts confer ownership; merely knowing an old ID does not."""
    from hermes_state_mutation_binding import BINDING_PREFIX
    row = conn.execute('SELECT session_key,chat_id,origin_json FROM sessions WHERE id=?', (sid,)).fetchone()
    if row is None or any(row):
        return
    if conn.execute('SELECT 1 FROM state_meta WHERE key=?', (BINDING_PREFIX + sid,)).fetchone():
        return  # authorize_history checks the already committed importer identity.
    if getattr(authority, '_native_legacy_transports', {}).get(actor.transport_id) != actor:
        raise RuntimeStoreError('permission_denied')


def resolve_local_target(authority, actor, sid):
    """Resolve physical history IDs without silently adopting canonical corruption."""
    db = authority.db
    row = db.get_session(sid)
    if row is None:
        raise RuntimeStoreError('not_found')
    from hermes_state_local import POLICY_PREFIX
    with db._read_ctx() as conn:
        from hermes_state_local_migration import LEGACY_PREFIX
        bound = conn.execute('SELECT 1 FROM state_meta WHERE key IN (?,?)',
            (POLICY_PREFIX + (row.get('chat_id') or sid), LEGACY_PREFIX + (row.get('chat_id') or sid))).fetchone()
    if bound or str(row.get('chat_id') or '').startswith('local-'):
        ref = SessionRef(authority.profile_id, row['chat_id'])
        authority.authorize(actor, ref, 'session:read')
        from gateway.session_local_recovery import restore_local_session
        return restore_local_session(authority, ref.session_id) if bound else ref
    return adopt_legacy_session(authority, actor, row)


def legacy_launch_params(row):
    """``(create params, api_mode)`` a legacy row resumes on: its cwd and model and, for main's
    in-process ACP server, the route it saved in ``model_config`` and rebuilt the agent from
    (``SessionManager._persist`` / ``_restore``, billing columns as its fallback). Older ACP rows
    hold the workspace only there. ``api_mode`` is no create field, so it is returned apart."""
    params = {key: row[key] for key in ('cwd', 'model') if row.get(key)}
    if row['source'] != 'acp':
        return params, None
    from hermes_state_sessions import _parse_model_config
    saved = _parse_model_config(row.get('model_config'))
    route = {'cwd': saved.get('cwd'), 'provider': saved.get('provider') or row.get('billing_provider'),
             'base_url': saved.get('base_url') or row.get('billing_base_url'), 'api_mode': saved.get('api_mode')}
    route = {key: value.strip() for key, value in route.items() if isinstance(value, str) and value.strip()}
    if 'cwd' not in params and Path(route.get('cwd', '')).is_absolute():
        params['cwd'] = route['cwd']
    params.update({key: route[key] for key in ('provider', 'base_url') if key in route})
    return params, route.get('api_mode')


def adopt_legacy_session(authority, actor, row):
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if (getattr(authority, '_native_legacy_transports', {}).get(actor.transport_id) != actor
            or not {'session:read', 'session:create'} <= actor.capabilities):
        raise RuntimeStoreError('not_found')
    from hermes_state_local_migration import LEGACY_CREATION_SOURCES, validate_legacy_row
    validate_legacy_row(authority.db, row)
    from gateway.session_local_recovery import restore_local_session
    commit_row_adoption(authority, actor, row, LEGACY_CREATION_SOURCES.get(row['source'], row['source']),
                        'legacy:' + row['id'])
    return restore_local_session(authority, row['id'])


def commit_row_adoption(authority, actor, row, creation_source, request_id, **receipt_extra):
    """Bind an existing row as *actor*'s local session under one receipt: its frozen launch policy,
    its local route and ``bind_legacy_target``'s identity rewrite commit together. The caller has
    validated the row and publishes it (``restore_local_session``)."""
    authority._require_admission_open()
    from gateway.run import _load_gateway_config, _resolve_gateway_model
    from gateway.session_policy import build_policy, bind_launch_key
    from hermes_state_local import commit_local_session
    from hermes_cli.active_sessions import active_session_liveness_guard, ActiveSessionRegistryError

    sid = row['id']
    params, api_mode = legacy_launch_params(row)
    params['source'] = creation_source
    private = {}
    policy = build_policy(params, _load_gateway_config(), private_secrets=private)
    if api_mode:
        config = json.loads(policy.config_json)
        config.setdefault('model', {})['api_mode'] = api_mode
        policy = replace(policy, config_json=json.dumps(config))
    if policy.model is None:
        policy = replace(policy, model=_resolve_gateway_model(policy.config()))
    policy = bind_launch_key(authority, sid, policy, None, config_secrets=private)
    from gateway.session_local_recovery import local_source
    source = local_source(authority, sid, actor.subject)
    route = authority.runner.session_store._generate_session_key(source)
    now = datetime.fromtimestamp(row['started_at'])
    entry = SessionEntry(route, row['id'], now, now, origin=source, platform=Platform.LOCAL)
    receipt = {'profile_id': authority.profile_id, 'principal_id': actor.subject, 'request_id': request_id,
               'session_id': sid, 'route': route, 'entry': entry.to_dict(), 'policy': asdict(policy),
               'legacy_session_id': row['id'], 'lineage': [row['id']], **receipt_extra}
    try:
        # Same lock order as old Bot Chat: mailbox (if any), liveness, SQLite.
        with active_session_liveness_guard(row['id'], registry_home=Path(authority.db.db_path).parent) as active:
            if active:
                raise RuntimeStoreError('runtime_coordination_required')
            return commit_local_session(authority.db, epoch=authority.epoch, receipt=receipt)
    except ActiveSessionRegistryError as exc:
        raise RuntimeStoreError('runtime_coordination_required') from exc

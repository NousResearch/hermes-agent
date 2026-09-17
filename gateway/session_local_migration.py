"""Owner-only adoption of pre-authority local transcripts.

An old transcript has no authenticated subject. Only a redeemed same-profile
native control ticket may adopt it; remote login is not evidence of ownership.
The historical ID becomes the stable logical owner without copying its messages.
Compression and reset may later advance the physical target, never FIFO identity.
"""
from dataclasses import asdict, replace
from datetime import datetime
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


def adopt_legacy_session(authority, actor, row):
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if (getattr(authority, '_native_legacy_transports', {}).get(actor.transport_id) != actor
            or not {'session:read', 'session:create'} <= actor.capabilities):
        raise RuntimeStoreError('not_found')
    from hermes_state_local_migration import validate_legacy_row
    validate_legacy_row(authority.db, row)
    authority._require_admission_open()
    from gateway.run import _load_gateway_config, _resolve_gateway_model
    from gateway.session_policy import build_policy, bind_launch_key
    from gateway.session_local_recovery import restore_local_session
    from hermes_state_local import commit_local_session
    from hermes_cli.active_sessions import active_session_liveness_guard, ActiveSessionRegistryError

    request_id = 'legacy:' + row['id']
    sid = row['id']
    params = {'source': row['source']}
    if row.get('cwd'):
        params['cwd'] = row['cwd']
    if row.get('model'):
        params['model'] = row['model']
    private = {}
    policy = build_policy(params, _load_gateway_config(), private_secrets=private)
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
               'legacy_session_id': row['id'], 'lineage': [row['id']]}
    try:
        # Same lock order as old Bot Chat: mailbox (if any), liveness, SQLite.
        with active_session_liveness_guard(row['id'], registry_home=Path(authority.db.db_path).parent) as active:
            if active:
                raise RuntimeStoreError('runtime_coordination_required')
            commit_local_session(authority.db, epoch=authority.epoch, receipt=receipt)
    except ActiveSessionRegistryError as exc:
        raise RuntimeStoreError('runtime_coordination_required') from exc
    return restore_local_session(authority, sid)

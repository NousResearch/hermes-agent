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
    """``(create params, api_mode)`` a legacy row resumes on.

    The old local surfaces persisted their effective route and runtime choices in
    ``model_config``. Preserve those choices while adopting them, rather than
    combining their model with a later profile default. Older ACP rows also hold
    their workspace only there. ``api_mode`` is no create field, so it is
    returned apart.
    """
    params = {key: row[key] for key in ('cwd', 'model') if row.get(key)}
    from hermes_state_sessions import _parse_model_config
    saved = _parse_model_config(row.get('model_config'))
    from hermes_state_gateway import SessionGatewayMixin
    route = SessionGatewayMixin.session_gateway_runtime(row)
    route = {key: value.strip() for key, value in route.items() if isinstance(value, str) and value.strip()}
    billing_base_url = row.get('billing_base_url')
    if 'base_url' not in route and isinstance(billing_base_url, str) and billing_base_url.strip():
        route['base_url'] = billing_base_url.strip()
        # The reader returns a bare ``custom`` billing provider with its endpoint and drops ``auto``:
        # an endpoint left without a provider is a custom route, never one to pair with the
        # profile's provider.
        route.setdefault('provider', 'custom')
    if row['source'] == 'acp' and 'cwd' not in params and Path(str(saved.get('cwd') or '')).is_absolute():
        params['cwd'] = saved['cwd']
    params.update({key: route[key] for key in ('provider', 'base_url') if key in route})
    reasoning = saved.get('reasoning_config')
    if isinstance(reasoning, dict):
        reasoning = 'none' if reasoning.get('enabled') is False else reasoning.get('effort')
    if isinstance(reasoning, str):
        from hermes_constants import parse_reasoning_effort
        if parse_reasoning_effort(reasoning) is not None:
            params['reasoning'] = reasoning
    tier = saved.get('service_tier')
    if isinstance(tier, str):
        from agent.fast_mode import parse_exact_service_tier
        try:
            parse_exact_service_tier(tier)
        except ValueError:
            pass
        else:
            params['service_tier'] = tier
    return params, route.get('api_mode')


def _endpoint_owner(saved, config):
    """The provider whose credential may serve the saved endpoint. A named entry that now points
    elsewhere owns a credential for THAT host: rebind to the entry still serving the saved URL, or
    refuse by name rather than send the moved entry's key to the old host."""
    from gateway.session_route_identity import resolved_route, same_route
    identity, entry_url = resolved_route({'provider': saved['provider']}, config)
    if not entry_url or same_route(saved, {'provider': identity, 'base_url': entry_url}, config):
        return saved['provider']
    names = list(config.get('providers') or {}) if isinstance(config.get('providers'), dict) else []
    legacy = config.get('custom_providers')
    names += [e.get('name') for e in legacy if isinstance(e, dict)] if isinstance(legacy, list) else []
    for name in names:
        owner, url = resolved_route({'provider': str(name or '')}, config)
        if url and same_route(saved, {'provider': owner, 'base_url': url}, config):
            return owner
    raise RuntimeStoreError('launch_credentials_unavailable', field='provider',
                            detail=f'{identity} no longer serves the saved endpoint; select a model to continue')


def adopt_legacy_session(authority, actor, row):
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if (getattr(authority, '_native_legacy_transports', {}).get(actor.transport_id) != actor
            or not {'session:read', 'session:create'} <= actor.capabilities):
        raise RuntimeStoreError('not_found')
    from hermes_state_local_migration import LEGACY_CREATION_SOURCES, validate_legacy_row
    validate_legacy_row(authority.db, row)
    authority._require_admission_open()
    from gateway.run import _load_gateway_config, _resolve_gateway_model
    from gateway.session_policy import build_policy, bind_launch_key
    from gateway.session_local_recovery import restore_local_session
    from hermes_state_local import commit_local_session
    from hermes_cli.active_sessions import active_session_liveness_guard, ActiveSessionRegistryError

    request_id = 'legacy:' + row['id']
    sid = row['id']
    params, api_mode = legacy_launch_params(row)
    params['source'] = LEGACY_CREATION_SOURCES.get(row['source'], row['source'])
    private = {}
    config = _load_gateway_config()
    model_cfg = config.get('model')
    if isinstance(model_cfg, dict) and ('provider' in params or 'base_url' in params):
        from gateway.session_route_identity import ROUTE_BOUND_KEYS, resolved_route, same_route
        saved = {'provider': params.get('provider', model_cfg.get('provider')), 'base_url': params.get('base_url')}
        # ``auto`` matches every route, so an ``auto`` side proves nothing about the saved endpoint.
        if ('auto' not in (resolved_route(model_cfg, config)[0], resolved_route(saved, config)[0])
                and same_route(saved, model_cfg, config)):
            # The profile still selects this endpoint: resume on the profile's route, so its own
            # credential (inline key, ${ENV} key, key_env) keeps its normal precedence.
            params.pop('provider', None)
            params.pop('base_url', None)
        else:
            # Another route than the profile's: its endpoint, wire mode, context pin and credential
            # (inline key, key_env pointer, api_key_from) stay with the profile, never paired with the
            # legacy route. ``provider`` stays: the saved route's own provider is in params, or it is
            # the profile's when the saved row carried only a base_url (resolved by _endpoint_owner).
            dropped = tuple(k for k in ROUTE_BOUND_KEYS if k != 'provider')
            config['model'] = {k: v for k, v in model_cfg.items() if k not in dropped}
            if saved['base_url']:
                params['provider'] = _endpoint_owner(saved, config)
    policy = build_policy(params, config, private_secrets=private)
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

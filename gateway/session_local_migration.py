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
    from hermes_state_mutation_binding import BINDING_PREFIX, _login_label, _owner_user
    row = conn.execute('SELECT * FROM sessions WHERE id=?', (sid,)).fetchone()
    if row is None or row['session_key'] or row['chat_id'] or row['origin_json']:
        return
    if conn.execute('SELECT 1 FROM state_meta WHERE key=?', (BINDING_PREFIX + sid,)).fetchone():
        return  # authorize_history checks the already committed importer identity.
    if _owner_user(row) and row['user_id'] in (actor.subject, _login_label(actor.subject)):
        return  # The login that created it (a served remote Desktop); authorize_history re-checks it.
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
        if bound and row.get('chat_id'):
            from hermes_state_local import local_lineage_owner
            # A copied chat_id is not evidence that this row belongs to the owner's lineage.
            if local_lineage_owner(conn, sid) != row['chat_id']:
                raise RuntimeStoreError('storage_unavailable')
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


def bot_chat_row(row):
    """Main's Bot Chat rule for a stored row: the ``follow_profile_config`` marker, or, for a row
    written before the marker existed, the exact title ``Bot Chat``."""
    from hermes_state_sessions import _parse_model_config
    saved = _parse_model_config(row.get('model_config'))
    return saved.get('follow_profile_config') is True or str(row.get('title') or '').strip() == 'Bot Chat'


def composer_pick_holds(row, config):
    """Whether *row*'s composer pick still holds: the profile (*config*) targets what it diverged from."""
    from gateway.session_policy import profile_target
    from hermes_state_sessions import _parse_model_config
    composer = _parse_model_config(row.get('model_config')).get('composer_override_profile')
    return isinstance(composer, dict) and (
        str(composer.get('model') or '').strip(), str(composer.get('provider') or '').strip(),
    ) == profile_target(config)


def legacy_bot_chat_launch(row, params, api_mode, config):
    """``(params, api_mode)`` for adoption. A Bot Chat (``bot_chat_row``) follows its profile.

    A composer pick is kept only while the profile still targets what the pick diverged from, and
    is adopted the way a create records one: held until the next profile edit. Otherwise only the
    workspace and source carry over, so the route last run is not frozen and its wire mode is not
    either. Any other row is unchanged.
    """
    if not bot_chat_row(row):
        return params, api_mode
    if not ('model' in params and composer_pick_holds(row, config)):
        params = {key: params[key] for key in ('cwd', 'source') if key in params}
        api_mode = None
    return {**params, 'follow_profile_config': True}, api_mode


def follow_bot_chat(authority, session_id, policy, live, picked=()):
    """*policy* following its profile when the session is a Bot Chat an earlier build committed
    frozen (its delivery door, a CLI titled create, an adoption), else *policy* itself.

    The caller is a client opening the chat (``session.resume``), which adopts it on the same rule
    and composer pick as adoption above; *live* is the profile's config the open read. A request
    that names ``follow_profile_config`` (``false`` included) is the client's own choice, and a
    bypass launch never reads the profile. The request is not rewritten. A committed ``/model``
    (*picked*, the session's policy edits) holds like a composer pick, and a committed ``/reasoning``
    stays the session's level, recorded as its session pin.
    The caller re-binds the private values without the followed sections, so a frozen key never
    reaches the profile's route.
    """
    from gateway.session_policy import FOLLOW_ADOPTED, SESSION_PINS, present_sections, profile_target
    if (policy.follows_profile or policy.ignore_user_config
            or 'follow_profile_config' in json.loads(policy.request_json)):
        return policy
    db = authority.db
    owner = db.get_session(session_id) or {}
    if not bot_chat_row(owner):
        # Compression can carry the canonical title to a later segment of the same conversation.
        titled = db.get_session_by_title('Bot Chat')
        if titled is None:
            return policy
        from hermes_state_local import local_conversation_ids
        with db._read_ctx() as conn:
            if titled['id'] not in local_conversation_ids(conn, session_id):
                return policy
    live = present_sections(live)
    config = json.loads(policy.config_json)
    model = config.get('model') if isinstance(config.get('model'), dict) else {}
    if composer_pick_holds(owner, live) or ('model' in picked and model.get('selection_profile') is None):
        # Recorded the way a create records a pick: it holds until the profile's model or provider moves.
        # A ``/model`` committed while the chat was frozen is such a pick too (its receipt, in *picked*).
        config.setdefault('model', {})['selection_profile'] = list(profile_target(live))
    pins = config.get(SESSION_PINS) if isinstance(config.get(SESSION_PINS), dict) else {}
    if 'reasoning' in picked and not isinstance(pins.get('reasoning'), bool):
        config[SESSION_PINS] = {**pins, 'reasoning': True}
    return replace(policy, config_json=json.dumps({**config, FOLLOW_ADOPTED: True}))


def _route_is_usage_stamp(row):
    """True when the row's only route record is the ``billing_provider='custom'`` + ``billing_base_url``
    usage stamp: no ``gateway_runtime`` and no top-level ``provider``/``base_url`` in ``model_config``.
    That stamp records which endpoint billed the last turn (the profile's route at the time), not a
    route the user picked for this session."""
    from hermes_state_sessions import _parse_model_config
    saved = _parse_model_config(row.get('model_config'))
    runtime = saved.get('gateway_runtime')
    if isinstance(runtime, dict) and runtime.get('provider'):
        return False
    if any(isinstance(saved.get(k), str) and saved[k].strip() for k in ('provider', 'base_url')):
        return False
    return str(row.get('billing_provider') or '').strip().lower() == 'custom'


def _configured_owner(saved, config):
    """The configured named entry (``providers:``/``custom_providers:``) serving the saved URL, or None."""
    from gateway.session_route_identity import resolved_route, same_route
    names = list(config.get('providers') or {}) if isinstance(config.get('providers'), dict) else []
    legacy = config.get('custom_providers')
    names += [e.get('name') for e in legacy if isinstance(e, dict)] if isinstance(legacy, list) else []
    for name in names:
        owner, url = resolved_route({'provider': str(name or '')}, config)
        if url and same_route(saved, {'provider': owner, 'base_url': url}, config):
            return owner
    return None


def _endpoint_owner(saved, config):
    """The provider whose credential may serve the saved endpoint. A named entry that now points
    elsewhere owns a credential for THAT host: rebind to the entry still serving the saved URL, or
    refuse by name rather than send the moved entry's key to the old host."""
    from gateway.session_route_identity import resolved_route, same_route
    identity, entry_url = resolved_route({'provider': saved['provider']}, config)
    if not entry_url or same_route(saved, {'provider': identity, 'base_url': entry_url}, config):
        return saved['provider']
    owner = _configured_owner(saved, config)
    if owner:
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
    validate_legacy_row(authority.db, row, subject=actor.subject)
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
    params, api_mode = legacy_bot_chat_launch(row, params, api_mode, config)
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
        elif (_route_is_usage_stamp(row) and saved['base_url']
                and resolved_route(saved, config)[0] == 'custom' and not _configured_owner(saved, config)):
            # T21/F1: the only record is the usage stamp of a bare custom endpoint that no configured
            # entry serves any more (the profile's server moved). Pre-authority resume healed that stamp
            # to nothing and used the profile route; do the same rather than freeze a dead endpoint into
            # the policy, where every turn retries it until the cycle gives up.
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
    # restore_policy rejects a blank model, and the commit below is what the next open reads.
    # A following Bot Chat drops the saved name out of the request, so a profile with no
    # default would store ''. Put a name on the receipt only: the profile's resolved model,
    # else the row's. The request is unchanged, so a later profile default still wins.
    named = str(policy.model or '').strip() or (_resolve_gateway_model(policy.config()) or '').strip()
    if not named:
        named = str(row.get('model') or '').strip()
    if not named:
        raise RuntimeStoreError('invalid_params', field='model',
                                detail='the profile names no model and the saved session has none')
    if policy.model != named:
        policy = replace(policy, model=named)
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

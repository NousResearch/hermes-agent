"""Owner-only adoption of pre-authority local transcripts.

An old transcript has no authenticated subject. Only the machine owner (a redeemed same-profile
native control ticket, or the owner signed in to the Webapp: gateway.owner_identity) may adopt it.
The historical ID becomes the stable logical owner without copying its messages.
Compression and reset may later advance the physical target, never FIFO identity.
"""
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
from urllib.parse import urlsplit

from gateway.config import Platform
from gateway.session import SessionEntry, SessionSource
from gateway.session_contract import SessionRef
from gateway.session_persistence import authority_persistence, owner_rule
from persistence.core.errors import AuthorityRefusal
from persistence.conversation.sessions import parse_model_config


def bind_native_transport(authority, actor, identity):
    # The machine owner signed in to the Webapp is the same person native Desktop is (owner_identity).
    from gateway.owner_identity import host_owner_subject
    native = (identity.get('provider') == 'local' and identity.get('profile_id') == authority.profile_id
              and identity.get('instance_id') == authority.instance_id)
    if ((native or actor.subject == host_owner_subject())
            and actor.profile_id == authority.profile_id
            and 'session:create' in actor.capabilities):
        if not hasattr(authority, '_native_legacy_transports'):
            authority._native_legacy_transports = {}
        authority._native_legacy_transports[actor.transport_id] = actor


def unbind_native_transport(authority, actor):
    getattr(authority, '_native_legacy_transports', {}).pop(actor.transport_id, None)


def legacy_row(row):
    """The columns of *row* (a ``SessionRecord``) ``validate_legacy_row`` reads, as the stored row holds them."""
    return {'source': row.source, 'user_id': row.user_id, 'session_key': row.session_key, 'chat_id': row.chat_id,
            'origin_json': None if row.origin is None else json.dumps(row.origin), 'profile_name': row.profile_name}


def resolve_local_target(authority, actor, sid):
    """Resolve physical history IDs without silently adopting canonical corruption."""
    store = authority_persistence(authority)
    row = store.sessions.get(sid)
    if row is None:
        raise AuthorityRefusal('not_found')
    marks = store.bindings.binding_marks(row.chat_id or sid, lineage_of=sid if row.chat_id else None)
    bound = marks.policy or marks.legacy
    # A copied chat_id is not evidence that this row belongs to the owner's lineage.
    if bound and row.chat_id and marks.lineage_owner != row.chat_id:
        raise AuthorityRefusal('storage_unavailable')
    if bound or str(row.chat_id or '').startswith('local-'):
        ref = SessionRef(authority.profile_id, row.chat_id)
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
    params = {key: getattr(row, key) for key in ('cwd', 'model') if getattr(row, key)}
    saved = parse_model_config(row.model_config)
    from persistence.gateway.routing import gateway_runtime
    route = gateway_runtime(row.model_config, billing_provider=row.billing_provider, billing_base_url=row.billing_base_url)
    route = {key: value.strip() for key in ('provider', 'base_url', 'api_mode')
             if isinstance(value := getattr(route, key), str) and value.strip()}
    billing_base_url = row.billing_base_url
    if 'base_url' not in route and isinstance(billing_base_url, str) and billing_base_url.strip():
        route['base_url'] = billing_base_url.strip()
        # The reader returns a bare ``custom`` billing provider with its endpoint and drops ``auto``:
        # an endpoint left without a provider is a custom route, never one to pair with the
        # profile's provider.
        route.setdefault('provider', 'custom')
    if (route.get('provider', 'custom') != 'custom'
            and urlsplit(route.get('base_url', '')).scheme not in ('', 'http', 'https')):
        # A provider's own pseudo-address (process://, moa://) is not an endpoint the session picked,
        # and replayed as a launch override it is refused (#142779, #147676). The provider is the route.
        route.pop('base_url')
    if row.source == 'acp' and 'cwd' not in params and Path(str(saved.get('cwd') or '')).is_absolute():
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
    written before the marker existed, the exact title ``Bot Chat``. False for no row (None)."""
    if row is None:
        return False
    return parse_model_config(row.model_config).get('follow_profile_config') is True or str(row.title or '').strip() == 'Bot Chat'


def composer_pick_holds(row, config):
    """Whether *row*'s composer pick still holds: the profile (*config*) targets what it diverged from."""
    from gateway.session_policy import profile_target
    composer = parse_model_config(row.model_config if row is not None else None).get('composer_override_profile')
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
    store = authority_persistence(authority)
    owner = store.sessions.get(session_id)
    if not bot_chat_row(owner):
        # Compression can carry the canonical title to a later segment of the same conversation.
        titled = store.sessions.get_by_title('Bot Chat')
        if titled is None:
            return policy
        if titled.session_id not in store.bindings.conversation_ids(session_id):
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
    saved = parse_model_config(row.model_config)
    runtime = saved.get('gateway_runtime')
    if isinstance(runtime, dict) and runtime.get('provider'):
        return False
    if any(isinstance(saved.get(k), str) and saved[k].strip() for k in ('provider', 'base_url')):
        return False
    return str(row.billing_provider or '').strip().lower() == 'custom'


def _profile_default(model_cfg):
    return str(model_cfg.get('default') or '').strip() if isinstance(model_cfg, dict) else str(model_cfg or '').strip()


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
    raise AuthorityRefusal('launch_credentials_unavailable', field='provider',
                           detail=f'{identity} no longer serves the saved endpoint; select a model to continue')


def adopts_client_history(authority, actor, row):
    """Whether this resume adopts *row* (a ``SessionRecord``), a phone app's self-labelled chat
    (``session_rules.legacy_local.client_history_row``), as a writable conversation: only for the
    native owner adoption requires, and only when it validates for that owner. A read-only view of it
    already open (a metadata read came first) is promoted in place (restore_local_session); a session
    this authority serves any other way is left alone."""
    from gateway.session_history_mount import is_history_mount
    from session_rules.legacy_local import client_history_row, validate_legacy_row
    rule = owner_rule()
    live = authority.sessions.get(row.session_id)
    if (not client_history_row(legacy_row(row), rule) or (live is not None and not is_history_mount(live))
            or getattr(authority, '_native_legacy_transports', {}).get(actor.transport_id) != actor):
        return False
    try:
        validate_legacy_row(authority_persistence(authority).sessions.store_owner(), legacy_row(row),
                            subject=actor.subject, rule=rule)
    except AuthorityRefusal:
        return False
    return True


def adopt_legacy_session(authority, actor, row):
    if actor.profile_id != authority.profile_id:
        raise AuthorityRefusal('profile_mismatch')
    if (getattr(authority, '_native_legacy_transports', {}).get(actor.transport_id) != actor
            or not {'session:read', 'session:create'} <= actor.capabilities):
        raise AuthorityRefusal('not_found')
    from session_rules.legacy_local import legacy_creation_source, validate_legacy_row
    store, rule = authority_persistence(authority), owner_rule()
    validate_legacy_row(store.sessions.store_owner(), legacy_row(row), subject=actor.subject, rule=rule)
    authority._require_admission_open()
    from gateway.run import _load_gateway_config, _resolve_gateway_model
    from gateway.session_local import commit_local_session
    from gateway.session_policy import build_policy, bind_launch_key
    from gateway.session_local_recovery import restore_local_session
    from hermes_cli.active_sessions import active_session_liveness_guard, ActiveSessionRegistryError

    request_id = 'legacy:' + row.session_id
    sid = row.session_id
    params, api_mode = legacy_launch_params(row)
    params['source'] = legacy_creation_source(legacy_row(row), rule)
    private = {}
    config = _load_gateway_config()
    params, api_mode = legacy_bot_chat_launch(row, params, api_mode, config)
    model_cfg = config.get('model')
    healed = None
    if 'provider' not in params and params.get('model') and params['model'] != _profile_default(model_cfg):
        # A row with a model but no provider would run on the profile's slot, which serves another model
        # (#132648); its saved URL/wire may be stamps of exactly those misrouted turns.
        from hermes_cli.runtime_provider_custom import empty_provider_identity
        if healed := empty_provider_identity(params.get('base_url'), params['model']):
            params['provider'] = healed
            params.pop('base_url', None)
            api_mode = None
    # A healed entry is its own route, even when it shares a relay URL with the profile's entry.
    if isinstance(model_cfg, dict) and ('provider' in params or 'base_url' in params):
        from gateway.session_route_identity import ROUTE_BOUND_KEYS, resolved_route, same_route
        saved = {'provider': params.get('provider', model_cfg.get('provider')), 'base_url': params.get('base_url')}
        # ``auto`` matches every route, so an ``auto`` side proves nothing about the saved endpoint.
        if (not healed and 'auto' not in (resolved_route(model_cfg, config)[0], resolved_route(saved, config)[0])
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
        named = str(row.model or '').strip()
    if not named:
        raise AuthorityRefusal('invalid_params', field='model',
                               detail='the profile names no model and the saved session has none')
    if policy.model != named:
        policy = replace(policy, model=named)
    policy = bind_launch_key(authority, sid, policy, None, config_secrets=private)
    from gateway.session_local_recovery import local_source
    source = local_source(authority, sid, actor.subject)
    route = authority.runner.session_store._generate_session_key(source)
    now = datetime.fromtimestamp(row.started_at)
    entry = SessionEntry(route, sid, now, now, origin=source, platform=Platform.LOCAL)
    receipt = {'profile_id': authority.profile_id, 'principal_id': actor.subject, 'request_id': request_id,
               'session_id': sid, 'route': route, 'entry': entry.to_dict(), 'policy': asdict(policy),
               'legacy_session_id': sid, 'lineage': [sid]}
    try:
        # Same lock order as old Bot Chat: mailbox (if any), liveness, SQLite.
        with active_session_liveness_guard(sid, registry_home=store.scope.profile_dir) as active:
            if active:
                raise AuthorityRefusal('runtime_coordination_required')
            commit_local_session(store, epoch=authority.epoch, receipt=receipt, owner=rule)
    except ActiveSessionRegistryError as exc:
        raise AuthorityRefusal('runtime_coordination_required') from exc
    return restore_local_session(authority, sid)

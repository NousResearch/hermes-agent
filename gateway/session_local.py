"""Fresh local routes owned by the authenticated gateway, not by a viewer.

Source and supported launch settings are frozen per route, never process env.
"""
from __future__ import annotations

import json
import logging
import uuid
from contextlib import contextmanager

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource
from gateway.session_contract import CANONICAL_GATEWAY_PROTOCOL
from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)


class LocalSessionAdapter(BasePlatformAdapter):
    """Delivery stays in the canonical event stream; no second local transport."""

    def __init__(self, authority):
        super().__init__(PlatformConfig(enabled=True), Platform.LOCAL)
        self.authority = authority
        self.sources = {}
        self.policies = {}

    def authorize_source(self, source):
        saved = self.sources.get(source.chat_id)
        if saved is None:
            return False
        original, identity = saved
        return (source is original and self._identity(source) == identity
                and any(live.source is source for live in self.authority.sessions.values()))

    @staticmethod
    def _identity(source):
        return source.platform, source.chat_id, source.user_id, source.profile, source.chat_type

    def register_source(self, source):
        self.sources[source.chat_id] = source, self._identity(source)

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        self.sources.clear()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        # TurnRunner already publishes model/tool deltas and settlement exactly once.
        return SendResult(success=True, message_id=uuid.uuid4().hex)

    async def edit_message(self, chat_id, message_id, content, *, finalize=False):
        return SendResult(success=True, message_id=message_id)

    async def _record_delivery_obligation(self, *args, **kwargs):
        # No platform ACK to lose: viewers replay the durable transcript. A ledger row would only
        # park a raw copy of the reply in state.db that no redelivery can ever use.
        return None

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {'id': chat_id, 'type': 'dm'}

    async def send_exec_approval(self, **kwargs):
        # The existing callback owns registration in PendingControls. An ACK here
        # selects that rendering path without a plaintext fallback or a second waiter.
        return SendResult(success=True, message_id=uuid.uuid4().hex)

    async def send_clarify(self, chat_id, question, choices, clarify_id, session_key, metadata=None):
        error = self._unanswerable_clarify(chat_id)
        if error is not None:
            # Acknowledging this question would park the turn on an answer nobody can submit.
            # A structured decline releases the wait at once and skips the plain-text re-ask
            # (an ordinary failure would fall back to a send() that ACKs, and the turn would
            # wait the whole clarify timeout).
            from gateway.relay.egress import EGRESS_DECLINE_CODE
            return SendResult(success=False, error=error, error_kind='forbidden',
                              raw_response={'success': False, 'code': EGRESS_DECLINE_CODE, 'error': error})
        # Local sessions keep the base prompt: a choice question also captures typed text.
        return await super().send_clarify(chat_id, question, choices, clarify_id, session_key, metadata)

    def _unanswerable_clarify(self, chat_id):
        """Why no one can answer a question asked in this chat now, or None to ask it.

        Hosted rooms (controls expose approvals only) and forwarded A2A conversations (the
        author is a remote peer with no prompt surface) never have an answering viewer. Any
        other chat can be answered by an attached viewer holding ``session:respond``; with none
        attached, a producer turn (Bot Chat DM, plugin/heartbeat injection) declines, while a
        viewer's own turn keeps waiting so its author can re-attach and answer."""
        policy = self.policies.get(chat_id)
        if policy is None:
            return None
        from gateway.session_a2a import is_forward_policy
        if policy.source == 'bot_room':
            return 'Clarification is unavailable in hosted rooms'
        if is_forward_policy(policy):
            return 'Clarification is unavailable to a forwarded peer'
        live = self.authority.sessions.get(chat_id)
        if live is None:
            return None
        with live.event_stream.lock:
            if any('session:respond' in member.capabilities for member in live.subscribers.values()):
                return None
        from hermes_state_runtime import list_session_admissions
        started = next((row for row in list_session_admissions(self.authority.db, session_id=chat_id)
                        if row['status'] == 'started'), None)
        if started is not None and 'local_automation_v1' in started['payload']:
            return 'Clarification is unavailable: no viewer can answer this chat'
        return None


def authorize_local_source(runner, source):
    """None for legacy/nonlocal routes; registered local adapters fail closed."""
    if source.platform != Platform.LOCAL:
        return None
    # A forged profile must not miss this adapter and fall into a messaging
    # allow-all policy. Local source identity is owned by this authority alone.
    from gateway.session_authorities import all_authorities
    owners = all_authorities(runner)
    adapter = runner._adapters_for_profile(source.profile).get(Platform.LOCAL)
    if not isinstance(adapter, LocalSessionAdapter):
        # A LOCAL source naming a profile this runtime does not serve must fail closed rather
        # than fall into a messaging allow-all policy; legacy runners (no authority) stay None.
        return False if owners else None
    # The adapter's authority must be one THIS runtime owns (the launch authority or a served
    # secondary's), never a stale/foreign object that happens to carry a LOCAL adapter.
    if not any(adapter.authority is owner for owner in owners):
        return False
    return adapter.authorize_source(source)


def _bypass_policy(params, *, private_secrets, client_local=False):
    """Frozen snapshot for --safe-mode / --ignore-user-config: code defaults plus the explicit
    launch options. The profile's config.yaml/.env are never read, so a malformed profile cannot
    block or shape the session; there is no default model to inherit, so one must be explicit."""
    import copy
    from gateway.session_policy import build_policy
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    if not params.get('model'):
        raise RuntimeStoreError('invalid_params', field='model',
                                detail='a bypass launch has no profile to inherit a model from')
    defaults = copy.deepcopy(DEFAULT_CONFIG)
    if not isinstance(defaults.get('model'), dict):  # code default is the empty string
        defaults['model'] = {}
    return build_policy(params, defaults, private_secrets=private_secrets, profile_terminal=False,
                        client_local=client_local)


def load_profile_config(home=None):
    """The profile's effective config.yaml for a NEW session's policy, ``${VAR}`` refs expanded against
    the profile's CURRENT ``.env`` (the source tool env refreshes from), not the env this daemon
    started with: a single-profile owner binds no secret scope and never reloads ``.env`` for local
    turns. A bound scope (a served profile's owner scope) already is that source and is kept; a miss
    on the launch profile still falls to the process env (systemd / ``op run`` injection)."""
    from pathlib import Path
    from gateway.run import _load_gateway_config
    with profile_env_scope(home):
        return _load_gateway_config(*(() if home is None else (Path(home) / 'config.yaml',)))


@contextmanager
def profile_env_scope(home=None):
    """Bind the profile's current ``.env`` (see ``load_profile_config``) unless a scope is bound."""
    from pathlib import Path
    from agent.secret_scope import build_profile_secret_scope, current_secret_scope, reset_secret_scope, set_secret_scope
    from gateway.run import _gateway_config_home
    if current_secret_scope() is not None:
        yield
        return
    home = Path(home) if home is not None else _gateway_config_home()  # default: the active gateway home
    token = set_secret_scope(build_profile_secret_scope(home), profile_home=str(home))
    try:
        yield
    finally:
        reset_secret_scope(token)


def project_plugins_warning(policy):
    """The plain notice for a session whose client opted into project plugins (its forwarded
    ``HERMES_ENABLE_PROJECT_PLUGINS``) in a folder that has some: the gateway never loads them."""
    from pathlib import Path
    from utils import is_truthy_value
    # The client's folder is a path on this host only for a client the gateway proved local.
    env = policy.client_env() if policy is not None and not policy.client_env_remote else None
    folder = Path(policy.cwd) / '.hermes' / 'plugins' if env else None
    if not env or not is_truthy_value(env.get('HERMES_ENABLE_PROJECT_PLUGINS')) or not folder.is_dir():
        return None
    try:
        if not any(folder.iterdir()):
            return None
    except OSError:  # an unlistable folder skips the optional notice, never the session
        return None
    return (f'project plugins in {folder} are not loaded: the gateway that runs Hermes sessions does not '
            'load ./.hermes/plugins, so HERMES_ENABLE_PROJECT_PLUGINS has no effect here.')


def create_local_session(authority, actor, params, *, trusted_policy=None, trusted_secrets=None, skills_prompt=None,
                         client_local=False):
    """Stage, commit and publish a local session inline (synchronous callers)."""
    from gateway.session_local_recovery import restore_local_session
    from hermes_state_local import commit_local_session
    staged = stage_local_session(authority, actor, params, trusted_policy=trusted_policy,
                                 trusted_secrets=trusted_secrets, skills_prompt=skills_prompt,
                                 client_local=client_local)
    if isinstance(staged, dict):
        commit_local_session(authority.db, epoch=authority.epoch, receipt=staged)
        staged = restore_local_session(authority, staged['session_id'])
    return staged


def staged_creation_follows(authority, actor, request_id):
    """Whether a retry of *request_id* may ask to follow the profile.

    True when nothing is staged under that identity, or the staged request already follows.
    False only when an older build committed the session without ``follow_profile_config`` and
    stopped before its title. The next attempt reuses the identity, and a different request is
    refused for good. The retry then carries the frozen request.
    """
    if not isinstance(request_id, str) or not request_id:
        return True
    from gateway.session_local_recovery import local_identity
    from hermes_state_local import local_receipt
    sid = local_identity(authority.profile_id, actor.subject, request_id)
    if authority.db.get_session(sid) is None:
        return True
    try:
        request = json.loads(local_receipt(authority.db, sid)['policy']['request_json'])
    except (KeyError, TypeError, ValueError, RuntimeStoreError):
        return True
    return isinstance(request, dict) and request.get('follow_profile_config') is True


def stage_local_session(authority, actor, params, *, trusted_policy=None, trusted_secrets=None, skills_prompt=None,
                        client_local=False):
    """Everything before the creation commit: the existing session's ref on an exact retry, else the
    receipt ``commit_local_session`` writes (the async RPC commits it off the owner loop).
    ``client_local``: the gateway's own proof that the client is on this host (its connection redeemed a
    ticket minted over the private control socket), never a value the client sends."""
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if 'session:create' not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    from gateway.session_policy import build_policy, BYPASS_FIELDS
    from gateway.run import _resolve_gateway_model
    from dataclasses import replace
    private_secrets = dict(trusted_secrets or {})
    client_key = params.get('api_key')  # the client's own; an alias's key is the owner's expansion
    if trusted_policy is not None:
        policy = trusted_policy
    elif any(params.get(name) is True for name in BYPASS_FIELDS):
        policy = _bypass_policy(params, private_secrets=private_secrets, client_local=client_local)
    else:
        from gateway.session_local_route import resolve_launch_route
        raw_model, sent = params.get('model'), params
        with profile_env_scope():  # an alias's ``key_env`` / ``${VAR}`` key reads the same .env
            config = load_profile_config()
            params = resolve_launch_route(params, config)
        # A client's session.create is the one caller whose named folder may select the coding posture.
        policy = build_policy(params, config, private_secrets=private_secrets, workspace_posture=True,
                              client_local=client_local)
        if params is not sent:  # resolved, even to a model of the alias's own name
            # Owner-stamped provenance (not create fields): a remote client cannot re-resolve the typed
            # `-m` alias against its own home, so resume and an exact retry compare a repeat against the
            # name, and ``route_derived`` names the fields the route (not the client) filled.
            import json
            derived = sorted(key for key in ('provider', 'base_url') if key in params and key not in sent)
            policy = replace(policy, request_json=json.dumps(
                {**json.loads(policy.request_json), 'model_requested': raw_model, 'route_derived': derived},
                sort_keys=True))
        if policy.model is None:
            policy = replace(policy, model=_resolve_gateway_model(policy.config()))
    request_id = params.get('request_id', uuid.uuid4().hex)
    if not isinstance(request_id, str) or not request_id or len(request_id) > 256:
        raise RuntimeStoreError('invalid_params', field='request_id',
                                detail='must be a non-empty string of at most 256 characters')
    from dataclasses import asdict
    from gateway.session import SessionEntry
    from gateway.session_lifecycle import _now
    from gateway.session_local_recovery import local_identity, restore_local_session
    authority._require_admission_open()
    sid = local_identity(authority.profile_id, actor.subject, request_id)
    from gateway.session_policy import bind_launch_key, restore_policy, launch_key
    if authority.db.get_session(sid) is not None:
        from hermes_state_local import local_receipt
        saved = local_receipt(authority.db, sid)
        original = restore_policy(saved['policy'])
        from hermes_state_local import same_create_retry
        if not same_create_retry(saved['policy'], policy.request_json, params):
            raise RuntimeStoreError('invalid_params', field='request_id',
                                    detail='a resumed request_id must carry the same frozen request')
        import hmac
        import json
        if 'model_requested' in json.loads(original.request_json):
            # An owner-resolved alias: its key came from the profile (a `.env` rotation is not a changed
            # request) and the client sent none, so one it sends now is a different request.
            if client_key is not None:
                raise RuntimeStoreError('admission_conflict')
        elif original.credential_ref is not None or params.get('api_key') is not None:
            key = launch_key(authority, original)
            if key is None or not hmac.compare_digest(key, params.get('api_key') or ''):
                raise RuntimeStoreError('admission_conflict')
        return restore_local_session(authority, sid)
    from gateway.session_policy import launch_skills
    if launch_skills(params) and skills_prompt is None:
        # `-s` is rendered off-loop by the RPC handler (render_launch_skills); a caller that
        # skipped it must not mint a session silently missing its preload.
        raise RuntimeStoreError('invalid_params', field='skills',
                                detail='the caller must render -s skills before minting the session')
    policy = replace(policy, skills_prompt=skills_prompt)
    if warning := project_plugins_warning(policy):
        logger.warning('session %s: %s', sid, warning)
    client_request = {k: params[k] for k in ('client_env',) if k in params}
    policy = bind_launch_key(authority, sid, policy, params.get("api_key"), config_secrets=private_secrets,
                             owner_key=client_key is None,  # an overlapping exact retry takes the held one
                             client_request=client_request)
    from gateway.session_local_recovery import local_source
    source = local_source(authority, sid, actor.subject)
    route = authority.runner.session_store._generate_session_key(source)
    # Same clock as every other routing entry: the recovery sweeps compare against a
    # naive cutoff, and one aware entry aborts the whole iteration.
    now = _now()
    entry = SessionEntry(route, sid, now, now, origin=source, platform=Platform.LOCAL)
    # Prefill is read now, at creation (main's CLI: at startup), and held only if this attempt commits.
    from gateway.session_worker_construct import stage_session_prefill
    staged = stage_session_prefill(authority, policy, {
        'profile_id': authority.profile_id, 'principal_id': actor.subject, 'request_id': request_id,
        'session_id': sid, 'route': route, 'entry': entry.to_dict(), 'policy': asdict(policy)})
    # Owner memory, never stored: an overlapping create's commit judges this request's own map by the
    # winner's committed rules, as a sequential retry is.
    staged.client_request = client_request
    return staged


def publish_local_policy(authority, session_id, *, evict=True):
    from hermes_state_local import local_receipt
    from gateway.session_policy import restore_policy
    live = authority.sessions[session_id]
    from gateway.session_local_recovery import local_adapter_map
    adapter = local_adapter_map(authority)[Platform.LOCAL]
    adapter.policies[live.source.chat_id] = restore_policy(local_receipt(authority.db, session_id)['policy'])
    if evict:
        authority.runner._evict_cached_agent(live.route)


def _lazy_model(authority, policy):
    """The model a not-yet-built session will run: with no provider pinned, the free tier's agent
    build pins ``nous/welcome`` (``pin_model_for_route``), so the configured default is not it."""
    import json
    model = policy.current_model(authority) if policy else None
    if (policy is None or json.loads(policy.request_json).get('model')
            or str(policy.provider or '').strip().lower() not in {'', 'auto'}):
        return model
    from gateway.session_authorities import owner_scope
    from hermes_cli.anon_auth import GUEST_MODEL, free_tier_route
    with owner_scope(authority):
        return GUEST_MODEL if free_tier_route() else model


def _reasoning_fields(agent, policy, model, authority=None, override=None):
    """Main's session-info effort/tier fields, from the built agent or (lazy) what the policy's next turn
    resolves (a following session's current profile): a resumed Desktop slice holds its reasoning pill
    pending until ``reasoning_effort`` arrives."""
    from agent.fast_mode import parse_service_tier
    from agent.voice_turn_route import session_runtime_view
    if agent is None and policy is None:
        return {}
    view = session_runtime_view(agent) if agent is not None else None
    if view is None:
        from hermes_constants import resolve_reasoning_config
        effective, next_model = policy.effective(authority, secrets=False)
        route = effective.get('model') or {}
    config = (getattr(view, 'reasoning_config', None) if view is not None
              else override if override is not None else resolve_reasoning_config(effective, next_model or ''))
    tier = (getattr(view, 'service_tier', None) if view is not None
            else parse_service_tier((effective.get('agent') or {}).get('service_tier'))) or ''
    provider = getattr(view, 'provider', None) if view is not None else route.get('provider')
    base_url = getattr(view, 'base_url', None) if view is not None else route.get('base_url')
    return effort_fields(config, tier, model, provider, base_url, getattr(view, 'api_mode', None))


def effort_fields(config, tier, model, provider, base_url, api_mode=None):
    """Session-info ``reasoning_effort`` / ``_wire`` / ``service_tier`` / ``fast`` for one resolved route."""
    from agent.fast_mode import STATIC_TIERS
    from agent.reasoning_effort import clamp_effort, route_supported_efforts
    from hermes_cli.models import resolve_fast_mode_overrides
    tier = tier or ''
    effort = ''
    if isinstance(config, dict):
        effort = 'none' if config.get('enabled') is False else str(config.get('effort') or '')
    wire = ''
    if effort and effort != 'none':
        wire = str(clamp_effort(effort, route_supported_efforts(provider, model, api_mode)) or '')
    try:
        fast = tier in STATIC_TIERS and resolve_fast_mode_overrides(
            model, provider=provider or None, base_url=base_url or None, tier=tier) is not None
    except Exception:  # health: allow BLE001 -- main's _fast_tier_applies gate: unknown means not Fast
        fast = False
    return {'reasoning_effort': effort, 'reasoning_effort_wire': wire, 'service_tier': tier, 'fast': fast}


def local_session_info(authority, ref):
    live = authority.sessions[ref.session_id]
    agent = authority.agent(ref)
    from gateway.session_authorities import served_profile_name
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(authority.runner, live.source)
    info = {'source': policy.source if policy else live.source.platform.value,
            'model': (getattr(agent, 'model', policy.current_model(authority) if policy else None) if agent is not None
                      else _lazy_model(authority, policy)),
            'lazy': agent is None,
            'profile_id': authority.profile_id, 'desktop_protocol': CANONICAL_GATEWAY_PROTOCOL,
            # Main's TUI labels a named profile's composer (``alpha ❯``) from this field.
            'profile_name': served_profile_name(authority.profile_id),
            **({'cwd': policy.cwd} if policy else {}),
            # An ACP session's edit approval policy: every attached surface reads the same one.
            **({'edit_approval_policy': policy.edit_approval_policy} if policy and policy.source == 'acp' else {})}
    peek = getattr(authority.runner, '_peek_session_state', None)
    state = peek(live.route) if peek is not None else None
    override = state.conversation.reasoning_override if state is not None else None  # a typed /reasoning
    if agent is None and policy is None and live.source.platform != Platform.LOCAL:
        # A messaging chat not built yet: what its next turn resolves (its /model and /reasoning picks
        # included), so a viewer's model and reasoning pill never wait on a field that never comes.
        from gateway.session_messaging_picks import messaging_route_info
        info.update(messaging_route_info(authority.runner, live))
    else:
        info.update(_reasoning_fields(agent, policy, info['model'], authority, override))
    if policy:
        # The creation request the route was frozen with (secrets already extracted), so a
        # resuming client can tell "same flags again" from a real override attempt.
        import json
        info['launch_request'] = json.loads(policy.request_json)
        from agent.session_client_env import NOT_LOCAL_WARNING
        # Session-start notices main's TUI already renders as ``config_warning``.
        notices = (project_plugins_warning(policy), NOT_LOCAL_WARNING if policy.client_env_remote else None)
        if warning := ' '.join(filter(None, notices)):
            info['config_warning'] = warning
    info.update(_classic_cli_fields(authority, ref, live, agent, policy))
    return info


def _classic_cli_fields(authority, ref, live, agent, policy):
    """What main's HermesCLI read off itself for /status, the status bar and the exit summary: the
    stored title and times, the session's yolo, api_mode, the loaded tool count and a masked
    credential hint (never the key). Lazy sessions report what is known without building an agent."""
    import contextlib
    stored_id = ref.session_id
    with contextlib.suppress(Exception):
        from hermes_state_local import local_receipt
        stored_id = local_receipt(authority.db, ref.session_id)['entry']['session_id'] or stored_id
    row = {}
    with contextlib.suppress(Exception):
        row = authority.db.get_session(stored_id) or {}
    yolo = bool(policy is not None and getattr(policy, 'yolo', False))
    with contextlib.suppress(Exception):
        from tools.approval import is_session_yolo_enabled
        yolo = yolo or is_session_yolo_enabled(live.route)
    api_mode = getattr(agent, 'api_mode', None) if agent is not None else None
    if api_mode is None and policy is not None:
        with contextlib.suppress(Exception):
            api_mode = (policy.effective(authority, secrets=False)[0].get('model') or {}).get('api_mode')
    hint = None
    if agent is not None and getattr(agent, 'api_key', None):
        from agent.redact import mask_secret
        hint = mask_secret(str(agent.api_key))
    tools = getattr(agent, 'tools', None) if agent is not None else None
    started = row.get('started_at')
    return {'title': row.get('title'), 'started_at': started,
            'last_active_at': row.get('last_activity_at') or started, 'yolo': yolo,
            'api_mode': api_mode or None, 'tool_count': len(tools) if isinstance(tools, list) else None,
            'credential_hint': hint}

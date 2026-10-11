"""Fresh local routes owned by the authenticated gateway, not by a viewer.

Source and supported launch settings are frozen per route, never process env.
"""
from __future__ import annotations

import uuid

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource
from gateway.session_contract import CANONICAL_GATEWAY_PROTOCOL
from hermes_state_runtime import RuntimeStoreError


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


def _bypass_policy(params, *, private_secrets):
    """Frozen snapshot for --safe-mode / --ignore-user-config: code defaults plus the explicit
    launch options. The profile's config.yaml/.env are never read, so a malformed profile cannot
    block or shape the session; there is no default model to inherit, so one must be explicit."""
    import copy
    from gateway.session_policy import build_policy
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    if not params.get('model'):
        raise RuntimeStoreError('invalid_params')
    defaults = copy.deepcopy(DEFAULT_CONFIG)
    if not isinstance(defaults.get('model'), dict):  # code default is the empty string
        defaults['model'] = {}
    return build_policy(params, defaults, private_secrets=private_secrets, profile_terminal=False)


def create_local_session(authority, actor, params, *, trusted_policy=None, trusted_secrets=None, skills_prompt=None):
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if 'session:create' not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    from gateway.session_policy import build_policy, BYPASS_FIELDS
    from gateway.run import _load_gateway_config, _resolve_gateway_model
    from dataclasses import replace
    private_secrets = dict(trusted_secrets or {})
    if trusted_policy is not None:
        policy = trusted_policy
    elif any(params.get(name) is True for name in BYPASS_FIELDS):
        policy = _bypass_policy(params, private_secrets=private_secrets)
    else:
        from gateway.session_local_route import resolve_launch_route
        config = _load_gateway_config()
        params = resolve_launch_route(params, config)
        policy = build_policy(params, config, private_secrets=private_secrets)
        if policy.model is None:
            policy = replace(policy, model=_resolve_gateway_model(policy.config()))
    request_id = params.get('request_id', uuid.uuid4().hex)
    if not isinstance(request_id, str) or not request_id or len(request_id) > 256:
        raise RuntimeStoreError('invalid_params')
    from dataclasses import asdict
    from gateway.session import SessionEntry
    from gateway.session_lifecycle import _now
    from gateway.session_local_recovery import local_identity, restore_local_session
    from hermes_state_local import commit_local_session
    authority._require_admission_open()
    sid = local_identity(authority.profile_id, actor.subject, request_id)
    from gateway.session_policy import bind_launch_key, restore_policy, launch_key
    if authority.db.get_session(sid) is not None:
        from hermes_state_local import local_receipt
        saved = local_receipt(authority.db, sid)
        original = restore_policy(saved['policy'])
        if original.request_json != policy.request_json:
            raise RuntimeStoreError('invalid_params')
        if original.credential_ref is not None or params.get('api_key') is not None:
            import hmac
            key = launch_key(authority, original)
            if key is None or not hmac.compare_digest(key, params.get('api_key') or ''):
                raise RuntimeStoreError('admission_conflict')
        return restore_local_session(authority, sid)
    from gateway.session_policy import launch_skills
    if launch_skills(params) and skills_prompt is None:
        # `-s` is rendered off-loop by the RPC handler (render_launch_skills); a caller that
        # skipped it must not mint a session silently missing its preload.
        raise RuntimeStoreError('invalid_params')
    policy = replace(policy, skills_prompt=skills_prompt)
    policy = bind_launch_key(authority, sid, policy, params.get("api_key"), config_secrets=private_secrets)
    from gateway.session_local_recovery import local_source
    source = local_source(authority, sid, actor.subject)
    route = authority.runner.session_store._generate_session_key(source)
    # Same clock as every other routing entry: the recovery sweeps compare against a
    # naive cutoff, and one aware entry aborts the whole iteration.
    now = _now()
    entry = SessionEntry(route, sid, now, now, origin=source, platform=Platform.LOCAL)
    commit_local_session(authority.db, epoch=authority.epoch, receipt={
        'profile_id': authority.profile_id, 'principal_id': actor.subject, 'request_id': request_id,
        'session_id': sid, 'route': route, 'entry': entry.to_dict(), 'policy': asdict(policy)})
    return restore_local_session(authority, sid)


def publish_local_policy(authority, session_id):
    from hermes_state_local import local_receipt
    from gateway.session_policy import restore_policy
    live = authority.sessions[session_id]
    from gateway.session_local_recovery import local_adapter_map
    adapter = local_adapter_map(authority)[Platform.LOCAL]
    adapter.policies[live.source.chat_id] = restore_policy(local_receipt(authority.db, session_id)['policy'])
    authority.runner._evict_cached_agent(live.route)


def _lazy_model(authority, policy):
    """The model a not-yet-built session will run: with no provider pinned, the free tier's agent
    build pins ``nous/welcome`` (``pin_model_for_route``), so the configured default is not it."""
    import json
    model = policy.model if policy else None
    if (policy is None or json.loads(policy.request_json).get('model')
            or str(policy.provider or '').strip().lower() not in {'', 'auto'}):
        return model
    from gateway.session_authorities import owner_scope
    from hermes_cli.anon_auth import GUEST_MODEL, free_tier_route
    with owner_scope(authority):
        return GUEST_MODEL if free_tier_route() else model


def local_session_info(authority, ref):
    live = authority.sessions[ref.session_id]
    agent = authority.agent(ref)
    from gateway.session_authorities import served_profile_name
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(authority.runner, live.source)
    info = {'source': policy.source if policy else live.source.platform.value,
            'model': (getattr(agent, 'model', policy.model if policy else None) if agent is not None
                      else _lazy_model(authority, policy)),
            'lazy': agent is None,
            'profile_id': authority.profile_id, 'desktop_protocol': CANONICAL_GATEWAY_PROTOCOL,
            # Main's TUI labels a named profile's composer (``alpha ❯``) from this field.
            'profile_name': served_profile_name(authority.profile_id),
            **({'cwd': policy.cwd} if policy else {})}
    if policy:
        # The creation request the route was frozen with (secrets already extracted), so a
        # resuming client can tell "same flags again" from a real override attempt.
        import json
        info['launch_request'] = json.loads(policy.request_json)
    return info

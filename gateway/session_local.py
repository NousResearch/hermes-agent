"""Fresh local routes owned by the authenticated gateway, not by a viewer.

Source and supported launch settings are frozen per route, never process env.
"""
from __future__ import annotations

import uuid

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource
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

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {'id': chat_id, 'type': 'dm'}

    async def send_exec_approval(self, **kwargs):
        # The existing callback owns registration in PendingControls. An ACK here
        # selects that rendering path without a plaintext fallback or a second waiter.
        return SendResult(success=True, message_id=uuid.uuid4().hex)


def authorize_local_source(runner, source):
    """None for legacy/nonlocal routes; registered local adapters fail closed."""
    if source.platform != Platform.LOCAL:
        return None
    # A forged profile must not miss this adapter and fall into a messaging
    # allow-all policy. Local source identity is owned by this authority alone.
    adapter = runner._adapters_for_profile(source.profile).get(Platform.LOCAL)
    if not isinstance(adapter, LocalSessionAdapter):
        return None
    from gateway.session_authorities import authority_for_profile_id
    owner = authority_for_profile_id(runner, adapter.authority.profile_id)
    if owner is None or adapter.authority is not owner:
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


def create_local_session(authority, actor, params, *, trusted_policy=None, trusted_secrets=None):
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
        policy = build_policy(params, _load_gateway_config(), private_secrets=private_secrets)
        if policy.model is None:
            policy = replace(policy, model=_resolve_gateway_model(policy.config()))
    request_id = params.get('request_id', uuid.uuid4().hex)
    if not isinstance(request_id, str) or not request_id or len(request_id) > 256:
        raise RuntimeStoreError('invalid_params')
    from dataclasses import asdict
    from datetime import datetime, timezone
    from gateway.session import SessionEntry
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
    policy = bind_launch_key(authority, sid, policy, params.get("api_key"), config_secrets=private_secrets)
    from gateway.session_local_recovery import local_source
    source = local_source(authority, sid, actor.subject)
    route = authority.runner.session_store._generate_session_key(source)
    now = datetime.now(timezone.utc)
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


def local_session_info(authority, ref):
    live = authority.sessions[ref.session_id]
    agent = authority.agent(ref)
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(authority.runner, live.source)
    return {'source': policy.source if policy else live.source.platform.value,
            'model': getattr(agent, 'model', policy.model if policy else None), 'lazy': agent is None,
            'profile_id': authority.profile_id, **({'cwd': policy.cwd} if policy else {})}

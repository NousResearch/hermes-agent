"""Agent-construction inputs the owner resolves for a managed worker (``construct_v1``).

The in-process turn passes these to ``AIAgent`` from the owner's runner state; a fresh worker
cannot reconstruct them (its config readers serve the frozen session snapshot, and a bypass
worker must never reload the profile), so the owner resolves them per turn, mode-appropriately,
and they ride the hydrated request beside ``turn_v1``. The child validates the closed shape.
"""
import json
from pathlib import Path

KEYS = frozenset({'fallback_model', 'ephemeral_system_prompt', 'prefill_messages'})


def construct_inputs(authority, policy, chat_id):
    """Owner side, per turn (off the owner loop: config reads)."""
    from gateway.session_authorities import owner_scope
    with owner_scope(authority):
        return {'fallback_model': _fallback_chain(authority, policy),
                'ephemeral_system_prompt': _ephemeral_prompt(authority, policy, chat_id),
                'prefill_messages': session_prefill(authority, policy, chat_id)}


def _fallback_chain(authority, policy):
    """The chain ``_build_fresh_agent`` gives an in-process turn. Ordinary sessions keep the
    profile refresh semantics (re-read per turn, last-known-good per home); a config-only session
    takes its frozen explicit snapshot (never the profile); safe mode carries none."""
    if policy.safe_mode:
        return None
    if policy.ignore_user_config:
        from hermes_cli.fallback_config import get_fallback_chain
        return get_fallback_chain(policy.config(authority)) or None
    refresh = getattr(authority.runner, '_refresh_fallback_model', None)
    return refresh() if refresh is not None else None


def _ephemeral_prompt(authority, policy, chat_id):
    """``TurnRunner``'s combined ephemeral prompt for a LOCAL turn (no platform context block):
    the gateway prompt (``agent.system_prompt`` / ``display.personality`` / channel override) plus
    the frozen ``-s`` skills blocks. API-time only, never part of the persisted system prefix.
    A bypass session never reads the profile: only its frozen launch additions apply."""
    configured = None
    resolve = getattr(authority.runner, '_get_system_prompt_for_channel', None)
    if not policy.ignore_user_config and resolve is not None:
        from gateway.config import Platform
        with policy.client_env_scope():
            configured = resolve(Platform.LOCAL, chat_id or '')
    return '\n\n'.join(p for p in (configured, policy.skills_prompt) if p) or None


def session_routing(authority, policy):
    """The session's ``provider_routing``: its own profile's, as of its creation (main's CLI read it
    once at startup). Never the runner's daemon-start copy."""
    routing = policy.config(authority, secrets=False).get('provider_routing')
    return routing if isinstance(routing, dict) else {}


def session_prefill(authority, policy, session_id):
    """The session's prefill examples, for the in-process turn and a managed worker alike: the
    snapshot ``stage_session_prefill`` took when the session was created (main's CLI read the file
    once at startup). A session this owner did not create (restored after a restart, a branch)
    reads it on first use and keeps that. A bypass session never loads it."""
    if policy.ignore_user_config:
        return None
    held = _held_prefill(authority)
    if session_id not in held:
        held.setdefault(session_id, (_read_prefill(authority, policy),))
    return held[session_id][0]


class StagedReceipt(dict):
    """A creation receipt plus owner memory ``commit_local_session`` publishes only from the
    transaction that creates the session (never persisted: the receipt itself is the dict)."""
    on_create: '_PrefillPublication | None' = None
    # The request's own ``client_env`` (stage_local_session), for an overlapping create's retry check.
    client_request: 'dict | None' = None


class _PrefillPublication:
    def __init__(self, authority, policy, session_id):
        # A box per attempt: ownership is identity, even when two attempts read the same contents.
        self.authority, self.policy, self.session_id, self.box = authority, policy, session_id, None

    def read(self):
        """This attempt's snapshot, read by its commit before the transaction (off the owner loop)."""
        self.box = (_read_prefill(self.authority, self.policy),)

    def publish(self):
        _held_prefill(self.authority)[self.session_id] = self.box

    def withdraw(self):
        held = _held_prefill(self.authority)
        if held.get(self.session_id) is self.box:
            held.pop(self.session_id, None)


def stage_session_prefill(authority, policy, receipt):
    """Stage a new session's prefill snapshot: the ``prefill_messages_file`` of its config, relative
    to its own profile home, read by this attempt's commit (``commit_local_session``, which the async
    creates run in ``tracked_write``'s worker thread), never on the owner loop. The snapshot is held
    in the authority's memory only if this attempt's commit creates the session: an overlapping
    retry or conflicting create never replaces the winner's, and a refused or failed create leaves
    nothing. Never evicted by other sessions; ``release_session_prefill`` drops it with the session."""
    staged = StagedReceipt(receipt)
    if not policy.ignore_user_config:
        staged.on_create = _PrefillPublication(authority, policy, receipt['session_id'])
    return staged


def release_session_prefill(authority, session_ids):
    held = getattr(authority, '_local_prefill', None) or {}
    for sid in session_ids:
        held.pop(str(sid), None)


def _held_prefill(authority):
    held = getattr(authority, '_local_prefill', None)
    if held is None:
        held = authority._local_prefill = {}
    return held


def _read_prefill(authority, policy):
    from hermes_cli.cli_config_load import _load_prefill_messages, _resolve_prefill_messages_file
    home = Path(str(authority.profile_id))
    if not home.is_absolute():
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    with policy.client_env_scope():  # HERMES_PREFILL_MESSAGES_FILE: a local session's client's
        path = _resolve_prefill_messages_file(policy.config(authority, secrets=False))
    return _load_prefill_messages(path, home) or None


def construct_kwargs(frame):
    """Child side: the validated ``AIAgent`` keyword arguments; a frame without the object
    (older owner) constructs exactly as before."""
    inputs = json.loads(frame['policy'].get('request_json') or '{}').get('construct_v1')
    if inputs is None:
        return {}
    if (not isinstance(inputs, dict) or set(inputs) != KEYS
            or not _dict_list(inputs['fallback_model']) or not _dict_list(inputs['prefill_messages'])
            or not isinstance(inputs['ephemeral_system_prompt'], (str, type(None)))):
        raise ValueError('invalid_managed_worker_bootstrap')
    return dict(inputs)


def _dict_list(value):
    return value is None or (isinstance(value, list) and all(isinstance(e, dict) for e in value))


def route_runtime(policy, frozen, model, key, *, fallback=True):
    """``(runtime, fallback_entry)`` for a launch-policy route: the session's FROZEN config and launch
    credential, walking its own fallback chain on an AuthError unless ``fallback=False`` (a session
    control asking which route the session itself runs). The one resolver an in-process turn
    (``_resolve_policy_agent_runtime``) and a managed worker (``worker_route``) share."""
    from hermes_cli.runtime_provider import (
        frozen_runtime_config, resolve_runtime_provider, resolve_runtime_with_fallback,
    )
    from hermes_cli.runtime_provider_custom import _resolve_named_custom_runtime
    provider = (frozen.get('model') or {}).get('provider')
    launch_url = json.loads(policy.request_json).get('base_url')
    if launch_url:
        # The creation request is immutable identity, not the current selection: once a canonical
        # model mutation moved the route off the launch endpoint, the policy's selection wins.
        from hermes_cli.route_identity import normalize_route_base_url
        if normalize_route_base_url(launch_url) != normalize_route_base_url((frozen.get('model') or {}).get('base_url')):
            launch_url = None
    # Every rung reads the session's FROZEN config, never live config.yaml: a later
    # ``model.base_url`` edit must not carry this route's frozen credential to the new host
    # (R2-M2). The auth-store pool is not config, so refresh/rotation still applies.
    with frozen_runtime_config(frozen):
        runtime = _resolve_named_custom_runtime(requested_provider=provider,
            explicit_api_key=key, explicit_base_url=launch_url,
            target_model=model, config=frozen)
        if runtime is not None:
            return runtime, None
        # Only a bound launch key is explicit: the frozen ``model.api_key`` is config, which the
        # resolver already reads from the frozen scope, so a URL-matched credential pool still
        # serves (and rotates on) the frozen endpoint.
        # Same resolution-time walker the in-process one-shot used (#81209): an AuthError
        # from the frozen primary (expired token, Portal down, exhausted pool) tries the
        # route's own ``fallback_providers`` before the turn is refused. Only a launch URL is
        # explicit (classic one-shot parity): the frozen ``model.base_url`` is resolved by
        # the provider chain, which keeps the credential pool an explicit URL would drop.
        if not fallback:
            # Takes no config argument: it reads `frozen` only through the frozen_runtime_config scope above, so it must stay inside that `with`.
            return resolve_runtime_provider(requested=provider, target_model=model,
                explicit_base_url=launch_url, explicit_api_key=key), None
        return resolve_runtime_with_fallback(frozen, requested=provider,
            explicit_api_key=key, explicit_base_url=launch_url, target_model=model)


def worker_route(policy, construct):
    """Child side: the ``api_key`` / ``base_url`` / ``api_mode`` overrides for the worker's agent, resolved
    as the in-process turn resolves its route (``route_runtime`` with the launch key, if one was bound),
    from this process's profile home (its HERMES_HOME and ``.env``). The frozen ``model.base_url`` alone
    is not the route: a per-model api_mode (opencode-go Responses models) and a resolved endpoint
    (Bedrock's regional URL) come only from resolution. A fallback entry's route belongs to another
    provider and model, so the primary never carries it; the agent's own fallback chain switches them
    together (#139155).

    A primary AuthError walks the chain the agent will actually get: the owner's per-turn
    ``construct_v1`` chain (an ordinary session's refreshed profile fallbacks) when it was sent,
    the frozen snapshot's otherwise (older owner)."""
    frozen, model = policy.effective()
    if 'fallback_model' in construct:
        frozen = {**frozen, 'fallback_providers': construct['fallback_model'] or [], 'fallback_model': None}
    runtime, fallback_entry = route_runtime(policy, frozen, model, construct['api_key'])
    if fallback_entry is not None:
        return {}
    return {'api_key': construct['api_key'] or runtime.get('api_key'), 'base_url': runtime.get('base_url'),
            'api_mode': runtime.get('api_mode')}


def checkpoint_agent_kwargs(config):
    """``AIAgent`` checkpoint arguments from a config dict (every turn: the gateway's live config, a
    managed worker's frozen session snapshot). The gateway bypasses ``load_config()``, so defaults
    are here; legacy ``checkpoints: true`` works."""
    cp_cfg = config.get("checkpoints", {}) if isinstance(config, dict) else {}
    if isinstance(cp_cfg, bool):
        cp_cfg = {"enabled": cp_cfg}
    elif not isinstance(cp_cfg, dict):
        cp_cfg = {}
    from hermes_cli.config import DEFAULT_CONFIG
    defaults = DEFAULT_CONFIG["checkpoints"]
    return {
        "checkpoints_enabled": cp_cfg.get("enabled", defaults["enabled"]),
        "checkpoint_max_snapshots": cp_cfg.get("max_snapshots", defaults["max_snapshots"]),
        "checkpoint_max_total_size_mb": cp_cfg.get("max_total_size_mb", defaults["max_total_size_mb"]),
        "checkpoint_max_file_size_mb": cp_cfg.get("max_file_size_mb", defaults["max_file_size_mb"])}


def routing_agent_kwargs(config, model, provider, base_url):
    """The provider-routing ``AIAgent`` arguments every in-process surface passes (the CLI's
    ``provider_routing`` / ``openrouter.min_coding_score`` / ``agent.service_tier``, a static fast
    tier's request overrides), from a managed worker's frozen session config."""
    from agent.fast_mode import STATIC_TIERS, parse_service_tier
    from hermes_cli.models import resolve_fast_mode_overrides
    pr = config.get('provider_routing') or {}
    score = (config.get('openrouter') or {}).get('min_coding_score')
    tier = parse_service_tier((config.get('agent') or {}).get('service_tier', ''))
    overrides = resolve_fast_mode_overrides(model, provider=provider, base_url=base_url, tier=tier) if tier in STATIC_TIERS else None
    return {'providers_allowed': pr.get('only'), 'providers_ignored': pr.get('ignore'), 'providers_order': pr.get('order'),
            'provider_sort': pr.get('sort'), 'provider_require_parameters': pr.get('require_parameters', False),
            'provider_data_collection': pr.get('data_collection'), 'service_tier': tier, 'request_overrides': overrides,
            'openrouter_min_coding_score': score if isinstance(score, (int, float)) and 0 <= score <= 1 else None}

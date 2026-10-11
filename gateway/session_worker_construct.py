"""Agent-construction inputs the owner resolves for a managed worker (``construct_v1``).

The in-process turn passes these to ``AIAgent`` from the owner's runner state; a fresh worker
cannot reconstruct them (its config readers serve the frozen session snapshot, and a bypass
worker must never reload the profile), so the owner resolves them per turn, mode-appropriately,
and they ride the hydrated request beside ``turn_v1``. The child validates the closed shape.
"""
import json

KEYS = frozenset({'fallback_model', 'ephemeral_system_prompt', 'prefill_messages'})


def construct_inputs(authority, policy, chat_id):
    """Owner side, per turn (off the owner loop: config reads)."""
    from gateway.session_authorities import owner_scope
    with owner_scope(authority):
        return {'fallback_model': _fallback_chain(authority, policy),
                'ephemeral_system_prompt': _ephemeral_prompt(authority, policy, chat_id),
                'prefill_messages': _prefill(authority, policy)}


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
        configured = resolve(Platform.LOCAL, chat_id or '')
    return '\n\n'.join(p for p in (configured, policy.skills_prompt) if p) or None


def _prefill(authority, policy):
    """The runner's prefill examples, exactly as the in-process constructor passes them. A bypass
    session never loads the profile's prefill file (and the safe policy drops prefill anyway)."""
    if policy.ignore_user_config:
        return None
    return getattr(authority.runner, '_prefill_messages', None) or None


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

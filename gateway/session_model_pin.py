"""Opt-in ``gateway.pin_session_model`` (default off): a messaging session keeps the model, provider
and reasoning config it last ran on, instead of re-reading ``model.default`` and
``agent.reasoning_effort`` from config.yaml on every message.

Without it, editing the config (or running ``hermes model``) moves every live chat to the new route
on its next message, and that switch re-sends the whole conversation uncached. With it, the edit
reaches the next session (``/new``). An explicit ``/model`` still wins, and a session whose last turn
ran on a provider fallback is not pinned, so turn-scoped fallback restore behaves as before.
"""

import copy
import json
import logging

logger = logging.getLogger(__name__)


def pin_enabled(user_config: dict | None = None) -> bool:
    from gateway.run import _load_gateway_config
    cfg = user_config if user_config is not None else _load_gateway_config()
    return bool(((cfg or {}).get("gateway") or {}).get("pin_session_model"))


def _recorded_runtime(row: dict | None) -> dict | None:
    """The ``model_config.gateway_runtime`` a finished gateway turn recorded on ``row``, else None."""
    if not row or not row.get("model"):
        return None
    raw = row.get("model_config")
    try:
        config = json.loads(raw) if isinstance(raw, str) and raw else raw
    except ValueError:
        return None
    runtime = config.get("gateway_runtime") if isinstance(config, dict) else None
    return runtime if isinstance(runtime, dict) and runtime.get("provider") else None


def _stored_route(runner, session_key: str | None) -> tuple[dict, dict] | None:
    """``(row, gateway_runtime)`` from the newest segment of the session's compression lineage that
    recorded one, when that turn ran on the session's own route (not a provider fallback)."""
    store, sdb = getattr(runner, "session_store", None), getattr(runner, "_session_db", None)
    session_id = store.peek_session_id(session_key) if session_key and store is not None else None
    if not session_id or sdb is None:
        return None
    db = sdb._db
    lineage = db.get_compression_lineage(session_id) or [session_id]
    # A rotated child starts without a recorded route; its compression parents hold the pin.
    segments = lineage[: lineage.index(session_id) + 1] if session_id in lineage else [session_id]
    for segment_id in reversed(segments):
        row = db.get_session(segment_id)
        runtime = _recorded_runtime(row)
        if runtime is not None:
            return None if runtime.get("fallback_active") else (row, runtime)
    return None


def pinned_route(runner, session_key, user_config, model: str, runtime_kwargs: dict) -> tuple[str, dict]:
    """The session's stored ``(model, runtime_kwargs)`` while pinned, else the config-resolved pair.

    Credentials are never stored: the stored provider is re-resolved for the stored model."""
    if not session_key or not pin_enabled(user_config) or runner._session_model_override(session_key) is not None:
        return model, runtime_kwargs
    stored = _stored_route(runner, session_key)
    if stored is None:
        return model, runtime_kwargs
    pinned_model, provider = str(stored[0]["model"]).strip(), str(stored[1]["provider"]).strip()
    if (pinned_model, provider) == (model, runtime_kwargs.get("provider")):
        return model, runtime_kwargs  # config still names the stored route
    if provider.lower() == "custom":
        # A named custom entry runs as the bare "custom" provider; recover its identity to re-resolve it.
        from hermes_cli.runtime_provider import canonical_custom_identity
        provider = canonical_custom_identity(base_url=stored[1].get("base_url"), model=pinned_model) or provider
    from gateway.run import _resolve_runtime_agent_kwargs_for_provider
    try:
        pinned = _resolve_runtime_agent_kwargs_for_provider(provider, target_model=pinned_model)
    except RuntimeError:
        logger.warning("Pinned session route %s via %s is unavailable; following config (%s)",
                       pinned_model, provider, model, exc_info=True)
        return model, runtime_kwargs
    pinned.pop("model", None)
    # A notice about the config route's credentials would describe a route this turn does not run.
    runner._pre_agent_fallback_notice = None
    logger.info("Session %s stays on %s via %s (gateway.pin_session_model; config names %s)",
                session_key, pinned_model, provider, model)
    return pinned_model, pinned


def pinned_reasoning_config(runner, session_key, model: str) -> dict | None:
    """The reasoning config the session last ran ``model`` with while pinned, else config's."""
    if pin_enabled():
        stored = _stored_route(runner, session_key)
        if stored is not None and stored[0].get("model") == model and "reasoning_config" in stored[1]:
            return copy.deepcopy(stored[1]["reasoning_config"])
    return runner._load_reasoning_config(model)


def recorded_reasoning(agent) -> dict:
    """Extra ``gateway_runtime`` fields a pinned session records after each turn."""
    return {"reasoning_config": copy.deepcopy(getattr(agent, "reasoning_config", None))} if pin_enabled() else {}

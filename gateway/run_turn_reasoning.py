"""Value-only plugin reasoning selection, after native session/request policy resolution."""
import json
import logging

from agent.reasoning_effort import EFFORT_LADDER
from agent.turn_reasoning import TURN_REASONING_HOOK, _config_for_route

logger = logging.getLogger(__name__)


def _provenance(turn, policy, baseline):
    from gateway.session_cli_turn import cli_turn
    from gateway.session_api_turn import api_execution
    override = cli_turn(turn._ctx.inbound_message_id).get("model_override") or {}
    api = api_execution.get()
    options = (api["settings"].get("model_options") or {}) if api else {}
    from gateway.platforms.api_server_request_options import _request_reasoning_config
    if "reasoning" in override or _request_reasoning_config(options) is not None:
        return {"config": baseline, "source": "request", "pinned": True, "raw_effort": None}
    peek = getattr(turn._runner, "_peek_session_state", None)
    state = peek(turn._ctx.session_key) if peek is not None and turn._ctx.session_key else None
    if state is not None and state.conversation.reasoning_override is not None:
        return {"config": baseline, "source": "session_override", "pinned": True, "raw_effort": None}
    if policy:
        from gateway.session_authorities import active_authority
        from gateway.session_policy import session_pinned
        config = policy.config(active_authority(turn._runner), secrets=False)
        raw = (config.get("agent") or {}).get("reasoning_effort")
        pinned = session_pinned(config, "reasoning", "reasoning" in json.loads(policy.request_json))
    else:
        raw, pinned = None, False
    return {"config": baseline, "source": "session" if pinned else "default", "pinned": pinned,
            "raw_effort": raw if isinstance(raw, (str, bool)) else None}


def _text_history(history):
    return [{"role": row["role"], "content": row["content"][:2000]} for row in (history or [])[-6:]
            if row.get("role") in {"user", "assistant"} and isinstance(row.get("content"), str)]


def select_reasoning(turn, policy, baseline, model, runtime, platform):
    """Return ``(native config, optional normalized effort)``; no mutable host handles reach plugins."""
    from hermes_cli.plugins import has_hook, invoke_hook
    if not has_hook(TURN_REASONING_HOOK):
        return baseline, None
    from hermes_constants import get_hermes_home
    ctx = turn._ctx
    provenance = _provenance(turn, policy, dict(baseline) if baseline is not None else None)
    manual_choice = provenance["source"] in {"request", "session_override"} or (
        provenance["pinned"] and provenance["raw_effort"] != "none")
    results = invoke_hook(
        TURN_REASONING_HOOK, platform=platform, model=model, provider=runtime.get("provider") or "",
        api_mode=runtime.get("api_mode") or "", profile_home=str(get_hermes_home()),
        session_id=ctx.session_id, session_key=ctx.session_key, turn_id=ctx.inbound_message_id,
        user_message=ctx.title_user_message if ctx.title_user_message is not None else ctx.message,
        conversation_history=_text_history(ctx.history), reasoning=provenance,
    )
    if manual_choice:
        return baseline, None
    for result in results:
        if isinstance(result, str) and result in EFFORT_LADDER:
            return _config_for_route(result, runtime.get("provider"), model, runtime.get("api_mode")), result
        logger.warning("Ignoring invalid %s result (expected a canonical effort string)", TURN_REASONING_HOOK)
    return baseline, None

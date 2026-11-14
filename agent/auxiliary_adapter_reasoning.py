"""Reasoning effort on the auxiliary adapters' native requests (Codex Responses, Anthropic Messages).

Aux callers speak Chat Completions, which carries effort in two shapes: OpenRouter-style
``extra_body.reasoning`` and the standard top-level ``reasoning_effort``. The ``custom`` profile
projects keyed ``providers:`` entries onto the top-level shape (strict gateways 400 on the nested
one, #75089), so a ``codex_responses`` or ``anthropic_messages`` custom provider reaches its adapter
with only ``reasoning_effort`` set. Both shapes are the same request; the nested one wins.
"""

from typing import Any, Dict, Optional


def _requested_reasoning(extra_body: Any, top_level_effort: Any) -> Optional[Dict[str, Any]]:
    """The caller's reasoning request as an ``extra_body.reasoning``-shaped dict, or None."""
    reasoning_cfg = extra_body.get("reasoning") if isinstance(extra_body, dict) else None
    if isinstance(reasoning_cfg, dict):
        return reasoning_cfg
    from agent.reasoning_effort import EFFORT_LADDER

    effort = str(top_level_effort or "").strip().lower()
    # Non-ladder values (Groq's "default") are not a reasoning level: leave the server default.
    if effort not in EFFORT_LADDER:
        return None
    return {"enabled": effort != "none", "effort": effort}


def _codex_aux_reasoning_kwargs(
    extra_body: Dict[str, Any], top_level_effort: Any, *, model: str, host: str, is_codex_backend: bool, is_xai: bool,
) -> Dict[str, Any]:
    """Responses ``reasoning``/``include`` kwargs for an aux request; ``{}`` keeps the model default."""
    reasoning_cfg = _requested_reasoning(extra_body, top_level_effort)
    if reasoning_cfg is None:
        return {}
    # Shared per-model vocabulary with the main transport ("max" only where the model publishes it; "minimal"/"ultra"
    # clamp to a listed level; ``()`` = the model takes no ``reasoning`` field at all — gpt-4o/4.1 on api.openai.com,
    # #76255). ``enabled: False`` goes on the wire as ``effort: none`` where the vocabulary has it,
    # since an omitted field leaves the model's default effort on (#75227).
    from agent.reasoning_effort import clamp_effort
    from agent.transports.codex import _codex_efforts_for_route

    supported = _codex_efforts_for_route(model, host, is_codex_backend=is_codex_backend)
    if supported and reasoning_cfg.get("enabled") is not False:
        # Truthy-only: Codex 400s on e.g. {"effort": null}, so falsy → default.
        effort = clamp_effort(reasoning_cfg.get("effort") or "medium", supported)
        return {"reasoning": {"effort": effort, "summary": "auto"}, "include": ["reasoning.encrypted_content"]}
    if "none" in supported and not is_xai:
        return {"reasoning": {"effort": "none"}}
    return {}

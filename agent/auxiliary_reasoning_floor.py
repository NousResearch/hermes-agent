"""Reasoning floor for auxiliary requests: when a route refuses to switch reasoning OFF, step it UP.

Aux lanes that want speed over thought (title generation: ``max_tokens=64``, JSON body) send the
provider's thinking-off encoding — top-level ``reasoning_effort: "none"`` on the custom profile,
``extra_body.reasoning: {"enabled": false}`` on OpenRouter-shaped relays, ``_reasoning_config`` on the
Anthropic Messages adapters. Some endpoints understand the field but refuse the disable ("Reasoning is
mandatory for this endpoint and cannot be disabled" — the Nous Portal on gpt-6-astra); before this
module every title call there 400'd and the session stayed untitled.

The recovery is a *step up*, not a strip: the same request goes out again at the lowest effort every
reasoning wire accepts (``low``; ``minimal`` is rejected by o-series and Claude), and the (route, model)
pair is memoised so the next disabled-reasoning aux call on it starts at the floor instead of burning
the guaranteed 400 first. Dropping the field would also succeed once, but it says nothing about the
next call and hands the effort choice back to the provider default (often ``medium`` or higher, the
opposite of what a thinking-off caller asked for).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

REASONING_FLOOR_EFFORT = "low"

# (route key, model) pairs that refused a reasoning disable in this process.
_FLOORED_ROUTES: set[tuple[str, str]] = set()

_DISABLED_EFFORTS = {"none", "off", "disabled", "false", "0"}


def _route_key(provider: Optional[str], base_url: Optional[str]) -> str:
    """Endpoint host:port when known (a base_url override turns a named provider into ``custom``), else
    the provider name — the same key shape ``auxiliary_structured_output`` uses."""
    return (urlparse(base_url or "").netloc or "").lower() or str(provider or "").strip().lower()


def _is_disabled(reasoning_config: Any) -> bool:
    return isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False


def floor_reasoning_config(reasoning_config: Any) -> Dict[str, Any]:
    """The caller's disabled ``reasoning_config`` lifted to the floor; anything else returned as-is."""
    if _is_disabled(reasoning_config):
        return {"enabled": True, "effort": REASONING_FLOOR_EFFORT}
    return reasoning_config


def _thinking_type(extra_body: Any) -> str:
    if not isinstance(extra_body, dict):
        return ""
    thinking = extra_body.get("thinking")
    if not isinstance(thinking, dict):
        return ""
    return str(thinking.get("type", "")).strip().lower()


def _wire_enables_reasoning(kwargs: Dict[str, Any], extra_body: Any) -> bool:
    """True when the wire already asks for thinking on, so a floor must not overwrite it."""
    effort = str(kwargs.get("reasoning_effort", "")).strip().lower()
    if effort and effort not in _DISABLED_EFFORTS:
        return True
    kind = _thinking_type(extra_body)
    if kind and kind != "disabled":
        return True
    if isinstance(extra_body, dict):
        reasoning = extra_body.get("reasoning")
        if isinstance(reasoning, dict) and not _is_disabled(reasoning):
            nested = str(reasoning.get("effort", "")).strip().lower()
            if nested not in _DISABLED_EFFORTS and (reasoning.get("enabled") is True or nested):
                return True
    return False


def _adaptive_thinking_off_fields(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Adaptive thinking+output_config at the floor, or None when *kwargs* is not that thinking-off.

    OpenAI-compat relays encode adaptive-Claude thinking-off as ``thinking.type=disabled``
    (opus 4.6) or as no thinking field at all (mandatory families). Neither shape is
    ``reasoning_effort`` / ``extra_body.reasoning`` / ``_reasoning_config``, so the floor
    ladder used to return None and a route that refuses the disable 400'd on every aux call.
    Native Messages already carries the disable on ``_reasoning_config``; injecting a second
    thinking block there would passthrough beside the adapter's own field.
    """
    model = str(kwargs.get("model") or "")
    extra_body = kwargs.get("extra_body")
    from agent.anthropic_adapter import (
        _MANDATORY_THINKING_CLAUDE_SUBSTRINGS,
        _is_claude_model,
        _model_matches,
        _supports_adaptive_thinking,
        adaptive_thinking_wire_fields,
    )

    adaptive = adaptive_thinking_wire_fields(
        {"enabled": True, "effort": REASONING_FLOOR_EFFORT}, model,
    )
    if not adaptive:
        return None
    thinking_off = _thinking_type(extra_body) == "disabled"
    mandatory_omit = (
        "_reasoning_config" not in kwargs
        and _is_claude_model(model)
        and _supports_adaptive_thinking(model)
        and _model_matches(model, _MANDATORY_THINKING_CLAUDE_SUBSTRINGS)
        and not _wire_enables_reasoning(kwargs, extra_body)
    )
    if not thinking_off and not mandatory_omit:
        return None
    return adaptive


def _merge_adaptive_floor(extra_body: Any, adaptive: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay *adaptive* onto *extra_body*, keeping a structured ``output_config.format``."""
    merged = dict(extra_body) if isinstance(extra_body, dict) else {}
    existing_output = merged.get("output_config")
    merged.pop("thinking", None)
    merged.update(adaptive)
    if isinstance(existing_output, dict):
        preserved = {key: value for key, value in existing_output.items() if key != "effort"}
        preserved.update(adaptive.get("output_config") or {})
        merged["output_config"] = preserved
    return merged


def with_reasoning_floor(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Copy of *kwargs* with every thinking-OFF encoding lifted to ``REASONING_FLOOR_EFFORT``:
    top-level ``reasoning_effort``, ``extra_body.reasoning`` (OpenRouter shape), the adapter's private
    ``_reasoning_config``, and adaptive-Claude shapes (``thinking.type=disabled``, or the empty omit
    mandatory families emit). ``None`` when nothing was disabled, so the ladder never re-sends an
    unchanged request."""
    changed = False
    retry = dict(kwargs)
    adaptive = _adaptive_thinking_off_fields(kwargs)
    effort_off = str(retry.get("reasoning_effort", "")).strip().lower() in _DISABLED_EFFORTS
    if effort_off and adaptive is None:
        retry["reasoning_effort"] = REASONING_FLOOR_EFFORT
        changed = True
    if _is_disabled(retry.get("_reasoning_config")):
        retry["_reasoning_config"] = floor_reasoning_config(retry["_reasoning_config"])
        changed = True
    extra_body = retry.get("extra_body")
    if isinstance(extra_body, dict):
        reasoning = extra_body.get("reasoning")
        if _is_disabled(reasoning) or (
            isinstance(reasoning, dict) and str(reasoning.get("effort", "")).strip().lower() in _DISABLED_EFFORTS
        ):
            retry["extra_body"] = {**extra_body, "reasoning": {"enabled": True, "effort": REASONING_FLOOR_EFFORT}}
            changed = True
    if adaptive is not None:
        # ``reasoning_effort: none`` beside adaptive thinking is the CometAPI 400 this floor
        # is recovering from; the adaptive shape is the effort knob on that wire.
        if effort_off:
            retry.pop("reasoning_effort", None)
        retry["extra_body"] = _merge_adaptive_floor(retry.get("extra_body"), adaptive)
        changed = True
    return retry if changed else None


def remember_reasoning_floor(
    provider: Optional[str], base_url: Optional[str], rejected_kwargs: Dict[str, Any], error: BaseException,
) -> None:
    """Record that this route's ``rejected_kwargs["model"]`` refuses to disable reasoning (the ladder
    calls this after the stepped-up retry succeeded)."""
    _FLOORED_ROUTES.add((_route_key(provider, base_url), str(rejected_kwargs.get("model") or "")))


_NOUS_PROVIDERS = {"nous", "nous-portal", "nousresearch"}


def _catalog_marks_mandatory(provider: Optional[str], base_url: Optional[str], model: Optional[str]) -> bool:
    """True when the route's ``/v1/models`` catalog (OpenRouter, Nous Portal) flags *model*
    ``reasoning.mandatory``. Cache-only — memory, then the disk mirror — so it never blocks; a cold
    catalog is warmed in the background, and the mirror it writes answers every later call and process.
    Without this an aux-only OpenRouter route (nothing else warms that catalog) paid the 400 in every
    process."""
    provider_norm = str(provider or "").strip().lower()
    host = (urlparse(base_url or "").hostname or "").lower()
    from hermes_cli import models_reasoning_caps as caps_mod
    if provider_norm == "openrouter" or host == "openrouter.ai" or host.endswith(".openrouter.ai"):
        lookup, warm = caps_mod.openrouter_model_reasoning_capabilities, caps_mod.warm_openrouter_reasoning_caps_async
    elif provider_norm in _NOUS_PROVIDERS:
        lookup, warm = caps_mod.nous_model_reasoning_capabilities, caps_mod.warm_nous_reasoning_caps_async
    else:
        return False
    try:
        caps = lookup(model)
        if caps is None:
            warm()
    except Exception:
        return False
    return bool(caps and caps.get("mandatory"))


def known_reasoning_floor(
    reasoning_config: Any, provider: Optional[str], base_url: Optional[str], model: Optional[str],
    task: Optional[str] = None,
) -> Any:
    """*reasoning_config* lifted to the floor when this route+model is known to refuse a disable — learned
    from an earlier 400 in this process, or flagged mandatory by the route's model catalog; unchanged
    otherwise. Runs before the profile projection so every wire shape starts at the floor."""
    if not _is_disabled(reasoning_config):
        return reasoning_config
    if (_route_key(provider, base_url), str(model or "")) not in _FLOORED_ROUTES and not _catalog_marks_mandatory(
        provider, base_url, model,
    ):
        return reasoning_config
    logger.info(
        "Auxiliary %s: %s (%s) cannot disable reasoning; sending effort=%s up front",
        task or "call", _route_key(provider, base_url) or "provider", model or "model", REASONING_FLOOR_EFFORT,
    )
    return floor_reasoning_config(reasoning_config)

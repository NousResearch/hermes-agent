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


def with_reasoning_floor(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Copy of *kwargs* with every thinking-OFF encoding lifted to ``REASONING_FLOOR_EFFORT``:
    top-level ``reasoning_effort``, ``extra_body.reasoning`` (OpenRouter shape) and the adapter's private
    ``_reasoning_config``. ``None`` when nothing was disabled, so the ladder never re-sends an unchanged
    request."""
    changed = False
    retry = dict(kwargs)
    if str(retry.get("reasoning_effort", "")).strip().lower() in _DISABLED_EFFORTS:
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
        "Auxiliary route cannot disable reasoning; sending effort=%s up front",
        REASONING_FLOOR_EFFORT,
    )
    return floor_reasoning_config(reasoning_config)

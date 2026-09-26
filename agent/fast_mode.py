"""Bounded fast-mode windows (``/fast auto`` and ``/fast cold``).

``agent.service_tier``: ``None`` (normal), ``"priority"`` (static fast, pinned into
``agent.request_overrides`` at build time), ``"auto"`` (every user turn opens a
window of ``agent.fast_auto_seconds``) or ``"cold"`` (only a session's first turn,
no prior history, opens it). The provider's fast override is layered onto request
kwargs only while the window is open; only per-request params (``service_tier`` /
``speed``) vary, so the request body stays byte-identical. Anthropic keeps a separate
prompt cache per speed, so each Anthropic window boundary re-writes the prefix at the
new speed.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

BOUNDED_MODES = frozenset({"auto", "cold"})
DEFAULT_WINDOW_SECONDS = 60
# Documented fast-mode rate-limit headers; a limit of 0 means the organization has no fast
# capacity for the model (https://platform.claude.com/docs/en/build-with-claude/fast-mode).
_FAST_LIMIT_HEADERS = ("anthropic-fast-input-tokens-limit", "anthropic-fast-output-tokens-limit")
# Every override ``resolve_fast_mode_overrides`` can pin for static ``/fast``.
_PINNED_FAST_OVERRIDES = {"speed": "fast", "service_tier": "priority"}


def begin_turn(agent: Any, conversation_history: Any) -> None:
    """Open (or refuse) the fast window at a user-turn boundary."""
    mode = getattr(agent, "service_tier", None)
    agent._fast_until = 0.0
    if mode not in BOUNDED_MODES:
        return
    if mode == "cold" and any(
        isinstance(m, dict) and m.get("role") in ("user", "assistant", "tool")
        for m in (conversation_history or ())
    ):
        return
    try:
        window = float(getattr(agent, "fast_auto_seconds", DEFAULT_WINDOW_SECONDS))
    except (TypeError, ValueError):
        window = DEFAULT_WINDOW_SECONDS
    agent._fast_until = time.monotonic() + max(window, 0.0)


def _route_fast_overrides(agent: Any) -> dict[str, Any]:
    """The fast override the shared gate allows for the agent's current model and route."""
    from hermes_cli.models import resolve_fast_mode_overrides
    base_url = getattr(agent, "base_url", None)
    if getattr(agent, "api_mode", None) == "anthropic_messages":
        base_url = getattr(agent, "_anthropic_base_url", None) or base_url
    return resolve_fast_mode_overrides(
        getattr(agent, "model", None), provider=getattr(agent, "provider", None), base_url=base_url
    ) or {}


def effective_request_overrides(agent: Any) -> dict[str, Any]:
    """``agent.request_overrides`` plus the fast override while the window is open, minus
    ``speed`` for a model this session learned has no fast capacity."""
    overrides = dict(getattr(agent, "request_overrides", None) or {})
    if getattr(agent, "service_tier", None) in BOUNDED_MODES and time.monotonic() < getattr(agent, "_fast_until", 0.0):
        overrides.update(_route_fast_overrides(agent))
    if "speed" in overrides and getattr(agent, "model", None) in (getattr(agent, "_fast_mode_unavailable_models", None) or ()):
        overrides.pop("speed", None)
    return overrides


def _pinned_fast_keys(overrides: Any) -> list[str]:
    if not isinstance(overrides, dict):
        return []
    return [key for key, value in _PINNED_FAST_OVERRIDES.items() if overrides.get(key) == value]


def regate_pinned_fast_overrides(agent: Any) -> None:
    """Re-run the fast-mode gate after the agent moved to another model/provider route.

    Static ``/fast`` pins the primary route's override (``speed`` or ``service_tier``) into
    ``agent.request_overrides``. A fallback or switched-to route must not inherit it: the
    chat-completions transport passes unknown overrides as top-level kwargs, so ``speed``
    on an OpenAI-compatible server fails every request with a ``TypeError``. Pinned values
    the new route's gate rejects are dropped. A restored primary is re-gated too: the
    switch_model snapshot keeps the pre-switch overrides (#75091).

    Only static ``/fast`` may ADD the new route's override. Without it the values came from
    config (e.g. ``delegation.request_overrides``), and swapping ``service_tier: priority``
    for Anthropic ``speed`` would switch on Fast Mode billing nobody asked for.

    While static ``/fast`` is still on, the primary snapshot counts as pinned too, so a later
    fast-capable rung of a fallback chain regains fast mode after an earlier rung dropped it.
    (``/fast off`` clears the live overrides but not the snapshot, hence the tier check.)"""
    overrides = dict(getattr(agent, "request_overrides", None) or {})
    static_fast = getattr(agent, "service_tier", None) == "priority"
    pinned = _pinned_fast_keys(overrides)
    primary = getattr(agent, "_primary_runtime", None)
    primary_pinned = (
        static_fast
        and isinstance(primary, dict)
        and bool(_pinned_fast_keys(primary.get("request_overrides")))
    )
    if not pinned and not primary_pinned:
        return
    try:
        allowed = _route_fast_overrides(agent)
    except Exception:
        # Never fail the switch over the gate: standard speed is always accepted.
        logger.debug("fast mode: gate failed for %s; continuing at standard speed",
                     getattr(agent, "model", None), exc_info=True)
        allowed = {}
    for key in pinned:
        if allowed.get(key) != overrides[key]:
            del overrides[key]
    if static_fast:
        overrides.update(allowed)
    agent.request_overrides = overrides


def fast_mode_unprovisioned(api_error: Any, api_kwargs: Any) -> bool:
    """True for a 429 on a ``speed: "fast"`` request whose fast-mode limit header is 0. The
    organization has no fast capacity for the model, so waiting or rotating keys cannot help."""
    if getattr(api_error, "status_code", None) != 429 or not isinstance(api_kwargs, dict):
        return False
    if (api_kwargs.get("extra_body") or {}).get("speed") != "fast":
        return False
    headers = getattr(getattr(api_error, "response", None), "headers", None)
    if headers is None:
        return False
    return any(str(headers.get(name, "")).strip() == "0" for name in _FAST_LIMIT_HEADERS)


def mark_fast_mode_unavailable(agent: Any) -> bool:
    """Stop sending ``speed`` for the current model for the rest of the session. False when the
    model was already marked, so the caller retries at most once per model."""
    model = getattr(agent, "model", None)
    unavailable = getattr(agent, "_fast_mode_unavailable_models", None)
    if not isinstance(unavailable, set):
        unavailable = agent._fast_mode_unavailable_models = set()
    if not model or model in unavailable:
        return False
    unavailable.add(model)
    return True

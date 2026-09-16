"""Bounded fast-mode windows (``/fast auto`` and ``/fast cold``).

``agent.service_tier``: ``None`` (normal), ``"priority"`` (static fast), ``"flex"``
(OpenRouter only), ``"auto"`` (every user turn opens a window of
``agent.fast_auto_seconds``) or ``"cold"`` (only a session's first turn, no prior
history, opens it).

Effective wire kwargs are resolved **per request** in
:func:`effective_request_overrides`: session ``/fast`` pin >
``agent.service_tier_overrides`` > global ``agent.service_tier`` >
raw user-supplied ``request_overrides`` ``service_tier``/``speed``.
Opt-in TTFT escalation overlays last. Only per-request params
(``service_tier`` / ``speed``) vary, so the prompt cache survives the
boundary. ``extra_body`` is never rewritten here.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

BOUNDED_MODES = frozenset({"auto", "cold"})
DEFAULT_WINDOW_SECONDS = 60
# * Keys the CLI/gateway/TUI loaders (and /fast) may bake into request_overrides.
TIER_WIRE_KEYS = ("service_tier", "speed")


def _agent_route(agent: Any) -> tuple[Any, Any, Any]:
    """``(model, provider, base_url)`` for the current request, Anthropic Messages URL included."""
    base_url = getattr(agent, "base_url", None)
    if getattr(agent, "api_mode", None) == "anthropic_messages":
        base_url = getattr(agent, "_anthropic_base_url", None) or base_url
    return getattr(agent, "model", None), getattr(agent, "provider", None), base_url


def logical_service_tier_source(agent: Any) -> tuple[str | None, bool]:
    """``(tier, configured)``: session pin, else config (per-model then global).

    Unpinned agents re-read config so ``/model``, fallback, cron, and delegated
    children pick the overlay for *their* model without per-surface resync.
    A session pin (including explicit ``None`` = normal) wins and is not
    inherited by children — the flag lives on this agent object only.
    *configured* is True for that pin and for an explicit ``normal`` in
    per-model or global config; empty / missing global is not a source.
    """
    from hermes_constants import parse_service_tier, resolve_service_tier_source

    if getattr(agent, "_service_tier_session_pinned", False) is True:
        return parse_service_tier(getattr(agent, "service_tier", None)), True
    agent_cfg: dict = {}
    with contextlib.suppress(Exception):
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        raw = cfg.get("agent")
        if isinstance(raw, dict):
            agent_cfg = raw
    return resolve_service_tier_source(
        agent_cfg,
        str(getattr(agent, "model", "") or ""),
        fallback=getattr(agent, "service_tier", None),
    )


def logical_service_tier(agent: Any) -> str | None:
    """Canonical tier for this request: session pin, else config (per-model then global)."""
    return logical_service_tier_source(agent)[0]


def set_framework_baked_tier_keys(agent: Any, mapped: dict[str, Any] | None) -> None:
    """Record which ``service_tier``/``speed`` keys the framework wrote onto *agent*.

    Loaders and ``/fast`` call this when they merge a resolved mapping into
    ``request_overrides``. Empty *mapped* clears the marker (explicit normal).
    """
    agent._framework_baked_tier_keys = frozenset(
        key for key in TIER_WIRE_KEYS if mapped and key in mapped
    )


def release_framework_baked_tier_keys(agent: Any) -> None:
    """Drop framework-baked wire keys from live overrides and the primary snapshot.

    Raw user keys stay. Used when a reused agent leaves a ``/fast`` session
    (CLI ``/new``). TUI/gateway ``/new`` rebuild or evict the agent instead.
    """
    baked = getattr(agent, "_framework_baked_tier_keys", None) or ()
    if baked:
        overrides = getattr(agent, "request_overrides", None)
        if isinstance(overrides, dict):
            cleaned = dict(overrides)
            for key in baked:
                cleaned.pop(key, None)
            agent.request_overrides = cleaned
        rt = getattr(agent, "_primary_runtime", None)
        if isinstance(rt, dict):
            snap = rt.get("request_overrides")
            if isinstance(snap, dict):
                cleaned_snap = dict(snap)
                for key in baked:
                    cleaned_snap.pop(key, None)
                rt["request_overrides"] = cleaned_snap
    set_framework_baked_tier_keys(agent, None)


def strip_inherited_framework_baked_overrides(
    overrides: dict[str, Any] | None,
    parent: Any,
) -> dict[str, Any]:
    """Copy *overrides* without the parent's framework-baked ``service_tier``/``speed``.

    Explicit ``delegation.request_overrides`` are merged *after* this strip so
    user-configured child keys still reach the wire unmarked.
    """
    copied = dict(overrides or {})
    baked = getattr(parent, "_framework_baked_tier_keys", None)
    if not isinstance(baked, (frozenset, set, tuple, list)):
        return copied
    for key in baked:
        copied.pop(key, None)
    return copied


def _framework_replaces_tier_keys(agent: Any) -> bool:
    """True when pin / per-model / global / an open auto-cold window owns the wire keys.

    Explicit ``normal`` is a framework source (strips raw/baked tier keys).
    Empty / missing config is not.
    """
    if getattr(agent, "_service_tier_session_pinned", False) is True:
        return True
    mode, configured = logical_service_tier_source(agent)
    if mode in BOUNDED_MODES:
        return time.monotonic() < getattr(agent, "_fast_until", 0.0)
    return configured


def begin_turn(
    agent: Any,
    conversation_history: Any,
    *,
    started_at: float | None = None,
) -> None:
    """Open (or refuse) the fast window at a user-turn boundary.

    *started_at* is the turn-start monotonic timestamp so the deadline stays
    anchored at conversation start even when this runs after primary restore.
    """
    mode = logical_service_tier(agent)
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
    origin = started_at if started_at is not None else time.monotonic()
    agent._fast_until = origin + max(window, 0.0)


def _apply_escalation_overlay(agent: Any, overrides: dict[str, Any]) -> dict[str, Any]:
    """Last-step TTFT ladder overlay. No-op when escalation is disabled or gated."""
    try:
        from agent.service_tier_escalation import apply_escalation_to_overrides

        return apply_escalation_to_overrides(agent, overrides)
    except Exception:
        return overrides


def effective_request_overrides(agent: Any) -> dict[str, Any]:
    """``agent.request_overrides`` plus the resolved service-tier wire keys.

    Framework-baked ``service_tier`` / ``speed`` (loaders, ``/fast``) are always
    stripped from the copy so a stale mapping cannot leak. User-supplied keys in
    the same dict pass through when no framework tier source applies (default
    config, API/delegation raw overrides). When a framework source does apply
    (session pin, including explicit normal; per-model overlay; global tier
    including explicit ``normal``; open auto/cold window), both keys are
    replaced from that resolution — pin > per-model > global — and then
    opt-in TTFT escalation overlays last. Explicit ``normal`` strips the
    keys (no ``service_tier`` on the wire).
    Canonical ``agent.request_overrides`` / ``agent.service_tier`` are never
    mutated. Other keys (including ``extra_body``) are copied as-is.
    """
    overrides = dict(getattr(agent, "request_overrides", None) or {})
    baked = getattr(agent, "_framework_baked_tier_keys", None) or ()
    for key in TIER_WIRE_KEYS:
        if key in baked:
            overrides.pop(key, None)
    if _framework_replaces_tier_keys(agent):
        for key in TIER_WIRE_KEYS:
            overrides.pop(key, None)
    else:
        return _apply_escalation_overlay(agent, overrides)
    mode = logical_service_tier(agent)
    model, provider, base_url = _agent_route(agent)
    if mode in BOUNDED_MODES:
        if time.monotonic() >= getattr(agent, "_fast_until", 0.0):
            return _apply_escalation_overlay(agent, overrides)
        from hermes_cli.models import resolve_fast_mode_overrides

        overrides.update(
            resolve_fast_mode_overrides(model, provider=provider, base_url=base_url) or {}
        )
        return _apply_escalation_overlay(agent, overrides)
    from hermes_cli.models import resolve_service_tier_overrides

    mapped = resolve_service_tier_overrides(
        model, mode, provider=provider, base_url=base_url,
    )
    if mapped:
        overrides.update(mapped)
    return _apply_escalation_overlay(agent, overrides)

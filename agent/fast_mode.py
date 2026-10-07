"""Bounded fast-mode windows (``/fast auto`` and ``/fast cold``).

``agent.service_tier``: ``None`` (normal), ``"priority"`` / ``"ultrafast"`` /
``"flex"`` (static tiers), ``"auto"`` (every user turn opens a window of
``agent.fast_auto_seconds``) or ``"cold"`` (only a session's first turn, no
prior history, opens it).

Effective wire kwargs are resolved **per request** in
:func:`effective_request_overrides`: session ``/fast`` pin >
``agent.service_tier_overrides`` > global ``agent.service_tier`` >
raw user-supplied ``request_overrides`` ``service_tier``/``speed``.
Opt-in TTFT escalation overlays last. Only per-request params
(``service_tier`` / ``speed``) vary, so the prompt cache survives the
boundary. ``extra_body`` is never rewritten here. Anthropic keeps a
separate prompt cache per speed, so each Anthropic window boundary
re-writes the prefix at the new speed.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

BOUNDED_MODES = frozenset({"auto", "cold"})
DEFAULT_WINDOW_SECONDS = 60
# * Keys the CLI/gateway/TUI loaders (and /fast) may bake into request_overrides.
TIER_WIRE_KEYS = ("service_tier", "speed")
# Documented fast-mode rate-limit headers; a limit of 0 means the organization has no fast
# capacity for the model (https://platform.claude.com/docs/en/build-with-claude/fast-mode).
_FAST_LIMIT_HEADERS = ("anthropic-fast-input-tokens-limit", "anthropic-fast-output-tokens-limit")
#: Tiers sent on every request of the session. Ultrafast is OpenAI-only and gated per model;
#: flex is OpenRouter-only (mapped in ``hermes_cli.models.resolve_service_tier_overrides``).
STATIC_TIERS = frozenset({"priority", "ultrafast", "flex"})
# Codex app-server names for wire tiers it accepts (turn/start.serviceTier); a tier missing here is not sent.
CODEX_TIER_WORDS: dict[str, str] = {"priority": "fast"}
NORMAL_TIER_WORDS = frozenset({"", "normal", "default", "standard", "off", "none"})
# User/config word -> agent.service_tier. The single table every surface (config loaders, /fast
# on CLI / gateway / TUI) parses through, so a new tier is one edit.
SERVICE_TIER_WORDS: dict[str, str] = {
    "fast": "priority", "priority": "priority", "on": "priority",
    "flex": "flex",
    "ultrafast": "ultrafast", "auto": "auto", "cold": "cold",
}


def parse_service_tier(raw: Any) -> str | None:
    """``agent.service_tier`` for a user/config word; None for normal and for unknown words."""
    value = str(raw or "").strip().lower()
    return None if value in NORMAL_TIER_WORDS else SERVICE_TIER_WORDS.get(value)


def parse_exact_service_tier(raw: Any) -> str:
    """Strict :func:`parse_service_tier` for an explicit client pick: ``""`` pins normal, an unknown
    word raises ``ValueError`` instead of silently reading as normal."""
    value = str(raw or "").strip().lower()
    if value not in NORMAL_TIER_WORDS and value not in SERVICE_TIER_WORDS:
        raise ValueError(f"unknown service tier: {value}")
    return parse_service_tier(value) or ""


def service_tier_word(tier: Any) -> str:
    """The user-facing word for a stored tier (``priority`` -> ``fast``, None/"" -> ``normal``)."""
    return {"priority": "fast", None: "normal", "": "normal"}.get(tier, tier)


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
    from hermes_constants import resolve_service_tier_source

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


def _drop_unprovisioned_speed(agent: Any, overrides: dict[str, Any]) -> dict[str, Any]:
    """Strip ``speed`` when this session learned the model has no fast capacity."""
    if "speed" in overrides and getattr(agent, "model", None) in (
        getattr(agent, "_fast_mode_unavailable_models", None) or ()
    ):
        overrides.pop("speed", None)
    return overrides


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
    Models this session learned have no Anthropic fast capacity lose ``speed``.
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
        return _apply_escalation_overlay(agent, _drop_unprovisioned_speed(agent, overrides))
    mode = logical_service_tier(agent)
    model, provider, base_url = _agent_route(agent)
    if mode in BOUNDED_MODES:
        if time.monotonic() >= getattr(agent, "_fast_until", 0.0):
            return _apply_escalation_overlay(agent, _drop_unprovisioned_speed(agent, overrides))
        from hermes_cli.models import resolve_fast_mode_overrides

        overrides.update(
            resolve_fast_mode_overrides(model, provider=provider, base_url=base_url) or {}
        )
        return _apply_escalation_overlay(agent, _drop_unprovisioned_speed(agent, overrides))
    from hermes_cli.models import resolve_service_tier_overrides

    mapped = resolve_service_tier_overrides(
        model, mode, provider=provider, base_url=base_url,
    )
    if mapped:
        overrides.update(mapped)
    return _apply_escalation_overlay(agent, _drop_unprovisioned_speed(agent, overrides))


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

"""Usage-aware routing for the MAIN interactive orchestrator model.

This is deliberately narrow in scope: it decides which native provider/model
*drives the top-level conversation* (the CLI's or gateway's main agent loop),
not which model a delegated sub-agent runs on (see
:mod:`agent.delegation_routing` for that). The two concerns share the same
usage-cache substrate (:mod:`agent.delegation_usage_cache`) but are
independent knobs: an operator may run the orchestrator on Codex while
delegating sub-tasks anywhere, or vice versa.

Design constraints (mirrored from ``agent.delegation_routing``):

* Pure and side-effect-free. This module takes an already-resolved
  :class:`agent.delegation_routing.ProviderUsage` reading and already-known
  "what provider is currently active" state, and returns a deterministic
  :class:`OrchestratorRouteDecision`. It never touches the network, the
  filesystem, or credentials, and never calls
  ``agent.delegation_usage_cache.refresh_provider_now`` /
  ``agent.account_usage.fetch_account_usage`` itself — callers read the
  cache (via ``build_usage_view``) and pass the result in.
* ``claude-p`` can never be named or selected as the orchestrator here — it
  is a child-worker-only backend (see ``agent.delegation_routing.
  CLAUDE_P_PROVIDER``). Config naming it as primary or fallback is a hard,
  loud error, and the decision function itself never returns it even if a
  caller passes it in as "current" through some future misconfiguration.
* Fallback ("Agy") is entered when the cached PRIMARY provider's remaining
  percent is ``<= switch_at_remaining_percent``. Recovery back to the
  primary only happens on a FRESH reading strictly above
  ``restore_above_remaining_percent`` — a stale good reading, or any
  "unknown" reading, preserves whatever route is currently active (or the
  primary, at startup / no prior route).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from agent.delegation_routing import CLAUDE_P_PROVIDER, NATIVE_ROUTABLE_PROVIDERS, ProviderUsage

__all__ = [
    "DEFAULT_RESTORE_ABOVE_REMAINING_PERCENT",
    "DEFAULT_SWITCH_AT_REMAINING_PERCENT",
    "DEFAULT_USAGE_STALE_SECONDS",
    "DEFAULT_USAGE_TTL_SECONDS",
    "OrchestratorRouteDecision",
    "OrchestratorRoutingConfig",
    "OrchestratorRoutingConfigError",
    "decide_orchestrator_route",
    "parse_orchestrator_routing_config",
]


class OrchestratorRoutingConfigError(ValueError):
    """Raised when ``agent.orchestrator_usage_routing`` config is malformed.

    Loud by design, mirroring ``agent.delegation_routing.RouteConfigError``:
    a bad percent or an unsupported provider must never silently degrade
    into "feature quietly does nothing" (that's what ``enabled: false`` is
    for) or, worse, silently target the wrong provider.
    """


DEFAULT_SWITCH_AT_REMAINING_PERCENT: float = 10.0
DEFAULT_RESTORE_ABOVE_REMAINING_PERCENT: float = 10.0
DEFAULT_USAGE_TTL_SECONDS: int = 900
DEFAULT_USAGE_STALE_SECONDS: int = 7200


@dataclass(frozen=True)
class OrchestratorRoutingConfig:
    """Validated ``agent.orchestrator_usage_routing`` config block.

    ``enabled=False`` (including the "block absent" case) is the fully
    inert state: :func:`decide_orchestrator_route` always returns "keep
    current, no switch" for an inert config without needing any other field
    to be valid.
    """

    enabled: bool = False
    primary_provider: str = ""
    primary_model: str = ""
    fallback_provider: str = ""
    fallback_model: str = ""
    switch_at_remaining_percent: float = DEFAULT_SWITCH_AT_REMAINING_PERCENT
    restore_above_remaining_percent: float = DEFAULT_RESTORE_ABOVE_REMAINING_PERCENT
    usage_ttl_seconds: int = DEFAULT_USAGE_TTL_SECONDS
    usage_stale_seconds: int = DEFAULT_USAGE_STALE_SECONDS
    primary_usage_window_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class OrchestratorRouteDecision:
    """Structured, loggable outcome of :func:`decide_orchestrator_route`.

    ``reason`` is always safe to print/log at info level: it names
    providers/models and percentages only, never credentials or raw
    exception text.
    """

    provider: str
    model: str
    reason: str
    switched: bool = False


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def _require_str(raw: Mapping[str, Any], key: str, *, where: str) -> str:
    if key not in raw:
        raise OrchestratorRoutingConfigError(f"{where}: '{key}' is required")
    value = str(raw.get(key) or "").strip()
    if not value:
        raise OrchestratorRoutingConfigError(
            f"{where}: '{key}' is required and must be a non-empty string"
        )
    return value


def _parse_percent(raw: Any, key: str, *, where: str) -> float:
    if raw is None:
        raise OrchestratorRoutingConfigError(f"{where}: '{key}' is required")
    if isinstance(raw, bool):
        raise OrchestratorRoutingConfigError(f"{where}: '{key}' must be a number between 0 and 100")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        raise OrchestratorRoutingConfigError(
            f"{where}: '{key}' must be a number between 0 and 100"
        ) from None
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise OrchestratorRoutingConfigError(
            f"{where}: '{key}'={raw!r} is out of range (expected 0-100)"
        )
    return value


def _parse_positive_int(raw: Any, key: str, *, where: str) -> int:
    if raw is None:
        raise OrchestratorRoutingConfigError(f"{where}: '{key}' is required")
    if isinstance(raw, bool):
        raise OrchestratorRoutingConfigError(f"{where}: '{key}' must be a positive integer")
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        raise OrchestratorRoutingConfigError(
            f"{where}: '{key}' must be a positive integer"
        ) from None
    if value <= 0:
        raise OrchestratorRoutingConfigError(f"{where}: '{key}'={raw!r} must be > 0")
    return value


def _parse_window_prefixes(raw: Any, *, where: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise OrchestratorRoutingConfigError(
            f"{where}: 'primary_usage_window_prefixes' must be a list of strings"
        )
    return tuple(str(item).strip() for item in raw if str(item).strip())


def _require_native_provider(value: str, key: str, *, where: str) -> str:
    normalized = value.strip().lower()
    if normalized == CLAUDE_P_PROVIDER:
        raise OrchestratorRoutingConfigError(
            f"{where}: '{key}'={value!r} is not allowed — claude-p is a "
            f"child-worker-only backend and can never be the main orchestrator"
        )
    if normalized not in NATIVE_ROUTABLE_PROVIDERS:
        supported = ", ".join(sorted(NATIVE_ROUTABLE_PROVIDERS))
        raise OrchestratorRoutingConfigError(
            f"{where}: unsupported '{key}' {value!r} (supported: {supported})"
        )
    return normalized


def parse_orchestrator_routing_config(
    full_config: Optional[Mapping[str, Any]],
) -> OrchestratorRoutingConfig:
    """Parse+validate the ``agent.orchestrator_usage_routing`` config block.

    Returns an inert, disabled config (no further validation) when the
    block is absent or ``enabled: false`` — mirroring
    ``agent.delegation_routing.load_route_catalog``'s early return for
    ``routing.enabled is false``, so a half-written or future-staged block
    cannot break turns while the feature is off.
    """
    cfg = full_config if isinstance(full_config, Mapping) else {}
    agent_cfg = cfg.get("agent")
    agent_cfg = agent_cfg if isinstance(agent_cfg, Mapping) else {}
    block = agent_cfg.get("orchestrator_usage_routing")

    if block is None:
        return OrchestratorRoutingConfig(enabled=False)
    if not isinstance(block, Mapping):
        raise OrchestratorRoutingConfigError(
            "agent.orchestrator_usage_routing must be a mapping"
        )

    where = "agent.orchestrator_usage_routing"

    # Inert-when-disabled, BEFORE validating the rest — same idiom as
    # delegation_routing.load_route_catalog's routing.enabled gate.
    if not bool(block.get("enabled", False)):
        return OrchestratorRoutingConfig(enabled=False)

    primary_provider = _require_native_provider(
        _require_str(block, "primary_provider", where=where), "primary_provider", where=where
    )
    fallback_provider = _require_native_provider(
        _require_str(block, "fallback_provider", where=where), "fallback_provider", where=where
    )
    primary_model = _require_str(block, "primary_model", where=where)
    fallback_model = _require_str(block, "fallback_model", where=where)

    switch_at = _parse_percent(
        block.get("switch_at_remaining_percent"), "switch_at_remaining_percent", where=where
    )
    restore_above = _parse_percent(
        block.get("restore_above_remaining_percent"), "restore_above_remaining_percent", where=where
    )
    usage_ttl_seconds = _parse_positive_int(
        block.get("usage_ttl_seconds"), "usage_ttl_seconds", where=where
    )
    usage_stale_seconds = _parse_positive_int(
        block.get("usage_stale_seconds"), "usage_stale_seconds", where=where
    )
    if usage_stale_seconds < usage_ttl_seconds:
        raise OrchestratorRoutingConfigError(
            f"{where}: 'usage_stale_seconds' must be greater than or equal to "
            f"'usage_ttl_seconds'"
        )

    window_prefixes = _parse_window_prefixes(
        block.get("primary_usage_window_prefixes"), where=where
    )

    return OrchestratorRoutingConfig(
        enabled=True,
        primary_provider=primary_provider,
        primary_model=primary_model,
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
        switch_at_remaining_percent=switch_at,
        restore_above_remaining_percent=restore_above,
        usage_ttl_seconds=usage_ttl_seconds,
        usage_stale_seconds=usage_stale_seconds,
        primary_usage_window_prefixes=window_prefixes,
    )


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def _normalize_provider(value: Optional[str]) -> str:
    return str(value or "").strip().lower()


def decide_orchestrator_route(
    *,
    current_provider: Optional[str],
    usage: ProviderUsage,
    config: OrchestratorRoutingConfig,
    current_model: Optional[str] = None,
) -> OrchestratorRouteDecision:
    """Pick the orchestrator's provider/model for the NEXT turn.

    ``current_provider``/``current_model`` describe what is actually driving
    the conversation right now (``None``/empty at startup, before any turn
    has run). ``usage`` is the cached reading for the configured PRIMARY
    provider (freshness one of ``fresh``/``stale``/``unknown``, per
    :mod:`agent.delegation_usage_cache`). Never performs I/O.

    ``current_model`` is used only to echo back an unchanged decision
    exactly (e.g. when the config is disabled, or config.enabled is True but
    the config's own model name should be preferred) — it never influences
    *which provider* is selected.
    """
    primary = (config.primary_provider, config.primary_model, "primary orchestrator")
    fallback = (config.fallback_provider, config.fallback_model, "Agy fallback")

    if not config.enabled:
        # Fully inert: keep whatever is currently active. At startup (no
        # prior route) that is the primary by construction of the caller,
        # but this function itself just preserves "current" — never invents
        # a switch when the feature is off. The config carries no validated
        # provider/model names in this state (it short-circuits before
        # parsing them), so the current provider/model are echoed back
        # verbatim rather than guessed.
        current = _normalize_provider(current_provider)
        return OrchestratorRouteDecision(
            provider=current,
            model=str(current_model or ""),
            reason="orchestrator usage routing disabled — keeping current route",
            switched=False,
        )

    current = _normalize_provider(current_provider)
    # Defense in depth: claude-p must never be selectable here even if a
    # caller's "current provider" bookkeeping was corrupted upstream.
    on_fallback = current == config.fallback_provider

    if usage.freshness != "fresh":
        if on_fallback:
            return OrchestratorRouteDecision(
                provider=fallback[0], model=fallback[1],
                reason=(
                    f"primary usage reading {usage.freshness} — preserving current "
                    f"route ({fallback[2]})"
                ),
                switched=False,
            )
        return OrchestratorRouteDecision(
            provider=primary[0], model=primary[1],
            reason=(
                f"primary usage reading {usage.freshness} — preserving current "
                f"route ({primary[2]})"
            ),
            switched=False,
        )

    remaining = usage.remaining_percent
    low_usage = remaining is not None and remaining <= config.switch_at_remaining_percent

    if not on_fallback:
        if low_usage:
            return OrchestratorRouteDecision(
                provider=fallback[0], model=fallback[1],
                reason=(
                    f"primary remaining {remaining:.1f}% <= "
                    f"switch_at_remaining_percent={config.switch_at_remaining_percent:.1f}% "
                    f"({usage.freshness}) — switching to {fallback[2]}"
                ),
                switched=True,
            )
        return OrchestratorRouteDecision(
            provider=primary[0], model=primary[1],
            reason=(
                f"primary remaining "
                f"{'unknown' if remaining is None else f'{remaining:.1f}%'} is "
                f"healthy — staying on {primary[2]}"
            ),
            switched=False,
        )

    # Currently on the fallback: only a FRESH good reading recovers.
    good_reading = remaining is not None and remaining > config.restore_above_remaining_percent
    if usage.freshness == "fresh" and good_reading:
        return OrchestratorRouteDecision(
            provider=primary[0], model=primary[1],
            reason=(
                f"fresh primary remaining {remaining:.1f}% > "
                f"restore_above_remaining_percent="
                f"{config.restore_above_remaining_percent:.1f}% — recovering to "
                f"{primary[2]}"
            ),
            switched=True,
        )
    return OrchestratorRouteDecision(
        provider=fallback[0], model=fallback[1],
        reason=(
            f"primary remaining "
            f"{'unknown' if remaining is None else f'{remaining:.1f}%'} "
            f"({usage.freshness}) does not clear the fresh-recovery bar — "
            f"staying on {fallback[2]}"
        ),
        switched=False,
    )

# -*- coding: utf-8 -*-
"""Stage 2: per-task model auto-router.

Reads the user's ``model_aliases`` (Stage 1) + their capability metadata from
``models_dev``, then picks the best alias for a task using real turn signal:

* vision attachments present     -> require a multimodal-capable alias, else fall back
* long text / big context budget -> prefer high-context tier, cap by ``max_output``
* deep-research intent           -> prefer the highest reasoning tier
* otherwise                      -> fast/cheap default

Manual override lives in ``resolve()`` (pass an explicit alias name). No rewrite of
the 900-line switch handler -- this is a small, self-contained module that the
handler calls on its ``auto`` subcommand path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # static analysis only; runtime import stays inside _tier_of
    from .models_dev import ModelInfo


# --- Routing tier table -----------------------------------------------------
# Tier = how much "brain" to spend. Lower index wins ties; higher index is the
# deep-reasoning choice. Kept as a list so ordering == priority.
_TIER_ORDER = ("fast", "embed", "researcher")

# Human-readable labels for log lines only (never authoritative).
_TIER_LABELS = {
    "fast": "fast/cheap subagent prep + vision",
    "researcher": "deep research + orchestration",
    "embed": "RAG / embeddings",
}

# Aliases whose target model is self-hosted / not in models.dev. These are best-effort
# skipped providers; ``custom`` (LM Studio) aliases always return None from get_model_info,
# so we let the call run and fall back to the name heuristic naturally instead of special-casing.
_UNKNOWN_MODELS_DEV_PROVIDERS = {"ollama"}

# Soft context-budget signal: a large input nudges toward high-context tiers. Exposed as a
# named module constant (not a magic number) so it stays discoverable/tunable without any
# runtime config plumbing -- see SKILL.md Pitfalls ("soft heuristic, never authoritative").
_CONTEXT_THRESHOLD_BYTES = 64_000


@dataclass
class RouteDecision:
    """What ``resolve()`` returns; kept flat so callers can log/report it."""

    alias: str                       # chosen model_aliases key
    provider_model: Optional[str]    # the literal target from config (for logs)
    tier: str                        # which _TIER_ORDER slot won
    reason: str                      # short why (vision / context / depth / default)


# --- Capability helpers -----------------------------------------------------

def _tier_of(alias: str, aliases_config: dict) -> tuple[int, bool]:
    """Return (priority_index, is_multimodal). Lower index = higher priority.

    multimodal flag comes from whether the alias's target model carries the
    ``attachment`` capability; missing metadata defaults to False so a non-vision
    alias never masquerades as capable of reading images.
    """
    idx = _TIER_ORDER.index(alias) if alias in _TIER_ORDER else len(_TIER_ORDER)

    spec = aliases_config.get(alias, {})
    provider = str(spec.get("provider", "")).lower() or "custom"
    model_id = str(spec.get("model", "")).lower()

    # Multimodality: read from agent.models_dev when the alias is a known hosted
    # provider (models.dev covers those); for custom/LM Studio aliases we cannot
    # query models.dev, so fall back to a best-effort heuristic on the target name.
    multimodal = False
    try:
        from agent import models_dev

        if provider not in _UNKNOWN_MODELS_DEV_PROVIDERS:
            info = models_dev.get_model_info(provider, model_id)
            if info is not None:
                multimodal = bool(getattr(info, "attachment", False)) or bool(
                    getattr(info, "supports_vision", False)
                )
    except Exception:
        pass

    # Last resort: heuristic on the target name (never authoritative).
    if not multimodal:
        multimodal = "vision" in model_id or "multimodal" in model_id

    return idx, multimodal


# --- Public entry point -----------------------------------------------------

def resolve(
    aliases_config: dict,
    *,
    has_vision: bool = False,
    context_bytes: int = 0,
    deep_intent: bool = False,
    prefer_alias: Optional[str] = None,
) -> RouteDecision:
    """Pick the best alias for one task.

    Parameters
    ----------
    aliases_config : dict
        The ``model_aliases`` mapping from config.yaml, e.g.
        ``{"researcher": {"model": "ornith-1.5-35b-a3b"}, ...}``.
    has_vision : bool
        True when the current turn carries image/video attachments.
    context_bytes : int
        Rough size of the task text (>= 0). Used as a soft signal only -- the
        router does NOT parse the full context window; it nudges toward high-context
        tiers for large inputs and away from aliases that cap at small ``max_output``.
    deep_intent : bool
        True when the user explicitly wants "deep" work (e.g. subcommand hint or a
        known research trigger). Lets them steer to the top tier on demand.
    prefer_alias : str | None
        Manual override: an exact alias name from Stage 1, bypassing scoring.

    Returns
    -------
    RouteDecision
        The chosen ``alias`` + its provider/model and a short ``reason`` string.
    """
    if not aliases_config:
        return RouteDecision(
            alias="",
            provider_model=None,
            tier="none",
            reason="no model_aliases configured -- run Stage 1 first",
        )

    # Manual override wins everything (that's what /model auto <alias> maps to).
    if prefer_alias in aliases_config:
        cfg = aliases_config[prefer_alias]
        tier = _TIER_ORDER.index(prefer_alias) if prefer_alias in _TIER_ORDER else "custom"
        return RouteDecision(
            alias=prefer_alias,
            provider_model=str(cfg.get("model")),
            tier=str(tier),
            reason=f"manual override: /model auto {prefer_alias}",
        )

    # Multimodal requirement is a hard gate: an alias whose target lacks the
    # attachment capability cannot serve a task that ships images.
    candidates = list(aliases_config.keys())
    if has_vision:
        eligible = [a for a in candidates]  # scored below; multimodal flag breaks ties
        modal_eligible = [
            a for a in eligible if _tier_of(a, aliases_config)[1]
        ]
        pool = modal_eligible or eligible

    else:
        pool = candidates

    if not pool:
        return RouteDecision(
            alias=list(aliases_config.keys())[0],
            provider_model=str(aliases_config[list(aliases_config.keys())[0]].get("model")),
            tier="custom",
            reason="vision-only aliases absent -- fell back to first available",
        )

    # Score each candidate: tier priority, then context fit, then depth preference.
    def score(alias: str) -> tuple[int, int, int]:
        idx, _ = _tier_of(alias, aliases_config)
        ctx_penalty = 0 if context_bytes <= _CONTEXT_THRESHOLD_BYTES else (idx + 1)  # large input prefers high tier
        depth_bonus = 0 if not deep_intent else idx * -2         # low index wins when deep
        return (ctx_penalty, depth_bonus, idx)

    best = min(pool, key=score)
    _, multimodal = _tier_of(best, aliases_config)
    cfg = aliases_config[best]
    reason_bits = [f"tier={_TIER_LABELS.get(best, 'custom')}"]
    if has_vision:
        reason_bits.append("vision attachments")
    if context_bytes > _CONTEXT_THRESHOLD_BYTES:
        reason_bits.append(f"{context_bytes // 1024}KB input")
    if deep_intent:
        reason_bits.append("deep intent")
    return RouteDecision(
        alias=best,
        provider_model=str(cfg.get("model")),
        tier=_TIER_LABELS.get(best, "custom"),
        reason="; ".join(reason_bits),
    )


def build_aliases_from_config(config_path: str) -> dict:
    """Read ``model_aliases`` straight from a config.yaml path.

    Thin wrapper so the CLI can hand this module config without importing the full
    runtime config stack. Returns an empty dict if the file is missing/unreadable.
    """
    import yaml

    aliases: dict = {}
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        raw = data.get("model_aliases") or {}
        for name, spec in raw.items():
            if isinstance(spec, dict) and spec.get("model"):
                aliases[name] = {"model": str(spec["model"])}
    except Exception:
        return {}
    return aliases


# --- Test hook --------------------------------------------------------------

def _reset_tier_order(order=None) -> None:  # pragma: no cover - test seam
    global _TIER_ORDER
    if order is not None:
        _TIER_ORDER = tuple(order)

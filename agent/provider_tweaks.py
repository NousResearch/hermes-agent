"""Provider-specific OpenRouter routing tweaks.

Central registry for known-buggy OpenRouter endpoints whose upstream providers
silently drop tool-call streams, stall mid-arguments, or otherwise fail in ways
that are not user-configurable.  Consumed by ``run_agent.py`` when building
``provider_preferences`` for chat completion requests against OpenRouter.

Design principles:

* Only applies to OpenRouter ``base_url`` — other provider chains route through
  different infrastructure and may not have the same endpoint issues.
* User-provided preferences always win.  We only layer defaults in where the
  user hasn't specified ``only``, ``order``, or ``ignore``.
* Additions must be backed by a concrete upstream-bug reference (vendor repo
  issue, reproducible empirical evidence) — this is not for speculative
  provider preferences.

Registry format (``_KNOWN_BROKEN_ROUTES``):
    key: lowercase model-slug substring that identifies the affected family
    value: {
        "ignore": [list of OpenRouter provider tags to skip, e.g. "minimax"],
        "order":  [list of OpenRouter provider tags to prefer in order],
        "reason": "human-readable one-liner used in logs",
        "ref":    "issue/PR reference for the upstream bug",
    }
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Ordered list: first matching entry wins.  Match is substring-in-lower-model.
_KNOWN_BROKEN_ROUTES: List[Dict[str, Any]] = [
    {
        # MiniMax direct OpenRouter endpoint has documented non-terminating
        # streams on tool-calling workflows (MiniMax-M2 issue #109, Apr 2026;
        # OpenClaw #1622).  Empirically reproduced 4/4 times on 2026-04-18:
        # streaming a write_file tool call returned zero bytes and closed
        # silently at ~40s from both minimax/fp8 and minimax/highspeed tags.
        # Fireworks, Together, NovitaAI, Google-Vertex, AtlasCloud all work.
        "match": "minimax/",
        "ignore": ["minimax"],
        "order": [
            "fireworks",       # m2.7: best throughput + uptime
            "novitaai",        # m2:   best tool-call error rate (0.19%)
            "google-vertex",   # m2:   fastest latency
            "atlascloud",
            "together",        # fp4 quant — last resort
        ],
        "reason": "Minimax direct endpoint drops tool-call streams",
        "ref": "MiniMax-M2#109, OpenClaw#1622, Hermes-PR#12072",
    },
]


def get_provider_tweaks(model: Optional[str], base_url: Optional[str]) -> Dict[str, Any]:
    """Return known-broken-endpoint tweaks for a given model/base_url pair.

    Returns an empty dict when no tweaks apply (non-OpenRouter endpoint,
    unknown model, etc.) so callers can do ``if tweaks:`` cheaply.

    Returned keys when applicable:
        ignore: list[str] — OpenRouter provider tags to exclude
        order:  list[str] — OpenRouter provider tags to prefer in order
        reason: str       — human-readable reason (for logging)
        ref:    str       — upstream bug reference (for logging)
    """
    if not model or not base_url:
        return {}
    url_lower = base_url.lower()
    # Only OpenRouter-compatible endpoints understand the ``provider`` object.
    if "openrouter.ai" not in url_lower:
        return {}
    model_lower = model.lower()
    for entry in _KNOWN_BROKEN_ROUTES:
        if entry["match"] in model_lower:
            return {
                "ignore": list(entry.get("ignore") or []),
                "order": list(entry.get("order") or []),
                "reason": entry.get("reason", ""),
                "ref": entry.get("ref", ""),
            }
    return {}


def merge_provider_tweaks(
    provider_preferences: Dict[str, Any],
    tweaks: Dict[str, Any],
    *,
    log_label: str = "",
) -> Dict[str, Any]:
    """Merge auto-tweaks into user-supplied provider preferences.

    User-provided fields always win — this function never overrides ``only``,
    ``ignore``, or ``order`` that the user has already set.  It only supplies
    defaults where those fields are absent.

    When the user has set ``only`` (whitelist mode), the tweaks are fully
    ignored: a whitelist already constrains routing to a known-good subset,
    and layering ``ignore``/``order`` on top would be confusing.

    Emits a single INFO log line when tweaks are actually applied so the
    behaviour is visible in agent.log without spamming every request.
    """
    if not tweaks:
        return provider_preferences or {}
    result = dict(provider_preferences or {})
    # Whitelist already narrows routing — don't layer on.
    if result.get("only"):
        return result

    applied: List[str] = []
    if tweaks.get("ignore") and "ignore" not in result:
        result["ignore"] = list(tweaks["ignore"])
        applied.append(f"ignore={tweaks['ignore']}")
    if tweaks.get("order") and "order" not in result:
        result["order"] = list(tweaks["order"])
        applied.append(f"order={tweaks['order']}")

    if applied:
        logger.info(
            "Provider tweaks applied%s: %s (reason: %s; ref: %s)",
            f" [{log_label}]" if log_label else "",
            ", ".join(applied),
            tweaks.get("reason", "?"),
            tweaks.get("ref", "?"),
        )
    return result


__all__ = ["get_provider_tweaks", "merge_provider_tweaks"]

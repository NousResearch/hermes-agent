"""``openrouter.min_coding_score``: the coding-score floor of OpenRouter's Pareto Code router.

Every surface that builds an ``AIAgent`` from config (CLI, TUI/Desktop, messaging gateway turns,
managed workers, cron, ``/bg``, API server, ACP) passes ``openrouter_min_coding_score`` from
:func:`min_coding_score`. A surface that leaves it out sends ``openrouter/pareto-code`` requests
without the ``pareto-router`` plugin, and OpenRouter then routes to its strongest (priciest) tier.
Delegated children and fallback activation keep the value the parent agent was built with.
"""
from __future__ import annotations

from typing import Any


def pareto_score(raw: Any) -> float | None:
    """``raw`` as a float in [0, 1]; ``None`` for unset (``None`` / ``""``), non-numeric or out of range.
    Same rule the request builders apply (``agent/transports/chat_completions.py::_pareto_score``)."""
    if raw is None or raw == "":
        return None
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return None
    return score if 0.0 <= score <= 1.0 else None


def min_coding_score(config: Any) -> float | None:
    """The validated ``openrouter.min_coding_score`` of a loaded config dict (``None`` when unset/invalid)."""
    section = config.get("openrouter") if isinstance(config, dict) else None
    return pareto_score(section.get("min_coding_score")) if isinstance(section, dict) else None

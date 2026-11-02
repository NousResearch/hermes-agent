"""Global, route-independent cap for Hermes' working context window.

``model.context_length`` is an explicit pin for one configured route.  In
contrast, ``model.context_cap`` limits every resolved route without inflating
models whose own window is smaller.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def configured_context_cap(config: Optional[dict[str, Any]] = None) -> Optional[int]:
    """Return the positive global ``model.context_cap``, or ``None`` when unset/invalid."""
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly()
        except Exception:
            # Deliberate catch-all boundary: a broken/unreadable config must never break
            # context resolution, so fall back to "no cap" and log the reason.
            logger.debug("context_cap: config load failed; treating cap as unset", exc_info=True)
            return None
    if not isinstance(config, dict):
        return None
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    raw = model.get("context_cap")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        cap = int(raw)
    except (TypeError, ValueError):
        return None
    return cap if cap > 0 else None


def cap_context_length(context_length: int, config: Optional[dict[str, Any]] = None) -> int:
    """Clamp a resolved window to the global cap, never raising a smaller window."""
    resolved = int(context_length)
    cap = configured_context_cap(config)
    return min(resolved, cap) if cap is not None else resolved

"""Reasoning vocabulary overrides using the existing model_overrides lookup rules."""
import logging
from typing import Optional
from agent.models_dev import _override_for

logger = logging.getLogger(__name__)
_OVERRIDE_WARNED_KEYS: set = set()

def get_reasoning_effort_override(provider: str, model: str, *, catalog_hit: bool) -> Optional[tuple[str, ...]]:
    """Profile-scoped effort vocabulary override; absent/invalid differs from explicitly empty."""
    from agent.reasoning_effort import EFFORT_LADDER
    override = _override_for(provider, model, catalog_hit=catalog_hit)
    if not override or "supported_reasoning_efforts" not in override:
        return None
    raw = override["supported_reasoning_efforts"]
    if isinstance(raw, list) and all(isinstance(v, str) and v in EFFORT_LADDER[:-1] for v in raw):
        return tuple(level for level in EFFORT_LADDER if level in raw)
    from hermes_constants import hermes_home_key
    key = (hermes_home_key(), provider, model, "supported_reasoning_efforts")
    if key not in _OVERRIDE_WARNED_KEYS:
        _OVERRIDE_WARNED_KEYS.add(key)
        logger.warning("model_overrides: ignoring invalid supported_reasoning_efforts for %s/%s (expected a list of effort levels)", provider, model)
    return None

"""Flag resolution for Executive v2.

Default-off. Opt-in via env var or per-instance attribute.
No global config mutation. No config.yaml writes.
"""

from __future__ import annotations

import os
from typing import Any

_ENV_VAR = "HERMES_EXECUTIVE_V2_ENABLED"
_TRUTHY = {"1", "true", "yes", "on"}


def resolve_v2_enabled(agent: Any | None = None) -> bool:
    """Resolve whether Executive v2 is enabled.

    Resolution order (any truthy -> enabled):
    1. ``agent._executive_v2_enabled`` (per-instance flag).
    2. ``HERMES_EXECUTIVE_V2_ENABLED`` env var.

    Default: False.
    """
    try:
        if agent is not None and getattr(agent, "_executive_v2_enabled", None):
            return True
    except Exception:
        pass
    try:
        env = os.environ.get(_ENV_VAR, "")
        if env and env.strip().lower() in _TRUTHY:
            return True
    except Exception:
        pass
    return False

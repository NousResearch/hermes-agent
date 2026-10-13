"""Profile-scoped HTTP settings for the Telegram gateway transport."""

import logging
import math

from gateway.platforms._shared import extra_or_secret, platform_gate_env

logger = logging.getLogger(__name__)


def http_read_timeout(extra: dict) -> float:
    """Resolve scoped env → YAML → 20s; an invalid explicit value uses the default."""
    # HERMES_TELEGRAM_* is otherwise process-global in get_secret; this setting is profile-owned.
    raw = platform_gate_env("HERMES_TELEGRAM_HTTP_READ_TIMEOUT") or extra_or_secret(
        extra, "http_read_timeout", "", 20.0
    )
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        value = 0.0
    if isinstance(raw, bool) or not math.isfinite(value) or value <= 0:
        logger.warning("Invalid Telegram http_read_timeout; using the 20-second default")
        return 20.0
    return value

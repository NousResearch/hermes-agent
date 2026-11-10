"""Auxiliary timeout route classification and no-progress config parsing."""

from typing import Optional

from utils import base_url_hostname


def _is_local_aux_base_url(base_url: Optional[str]) -> bool:
    """Recognize loopback routes without treating an unknown host as local."""
    try:
        host = base_url_hostname(str(base_url or ""))
    except (TypeError, ValueError):
        return False
    return (host or "").strip().lower().strip("[]") in {
        "localhost", "127.0.0.1", "::1", "0.0.0.0",
    }


def task_no_progress_timeout(task: str, raw, *, default: float, logger) -> Optional[float]:
    """Parse an optional no-progress budget, retaining its existing warning contract."""
    if raw is None:
        return None
    try:
        value = float(raw)
    except (ValueError, TypeError):
        value = 0.0
    if isinstance(raw, bool) or value <= 0:
        # A typo must not silently leave the built-in timeout without explaining why.
        logger.warning(
            "auxiliary.%s.no_progress_timeout=%r is not a positive number of seconds; "
            "using the built-in %.0fs default", task, raw, default)
        return None
    return value

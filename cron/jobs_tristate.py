"""Strict true/false/null normalizers for per-job boolean overrides (``create_job`` / ``update_job``).

``None`` means "inherit the default"; anything but a real bool or None raises, because
``bool("false")`` is True and a coerced string would silently flip the override.
"""
from typing import Any, Optional


def normalize_catch_up(value: Any) -> Optional[bool]:
    """``catch_up``: True runs one catch-up for missed slots, False skips them, None = default."""
    if value is None or isinstance(value, bool):
        return value
    raise ValueError("catch_up must be true, false, or null (use the default).")


def normalize_wrap_response(value: Any) -> Optional[bool]:
    """``wrap_response``: True wraps delivered output, False delivers it raw, None inherits the
    global ``cron.wrap_response``."""
    if value is None or isinstance(value, bool):
        return value
    raise ValueError("wrap_response must be true, false, or null (inherit cron.wrap_response).")

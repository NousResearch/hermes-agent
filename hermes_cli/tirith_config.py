"""The one reader of ``security.tirith_enabled`` / ``security.tirith_fail_open`` and their
``TIRITH_ENABLED`` / ``TIRITH_FAIL_OPEN`` env overrides.

Kept apart from ``tools.tirith_security`` because two callers run exactly when that module
failed to import (``tools.approval``'s ImportError branches), and light on imports (only
``os`` here; ``hermes_cli.config`` loads lazily) because tools load while config is still
initializing.
"""

import os

_TRUTHY = frozenset({"1", "true", "yes"})  # the set tirith_security has always used


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    return default if val is None else val.lower() in _TRUTHY


def security_section(config: "dict | None" = None) -> dict:
    """``security`` from *config* (loaded read-only when omitted); ``{}`` when unreadable."""
    try:
        if config is None:
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly()
        return dict((config or {}).get("security", {}) or {})
    except Exception:
        return {}


def tirith_enabled(config: "dict | None" = None) -> bool:
    """``security.tirith_enabled`` (default True), overridden by ``TIRITH_ENABLED``."""
    return _env_bool("TIRITH_ENABLED", security_section(config).get("tirith_enabled", True))


def tirith_fail_open(config: "dict | None" = None) -> bool:
    """``security.tirith_fail_open`` (default True), overridden by ``TIRITH_FAIL_OPEN``."""
    return _env_bool("TIRITH_FAIL_OPEN", security_section(config).get("tirith_fail_open", True))


def fail_open_when_scanner_unavailable(config: "dict | None" = None) -> bool:
    """May a command run when the scanner cannot run at all? True when scanning is off
    (nothing to fail against), else the fail-open flag."""
    section = {"security": security_section(config)}
    return True if not tirith_enabled(section) else tirith_fail_open(section)

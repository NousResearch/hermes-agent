"""Configurable truncation limits for the ACP adapter.

Reads ``acp`` settings from ``config.yaml`` so power users can tune
truncation behaviour when Hermes is used as an ACP server (VS Code,
Zed, JetBrains, etc.) without patching the source.

Example ``config.yaml``::

    acp:
      tool_output_max_chars: 10000   # max chars for tool-result display
      title_max_chars: 120           # max chars for tool-call titles

Each key defaults to the pre-existing hardcoded value when missing.
"""

from __future__ import annotations

from typing import Any

# Hardcoded defaults — match the pre-existing values so behaviour is
# unchanged for users who don't set ``acp`` in config.yaml.
DEFAULT_TOOL_OUTPUT_MAX_CHARS = 5000   # _truncate_text default, _build_tool_complete_content
DEFAULT_TITLE_MAX_CHARS = 80           # build_tool_title terminal cmd, steer preview

# Module-level cache — populated on first call.
_cached_limits: dict | None = None


def _coerce_positive_int(value: Any, default: int) -> int:
    """Return ``value`` as a positive int, or ``default`` on any issue."""
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return default
    if iv <= 0:
        return default
    return iv


def get_acp_truncation_limits() -> dict[str, int]:
    """Return resolved ACP truncation limits, reading ``acp`` from config.

    Keys: ``tool_output_max_chars``, ``title_max_chars``.  Missing or
    invalid entries fall through to the ``DEFAULT_*`` constants.

    Result is cached for the process lifetime to avoid repeated disk I/O.
    Call ``_reset_acp_truncation_limits_cache()`` in tests that need a
    fresh read after config changes.
    """
    global _cached_limits
    if _cached_limits is not None:
        return _cached_limits
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        section = cfg.get("acp") if isinstance(cfg, dict) else None
        if not isinstance(section, dict):
            section = {}
    except Exception:
        section = {}

    _cached_limits = {
        "tool_output_max_chars": _coerce_positive_int(
            section.get("tool_output_max_chars"), DEFAULT_TOOL_OUTPUT_MAX_CHARS
        ),
        "title_max_chars": _coerce_positive_int(
            section.get("title_max_chars"), DEFAULT_TITLE_MAX_CHARS
        ),
    }
    return _cached_limits


def _reset_acp_truncation_limits_cache() -> None:
    """Reset the cached limits — for tests or after config hot-reload."""
    global _cached_limits
    _cached_limits = None


def get_tool_output_max_chars() -> int:
    """Maximum chars for ACP tool-result display (``_truncate_text`` default)."""
    return get_acp_truncation_limits()["tool_output_max_chars"]


def get_title_max_chars() -> int:
    """Maximum chars for ACP tool-call titles (terminal cmd, steer preview)."""
    return get_acp_truncation_limits()["title_max_chars"]

"""Shared Discord thread lifetime policy for adapters and the REST model tool."""
from typing import Any

VALID_THREAD_AUTO_ARCHIVE_MINUTES = frozenset({60, 1440, 4320, 10080})


def validate_thread_auto_archive_duration(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return 1440
    try:
        parsed = int(value)
    except ValueError:
        return 1440
    return parsed if parsed in VALID_THREAD_AUTO_ARCHIVE_MINUTES else 1440


def configured_thread_auto_archive_duration() -> int:
    """Resolve at call time under the served profile, without mutating configuration/env."""
    from gateway.config import PlatformConfig
    from gateway.config_loader import merge_platform_sections, platform_section
    from hermes_cli.config import load_config_readonly
    config = load_config_readonly() or {}
    platforms = merge_platform_sections(config, config.get("gateway"), {})
    section, _ = platform_section(config, "discord", platforms)
    extra = PlatformConfig.from_dict(platforms.get("discord", {})).extra
    value = extra.get("thread_auto_archive_duration", (section or {}).get("thread_auto_archive_duration", 1440))
    return validate_thread_auto_archive_duration(value)

"""Configuration helpers for the bundled A2A platform plugin."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9100
DEFAULT_MAX_CONCURRENCY = 16
DEFAULT_MAX_SESSIONS = 512
DEFAULT_MAX_TASKS = 2048
DEFAULT_MAX_TASK_HISTORY = 100
DEFAULT_TOOL_IO = "preview"
TOOL_IO_MODES = frozenset({"preview", "none", "full"})


def _positive_int(value: Any, default: int, *, maximum: int | None = None) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0 or (maximum is not None and parsed > maximum):
        return default
    return parsed


def _tool_io(value: Any) -> str:
    mode = str(value or "").strip().lower()
    return mode if mode in TOOL_IO_MODES else DEFAULT_TOOL_IO


@dataclass(frozen=True)
class A2ASettings:
    """Validated behavioral settings from the ``a2a`` config.yaml section."""

    enabled: bool = False
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    public_url: str | None = None
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    max_sessions: int = DEFAULT_MAX_SESSIONS
    max_tasks: int = DEFAULT_MAX_TASKS
    max_task_history: int = DEFAULT_MAX_TASK_HISTORY
    tool_io: str = DEFAULT_TOOL_IO

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "A2ASettings":
        data = values if isinstance(values, Mapping) else {}
        host = str(data.get("host") or DEFAULT_HOST).strip() or DEFAULT_HOST
        public_url = str(data.get("public_url") or "").strip() or None
        max_concurrency = _positive_int(
            data.get("max_concurrency"), DEFAULT_MAX_CONCURRENCY
        )
        max_tasks = max(
            max_concurrency,
            _positive_int(data.get("max_tasks"), DEFAULT_MAX_TASKS),
        )
        return cls(
            enabled=bool(data.get("enabled", False)),
            host=host,
            port=_positive_int(data.get("port"), DEFAULT_PORT, maximum=65535),
            public_url=public_url,
            max_concurrency=max_concurrency,
            max_sessions=_positive_int(data.get("max_sessions"), DEFAULT_MAX_SESSIONS),
            max_tasks=max_tasks,
            max_task_history=_positive_int(
                data.get("max_task_history"), DEFAULT_MAX_TASK_HISTORY
            ),
            tool_io=_tool_io(data.get("tool_io")),
        )


def load_a2a_settings() -> A2ASettings:
    """Load the merged ``a2a`` section from the active Hermes profile."""
    from hermes_cli.config import load_config

    config = load_config()
    section = config.get("a2a") if isinstance(config, dict) else None
    return A2ASettings.from_mapping(section)


def settings_from_platform_config(platform_config: Any) -> A2ASettings:
    """Build settings from a gateway ``PlatformConfig`` instance."""
    extra = getattr(platform_config, "extra", None)
    values = dict(extra) if isinstance(extra, Mapping) else {}
    values["enabled"] = bool(getattr(platform_config, "enabled", False))
    return A2ASettings.from_mapping(values)


def apply_yaml_config(
    _yaml_config: dict[str, Any], platform_config: dict[str, Any]
) -> dict[str, Any]:
    """Seed A2A-specific ``PlatformConfig.extra`` fields from config.yaml."""
    if not isinstance(platform_config, dict):
        return {}
    extra = platform_config.get("extra")
    seeded = dict(extra) if isinstance(extra, dict) else {}
    for key in (
        "host",
        "port",
        "public_url",
        "max_concurrency",
        "max_sessions",
        "max_tasks",
        "max_task_history",
        "tool_io",
    ):
        if key in platform_config:
            seeded[key] = platform_config[key]
    return seeded

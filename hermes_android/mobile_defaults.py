from __future__ import annotations

from typing import Any

from hermes_android.python_path import prefer_hermes_package_root

prefer_hermes_package_root()

from hermes_cli.config import load_config, save_config
from toolsets import validate_toolset

DEFAULT_ANDROID_API_SERVER_TOOLSETS = ["hermes-android-app"]


def _configured_api_server_toolsets(config: dict[str, Any] | None) -> list[str] | None:
    if not isinstance(config, dict):
        return None
    platform_toolsets = config.get("platform_toolsets")
    if not isinstance(platform_toolsets, dict):
        return None
    configured = platform_toolsets.get("api_server")
    if not isinstance(configured, list):
        return None
    cleaned = [str(item).strip() for item in configured if str(item).strip()]
    return cleaned or None


def should_force_android_api_server_toolsets(config: dict[str, Any] | None) -> bool:
    configured = _configured_api_server_toolsets(config)
    if not configured:
        return True
    return not all(validate_toolset(name) for name in configured)


def resolved_android_api_server_toolsets(config: dict[str, Any] | None) -> list[str]:
    configured = _configured_api_server_toolsets(config)
    if configured and all(validate_toolset(name) for name in configured):
        return configured
    return list(DEFAULT_ANDROID_API_SERVER_TOOLSETS)


def ensure_android_defaults(config: dict[str, Any] | None = None, *, persist: bool = True) -> dict[str, Any]:
    loaded = load_config() if config is None else config
    platform_toolsets = loaded.setdefault("platform_toolsets", {})
    if not isinstance(platform_toolsets, dict):
        platform_toolsets = {}
        loaded["platform_toolsets"] = platform_toolsets

    current = platform_toolsets.get("api_server")
    if not isinstance(current, list) or not current:
        platform_toolsets["api_server"] = list(DEFAULT_ANDROID_API_SERVER_TOOLSETS)
        if persist:
            save_config(loaded)
    return loaded

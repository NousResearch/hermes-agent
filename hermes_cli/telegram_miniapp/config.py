"""Read-only config helpers for the Telegram Mini App sidecar."""

from __future__ import annotations

from typing import Any

from hermes_cli.config import cfg_get, load_config

from .server import MiniAppSettings, _default_allowed_origins


def settings_from_config(config: dict[str, Any] | None = None) -> MiniAppSettings:
    """Resolve Mini App settings without mutating config.yaml or .env."""
    cfg = load_config() if config is None else config
    section = cfg_get(cfg, "telegram_miniapp", default={})
    if not isinstance(section, dict):
        section = {}

    allowed = section.get("allowed_users") or []
    origins = section.get("cors_allowed_origins") or []
    cors_allowed_origins = {str(value) for value in origins if str(value).strip()}
    return MiniAppSettings(
        host=str(section.get("host") or "127.0.0.1"),
        port=int(section.get("port") or 9120),
        auth_ttl_seconds=int(section.get("auth_ttl_seconds") or 300),
        session_ttl_seconds=int(section.get("session_ttl_seconds") or 3600),
        allowed_users={str(value) for value in allowed if str(value).strip()},
        cors_allowed_origins=cors_allowed_origins or _default_allowed_origins(),
    )

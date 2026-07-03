"""Read-only config helpers for the Telegram Mini App sidecar."""

from __future__ import annotations

from typing import Any

from hermes_cli.config import cfg_get, get_hermes_home, load_config

from .server import MiniAppSettings, _default_allowed_origins


def _as_bool(value: Any) -> bool:
    """Strictly coerce a config value to bool, failing CLOSED.

    A plain ``bool(value)`` would make the string ``"false"`` (e.g. a quoted
    YAML value) truthy and silently ENABLE a security-sensitive switch. Here
    only a real boolean ``True`` or an explicit affirmative string turns it on;
    anything unrecognized stays off.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "on", "1"}
    return False


def settings_from_config(config: dict[str, Any] | None = None) -> MiniAppSettings:
    """Resolve Mini App settings without mutating config.yaml or .env.

    Public HTTPS smoke activation and the action gate are deliberately NOT read
    from durable config: ``enable_actions`` stays False here and can only be set
    by a per-run foreground CLI flag, exactly like ``public_smoke``. Config only
    supplies the owner allowlist that a future enabled run would consult.
    """
    cfg = load_config() if config is None else config
    section = cfg_get(cfg, "telegram_miniapp", default={})
    if not isinstance(section, dict):
        section = {}

    allowed = section.get("allowed_users") or []
    origins = section.get("cors_allowed_origins") or []
    action_owners = section.get("action_owners") or []
    cors_allowed_origins = {str(value) for value in origins if str(value).strip()}
    return MiniAppSettings(
        host=str(section.get("host") or "127.0.0.1"),
        port=int(section.get("port") or 9120),
        auth_ttl_seconds=int(section.get("auth_ttl_seconds") or 300),
        session_ttl_seconds=int(section.get("session_ttl_seconds") or 3600),
        allowed_users={str(value) for value in allowed if str(value).strip()},
        action_owners={str(value) for value in action_owners if str(value).strip()},
        hermes_home=str(get_hermes_home()),
        cors_allowed_origins=cors_allowed_origins or _default_allowed_origins(),
        action_initdata_ttl_seconds=int(section.get("action_initdata_ttl_seconds") or 900),
        # bridge_enabled is the durable switch the gateway-side bridge service
        # reads to decide whether to run its export/resolve cycle. It does NOT
        # register the sidecar decision route (that stays --enable-actions only)
        # and does nothing until a gateway wiring reads it; default off.
        bridge_enabled=_as_bool(section.get("bridge_enabled")),
        enable_actions=False,
        public_smoke=False,
    )

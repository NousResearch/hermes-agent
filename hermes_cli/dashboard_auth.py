"""Helpers for configuring native dashboard authentication."""

from __future__ import annotations

from typing import Any

from hermes_cli.config import load_config, save_config
from hermes_cli.web_server import hash_dashboard_password


def _dashboard_auth_from_config(config: dict[str, Any]) -> dict[str, Any]:
    dashboard = config.setdefault("dashboard", {})
    if not isinstance(dashboard, dict):
        dashboard = {}
        config["dashboard"] = dashboard
    auth = dashboard.setdefault("auth", {})
    if not isinstance(auth, dict):
        auth = {}
        dashboard["auth"] = auth
    return auth


def configure_dashboard_auth(*, username: str, password: str) -> dict[str, Any]:
    """Enable native dashboard auth and store a PBKDF2 password hash."""
    username = (username or "").strip()
    if not username:
        raise ValueError("dashboard auth username must not be empty")
    if not password:
        raise ValueError("dashboard auth password must not be empty")

    config = load_config()
    auth = _dashboard_auth_from_config(config)
    auth["enabled"] = True
    auth["username"] = username
    auth["password_hash"] = hash_dashboard_password(password)
    save_config(config)
    return dict(auth)


def disable_dashboard_auth() -> dict[str, Any]:
    """Disable native dashboard auth and clear the password hash."""
    config = load_config()
    auth = _dashboard_auth_from_config(config)
    auth["enabled"] = False
    auth["username"] = str(auth.get("username") or "")
    auth["password_hash"] = ""
    save_config(config)
    return dict(auth)


def dashboard_auth_status() -> dict[str, Any]:
    """Return safe dashboard auth status without exposing the password hash."""
    config = load_config()
    dashboard = config.get("dashboard", {})
    auth = dashboard.get("auth", {}) if isinstance(dashboard, dict) else {}
    auth = auth if isinstance(auth, dict) else {}
    username = str(auth.get("username") or "")
    password_hash = str(auth.get("password_hash") or "")
    return {
        "enabled": bool(auth.get("enabled")),
        "configured": bool(username and password_hash),
        "username": username,
    }

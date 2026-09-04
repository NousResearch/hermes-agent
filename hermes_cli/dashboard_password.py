"""Secure dashboard basic-auth password setup and rotation."""

from __future__ import annotations

import getpass
import secrets
import sys
from typing import Any

from hermes_constants import display_hermes_home

_USERNAME_ENV = "HERMES_DASHBOARD_BASIC_AUTH_USERNAME"
_PASSWORD_ENV = "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD"
_PASSWORD_HASH_ENV = "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH"
_SECRET_ENV = "HERMES_DASHBOARD_BASIC_AUTH_SECRET"


def _is_interactive() -> bool:
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def _configured_value(env_name: str, config_name: str) -> str:
    from hermes_cli.config import get_env_value_prefer_dotenv, load_config

    value = str(get_env_value_prefer_dotenv(env_name) or "").strip()
    if value:
        return value
    config = load_config()
    dashboard = config.get("dashboard", {}) if isinstance(config, dict) else {}
    basic = dashboard.get("basic_auth", {}) if isinstance(dashboard, dict) else {}
    return str(basic.get(config_name, "") or "").strip() if isinstance(basic, dict) else ""


def _read_password(*, generate: bool) -> tuple[str, bool]:
    if generate:
        return secrets.token_urlsafe(24), True
    if not _is_interactive():
        raise SystemExit(
            "Interactive password entry requires a TTY; use --generate for headless rotation."
        )
    try:
        password = getpass.getpass("Dashboard password: ")
        confirmation = getpass.getpass("Confirm dashboard password: ")
    except (EOFError, KeyboardInterrupt) as exc:
        raise SystemExit("Dashboard password rotation cancelled.") from exc
    if not password:
        raise SystemExit("Dashboard password cannot be empty.")
    if password != confirmation:
        raise SystemExit("Dashboard passwords do not match.")
    return password, False


def cmd_dashboard_password(args: Any) -> None:
    """Generate or securely prompt for a basic-auth password, then persist its hash."""
    from hermes_cli.config import save_env_values
    from plugins.dashboard_auth.basic import hash_password

    username = str(getattr(args, "username", None) or "").strip()
    if not username:
        username = _configured_value(_USERNAME_ENV, "username") or "admin"
    if not username.isascii() or "\n" in username or "\r" in username:
        raise SystemExit(
            "Dashboard username must contain only ASCII characters without line breaks; "
            "password was not changed."
        )

    password, generated = _read_password(generate=bool(getattr(args, "generate", False)))

    force_logout = bool(getattr(args, "force_logout", False))
    current_secret = _configured_value(_SECRET_ENV, "secret")
    if force_logout:
        secret_state = "rotated"
        secret = secrets.token_urlsafe(32)
    elif current_secret:
        secret_state = "preserved"
        secret = current_secret
    else:
        secret_state = "created"
        secret = secrets.token_urlsafe(32)

    updates = {
        _USERNAME_ENV: username,
        _PASSWORD_HASH_ENV: hash_password(password),
        _PASSWORD_ENV: None,
    }
    if secret_state != "preserved":
        updates[_SECRET_ENV] = secret

    if not save_env_values(updates):
        raise SystemExit("Dashboard password was not changed.")

    if generated:
        print(f"Generated dashboard password (displayed once): {password}")
    print(f"Dashboard password updated for user {username!r}.")
    print(f"Stored only its scrypt hash in {display_hermes_home()}/.env.")
    if secret_state == "rotated":
        print("The session-signing secret was rotated; existing sessions will be invalid after restart.")
    elif secret_state == "created":
        print("A session-signing secret was created; existing sessions will be invalid after restart.")
    else:
        print("The session-signing secret was preserved; existing sessions remain valid.")
    print("Restart the dashboard/backend before using the new password.")

"""Configure or rotate Dashboard username/password credentials."""

from __future__ import annotations

import getpass
import os
import secrets


_CREDENTIAL_ENV_OVERRIDES = (
    "HERMES_DASHBOARD_BASIC_AUTH_USERNAME",
    "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH",
    "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD",
    "HERMES_DASHBOARD_BASIC_AUTH_SECRET",
)


def cmd_dashboard_credentials(_args) -> None:
    """Configure new credentials or rotate existing credentials securely."""
    shadowing = [
        name for name in _CREDENTIAL_ENV_OVERRIDES if os.environ.get(name, "").strip()
    ]
    if shadowing:
        raise SystemExit(
            "Dashboard credentials are controlled by environment variable(s): "
            f"{', '.join(shadowing)}. Remove those overrides before using this command."
        )

    from hermes_cli.config import load_config, save_config
    from hermes_cli.plugins_cmd import ensure_basic_auth_plugin_enabled_in_config
    from plugins.dashboard_auth.basic import hash_password, verify_password

    cfg = load_config()
    dashboard = cfg.get("dashboard")
    basic = dashboard.get("basic_auth", {}) if isinstance(dashboard, dict) else {}
    if not isinstance(basic, dict):
        raise SystemExit("dashboard.basic_auth in config.yaml must be a mapping.")

    password_hash = str(basic.get("password_hash", "") or "").strip()
    plaintext = str(basic.get("password", "") or "")
    if not password_hash and plaintext:
        password_hash = hash_password(plaintext)

    if password_hash:
        print("Dashboard credentials are already configured.")
        try:
            change = input("Change username and password? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("Credential update cancelled.") from None
        if change not in {"y", "yes"}:
            print("Credential update cancelled.")
            return
        try:
            old_password = getpass.getpass("Old password: ")
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("Credential update cancelled.") from None
        if not verify_password(old_password, password_hash):
            raise SystemExit("old password is incorrect; configuration was not changed.")

    # Keep the first-run setup behavior: the initial secret must remain stable
    # across a restart. Existing credentials take the explicit rotation path
    # above and receive a fresh signing secret below.
    rotating = bool(password_hash)

    current_username = str(basic.get("username", "") or "").strip()
    default_username = current_username or "admin"
    try:
        username = input(f"New username [{default_username}]: ").strip() or default_username
        password = getpass.getpass("New password: ")
        confirmation = getpass.getpass("Confirm new password: ")
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("Credential update cancelled.") from None
    if not username:
        raise SystemExit("Username must not be empty.")
    if not password:
        raise SystemExit("Password must not be empty.")
    if password != confirmation:
        raise SystemExit("Passwords do not match; configuration was not changed.")
    dashboard = cfg.setdefault("dashboard", {})
    if not isinstance(dashboard, dict):
        raise SystemExit("dashboard in config.yaml must be a mapping.")
    basic = dashboard.setdefault("basic_auth", {})
    if not isinstance(basic, dict):
        raise SystemExit("dashboard.basic_auth in config.yaml must be a mapping.")

    basic["username"] = username
    basic["password_hash"] = hash_password(password)
    basic["password"] = ""
    # New credentials invalidate access and refresh tokens from the old secret.
    if rotating or not str(basic.get("secret", "") or "").strip():
        basic["secret"] = secrets.token_urlsafe(32)
    ensure_basic_auth_plugin_enabled_in_config(cfg)
    save_config(cfg)

    print(f"Dashboard credentials updated for user: {username}")
    print("After restart, existing dashboard sessions will be invalidated.")
    print(
        "The running dashboard keeps credentials in memory and does not reload "
        "them dynamically."
    )
    print(
        "Restart the Dashboard using the same command or service manager "
        "that started it."
    )
    print("For a manually started Dashboard:")
    print("  hermes dashboard --stop")
    print("  hermes dashboard --host <host> --port <port> --no-open")

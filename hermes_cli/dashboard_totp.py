"""``hermes dashboard totp`` — enrol / remove an authenticator app as the dashboard's 2nd factor.

Generates a fresh RFC 6238 secret, shows it as a QR code (when the optional ``qrcode`` package
is installed) plus the manual-entry key, asks for one code from the app to prove the pairing
worked, then persists ``dashboard.basic_auth.totp_secret`` in config.yaml (``--env`` writes
``HERMES_DASHBOARD_BASIC_AUTH_TOTP_SECRET`` to ``.env`` instead, for installs whose basic-auth
settings live there). The running dashboard must be restarted to pick it up.
"""

from __future__ import annotations

import sys

from hermes_cli.dashboard_auth.totp import (
    decode_totp_secret, generate_totp_secret, totp_provisioning_uri, verify_totp)

_ENV_KEY = "HERMES_DASHBOARD_BASIC_AUTH_TOTP_SECRET"
_ISSUER = "Hermes Dashboard"


def _current_basic_auth() -> tuple[dict, str, str]:
    """``(basic_auth config section, effective username, effective totp_secret)`` with the same
    env-over-config precedence the provider applies."""
    import os
    from hermes_cli.config import load_config
    section = (load_config().get("dashboard") or {}).get("basic_auth") or {}
    username = os.environ.get("HERMES_DASHBOARD_BASIC_AUTH_USERNAME", "").strip() or str(section.get("username") or "")
    existing = os.environ.get(_ENV_KEY, "").strip() or str(section.get("totp_secret") or "")
    return section, username, existing


def _render_qr(uri: str) -> bool:
    """Print the QR to the terminal; False when ``qrcode`` isn't installed."""
    try:
        import qrcode  # optional extra (pyproject ``messaging``/``feishu``/``dingtalk``)
    except ImportError:
        return False
    qr = qrcode.QRCode(border=2)
    qr.add_data(uri)
    qr.make(fit=True)
    qr.print_ascii(out=sys.stdout, invert=True)
    return True


def _persist(secret: str, *, use_env: bool) -> str:
    """Write the secret; returns a one-line description of where it went."""
    from hermes_cli.config import get_env_path, save_env_value
    if use_env:
        save_env_value(_ENV_KEY, secret)
        return f"{get_env_path()} ({_ENV_KEY})"
    from hermes_cli.config import load_config, save_config
    from hermes_cli.plugins_cmd import ensure_basic_auth_plugin_enabled_in_config
    cfg = load_config()
    cfg.setdefault("dashboard", {}).setdefault("basic_auth", {})["totp_secret"] = secret
    if ensure_basic_auth_plugin_enabled_in_config(cfg):
        print("  ✓ Re-enabled the bundled 'basic' auth plugin (was in plugins.disabled)")
    save_config(cfg)
    return "config.yaml (dashboard.basic_auth.totp_secret)"


def _clear(*, use_env: bool) -> None:
    import os
    from hermes_cli.config import get_env_value, load_config, save_config, save_env_value
    if use_env or get_env_value(_ENV_KEY) or os.environ.get(_ENV_KEY):
        # An empty env value is "unset" to the provider (resolve_env_or_cfg), so it can't
        # shadow anything; no need to delete the line.
        save_env_value(_ENV_KEY, "")
    cfg = load_config()
    basic = (cfg.get("dashboard") or {}).get("basic_auth")
    if isinstance(basic, dict) and basic.get("totp_secret"):
        basic["totp_secret"] = ""
        save_config(cfg)


def cmd_dashboard_totp(args) -> None:
    from hermes_cli.config import is_managed
    if is_managed():
        print("✗ This is a managed install; the orchestrator owns dashboard auth settings.", file=sys.stderr)
        sys.exit(1)

    use_env = bool(getattr(args, "env", False))
    section, username, existing = _current_basic_auth()

    if getattr(args, "disable", False):
        if not existing:
            print("Two-factor authentication is not enabled for the dashboard.")
            return
        _clear(use_env=use_env)
        print("✓ Two-factor authentication disabled. Restart the dashboard to apply.")
        return

    if not username:
        print("✗ Username/password dashboard auth is not configured yet.\n"
              "  Set dashboard.basic_auth (start `hermes dashboard` on a non-loopback host, or\n"
              "  set HERMES_DASHBOARD_BASIC_AUTH_USERNAME/PASSWORD in .env) and run this again.",
              file=sys.stderr)
        sys.exit(1)

    if existing and not getattr(args, "force", False):
        print("Two-factor authentication is already enabled for the dashboard.\n"
              "  Re-run with --force to pair a new authenticator (the old one stops working),\n"
              "  or --disable to turn it off.")
        return

    secret = generate_totp_secret()
    uri = totp_provisioning_uri(secret, account=username, issuer=_ISSUER)
    manual_key = " ".join(secret[i:i + 4] for i in range(0, len(secret), 4))

    print("\nPair an authenticator app with the Hermes dashboard\n")
    print("  Microsoft Authenticator:  +  →  Other account (Google, Facebook, etc.)  →  scan the QR")
    print("  (Google Authenticator, Authy, 1Password, Bitwarden, … work the same way.)\n")
    if not _render_qr(uri):
        print("  (Install the optional 'qrcode' package to get a scannable QR here.)\n")
    print("  Can't scan? Choose 'enter code manually' / 'enter a setup key' and type:")
    print(f"    Account:  {_ISSUER} ({username})")
    print(f"    Key:      {manual_key}")
    print("    Type:     time-based (TOTP), 6 digits, 30 seconds\n")
    if getattr(args, "show_uri", False):
        print(f"  otpauth URI: {uri}\n")

    if not getattr(args, "yes", False):
        if not sys.stdin.isatty():
            print("✗ Not a TTY: pass --yes to skip the confirmation code (not recommended).", file=sys.stderr)
            sys.exit(1)
        secret_bytes = decode_totp_secret(secret)
        for attempt in range(3):
            try:
                code = input("  Enter the 6-digit code the app shows now: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Cancelled — nothing was saved.")
                sys.exit(1)
            if verify_totp(secret_bytes, code) is not None:
                break
            print("  ✗ That code didn't match" + (", try again." if attempt < 2 else "."))
        else:
            print("\n  ✗ Giving up — nothing was saved. Check the phone's clock is set automatically.",
                  file=sys.stderr)
            sys.exit(1)

    where = _persist(secret, use_env=use_env)
    print(f"\n✓ Two-factor authentication enabled for user '{username}'.")
    print(f"  Saved to {where}.")
    print("  Restart the dashboard to apply; the login form will then ask for the authenticator code.")
    print("  Lost the phone? Run `hermes dashboard totp --disable` on this host.\n")

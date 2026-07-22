"""``hermes xchat ...`` CLI subcommands — registered by the plugin via
``ctx.register_cli_command()``.

Subcommands:

    setup       full first-time setup (token + user id + key generation/registration)
    register    (re)register the E2EE public keys only
    status      show token / key / registration state

Key registration is a rare, rate-limited write (only a few per 24h per
account). ``setup`` is safe to re-run: the private-key blob and the
registration payload are persisted to ``~/.hermes/xchat/`` BEFORE any
network call, so an interrupted run resumes the same identity instead of
minting a new one and burning the daily budget.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from hermes_cli.colors import Colors, color


def _state_dir() -> Path:
    from hermes_constants import get_hermes_home

    d = get_hermes_home() / "xchat"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _blob_path() -> Path:
    return _state_dir() / "private_keys.b64"


def _marker_path() -> Path:
    return _state_dir() / "registration.json"


def _read_marker() -> dict:
    try:
        return json.loads(_marker_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_marker(marker: dict) -> None:
    _marker_path().write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# argparse wiring


def register_cli(parser: argparse.ArgumentParser) -> None:
    """Wire up `hermes xchat ...` subcommands."""
    subs = parser.add_subparsers(dest="xchat_command", required=False)

    p_setup = subs.add_parser(
        "setup",
        help="First-time setup (OAuth token + E2EE key generation/registration)",
    )
    p_setup.add_argument(
        "--force",
        action="store_true",
        help="Generate a NEW identity even if one is already registered (dangerous)",
    )

    subs.add_parser("register", help="(Re)register the E2EE public keys with the X API")
    subs.add_parser("status", help="Show token / key / registration state")

    parser.set_defaults(func=dispatch)


def dispatch(args: argparse.Namespace) -> int:
    sub = getattr(args, "xchat_command", None)
    if sub in (None, "status"):
        return cmd_status()
    if sub == "setup":
        return cmd_setup(force=getattr(args, "force", False))
    if sub == "register":
        return cmd_register(force=False)
    print(color(f"Unknown xchat subcommand: {sub}", Colors.RED))
    return 1


def gateway_setup() -> None:
    """Zero-arg hook for the unified `hermes gateway setup` wizard."""
    cmd_setup(force=False)


# ---------------------------------------------------------------------------
# Helpers


def _get_env(key: str) -> str:
    val = os.getenv(key, "").strip()
    if val:
        return val
    # Fall back to the persisted .env (the CLI may run before env load).
    try:
        from hermes_cli.config import get_env_value

        return (get_env_value(key) or "").strip()
    except Exception:
        return ""


def _save_env(key: str, value: str) -> None:
    from hermes_cli.config import save_env_value

    save_env_value(key, value)
    os.environ[key] = value


def _run(coro):
    return asyncio.run(coro)


async def _fetch_user_id(api) -> str:
    me = await api.get_my_user()
    return str(me.get("id") or "")


# ---------------------------------------------------------------------------
# Commands


def cmd_status() -> int:
    token = _get_env("XCHAT_ACCESS_TOKEN")
    user_id = _get_env("XCHAT_USER_ID")
    marker = _read_marker()
    blob = _blob_path()

    print(color("X Chat integration status", Colors.BOLD))
    print(f"  access token:  {'✓ set' if token else '✗ missing'}")
    print(f"  refresh token: {'✓ set' if _get_env('XCHAT_REFRESH_TOKEN') else '– not set (no auto-renew)'}")
    print(f"  bot user id:   {user_id or '– not set'}")
    print(f"  key blob:      {'✓ ' + str(blob) if blob.exists() else '✗ missing'}")
    if marker.get("registered"):
        print(f"  registration:  ✓ version {marker.get('version')} ({marker.get('registered_at', '?')})")
    else:
        print("  registration:  ✗ not registered — run `hermes xchat setup`")
    return 0


def cmd_setup(*, force: bool) -> int:
    print(color("X Chat setup — end-to-end encrypted X DMs", Colors.BOLD))
    print(
        "You need an X developer app with OAuth 2.0 user-context enabled and a\n"
        "user access token carrying: dm.read dm.write users.read tweet.read\n"
        "(offline.access too if you want refresh tokens).\n"
        "Docs: https://docs.x.com/xchat/getting-started\n"
    )

    # 1. Token
    token = _get_env("XCHAT_ACCESS_TOKEN")
    if token:
        print("Found existing XCHAT_ACCESS_TOKEN.")
    else:
        import getpass

        token = getpass.getpass("Paste the OAuth2 user access token: ").strip()
        if not token:
            print(color("No token provided — aborting.", Colors.RED))
            return 1
        _save_env("XCHAT_ACCESS_TOKEN", token)
    refresh = _get_env("XCHAT_REFRESH_TOKEN")
    if not refresh:
        import getpass

        refresh = getpass.getpass(
            "Paste the OAuth2 refresh token (optional, Enter to skip): "
        ).strip()
        if refresh:
            _save_env("XCHAT_REFRESH_TOKEN", refresh)
            client_id = input("X app OAuth2 client id (needed for refresh): ").strip()
            if client_id:
                _save_env("XCHAT_CLIENT_ID", client_id)

    # 2. Bot user id
    from .api import XChatApi, XChatApiError

    api = XChatApi(token)
    user_id = _get_env("XCHAT_USER_ID")
    if not user_id:
        try:
            user_id = _run(_fetch_user_id(api))
        except XChatApiError as e:
            print(color(f"Could not resolve the bot's user id: {e}", Colors.RED))
            print("Check the token's scopes (users.read) and validity.")
            return 1
        finally:
            _run(api.aclose())
            api = None
        if not user_id:
            print(color("Could not resolve the bot's user id.", Colors.RED))
            return 1
        _save_env("XCHAT_USER_ID", user_id)
        print(f"Bot user id: {user_id}")
    else:
        _run(api.aclose())
        api = None

    # 3. Keys + registration
    rc = cmd_register(force=force)
    if rc != 0:
        return rc

    print()
    print(color("Setup complete.", Colors.GREEN))
    print("Enable the platform and start the gateway:")
    print("  hermes gateway start")
    return 0


def cmd_register(*, force: bool) -> int:
    """Generate + register the E2EE keys. Re-runnable / resume-safe."""
    token = _get_env("XCHAT_ACCESS_TOKEN")
    if not token:
        print(color("XCHAT_ACCESS_TOKEN not set — run `hermes xchat setup` first.", Colors.RED))
        return 1
    user_id = _get_env("XCHAT_USER_ID")

    from .api import XChatApi, XChatApiError, XChatRateLimited
    from .crypto import XChatCrypto

    marker = _read_marker()
    if marker.get("registered") and not force:
        print(
            f"Already registered (key version {marker.get('version')}). "
            "Use `hermes xchat setup --force` to mint a NEW identity."
        )
        return 0

    async def _register() -> int:
        nonlocal user_id
        api = XChatApi(
            token,
            refresh_token=_get_env("XCHAT_REFRESH_TOKEN"),
            client_id=_get_env("XCHAT_CLIENT_ID"),
            client_secret=_get_env("XCHAT_CLIENT_SECRET"),
        )
        try:
            if not user_id:
                user_id = await _fetch_user_id(api)
                if not user_id:
                    print(color("Could not resolve the bot's user id.", Colors.RED))
                    return 1
                _save_env("XCHAT_USER_ID", user_id)

            crypto = XChatCrypto()
            blob_path = _blob_path()
            resuming = blob_path.exists() and marker.get("body") and not force
            if resuming:
                crypto.load_keys(blob_path.read_text(encoding="utf-8").strip())
                body = marker["body"]
                version = str(marker.get("version") or "1")
                print(f"Resuming the saved identity ({blob_path}).")
            else:
                payload = crypto.generate_and_register_payload()
                body = payload["registration"]
                version = payload["version"]
                blob_path.write_text(payload["private_keys_b64"] + "\n", encoding="utf-8")
                try:
                    blob_path.chmod(0o600)
                except OSError:
                    pass
                _write_marker(
                    {"registered": False, "user_id": user_id, "version": version, "body": body}
                )
                print(f"Generated a new identity; private keys saved to {blob_path} (mode 600).")

            our_public_key = body["public_key"]["public_key"]

            # Reconcile: adopt an already-registered key instead of re-POSTing
            # (a prior POST may have applied server-side after erroring).
            try:
                existing = await api.get_public_keys(user_id)
            except XChatApiError:
                existing = []
            already = next(
                (k for k in existing if k.get("public_key") == our_public_key), None
            )
            if already:
                version = str(already.get("public_key_version") or version)
                print(f"Public key already registered (version {version}); skipping POST.")
            else:
                print(f"Registering public key version {version} …")
                try:
                    resp = await api.add_public_key(user_id, body)
                except XChatRateLimited as limited:
                    when = (
                        datetime.fromtimestamp(limited.reset_epoch, tz=timezone.utc).isoformat()
                        if limited.reset_epoch
                        else "the next window"
                    )
                    print(
                        color(
                            "Registration is rate limited (429). The daily budget is "
                            f"exhausted; wait until {when} and re-run — the saved "
                            "identity resumes, so no budget is wasted.",
                            Colors.RED,
                        )
                    )
                    return 1
                data = resp.get("data") or {}
                if isinstance(data, list):
                    data = data[0] if data else {}
                version = str(data.get("public_key_version") or version)

            _save_env("XCHAT_SIGNING_KEY_VERSION", version)
            _write_marker(
                {
                    "registered": True,
                    "user_id": user_id,
                    "version": version,
                    "body": body,
                    "registered_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            print(color(f"Key registration complete (version {version}).", Colors.GREEN))
            return 0
        finally:
            await api.aclose()

    try:
        return _run(_register())
    except XChatApiError as e:
        print(color(f"Registration failed: {e}", Colors.RED))
        return 1

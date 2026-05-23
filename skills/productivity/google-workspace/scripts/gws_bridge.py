#!/usr/bin/env python3
"""Bridge between Hermes OAuth token and gws CLI.

Refreshes the token if expired, then executes gws with the valid access token.

Multi-account support: pass ``--account EMAIL`` BEFORE the gws args, e.g.:

    gws_bridge.py --account user@example.com gmail +triage

If ``--account`` is omitted, the resolution chain (HERMES_GOOGLE_ACCOUNT env →
default pointer → legacy single-account token) decides which account is used.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure sibling modules (_hermes_home, google_account) are importable when
# run standalone.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from _hermes_home import get_hermes_home
import google_account


# The active account is resolved once at process start so all helper functions
# below see the same value (matters when the user passes --account).
_ACCOUNT: "str | None" = None


def _set_active_account(account: "str | None") -> None:
    global _ACCOUNT
    _ACCOUNT = account


def get_token_path() -> Path:
    """Token path for the active account.

    Resolution: explicit ``--account`` (parsed in main()) → HERMES_GOOGLE_ACCOUNT
    env → default account pointer → legacy ``~/.hermes/google_token.json``.
    """
    return google_account.resolve_token_path(_ACCOUNT)


def _normalize_authorized_user_payload(payload: dict) -> dict:
    normalized = dict(payload)
    if not normalized.get("type"):
        normalized["type"] = "authorized_user"
    return normalized


def refresh_token(token_data: dict) -> dict:
    """Refresh the access token using the refresh token."""
    import urllib.error
    import urllib.parse
    import urllib.request

    required_keys = ["client_id", "client_secret", "refresh_token", "token_uri"]
    missing = [k for k in required_keys if k not in token_data]
    if missing:
        print(f"ERROR: google_token.json is missing required fields: {', '.join(missing)}", file=sys.stderr)
        print("Please re-authenticate by running the Google Workspace setup script.", file=sys.stderr)
        sys.exit(1)

    params = urllib.parse.urlencode({
        "client_id": token_data["client_id"],
        "client_secret": token_data["client_secret"],
        "refresh_token": token_data["refresh_token"],
        "grant_type": "refresh_token",
    }).encode()

    req = urllib.request.Request(token_data["token_uri"], data=params)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"ERROR: Token refresh failed (HTTP {e.code}): {body}", file=sys.stderr)
        print("Re-run setup.py to re-authenticate.", file=sys.stderr)
        sys.exit(1)
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"ERROR: Token refresh failed (network): {e}", file=sys.stderr)
        sys.exit(1)

    token_data["token"] = result["access_token"]
    token_data["expiry"] = datetime.fromtimestamp(
        datetime.now(timezone.utc).timestamp() + result["expires_in"],
        tz=timezone.utc,
    ).isoformat()

    get_token_path().write_text(
        json.dumps(_normalize_authorized_user_payload(token_data), indent=2)
    )
    return token_data


def get_valid_token() -> str:
    """Return a valid access token, refreshing if needed."""
    token_path = get_token_path()
    if not token_path.exists():
        print(
            f"ERROR: No Google token found at {token_path}. "
            "Run setup.py --auth-url first.",
            file=sys.stderr,
        )
        sys.exit(1)

    token_data = json.loads(token_path.read_text())

    expiry = token_data.get("expiry", "")
    if expiry:
        exp_dt = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        if now >= exp_dt:
            token_data = refresh_token(token_data)

    return token_data["token"]


def _split_account_arg(argv: list[str]) -> tuple["str | None", list[str]]:
    """Strip a leading ``--account EMAIL`` (or ``--account=EMAIL``) from argv.

    Anything after this is passed verbatim to gws. We do this manually rather
    than with argparse because gws has its own arg syntax (subcommands,
    positional flags) and we don't want to interpret it.
    """
    if not argv:
        return None, argv
    first = argv[0]
    if first == "--account":
        if len(argv) < 2:
            print("ERROR: --account requires an email address.", file=sys.stderr)
            sys.exit(2)
        return argv[1], argv[2:]
    if first.startswith("--account="):
        return first.split("=", 1)[1], argv[1:]
    return None, argv


def main():
    """Refresh token if needed, then exec gws with remaining args."""
    if len(sys.argv) < 2:
        print("Usage: gws_bridge.py [--account EMAIL] <gws args...>", file=sys.stderr)
        sys.exit(1)

    account, gws_args = _split_account_arg(sys.argv[1:])
    if account is not None:
        try:
            account = google_account.normalize_email(account)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)
    _set_active_account(account)

    if not gws_args:
        print("Usage: gws_bridge.py [--account EMAIL] <gws args...>", file=sys.stderr)
        sys.exit(1)

    access_token = get_valid_token()
    env = os.environ.copy()
    env["GOOGLE_WORKSPACE_CLI_TOKEN"] = access_token

    result = subprocess.run(["gws"] + gws_args, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

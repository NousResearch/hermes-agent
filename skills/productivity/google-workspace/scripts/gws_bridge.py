#!/usr/bin/env python3
"""Bridge between Hermes OAuth token and gws CLI.

Refreshes the token if expired, then executes gws with the valid access token.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure sibling modules (_hermes_home) are importable when run standalone.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from _hermes_home import get_hermes_home
import auth_contexts as gauth

CURRENT_AUTH_CONTEXT = os.getenv("HERMES_GOOGLE_AUTH_CONTEXT", "default")


def get_token_path() -> Path:
    if CURRENT_AUTH_CONTEXT == "default" and not gauth.store_path().exists():
        return gauth.legacy_token_path()
    return gauth.materialize_token_file(CURRENT_AUTH_CONTEXT)


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

    refreshed = _normalize_authorized_user_payload(token_data)
    if CURRENT_AUTH_CONTEXT == "default" and gauth.legacy_token_path().exists() and not gauth.store_path().exists():
        get_token_path().write_text(json.dumps(refreshed, indent=2))
        try:
            get_token_path().chmod(0o600)
        except OSError:
            pass
    else:
        gauth.set_token_payload(CURRENT_AUTH_CONTEXT, refreshed)
    return token_data


def get_valid_token() -> str:
    """Return a valid access token, refreshing if needed."""
    token_path = get_token_path()
    if not token_path.exists():
        print("ERROR: No Google token found. Run setup.py --auth-url first.", file=sys.stderr)
        sys.exit(1)

    token_data = json.loads(token_path.read_text())

    expiry = token_data.get("expiry", "")
    if expiry:
        exp_dt = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        if now >= exp_dt:
            token_data = refresh_token(token_data)

    return token_data["token"]


def main():
    """Refresh token if needed, then exec gws with remaining args."""
    global CURRENT_AUTH_CONTEXT
    argv = sys.argv[1:]
    if "--auth-context" in argv:
        idx = argv.index("--auth-context")
        try:
            CURRENT_AUTH_CONTEXT = argv[idx + 1]
        except IndexError:
            print("ERROR: --auth-context requires a value", file=sys.stderr)
            sys.exit(2)
        del argv[idx:idx + 2]
    if not argv:
        print("Usage: gws_bridge.py [--auth-context NAME] <gws args...>", file=sys.stderr)
        sys.exit(1)

    access_token = get_valid_token()
    env = os.environ.copy()
    env["GOOGLE_WORKSPACE_CLI_TOKEN"] = access_token

    result = subprocess.run(["gws"] + argv, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _hermes_home import get_hermes_home

TOKEN_REFRESH_TIMEOUT_SECONDS = 30


def get_token_path() -> Path:
    return get_hermes_home() / "google_token.json"


def _parse_expiry(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _token_is_current(data: dict) -> bool:
    expiry = _parse_expiry(data.get("expiry"))
    if expiry is None:
        return True
    return expiry > datetime.now(timezone.utc) + timedelta(minutes=1)


def _write_token(path: Path, data: dict) -> None:
    data["type"] = "authorized_user"
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def get_valid_token() -> str:
    token_path = get_token_path()
    if not token_path.exists():
        print(f"Missing Google token file: {token_path}", file=sys.stderr)
        raise SystemExit(1)

    data = json.loads(token_path.read_text(encoding="utf-8"))
    token = data.get("token") or data.get("access_token")
    if token and _token_is_current(data):
        return token

    refresh_token = data.get("refresh_token")
    token_uri = data.get("token_uri", "https://oauth2.googleapis.com/token")
    client_id = data.get("client_id")
    client_secret = data.get("client_secret")
    if not (refresh_token and client_id and client_secret):
        print("Google token is expired and lacks refresh material.", file=sys.stderr)
        raise SystemExit(1)

    body = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        token_uri,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TOKEN_REFRESH_TIMEOUT_SECONDS) as response:
            refreshed = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"Google token refresh failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    access_token = refreshed.get("access_token")
    if not access_token:
        print("Google token refresh response did not include access_token.", file=sys.stderr)
        raise SystemExit(1)

    data["token"] = access_token
    expires_in = int(refreshed.get("expires_in") or 3600)
    data["expiry"] = (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()
    _write_token(token_path, data)
    return access_token


def main() -> None:
    token = get_valid_token()
    env = os.environ.copy()
    env["GOOGLE_WORKSPACE_CLI_TOKEN"] = token
    proc = subprocess.run(["gws", *sys.argv[1:]], env=env)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()

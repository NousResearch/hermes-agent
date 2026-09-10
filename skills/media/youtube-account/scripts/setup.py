"""Profile-scoped OAuth setup for read-only YouTube account access."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlparse


SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]
REDIRECT_URI = "http://localhost:1"
HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
CLIENT_PATH = HERMES_HOME / "youtube_client_secret.json"
TOKEN_PATH = HERMES_HOME / "youtube_token.json"
PENDING_PATH = HERMES_HOME / "youtube_oauth_pending.json"


def _ensure_dependencies() -> None:
    try:
        import google.auth  # noqa: F401
        import google_auth_oauthlib  # noqa: F401
        import googleapiclient  # noqa: F401
        return
    except ImportError:
        pass

    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError(
            "Google API dependencies are missing and uv is unavailable. "
            "Install google-auth, google-auth-oauthlib, and google-api-python-client."
        )
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "google-auth",
            "google-auth-oauthlib",
            "google-api-python-client",
        ],
        check=True,
    )


def _secure_file(path: Path) -> None:
    if os.name != "nt":
        path.chmod(0o600)
        return

    domain = os.environ.get("USERDOMAIN", "").strip()
    username = os.environ.get("USERNAME", getpass.getuser()).strip()
    identity = f"{domain}\\{username}" if domain else username
    result = subprocess.run(
        [
            "icacls",
            str(path),
            "/inheritance:r",
            "/grant:r",
            f"{identity}:F",
            "NT AUTHORITY\\SYSTEM:F",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not protect {path}: {result.stderr.strip()}")


def _write_private_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _secure_file(path)


def _client_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    client = payload.get("installed") or payload.get("web")
    if not isinstance(client, dict):
        raise ValueError("Expected a Google OAuth client JSON with installed or web settings")
    if not client.get("client_id") or not client.get("client_secret"):
        raise ValueError("OAuth client JSON is missing client_id or client_secret")
    return payload


def store_client(source: str) -> None:
    payload = _client_payload(Path(source).expanduser().resolve())
    _write_private_json(CLIENT_PATH, payload)
    print(f"OK: YouTube OAuth client saved to {CLIENT_PATH}")


def _credentials():
    _ensure_dependencies()
    from google.oauth2.credentials import Credentials

    return Credentials.from_authorized_user_file(str(TOKEN_PATH))


def check() -> bool:
    if not TOKEN_PATH.exists():
        print(f"NOT_AUTHENTICATED: No token at {TOKEN_PATH}")
        return False

    _ensure_dependencies()
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    try:
        credentials = _credentials()
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
            _write_private_json(TOKEN_PATH, json.loads(credentials.to_json()))
        if not credentials.valid:
            print("NOT_AUTHENTICATED: YouTube token is invalid")
            return False
        granted = set(credentials.scopes or [])
        if not set(SCOPES).issubset(granted):
            print("NOT_AUTHENTICATED: Token is missing youtube.readonly")
            return False
        (
            build("youtube", "v3", credentials=credentials, cache_discovery=False)
            .channels()
            .list(part="id", mine=True, maxResults=1)
            .execute()
        )
    except Exception as error:
        print(f"NOT_AUTHENTICATED: {type(error).__name__}")
        return False

    print(f"AUTHENTICATED: Read-only YouTube grant is valid at {TOKEN_PATH}")
    return True


def auth_url() -> None:
    if not CLIENT_PATH.exists():
        raise FileNotFoundError("No YouTube OAuth client is stored; run --client-secret first")

    _ensure_dependencies()
    from google_auth_oauthlib.flow import Flow

    flow = Flow.from_client_secrets_file(
        str(CLIENT_PATH),
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
        autogenerate_code_verifier=True,
    )
    url, state = flow.authorization_url(access_type="offline", prompt="consent")
    _write_private_json(
        PENDING_PATH,
        {
            "state": state,
            "code_verifier": flow.code_verifier,
            "redirect_uri": REDIRECT_URI,
        },
    )
    print(url)


def _code_and_state(value: str) -> tuple[str, str | None]:
    if not value.startswith("http"):
        return value, None
    query = parse_qs(urlparse(value).query)
    code = (query.get("code") or [""])[0]
    if not code:
        raise ValueError("Redirect URL does not contain an OAuth code")
    return code, (query.get("state") or [None])[0]


def exchange(value: str) -> None:
    if not CLIENT_PATH.exists() or not PENDING_PATH.exists():
        raise FileNotFoundError("Run --client-secret and --auth-url before --auth-code")

    pending = json.loads(PENDING_PATH.read_text(encoding="utf-8"))
    code, returned_state = _code_and_state(value)
    if returned_state and returned_state != pending.get("state"):
        raise ValueError("OAuth state mismatch; generate a fresh authorization URL")

    _ensure_dependencies()
    from google_auth_oauthlib.flow import Flow

    flow = Flow.from_client_secrets_file(
        str(CLIENT_PATH),
        scopes=SCOPES,
        redirect_uri=pending.get("redirect_uri", REDIRECT_URI),
        state=pending["state"],
        code_verifier=pending["code_verifier"],
    )
    flow.fetch_token(code=code)
    _write_private_json(TOKEN_PATH, json.loads(flow.credentials.to_json()))
    PENDING_PATH.unlink(missing_ok=True)
    print(f"OK: Read-only YouTube token saved to {TOKEN_PATH}")


def revoke() -> None:
    if not TOKEN_PATH.exists():
        PENDING_PATH.unlink(missing_ok=True)
        print("No YouTube token to revoke")
        return

    credentials = _credentials()
    token = credentials.refresh_token or credentials.token
    if token:
        try:
            request = urllib.request.Request(
                f"https://oauth2.googleapis.com/revoke?token={token}", method="POST"
            )
            urllib.request.urlopen(request, timeout=15).read()
        except Exception:
            pass
    TOKEN_PATH.unlink(missing_ok=True)
    PENDING_PATH.unlink(missing_ok=True)
    print("YouTube token deleted")


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up read-only YouTube OAuth")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--client-secret", metavar="PATH")
    actions.add_argument("--auth-url", action="store_true")
    actions.add_argument("--auth-code", metavar="CODE_OR_URL")
    actions.add_argument("--revoke", action="store_true")
    actions.add_argument("--install-deps", action="store_true")
    args = parser.parse_args()

    try:
        if args.check:
            raise SystemExit(0 if check() else 1)
        if args.client_secret:
            store_client(args.client_secret)
        elif args.auth_url:
            auth_url()
        elif args.auth_code:
            exchange(args.auth_code)
        elif args.revoke:
            revoke()
        elif args.install_deps:
            _ensure_dependencies()
            print("Dependencies installed")
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

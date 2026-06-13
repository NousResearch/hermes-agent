from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _hermes_home import display_hermes_home, get_hermes_home

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/contacts.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/documents.readonly",
]
REQUIRED_PACKAGES = ["google-api-python-client", "google-auth-oauthlib"]
REDIRECT_URI = "http://localhost:1/"

HERMES_HOME = get_hermes_home()
CLIENT_SECRET_PATH = HERMES_HOME / "google_client_secret.json"
TOKEN_PATH = HERMES_HOME / "google_token.json"
PENDING_AUTH_PATH = HERMES_HOME / "google_oauth_pending.json"


def _deps_available() -> bool:
    for module_name in ("googleapiclient", "google_auth_oauthlib"):
        try:
            importlib.import_module(module_name)
        except ImportError:
            return False
    return True


def install_deps() -> bool:
    if _deps_available():
        return True

    pip_cmd = [sys.executable, "-m", "pip", "install", *REQUIRED_PACKAGES]
    try:
        subprocess.check_call(pip_cmd)
        return True
    except subprocess.CalledProcessError:
        pass

    uv = shutil.which("uv")
    if not uv:
        print("Google dependencies missing. Install with: uv sync --extra google or pip install hermes-agent[google]")
        return False

    uv_cmd = [uv, "pip", "install", "--python", sys.executable, *REQUIRED_PACKAGES]
    try:
        subprocess.check_call(uv_cmd)
        return True
    except subprocess.CalledProcessError:
        print("Failed to install Google dependencies via uv")
        return False


def _ensure_deps() -> None:
    if install_deps():
        return
    raise SystemExit(1)


def _flow_class():
    _ensure_deps()
    from google_auth_oauthlib.flow import Flow

    return Flow


def _load_pending() -> dict:
    if not PENDING_AUTH_PATH.exists():
        print("No pending OAuth session. Run --auth-url first.")
        raise SystemExit(1)
    return json.loads(PENDING_AUTH_PATH.read_text(encoding="utf-8"))


def _extract_code_and_scopes(value: str) -> tuple[str, list[str] | None, str | None]:
    parsed = urlparse(value)
    if parsed.scheme and parsed.query:
        qs = parse_qs(parsed.query)
        code = qs.get("code", [""])[0]
        state = qs.get("state", [""])[0] or None
        raw_scope = qs.get("scope", [""])[0]
        scopes = unquote(raw_scope).split() if raw_scope else None
        return code, scopes, state
    return value.strip(), None, None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def get_auth_url() -> None:
    Flow = _flow_class()
    flow = Flow.from_client_secrets_file(
        str(CLIENT_SECRET_PATH),
        SCOPES,
        redirect_uri=REDIRECT_URI,
        autogenerate_code_verifier=True,
    )
    url, state = flow.authorization_url(access_type="offline", prompt="consent")
    _write_json(
        PENDING_AUTH_PATH,
        {"state": state, "code_verifier": getattr(flow, "code_verifier", "")},
    )
    print(url)


def exchange_auth_code(code_or_url: str) -> None:
    pending = _load_pending()
    code, callback_scopes, callback_state = _extract_code_and_scopes(code_or_url)
    expected_state = pending.get("state") or ""
    if callback_state and callback_state != expected_state:
        print("OAuth state mismatch; refusing token exchange.")
        raise SystemExit(1)
    if not code:
        print("No OAuth code found.")
        raise SystemExit(1)

    scopes = callback_scopes or SCOPES
    Flow = _flow_class()
    flow = Flow.from_client_secrets_file(
        str(CLIENT_SECRET_PATH),
        scopes,
        redirect_uri=REDIRECT_URI,
        state=expected_state,
        code_verifier=pending.get("code_verifier") or None,
    )
    try:
        flow.fetch_token(code=code)
    except Exception as exc:
        print(f"Token exchange failed: {exc}")
        raise SystemExit(1) from exc

    token_payload = json.loads(flow.credentials.to_json())
    token_payload["type"] = "authorized_user"
    granted = set(token_payload.get("scopes") or [])
    missing = sorted(set(SCOPES) - granted) if granted else []
    if missing:
        print("Warning: token was granted narrower scopes; missing: " + ", ".join(missing))
    _write_json(TOKEN_PATH, token_payload)
    PENDING_AUTH_PATH.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Google Workspace OAuth setup")
    parser.add_argument("--auth-url", action="store_true", help="Print an OAuth consent URL")
    parser.add_argument("--exchange-code", help="Exchange an auth code or redirect URL")
    args = parser.parse_args(argv)

    if args.auth_url:
        get_auth_url()
    elif args.exchange_code:
        exchange_auth_code(args.exchange_code)
    else:
        parser.print_help()
        print(f"\nCredential directory: {display_hermes_home()}")


if __name__ == "__main__":
    main()

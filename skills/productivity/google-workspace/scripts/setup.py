#!/usr/bin/env python3
"""Google Workspace OAuth2 setup for Hermes Agent.

Fully non-interactive — designed to be driven by the agent via terminal commands.
The agent mediates between this script and the user (works on CLI, Telegram, Discord, etc.)

Commands:
  setup.py --check                          # Is auth valid? Exit 0 = yes, 1 = no
  setup.py --client-secret /path/to.json    # Store OAuth client credentials
  setup.py --auth-url                       # Print the OAuth URL for user to visit
  setup.py --auth-code CODE                 # Exchange auth code for token
  setup.py --revoke                         # Revoke and delete stored token
  setup.py --install-deps                   # Install Python dependencies only

Agent workflow:
  1. Run --check. If exit 0, auth is good — skip setup.
  2. Ask user for client_secret.json path. Run --client-secret PATH.
  3. Run --auth-url. Send the printed URL to the user.
  4. User opens URL, authorizes, gets redirected to a page with a code.
  5. User pastes the code. Agent runs --auth-code CODE.
  6. Run --check to verify. Done.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

HERMES_HOME = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
TOKEN_PATH = HERMES_HOME / "google_token.json"
CLIENT_SECRET_PATH = HERMES_HOME / "google_client_secret.json"
PENDING_AUTH_PATH = HERMES_HOME / "google_oauth_pending.json"
LAST_AUTH_URL_PATH = HERMES_HOME / "google_oauth_last_url.txt"

SERVICE_SCOPE_GROUPS = {
    "email": [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/gmail.modify",
    ],
    "calendar": ["https://www.googleapis.com/auth/calendar"],
    "drive": ["https://www.googleapis.com/auth/drive.readonly"],
    "contacts": ["https://www.googleapis.com/auth/contacts.readonly"],
    "sheets": ["https://www.googleapis.com/auth/spreadsheets"],
    "docs": ["https://www.googleapis.com/auth/documents.readonly"],
}
SERVICE_ALIASES = {
    "all": "all",
    "email": "email",
    "gmail": "email",
    "mail": "email",
    "calendar": "calendar",
    "cal": "calendar",
    "drive": "drive",
    "contacts": "contacts",
    "people": "contacts",
    "sheets": "sheets",
    "docs": "docs",
    "documents": "docs",
}
DEFAULT_SERVICES = ["email", "calendar", "drive", "contacts", "sheets", "docs"]
ALL_SCOPES = [scope for service in DEFAULT_SERVICES for scope in SERVICE_SCOPE_GROUPS[service]]

REQUIRED_PACKAGES = ["google-api-python-client", "google-auth-oauthlib", "google-auth-httplib2"]

# OAuth redirect for "out of band" manual code copy flow.
# Google deprecated OOB, so we use a localhost redirect and tell the user to
# copy the code from the browser's URL bar (or the page body).
REDIRECT_URI = "http://localhost:1"
AUDIENCE_URL = "https://console.cloud.google.com/auth/audience"
PROJECT_SELECTOR_URL = "https://console.cloud.google.com/projectselector2/home/dashboard"
API_LIBRARY_URL = "https://console.cloud.google.com/apis/library"
CREDENTIALS_URL = "https://console.cloud.google.com/apis/credentials"


def install_deps():
    """Install Google API packages if missing. Returns True on success."""
    try:
        import googleapiclient  # noqa: F401
        import google_auth_oauthlib  # noqa: F401
        print("Dependencies already installed.")
        return True
    except ImportError:
        pass

    print("Installing Google API dependencies...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + REQUIRED_PACKAGES,
            stdout=subprocess.DEVNULL,
        )
        print("Dependencies installed.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies: {e}")
        print(f"Try manually: {sys.executable} -m pip install {' '.join(REQUIRED_PACKAGES)}")
        return False


def _ensure_deps():
    """Check deps are available, install if not, exit on failure."""
    try:
        import googleapiclient  # noqa: F401
        import google_auth_oauthlib  # noqa: F401
    except ImportError:
        if not install_deps():
            sys.exit(1)


def _dedupe(items: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _resolve_services(services_text: str | None) -> tuple[list[str], list[str]]:
    """Resolve a comma/space separated service list to canonical service names + scopes."""
    text = (services_text or "all").strip().lower()
    if not text or text == "all":
        services = list(DEFAULT_SERVICES)
        return services, list(ALL_SCOPES)

    raw_parts = [part.strip() for part in text.replace(" ", ",").split(",") if part.strip()]
    canonical = []
    unknown = []
    for part in raw_parts:
        alias = SERVICE_ALIASES.get(part)
        if alias == "all":
            return list(DEFAULT_SERVICES), list(ALL_SCOPES)
        if not alias:
            unknown.append(part)
            continue
        canonical.append(alias)

    if unknown:
        print(
            "ERROR: Unknown Google service(s): "
            + ", ".join(sorted(set(unknown)))
            + ". Supported values: all, email, calendar, drive, contacts, sheets, docs."
        )
        sys.exit(1)

    canonical = _dedupe(canonical)
    scopes = [scope for service in canonical for scope in SERVICE_SCOPE_GROUPS[service]]
    return canonical, scopes


def _stored_token_scopes() -> list[str] | None:
    if not TOKEN_PATH.exists():
        return None
    try:
        data = json.loads(TOKEN_PATH.read_text())
    except Exception:
        return None
    scopes = data.get("scopes")
    if isinstance(scopes, list) and scopes:
        return scopes
    return None


def _credentials_scopes(services_text: str | None = None) -> list[str]:
    requested_scopes = None
    if services_text:
        _, requested_scopes = _resolve_services(services_text)

    stored_scopes = _stored_token_scopes()
    if stored_scopes:
        if requested_scopes:
            missing = [scope for scope in requested_scopes if scope not in stored_scopes]
            if missing:
                print("TOKEN_MISSING_SCOPES: Stored token does not include the requested services.")
                print("Missing scopes:")
                for scope in missing:
                    print(f"  - {scope}")
                print("Re-run setup with a fresh auth URL for the services you need.")
                return []
        return stored_scopes

    return requested_scopes or list(ALL_SCOPES)


def check_auth(services_text: str | None = None):
    """Check if stored credentials are valid. Prints status, exits 0 or 1."""
    if not TOKEN_PATH.exists():
        print(f"NOT_AUTHENTICATED: No token at {TOKEN_PATH}")
        return False

    scopes = _credentials_scopes(services_text)
    if not scopes:
        return False

    _ensure_deps()
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    try:
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), scopes)
    except Exception as e:
        print(f"TOKEN_CORRUPT: {e}")
        return False

    if creds.valid:
        print(f"AUTHENTICATED: Token valid at {TOKEN_PATH}")
        return True

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            TOKEN_PATH.write_text(creds.to_json())
            print(f"AUTHENTICATED: Token refreshed at {TOKEN_PATH}")
            return True
        except Exception as e:
            print(f"REFRESH_FAILED: {e}")
            return False

    print("TOKEN_INVALID: Re-run setup.")
    return False


def store_client_secret(path: str):
    """Copy and validate client_secret.json to Hermes home."""
    src = Path(path).expanduser().resolve()
    if not src.exists():
        print(f"ERROR: File not found: {src}")
        sys.exit(1)

    try:
        data = json.loads(src.read_text())
    except json.JSONDecodeError:
        print("ERROR: File is not valid JSON.")
        sys.exit(1)

    if "installed" not in data and "web" not in data:
        print("ERROR: Not a Google OAuth client secret file (missing 'installed' key).")
        print(f"Download the correct file from: {CREDENTIALS_URL}")
        sys.exit(1)

    CLIENT_SECRET_PATH.write_text(json.dumps(data, indent=2))
    print(f"OK: Client secret saved to {CLIENT_SECRET_PATH}")


def _save_pending_auth(*, state: str, code_verifier: str, scopes: list[str], services: list[str], auth_url: str):
    """Persist the OAuth session bits needed for a later token exchange."""
    PENDING_AUTH_PATH.write_text(
        json.dumps(
            {
                "state": state,
                "code_verifier": code_verifier,
                "redirect_uri": REDIRECT_URI,
                "scopes": scopes,
                "services": services,
                "auth_url": auth_url,
            },
            indent=2,
        )
    )
    LAST_AUTH_URL_PATH.write_text(auth_url)


def _load_pending_auth() -> dict:
    """Load the pending OAuth session created by get_auth_url()."""
    if not PENDING_AUTH_PATH.exists():
        print("ERROR: No pending OAuth session found. Run --auth-url first.")
        sys.exit(1)

    try:
        data = json.loads(PENDING_AUTH_PATH.read_text())
    except Exception as e:
        print(f"ERROR: Could not read pending OAuth session: {e}")
        print("Run --auth-url again to start a fresh OAuth session.")
        sys.exit(1)

    if not data.get("state") or not data.get("code_verifier"):
        print("ERROR: Pending OAuth session is missing PKCE data.")
        print("Run --auth-url again to start a fresh OAuth session.")
        sys.exit(1)

    data.setdefault("scopes", list(ALL_SCOPES))
    data.setdefault("services", list(DEFAULT_SERVICES))
    return data


def _extract_code_and_state(code_or_url: str) -> tuple[str, str | None]:
    """Accept either a raw auth code or the full redirect URL pasted by the user."""
    if not code_or_url.startswith("http"):
        return code_or_url, None

    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(code_or_url)
    params = parse_qs(parsed.query)
    if "code" not in params:
        print("ERROR: No 'code' parameter found in URL.")
        print("When the browser lands on the localhost error page, copy the FULL address bar URL.")
        sys.exit(1)

    state = params.get("state", [None])[0]
    return params["code"][0], state


def _build_flow(scopes: list[str], *, state: str | None = None, code_verifier: str | None = None, autogenerate_code_verifier: bool = False):
    _ensure_deps()
    from google_auth_oauthlib.flow import Flow

    return Flow.from_client_secrets_file(
        str(CLIENT_SECRET_PATH),
        scopes=scopes,
        redirect_uri=REDIRECT_URI,
        state=state,
        code_verifier=code_verifier,
        autogenerate_code_verifier=autogenerate_code_verifier,
    )


def _create_auth_session(services_text: str | None, *, output_format: str = "plain", emit_output: bool = True) -> str:
    services, scopes = _resolve_services(services_text)
    flow = _build_flow(scopes, autogenerate_code_verifier=True)
    auth_url, state = flow.authorization_url(access_type="offline", prompt="consent")
    _save_pending_auth(
        state=state,
        code_verifier=flow.code_verifier,
        scopes=scopes,
        services=services,
        auth_url=auth_url,
    )

    if emit_output:
        if output_format == "json":
            print(
                json.dumps(
                    {
                        "success": True,
                        "auth_url": auth_url,
                        "auth_url_file": str(LAST_AUTH_URL_PATH),
                        "services": services,
                        "scopes": scopes,
                        "project_selector_url": PROJECT_SELECTOR_URL,
                        "api_library_url": API_LIBRARY_URL,
                        "credentials_url": CREDENTIALS_URL,
                        "audience_url": AUDIENCE_URL,
                        "instructions": [
                            "Open auth_url in your browser.",
                            "If the browser lands on a localhost error page, that is expected.",
                            "Copy the FULL redirected URL from the address bar and pass it to --auth-code.",
                        ],
                    },
                    indent=2,
                )
            )
        else:
            print(auth_url)
    return auth_url


def get_auth_url(services_text: str | None = None, *, output_format: str = "plain"):
    """Print the OAuth authorization URL. User visits this in a browser."""
    if not CLIENT_SECRET_PATH.exists():
        print("ERROR: No client secret stored. Run --client-secret first.")
        sys.exit(1)

    _create_auth_session(services_text, output_format=output_format)


def _print_recovery_url(auth_url: str, output_format: str):
    if output_format == "json":
        print(
            json.dumps(
                {
                    "success": False,
                    "fresh_auth_url": auth_url,
                    "auth_url_file": str(LAST_AUTH_URL_PATH),
                    "audience_url": AUDIENCE_URL,
                },
                indent=2,
            )
        )
    else:
        print("A fresh auth URL has been generated. Use this exact URL:")
        print(auth_url)
        print(f"If Google blocks access, add your account as a test user here: {AUDIENCE_URL}")


def exchange_auth_code(code: str, *, output_format: str = "plain"):
    """Exchange the authorization code for a token and save it."""
    if not CLIENT_SECRET_PATH.exists():
        print("ERROR: No client secret stored. Run --client-secret first.")
        sys.exit(1)

    pending_auth = _load_pending_auth()
    code, returned_state = _extract_code_and_state(code)
    if returned_state and returned_state != pending_auth["state"]:
        auth_url = _create_auth_session(
            ",".join(pending_auth.get("services", DEFAULT_SERVICES)),
            output_format=output_format,
            emit_output=False,
        )
        if output_format != "json":
            print("ERROR: OAuth state mismatch. Your browser redirect came from an older auth session.")
        _print_recovery_url(auth_url, output_format)
        sys.exit(1)

    flow = _build_flow(
        pending_auth.get("scopes", list(ALL_SCOPES)),
        state=pending_auth["state"],
        code_verifier=pending_auth["code_verifier"],
    )

    try:
        flow.fetch_token(code=code)
    except Exception as e:
        auth_url = _create_auth_session(
            ",".join(pending_auth.get("services", DEFAULT_SERVICES)),
            output_format=output_format,
            emit_output=False,
        )
        if output_format != "json":
            print(f"ERROR: Token exchange failed: {e}")
            print("The code may have expired or already been used.")
        _print_recovery_url(auth_url, output_format)
        sys.exit(1)

    creds = flow.credentials
    TOKEN_PATH.write_text(creds.to_json())
    PENDING_AUTH_PATH.unlink(missing_ok=True)
    if output_format == "json":
        print(
            json.dumps(
                {
                    "success": True,
                    "token_path": str(TOKEN_PATH),
                    "services": pending_auth.get("services", DEFAULT_SERVICES),
                },
                indent=2,
            )
        )
    else:
        print(f"OK: Authenticated. Token saved to {TOKEN_PATH}")


def revoke():
    """Revoke stored token and delete it."""
    if not TOKEN_PATH.exists():
        print("No token to revoke.")
        return

    scopes = _stored_token_scopes() or list(ALL_SCOPES)

    _ensure_deps()
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    try:
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), scopes)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())

        import urllib.request

        urllib.request.urlopen(
            urllib.request.Request(
                f"https://oauth2.googleapis.com/revoke?token={creds.token}",
                method="POST",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        )
        print("Token revoked with Google.")
    except Exception as e:
        print(f"Remote revocation failed (token may already be invalid): {e}")

    TOKEN_PATH.unlink(missing_ok=True)
    PENDING_AUTH_PATH.unlink(missing_ok=True)
    LAST_AUTH_URL_PATH.unlink(missing_ok=True)
    print(f"Deleted {TOKEN_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Google Workspace OAuth setup for Hermes")
    parser.add_argument(
        "--services",
        default="all",
        help="Comma-separated services to authorize: all, email, calendar, drive, contacts, sheets, docs",
    )
    parser.add_argument(
        "--format",
        choices=["plain", "json"],
        default="plain",
        help="Output format. Use json for agent-friendly parsing.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Check if auth is valid (exit 0=yes, 1=no)")
    group.add_argument("--client-secret", metavar="PATH", help="Store OAuth client_secret.json")
    group.add_argument("--auth-url", action="store_true", help="Print OAuth URL for user to visit")
    group.add_argument("--auth-code", metavar="CODE", help="Exchange auth code for token")
    group.add_argument("--revoke", action="store_true", help="Revoke and delete stored token")
    group.add_argument("--install-deps", action="store_true", help="Install Python dependencies")
    args = parser.parse_args()

    if args.check:
        sys.exit(0 if check_auth(args.services) else 1)
    if args.client_secret:
        store_client_secret(args.client_secret)
        return
    if args.auth_url:
        get_auth_url(args.services, output_format=args.format)
        return
    if args.auth_code:
        exchange_auth_code(args.auth_code, output_format=args.format)
        return
    if args.revoke:
        revoke()
        return
    if args.install_deps:
        sys.exit(0 if install_deps() else 1)


if __name__ == "__main__":
    main()

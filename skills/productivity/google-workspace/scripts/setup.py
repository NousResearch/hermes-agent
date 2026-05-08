#!/usr/bin/env python3
"""Google Workspace OAuth2 setup for Hermes Agent.

Fully non-interactive — designed to be driven by the agent via terminal commands.
The agent mediates between this script and the user (works on CLI, Telegram, Discord, etc.)

Commands:
  setup.py --check                          # Is auth valid? Exit 0 = yes, 1 = no
  setup.py --client-secret /path/to.json    # Store OAuth client credentials
  setup.py --auth-url [--services calendar] [--format json]
  setup.py --auth-code CODE [--format json] # Exchange auth code for token
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

from __future__ import annotations  # allow PEP 604 `X | None` on Python 3.9+

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure sibling modules (_hermes_home) are importable when run standalone.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from _hermes_home import display_hermes_home, get_hermes_home

HERMES_HOME = get_hermes_home()
TOKEN_PATH = HERMES_HOME / "google_token.json"
CLIENT_SECRET_PATH = HERMES_HOME / "google_client_secret.json"
PENDING_AUTH_PATH = HERMES_HOME / "google_oauth_pending.json"
LAST_AUTH_URL_PATH = HERMES_HOME / "google_oauth_last_url.txt"

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]
CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar"]
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
CONTACTS_SCOPES = ["https://www.googleapis.com/auth/contacts.readonly"]
SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
DOCS_SCOPES = ["https://www.googleapis.com/auth/documents"]

SCOPES = [
    *GMAIL_SCOPES,
    *CALENDAR_SCOPES,
    *DRIVE_SCOPES,
    *CONTACTS_SCOPES,
    *SHEETS_SCOPES,
    *DOCS_SCOPES,
]

SERVICE_SCOPE_MAP = {
    "email": GMAIL_SCOPES,
    "gmail": GMAIL_SCOPES,
    "calendar": CALENDAR_SCOPES,
    "drive": DRIVE_SCOPES,
    "contacts": CONTACTS_SCOPES,
    "sheets": SHEETS_SCOPES,
    "docs": DOCS_SCOPES,
    "all": SCOPES,
}

REQUIRED_PACKAGES = ["google-api-python-client", "google-auth-oauthlib", "google-auth-httplib2"]

# OAuth redirect for "out of band" manual code copy flow.
# Google deprecated OOB, so we use a localhost redirect and tell the user to
# copy the code from the browser's URL bar (or the page body).
REDIRECT_URI = "http://localhost:1"


def _normalize_authorized_user_payload(payload: dict) -> dict:
    normalized = dict(payload)
    if not normalized.get("type"):
        normalized["type"] = "authorized_user"
    return normalized


def _load_token_payload(path: Path = TOKEN_PATH) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _coerce_scope_list(raw) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        return [s.strip() for s in raw.split() if s.strip()]
    return [str(s).strip() for s in raw if str(s).strip()]


def _parse_services(raw: str | None) -> list[str]:
    services = [part.strip().lower() for part in (raw or "all").split(",") if part.strip()]
    if not services:
        services = ["all"]
    unknown = [service for service in services if service not in SERVICE_SCOPE_MAP]
    if unknown:
        choices = ", ".join(sorted(SERVICE_SCOPE_MAP))
        raise ValueError(f"Unknown Google service(s): {', '.join(unknown)}. Valid choices: {choices}")
    if "all" in services and len(services) > 1:
        raise ValueError("Use either 'all' or a comma-separated subset, not both.")
    return services


def _scopes_for_services(services: list[str]) -> list[str]:
    scopes: list[str] = []
    seen: set[str] = set()
    for service in services:
        for scope in SERVICE_SCOPE_MAP[service]:
            if scope not in seen:
                seen.add(scope)
                scopes.append(scope)
    return scopes


def _expected_scopes_from_payload(payload: dict) -> list[str]:
    requested_scopes = _coerce_scope_list(payload.get("requested_scopes"))
    if requested_scopes:
        return requested_scopes
    services = payload.get("services")
    if services:
        try:
            return _scopes_for_services(_parse_services(",".join(services) if isinstance(services, list) else str(services)))
        except ValueError:
            pass
    return list(SCOPES)


def _missing_scopes_from_payload(payload: dict) -> list[str]:
    raw = payload["scopes"] if "scopes" in payload else payload.get("scope")
    if raw is None:
        # Older tokens may not record scopes at all. Do not invent a warning.
        return []
    granted = set(_coerce_scope_list(raw))
    expected = _expected_scopes_from_payload(payload)
    return sorted(scope for scope in expected if scope not in granted)


def _format_missing_scopes(missing_scopes: list[str]) -> str:
    bullets = "\n".join(f"  - {scope}" for scope in missing_scopes)
    return (
        "Token is valid but missing required Google Workspace scopes:\n"
        f"{bullets}\n"
        "Run the Google Workspace setup again from this same Hermes profile to refresh consent."
    )


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

    # First choice: pip in the current interpreter. Works for most installs.
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + REQUIRED_PACKAGES,
            stdout=subprocess.DEVNULL,
        )
        print("Dependencies installed.")
        return True
    except subprocess.CalledProcessError as e:
        pip_error = e

    # Fallback: the interpreter has no pip (the Hermes Docker image's venv is
    # built with `uv sync`, which does not bootstrap pip). `uv pip install
    # --python <interpreter>` installs into that exact interpreter without
    # needing pip present. Targeting sys.executable keeps us on the venv the
    # script is actually running under, rather than guessing.
    uv = shutil.which("uv")
    if uv:
        try:
            subprocess.check_call(
                [uv, "pip", "install", "--python", sys.executable, "--quiet"]
                + REQUIRED_PACKAGES,
                stdout=subprocess.DEVNULL,
            )
            print("Dependencies installed.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install dependencies via uv: {e}")
            print(f"Manually: {uv} pip install --python {sys.executable} {' '.join(REQUIRED_PACKAGES)}")
            return False

    print(f"ERROR: Failed to install dependencies: {pip_error}")
    print(
        "On environments without pip (e.g. Nix, or the Hermes Docker image's "
        "uv-managed venv), install the optional extra instead:"
    )
    print("  hermes setup")
    print(f"Or manually: {sys.executable} -m pip install {' '.join(REQUIRED_PACKAGES)}")
    return False


def _ensure_deps():
    """Check deps are available, install if not, exit on failure."""
    try:
        import googleapiclient  # noqa: F401
        import google_auth_oauthlib  # noqa: F401
    except ImportError:
        if not install_deps():
            sys.exit(1)


def check_auth_live():
    """Check auth with a real API call to detect disabled_client/account issues."""
    # quiet=True suppresses the "AUTHENTICATED" print from check_auth so the
    # final status line reflects the live-call outcome (OK or FAILED).
    if not check_auth(quiet=True):
        return False
    try:
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH))
        service = build("calendar", "v3", credentials=creds)
        service.calendarList().list(maxResults=1).execute()
        print("LIVE_CHECK_OK: Real API call succeeded.")
        return True
    except Exception as e:
        err_str = str(e).lower()
        if "disabled_client" in err_str or "invalid_client" in err_str:
            print(f"LIVE_CHECK_FAILED: OAuth client or account disabled: {e}")
            print("  1. Check Google Cloud Console for disabled OAuth client")
            print("  2. Check myaccount.google.com for account status")
            print("  3. Do NOT retry with a disabled account")
        else:
            print(f"LIVE_CHECK_FAILED: {e}")
        return False


def check_auth(quiet: bool = False):
    """Check if stored credentials are valid. Prints status, exits 0 or 1."""
    if not TOKEN_PATH.exists():
        print(f"NOT_AUTHENTICATED: No token at {TOKEN_PATH}")
        return False

    _ensure_deps()
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    try:
        # Don't pass scopes — user may have authorized only a subset.
        # Passing scopes forces google-auth to validate them on refresh,
        # which fails with invalid_scope if the token has fewer scopes
        # than requested.
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH))
    except Exception as e:
        print(f"TOKEN_CORRUPT: {e}")
        return False

    payload = _load_token_payload(TOKEN_PATH)
    if creds.valid:
        missing_scopes = _missing_scopes_from_payload(payload)
        if missing_scopes:
            print(f"AUTHENTICATED (partial): Token valid but missing {len(missing_scopes)} scopes:")
            for s in missing_scopes:
                print(f"  - {s}")
        if not quiet:
            print(f"AUTHENTICATED: Token valid at {TOKEN_PATH}")
        return True

    if creds.expired and creds.refresh_token:
        try:
            existing_payload = _load_token_payload(TOKEN_PATH)
            creds.refresh(Request())
            refreshed_payload = _normalize_authorized_user_payload(json.loads(creds.to_json()))
            for key in ("services", "requested_scopes"):
                if existing_payload.get(key) and key not in refreshed_payload:
                    refreshed_payload[key] = existing_payload[key]
            TOKEN_PATH.write_text(json.dumps(refreshed_payload, indent=2))
            missing_scopes = _missing_scopes_from_payload(_load_token_payload(TOKEN_PATH))
            if missing_scopes:
                print(f"AUTHENTICATED (partial): Token refreshed but missing {len(missing_scopes)} scopes:")
                for s in missing_scopes:
                    print(f"  - {s}")
            if not quiet:
                print(f"AUTHENTICATED: Token refreshed at {TOKEN_PATH}")
            return True
        except Exception as e:
            err_str = str(e).lower()
            if "disabled_client" in err_str or "invalid_client" in err_str:
                print(f"OAUTH_CLIENT_DISABLED: {e}")
                print("  The OAuth client or Google account has been disabled.")
                print("  Steps to resolve:")
                print("    1. Check your Google Cloud Console — verify the OAuth client is not disabled")
                print("    2. Check if your Google account itself has been disabled at myaccount.google.com")
                print("    3. If the account is disabled, you can appeal at accounts.google.com/signin/recovery")
                print("    4. Do NOT retry API calls with a disabled account — this may worsen the situation")
                print("    5. If the OAuth client is disabled, create a new one in Google Cloud Console")
            elif "token_revoked" in err_str or "invalid_grant" in err_str:
                print(f"TOKEN_REVOKED: {e}")
                print("  Re-run setup to re-authenticate.")
            else:
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
        print("Download the correct file from: https://console.cloud.google.com/apis/credentials")
        sys.exit(1)

    CLIENT_SECRET_PATH.write_text(json.dumps(data, indent=2))
    print(f"OK: Client secret saved to {CLIENT_SECRET_PATH}")


def _save_pending_auth(
    *,
    state: str,
    code_verifier: str,
    services: list[str] | None = None,
    requested_scopes: list[str] | None = None,
):
    """Persist the OAuth session bits needed for a later token exchange."""
    PENDING_AUTH_PATH.write_text(
        json.dumps(
            {
                "state": state,
                "code_verifier": code_verifier,
                "redirect_uri": REDIRECT_URI,
                "services": services or ["all"],
                "requested_scopes": requested_scopes or list(SCOPES),
            },
            indent=2,
        )
    )


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
        sys.exit(1)

    state = params.get("state", [None])[0]
    return params["code"][0], state


def _start_auth_session(services: str | None = "all") -> dict:
    service_names = _parse_services(services)
    requested_scopes = _scopes_for_services(service_names)

    _ensure_deps()
    from google_auth_oauthlib.flow import Flow

    flow = Flow.from_client_secrets_file(
        str(CLIENT_SECRET_PATH),
        scopes=requested_scopes,
        redirect_uri=REDIRECT_URI,
        autogenerate_code_verifier=True,
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
    )
    _save_pending_auth(
        state=state,
        code_verifier=flow.code_verifier,
        services=service_names,
        requested_scopes=requested_scopes,
    )
    LAST_AUTH_URL_PATH.write_text(auth_url)
    return {
        "auth_url": auth_url,
        "services": service_names,
        "scopes": requested_scopes,
        "pending_path": str(PENDING_AUTH_PATH),
        "last_url_path": str(LAST_AUTH_URL_PATH),
    }


def get_auth_url(services: str | None = "all", output_format: str = "text"):
    """Print the OAuth authorization URL. User visits this in a browser."""
    if not CLIENT_SECRET_PATH.exists():
        if output_format == "json":
            print(json.dumps({"status": "error", "error": "missing_client_secret", "message": "No client secret stored. Run --client-secret first."}))
        else:
            print("ERROR: No client secret stored. Run --client-secret first.")
        sys.exit(1)

    try:
        payload = _start_auth_session(services)
    except ValueError as e:
        if output_format == "json":
            print(json.dumps({"status": "error", "error": "invalid_services", "message": str(e)}))
        else:
            print(f"ERROR: {e}")
        sys.exit(1)

    if output_format == "json":
        print(json.dumps(payload))
    else:
        # Print just the URL so the agent can extract it cleanly.
        print(payload["auth_url"])


def exchange_auth_code(code: str, output_format: str = "text"):
    """Exchange the authorization code for a token and save it."""
    if not CLIENT_SECRET_PATH.exists():
        if output_format == "json":
            print(json.dumps({"status": "error", "error": "missing_client_secret", "message": "No client secret stored. Run --client-secret first."}))
        else:
            print("ERROR: No client secret stored. Run --client-secret first.")
        sys.exit(1)

    pending_auth = _load_pending_auth()
    requested_scopes = _coerce_scope_list(pending_auth.get("requested_scopes")) or list(SCOPES)
    services = pending_auth.get("services") or ["all"]
    services_arg = ",".join(services) if isinstance(services, list) else str(services)
    raw_callback = code
    code, returned_state = _extract_code_and_state(code)
    if returned_state and returned_state != pending_auth["state"]:
        if output_format == "json":
            print(json.dumps({"status": "error", "error": "oauth_state_mismatch", "message": "OAuth state mismatch. Run --auth-url again to start a fresh session."}))
        else:
            print("ERROR: OAuth state mismatch. Run --auth-url again to start a fresh session.")
        sys.exit(1)

    _ensure_deps()
    from google_auth_oauthlib.flow import Flow
    from urllib.parse import parse_qs, urlparse

    # Extract granted scopes from the callback URL if the user pasted the full redirect URL.
    granted_scopes = list(requested_scopes)
    if isinstance(raw_callback, str) and raw_callback.startswith("http"):
        params = parse_qs(urlparse(raw_callback).query)
        scope_val = (params.get("scope") or [""])[0].strip()
        if scope_val:
            granted_scopes = scope_val.split()

    flow = Flow.from_client_secrets_file(
        str(CLIENT_SECRET_PATH),
        scopes=granted_scopes,
        redirect_uri=pending_auth.get("redirect_uri", REDIRECT_URI),
        state=pending_auth["state"],
        code_verifier=pending_auth["code_verifier"],
    )

    try:
        # Accept partial scopes — user may deselect some permissions in the consent screen
        os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
        flow.fetch_token(code=code)
    except Exception as e:
        if output_format == "json":
            payload = {
                "status": "error",
                "error": "token_exchange_failed",
                "message": str(e),
                "services": services if isinstance(services, list) else [services_arg],
            }
            try:
                fresh = _start_auth_session(services_arg)
                payload["fresh_auth_url"] = fresh["auth_url"]
                payload["services"] = fresh["services"]
                payload["scopes"] = fresh["scopes"]
            except Exception as fresh_error:
                payload["fresh_auth_error"] = str(fresh_error)
            print(json.dumps(payload))
        else:
            print(f"ERROR: Token exchange failed: {e}")
            print("The code may have expired. Run --auth-url to get a fresh URL.")
        sys.exit(1)

    creds = flow.credentials
    token_payload = _normalize_authorized_user_payload(json.loads(creds.to_json()))

    # Store only the scopes actually granted by the user, not what was requested.
    # creds.to_json() writes the requested scopes, which causes refresh to fail
    # with invalid_scope if the user only authorized a subset.
    actually_granted = list(creds.granted_scopes or []) if hasattr(creds, "granted_scopes") and creds.granted_scopes else []
    if actually_granted:
        token_payload["scopes"] = actually_granted
    elif granted_scopes != requested_scopes:
        # granted_scopes was extracted from the callback URL
        token_payload["scopes"] = granted_scopes

    token_payload["services"] = services
    token_payload["requested_scopes"] = requested_scopes

    missing_scopes = _missing_scopes_from_payload(token_payload)
    warnings = []
    if missing_scopes:
        warnings.append(f"Token missing some Google Workspace scopes: {', '.join(missing_scopes)}")
        if output_format != "json":
            print(warnings[-1].replace("Token", "WARNING: Token", 1))
            print("Some services may not be available.")

    TOKEN_PATH.write_text(json.dumps(token_payload, indent=2))
    PENDING_AUTH_PATH.unlink(missing_ok=True)
    profile_token_location = f"{display_hermes_home()}/google_token.json"
    if output_format == "json":
        print(json.dumps({
            "status": "authenticated",
            "token_path": str(TOKEN_PATH),
            "profile_token_location": profile_token_location,
            "services": services,
            "scopes": token_payload.get("scopes") or [],
            "missing_scopes": missing_scopes,
            "warnings": warnings,
        }))
    else:
        print(f"OK: Authenticated. Token saved to {TOKEN_PATH}")
        print(f"Profile-scoped token location: {profile_token_location}")


def revoke():
    """Revoke stored token and delete it."""
    if not TOKEN_PATH.exists():
        print("No token to revoke.")
        return

    token_payload = _load_token_payload(TOKEN_PATH)
    revoke_token = token_payload.get("refresh_token") or token_payload.get("token")

    try:
        if not revoke_token:
            raise ValueError("token file does not contain a token or refresh_token")

        import urllib.parse
        import urllib.request

        encoded_token = urllib.parse.quote(str(revoke_token), safe="")
        urllib.request.urlopen(
            urllib.request.Request(
                f"https://oauth2.googleapis.com/revoke?token={encoded_token}",
                method="POST",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            ),
            timeout=15,
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Check if auth is valid (exit 0=yes, 1=no)")
    group.add_argument("--check-live", action="store_true", help="Check auth with a real API call (detects disabled_client)")
    group.add_argument("--client-secret", metavar="PATH", help="Store OAuth client_secret.json")
    group.add_argument("--auth-url", action="store_true", help="Print OAuth URL for user to visit")
    group.add_argument("--auth-code", metavar="CODE", help="Exchange auth code for token")
    group.add_argument("--revoke", action="store_true", help="Revoke and delete stored token")
    group.add_argument("--install-deps", action="store_true", help="Install Python dependencies")
    parser.add_argument(
        "--services",
        default="all",
        help="Google service set for --auth-url: all, calendar, email, drive, contacts, sheets, docs; comma-separated",
    )
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    args = parser.parse_args()

    if args.check:
        sys.exit(0 if check_auth() else 1)
    if getattr(args, "check_live", False):
        sys.exit(0 if check_auth_live() else 1)
    elif args.client_secret:
        store_client_secret(args.client_secret)
    elif args.auth_url:
        get_auth_url(args.services, args.format)
    elif args.auth_code:
        exchange_auth_code(args.auth_code, args.format)
    elif args.revoke:
        revoke()
    elif args.install_deps:
        sys.exit(0 if install_deps() else 1)


if __name__ == "__main__":
    main()

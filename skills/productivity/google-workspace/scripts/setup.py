#!/usr/bin/env python3
"""Google Workspace OAuth2 setup for Hermes Agent.

Fully non-interactive — designed to be driven by the agent via terminal commands.
The agent mediates between this script and the user (works on CLI, Telegram, Discord, etc.)

Commands:
  setup.py --check                          # Is auth valid? Exit 0 = yes, 1 = no
  setup.py --client-secret /path/to.json    # Store OAuth client credentials
  setup.py --auth-url                       # Print the OAuth URL for user to visit
  setup.py --auth-code CODE                 # Exchange auth code for token
  setup.py --auth-code-file PATH            # Exchange code from a private file
  setup.py --revoke                         # Revoke and delete stored token
  setup.py --install-deps                   # Install Python dependencies only

Agent workflow:
  1. Run --check. If exit 0, auth is good — skip setup.
  2. Ask user for client_secret.json path. Run --client-secret PATH.
  3. Run --auth-url. Send the printed URL to the user.
  4. User opens URL, authorizes, and saves the redirected URL to a private file.
  5. Agent runs --auth-code-file PATH.
  6. Run --check to verify. Done.
"""

from __future__ import annotations  # allow PEP 604 `X | None` on Python 3.9+

import argparse
import json
import os
import shutil
import subprocess
import sys
from importlib.metadata import version as _distribution_version
from pathlib import Path

# Ensure sibling modules (_hermes_home) are importable when run standalone.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from _hermes_home import display_hermes_home, get_hermes_home
from utils import atomic_json_write as _write_private_json

HERMES_HOME = get_hermes_home()
TOKEN_PATH = HERMES_HOME / "google_token.json"
CLIENT_SECRET_PATH = HERMES_HOME / "google_client_secret.json"
PENDING_AUTH_PATH = HERMES_HOME / "google_oauth_pending.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/contacts.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/documents",
]

SERVICE_SCOPES = {
    "email": SCOPES[0:3],
    "calendar": [SCOPES[3]],
    "drive": [SCOPES[4]],
    "contacts": [SCOPES[5]],
    "sheets": [SCOPES[6]],
    "docs": [SCOPES[7]],
}

# Exact pins: keep in sync with pyproject.toml [project.optional-dependencies].google
# and tools/lazy_deps.py LAZY_DEPS['skill.google_workspace'].
# Pinning all protects against version drift and ensures the security floors
# (httplib2 GHSA-j5g9-f88f-gfj3, stale pyasn1/google-auth) are honoured
# regardless of install path.
REQUIRED_PACKAGES = [
    "google-api-python-client==2.194.0",
    "google-auth==2.55.1",
    "google-auth-oauthlib==1.3.1",
    "google-auth-httplib2==0.3.1",
    # GHSA-j5g9-f88f-gfj3 — Decompression Bomb DoS via unbounded gzip/deflate
    "httplib2==0.32.0",
    "pyasn1==0.6.4",
]

# OAuth redirect for "out of band" manual code copy flow.
# Google deprecated OOB, so we use a localhost redirect and tell the user to
# copy the code from the browser's URL bar (or the page body).
REDIRECT_URI = "http://localhost:1"


def _normalize_authorized_user_payload(payload: dict) -> dict:
    normalized = dict(payload)
    if not normalized.get("type"):
        normalized["type"] = "authorized_user"
    return normalized


def _scopes_for_services(value: str) -> list[str]:
    services = {service.strip() for service in value.split(",") if service.strip()}
    unknown = services - SERVICE_SCOPES.keys() - {"all"}
    if unknown:
        choices = ", ".join([*SERVICE_SCOPES, "all"])
        raise argparse.ArgumentTypeError(
            f"unknown service(s): {', '.join(sorted(unknown))}; choose from {choices}"
        )
    if not services or "all" in services:
        return list(SCOPES)

    selected = {scope for service in services for scope in SERVICE_SCOPES[service]}
    return [scope for scope in SCOPES if scope in selected]


def _load_token_payload(path: Path = TOKEN_PATH) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _missing_scopes_from_payload(
    payload: dict,
    required_scopes: list[str] | None = None,
) -> list[str]:
    raw = payload.get("scopes") or payload.get("scope")
    if not raw:
        return []
    granted = {s.strip() for s in (raw.split() if isinstance(raw, str) else raw) if s.strip()}
    required = required_scopes or SCOPES
    return sorted(scope for scope in required if scope not in granted)


def _format_missing_scopes(missing_scopes: list[str]) -> str:
    bullets = "\n".join(f"  - {scope}" for scope in missing_scopes)
    return (
        "Token is valid but missing required Google Workspace scopes:\n"
        f"{bullets}\n"
        "Run the Google Workspace setup again from this same Hermes profile to refresh consent."
    )


def _missing_required_packages() -> list[str]:
    """Return exact requirements absent or stale in this interpreter.

    All REQUIRED_PACKAGES entries are exact ``name==version`` pins, so a
    direct version comparison is sufficient — no ``packaging`` dependency
    needed in this standalone script.
    """
    missing = []
    for spec in REQUIRED_PACKAGES:
        name, _, wanted = spec.partition("==")
        try:
            if _distribution_version(name) != wanted:
                missing.append(spec)
        except Exception:
            missing.append(spec)
    return missing


def install_deps():
    """Install missing or stale Google API packages. Returns True on success."""
    missing = _missing_required_packages()
    if not missing:
        print("Dependencies already installed.")
        return True

    print("Installing Google API dependencies...")

    # First choice: pip in the current interpreter. Works for most installs.
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
            stdout=subprocess.DEVNULL,
        )
        remaining = _missing_required_packages()
        if remaining:
            print(f"ERROR: Dependencies remain stale after pip install: {' '.join(remaining)}")
            return False
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
                + missing,
                stdout=subprocess.DEVNULL,
            )
            remaining = _missing_required_packages()
            if remaining:
                print(f"ERROR: Dependencies remain stale after uv install: {' '.join(remaining)}")
                return False
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
    """Check exact dependency versions, install if stale, exit on failure."""
    if _missing_required_packages() and not install_deps():
        sys.exit(1)


def check_auth_live():
    """Refresh with Google to detect disabled clients without assuming a service scope."""
    # quiet=True suppresses the "AUTHENTICATED" print from check_auth so the
    # final status line reflects the live-call outcome (OK or FAILED).
    if not check_auth(quiet=True):
        return False
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH))
        if not creds.refresh_token:
            print("LIVE_CHECK_FAILED: Token has no refresh credential.")
            return False
        previous = _load_token_payload(TOKEN_PATH)
        creds.refresh(Request())
        refreshed = _normalize_authorized_user_payload(json.loads(creds.to_json()))
        if not refreshed.get("scopes") and previous.get("scopes"):
            refreshed["scopes"] = previous["scopes"]
        _write_private_json(TOKEN_PATH, refreshed, mode=0o600)
        print("LIVE_CHECK_OK: Real OAuth refresh succeeded.")
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

    if creds.valid:
        if not quiet:
            print(f"AUTHENTICATED: Token valid at {TOKEN_PATH}")
        return True

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _write_private_json(
                TOKEN_PATH,
                _normalize_authorized_user_payload(json.loads(creds.to_json())),
                mode=0o600,
            )
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
        data = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("ERROR: File is not valid JSON.")
        sys.exit(1)

    if "installed" not in data and "web" not in data:
        print("ERROR: Not a Google OAuth client secret file (missing 'installed' key).")
        print("Download the correct file from: https://console.cloud.google.com/apis/credentials")
        sys.exit(1)

    _write_private_json(CLIENT_SECRET_PATH, data, mode=0o600)
    print(f"OK: Client secret saved to {CLIENT_SECRET_PATH}")


def _save_pending_auth(*, state: str, code_verifier: str, scopes: list[str]):
    """Persist the OAuth session bits needed for a later token exchange."""
    _write_private_json(
        PENDING_AUTH_PATH,
        {
            "state": state,
            "code_verifier": code_verifier,
            "redirect_uri": REDIRECT_URI,
            "scopes": scopes,
        },
        mode=0o600,
    )


def _load_pending_auth() -> dict:
    """Load the pending OAuth session created by get_auth_url()."""
    if not PENDING_AUTH_PATH.exists():
        print("ERROR: No pending OAuth session found. Run --auth-url first.")
        sys.exit(1)

    try:
        data = json.loads(PENDING_AUTH_PATH.read_text(encoding="utf-8"))
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


def get_auth_url(scopes: list[str] | None = None):
    """Print the OAuth authorization URL. User visits this in a browser."""
    if not CLIENT_SECRET_PATH.exists():
        print("ERROR: No client secret stored. Run --client-secret first.")
        sys.exit(1)

    _ensure_deps()
    from google_auth_oauthlib.flow import Flow

    requested_scopes = scopes or list(SCOPES)
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
    if not isinstance(flow.code_verifier, str):
        raise RuntimeError("Google OAuth did not create a PKCE code verifier")
    _save_pending_auth(
        state=state,
        code_verifier=flow.code_verifier,
        scopes=requested_scopes,
    )
    # Print just the URL so the agent can extract it cleanly
    print(auth_url)


def exchange_auth_code(code: str):
    """Exchange the authorization code for a token and save it."""
    if not CLIENT_SECRET_PATH.exists():
        print("ERROR: No client secret stored. Run --client-secret first.")
        sys.exit(1)

    pending_auth = _load_pending_auth()
    raw_callback = code
    code, returned_state = _extract_code_and_state(code)
    if returned_state and returned_state != pending_auth["state"]:
        print("ERROR: OAuth state mismatch. Run --auth-url again to start a fresh session.")
        sys.exit(1)

    _ensure_deps()
    from google_auth_oauthlib.flow import Flow
    from urllib.parse import parse_qs, urlparse

    # Extract granted scopes from the callback URL if the user pasted the full redirect URL.
    requested_scopes = list(pending_auth.get("scopes") or SCOPES)
    flow_scopes = list(requested_scopes)
    if isinstance(raw_callback, str) and raw_callback.startswith("http"):
        params = parse_qs(urlparse(raw_callback).query)
        scope_val = (params.get("scope") or [""])[0].strip()
        if scope_val:
            flow_scopes = scope_val.split()

    flow = Flow.from_client_secrets_file(
        str(CLIENT_SECRET_PATH),
        scopes=flow_scopes,
        redirect_uri=pending_auth.get("redirect_uri", REDIRECT_URI),
        state=pending_auth["state"],
        code_verifier=pending_auth["code_verifier"],
    )

    try:
        # Accept partial scopes — user may deselect some permissions in the consent screen
        os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
        flow.fetch_token(code=code)
    except Exception as e:
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
    elif flow_scopes != SCOPES:
        token_payload["scopes"] = flow_scopes

    missing_scopes = _missing_scopes_from_payload(token_payload, requested_scopes)
    if missing_scopes:
        print(f"WARNING: Token missing some Google Workspace scopes: {', '.join(missing_scopes)}")
        print("Some services may not be available.")

    _write_private_json(TOKEN_PATH, token_payload, mode=0o600)
    PENDING_AUTH_PATH.unlink(missing_ok=True)
    print(f"OK: Authenticated. Token saved to {TOKEN_PATH}")
    print(f"Profile-scoped token location: {display_hermes_home()}/google_token.json")


def exchange_auth_code_file(path: str):
    """Exchange an OAuth callback stored outside shell history and process arguments."""
    callback_path = Path(path).expanduser().resolve()
    if not callback_path.is_file():
        print(f"ERROR: OAuth callback file not found: {callback_path}")
        sys.exit(1)

    exchange_auth_code(callback_path.read_text(encoding="utf-8").strip())
    callback_path.unlink()


def revoke():
    """Revoke stored token and delete it."""
    if not TOKEN_PATH.exists():
        print("No token to revoke.")
        return

    _ensure_deps()
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    try:
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH))
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())

        import urllib.request
        urllib.request.urlopen(
            urllib.request.Request(
                f"https://oauth2.googleapis.com/revoke?token={creds.token}",
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
    print(f"Deleted {TOKEN_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Google Workspace OAuth setup for Hermes")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Check if auth is valid (exit 0=yes, 1=no)")
    group.add_argument("--check-live", action="store_true", help="Check auth with a real API call (detects disabled_client)")
    group.add_argument("--client-secret", metavar="PATH", help="Store OAuth client_secret.json")
    group.add_argument("--auth-url", action="store_true", help="Print OAuth URL for user to visit")
    group.add_argument("--auth-code", metavar="CODE", help="Exchange auth code for token")
    group.add_argument(
        "--auth-code-file",
        metavar="PATH",
        help="Exchange an auth callback from a private file and delete it on success",
    )
    group.add_argument("--revoke", action="store_true", help="Revoke and delete stored token")
    group.add_argument("--install-deps", action="store_true", help="Install Python dependencies")
    parser.add_argument(
        "--services",
        default="all",
        type=_scopes_for_services,
        help="Comma-separated services: email,calendar,drive,contacts,sheets,docs,all",
    )
    args = parser.parse_args()

    if args.check:
        sys.exit(0 if check_auth() else 1)
    if getattr(args, "check_live", False):
        sys.exit(0 if check_auth_live() else 1)
    elif args.client_secret:
        store_client_secret(args.client_secret)
    elif args.auth_url:
        get_auth_url(args.services)
    elif args.auth_code:
        exchange_auth_code(args.auth_code)
    elif args.auth_code_file:
        exchange_auth_code_file(args.auth_code_file)
    elif args.revoke:
        revoke()
    elif args.install_deps:
        sys.exit(0 if install_deps() else 1)


if __name__ == "__main__":
    main()

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

from __future__ import annotations  # allow PEP 604 `X | None` on Python 3.9+

import argparse
import json
import os
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
SCOPES_CONFIG_PATH = HERMES_HOME / "google_scopes.json"

SCOPE_API_BASE = "https://www.googleapis.com/auth"
SERVICE_SCOPES = {
    "gmail": {
        "readonly": [f"{SCOPE_API_BASE}/gmail.readonly"],
        "readwrite": [
            f"{SCOPE_API_BASE}/gmail.readonly",
            f"{SCOPE_API_BASE}/gmail.send",
            f"{SCOPE_API_BASE}/gmail.modify",
        ],
    },
    "calendar": {
        "readonly": [f"{SCOPE_API_BASE}/calendar.readonly"],
        "readwrite": [f"{SCOPE_API_BASE}/calendar"],
    },
    "drive": {
        "readonly": [f"{SCOPE_API_BASE}/drive.readonly"],
        "readwrite": [f"{SCOPE_API_BASE}/drive"],
    },
    "sheets": {
        "readonly": [f"{SCOPE_API_BASE}/spreadsheets.readonly"],
        "readwrite": [f"{SCOPE_API_BASE}/spreadsheets"],
    },
    "docs": {
        "readonly": [f"{SCOPE_API_BASE}/documents.readonly"],
        "readwrite": [f"{SCOPE_API_BASE}/documents"],
    },
    "contacts": {
        # This skill never calls a contacts-write API, so the two modes
        # are intentionally identical — keeps the config schema uniform.
        "readonly": [f"{SCOPE_API_BASE}/contacts.readonly"],
        "readwrite": [f"{SCOPE_API_BASE}/contacts"],
    },
}

ALL_SERVICES = SERVICE_SCOPES.keys()
SERVICE_ALIASES = {"email": "gmail"}

# Legacy full-access scope list. Used as the fallback when no
# google_scopes.json exists — preserves the pre-config behavior for
# existing installs that haven't opted into per-service modes.
DEFAULT_SCOPES = list(
    dict.fromkeys(
        scope for svc in ALL_SERVICES for scope in SERVICE_SCOPES[svc]["readwrite"]
    )
)


def _parse_services_spec(spec: str) -> dict:
    """Parse a ``--services`` value into {service: 'readonly'|'readwrite'}.

    Forms accepted:
      ``all``                          → every service, readwrite
      ``all=ro``                       → every service, readonly
      ``gmail,calendar``               → those services, readwrite each
      ``gmail=ro,calendar=rw``         → explicit modes per service
      ``email`` (alias for gmail)      → handled via SERVICE_ALIASES
    """
    result: dict = {}
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "=" in token:
            name, mode = token.split("=", 1)
            name, mode = name.strip().lower(), mode.strip().lower()
        else:
            name, mode = token.lower(), "rw"

        name = SERVICE_ALIASES.get(name, name)

        if mode in {"ro", "readonly", "read-only", "r"}:
            mode_norm = "readonly"
        elif mode in {"rw", "readwrite", "read-write", "w"}:
            mode_norm = "readwrite"
        else:
            raise ValueError(
                f"Invalid mode '{mode}' for service '{name}'; expected ro or rw"
            )

        if name == "all":
            for svc in ALL_SERVICES:
                result[svc] = mode_norm
        elif name in SERVICE_SCOPES:
            result[name] = mode_norm
        else:
            valid = ", ".join(sorted({*SERVICE_SCOPES, "all", *SERVICE_ALIASES}))
            raise ValueError(f"Unknown service '{name}'; valid: {valid}")

    if not result:
        raise ValueError("--services value parsed to an empty set")

    return result


def _load_scope_config() -> dict | None:
    """Read ~/.hermes/google_scopes.json; return None if absent/unreadable.

    Returning None (rather than a default) preserves legacy behavior:
    callers fall back to the full SCOPES list, so installs that never
    set a config behave exactly as before.
    """
    if not SCOPES_CONFIG_PATH.exists():
        return None
    try:
        raw = json.loads(SCOPES_CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    cleaned = {
        k: v
        for k, v in raw.items()
        if k in SERVICE_SCOPES and v in {"readonly", "readwrite"}
    }
    return cleaned or None


def _save_scope_config(config: dict) -> None:
    SCOPES_CONFIG_PATH.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def _active_scopes() -> list:
    """OAuth scopes to request, derived from google_scopes.json.

    If no config exists, returns the full legacy SCOPES list so behavior
    matches the pre-config code path.
    """
    config = _load_scope_config()
    if config is None:
        return list(DEFAULT_SCOPES)
    scopes: list = []
    for svc, mode in config.items():
        scopes.extend(SERVICE_SCOPES[svc][mode])
    return list(dict.fromkeys(scopes))

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


def _missing_scopes_from_payload(payload: dict) -> list[str]:
    raw = payload.get("scopes") or payload.get("scope")
    if not raw:
        return []
    granted = {s.strip() for s in (raw.split() if isinstance(raw, str) else raw) if s.strip()}
    expected = _active_scopes()
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
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + REQUIRED_PACKAGES,
            stdout=subprocess.DEVNULL,
        )
        print("Dependencies installed.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies: {e}")
        print(
            "On environments without pip (e.g. Nix), install the optional extra instead:"
        )
        print("  pip install 'hermes-agent[google]'")
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
            creds.refresh(Request())
            TOKEN_PATH.write_text(
                json.dumps(
                    _normalize_authorized_user_payload(json.loads(creds.to_json())),
                    indent=2,
                )
            )
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


def _save_pending_auth(*, state: str, code_verifier: str):
    """Persist the OAuth session bits needed for a later token exchange."""
    PENDING_AUTH_PATH.write_text(
        json.dumps(
            {
                "state": state,
                "code_verifier": code_verifier,
                "redirect_uri": REDIRECT_URI,
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


def get_auth_url(format_json: bool = False):
    """Print the OAuth authorization URL. User visits this in a browser."""
    if not CLIENT_SECRET_PATH.exists():
        print("ERROR: No client secret stored. Run --client-secret first.")
        sys.exit(1)

    _ensure_deps()
    from google_auth_oauthlib.flow import Flow

    scopes = _active_scopes()
    flow = Flow.from_client_secrets_file(
        str(CLIENT_SECRET_PATH),
        scopes=scopes,
        redirect_uri=REDIRECT_URI,
        autogenerate_code_verifier=True,
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
    )
    _save_pending_auth(state=state, code_verifier=flow.code_verifier)
    if format_json:
        print(json.dumps({"auth_url": auth_url, "scopes": scopes}))
    else:
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

    requested_scopes = _active_scopes()
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

    missing_scopes = _missing_scopes_from_payload(token_payload)
    if missing_scopes:
        print(f"WARNING: Token missing some Google Workspace scopes: {', '.join(missing_scopes)}")
        print("Some services may not be available.")

    TOKEN_PATH.write_text(json.dumps(token_payload, indent=2))
    PENDING_AUTH_PATH.unlink(missing_ok=True)
    print(f"OK: Authenticated. Token saved to {TOKEN_PATH}")
    print(f"Profile-scoped token location: {display_hermes_home()}/google_token.json")


def revoke():
    """Revoke stored token and delete it."""
    if not TOKEN_PATH.exists():
        print("No token to revoke.")
        return

    _ensure_deps()
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    try:
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), _active_scopes())
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


def show_mode(format_json: bool = False):
    """Print the current scope config and the scopes that would be requested."""
    config = _load_scope_config()
    scopes = _active_scopes()
    if format_json:
        print(json.dumps({
            "config": config,
            "config_path": str(SCOPES_CONFIG_PATH),
            "config_present": SCOPES_CONFIG_PATH.exists(),
            "scopes": scopes,
        }, indent=2))
        return

    if config is None:
        print(f"No scope config at {SCOPES_CONFIG_PATH} — falling back to legacy full-access defaults.")
    else:
        print(f"Current scope config ({SCOPES_CONFIG_PATH}):")
        for svc in ALL_SERVICES:
            if svc in config:
                print(f"  {svc:9s} {config[svc]}")
    print(f"\nScopes that would be requested ({len(scopes)}):")
    for scope in scopes:
        print(f"  {scope}")


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
    group.add_argument("--show-mode", action="store_true", help="Print current per-service scope config")
    # Modifiers — used with --auth-url or --show-mode.
    parser.add_argument(
        "--services",
        metavar="SPEC",
        help=(
            "Per-service scope spec, e.g. 'gmail=ro,calendar=rw,drive=ro'. "
            "Bare names default to rw ('all' means every service). "
            "Saved to ~/.hermes/google_scopes.json and used for --auth-url. "
            "Examples: 'all', 'all=ro', 'gmail,calendar', 'gmail=ro,calendar=rw'."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for --auth-url and --show-mode (default: text)",
    )
    args = parser.parse_args()

    # Process --services modifier before any action that reads scope config.
    if args.services:
        try:
            parsed = _parse_services_spec(args.services)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        _save_scope_config(parsed)

    if args.check:
        sys.exit(0 if check_auth() else 1)
    if getattr(args, "check_live", False):
        sys.exit(0 if check_auth_live() else 1)
    elif args.client_secret:
        store_client_secret(args.client_secret)
    elif args.auth_url:
        get_auth_url(format_json=(args.format == "json"))
    elif args.auth_code:
        exchange_auth_code(args.auth_code)
    elif args.revoke:
        revoke()
    elif args.install_deps:
        sys.exit(0 if install_deps() else 1)
    elif getattr(args, "show_mode", False):
        show_mode(format_json=(args.format == "json"))


if __name__ == "__main__":
    main()

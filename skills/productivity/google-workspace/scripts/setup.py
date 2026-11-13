#!/usr/bin/env python3
"""Google Workspace OAuth2 setup for Hermes Agent.

Fully non-interactive — designed to be driven by the agent via terminal commands.
The agent mediates between this script and the user (works on CLI, Telegram, Discord, etc.)

Commands:
  setup.py --check                          # Is auth valid? Exit 0 = yes, 1 = no
  setup.py --check --services gmail          # Valid + has the Gmail scopes
  setup.py --check-live                      # Real API call on a granted service
  setup.py --client-secret /path/to.json    # Store OAuth client credentials
  setup.py --auth-url --services email       # Print OAuth URL (Gmail scopes only)
  setup.py --auth-code CODE                  # Exchange auth code for token
  setup.py --revoke                          # Revoke and delete stored token
  setup.py --install-deps                    # Install Python dependencies only

Scopes are requested per service so the consent screen only asks for what the
user actually needs. A narrow grant is valid, not broken: --check-live probes
the first service the token really holds instead of assuming Calendar.

Agent workflow:
  1. Run --check. If exit 0, auth is good — skip setup.
  2. Ask user for client_secret.json path. Run --client-secret PATH.
  3. Run --auth-url --services <list>. Send the printed URL to the user.
  4. User opens URL, authorizes, gets redirected to a page with a code.
  5. User pastes the code. Agent runs --auth-code CODE.
  6. Run --check-live to verify. Done.
"""

from __future__ import annotations  # allow PEP 604 `X | None` on Python 3.9+

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import pm
except ImportError:
    # A copied skill must not install into an unrelated Python environment.
    pm = None

# Ensure sibling modules (_hermes_home) are importable when run standalone.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from _hermes_home import display_hermes_home, get_hermes_home

HERMES_HOME = get_hermes_home()
TOKEN_PATH = HERMES_HOME / "google_token.json"
CLIENT_SECRET_PATH = HERMES_HOME / "google_client_secret.json"
PENDING_AUTH_PATH = HERMES_HOME / "google_oauth_pending.json"

# Scope groups per service. Requesting only what the user needs keeps the
# consent screen honest and makes --check/--check-live meaningful: a token
# that deliberately lacks a scope must not be reported as broken.
SCOPE_GROUPS: dict[str, list[str]] = {
    "gmail": [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/gmail.modify",
    ],
    "calendar": ["https://www.googleapis.com/auth/calendar"],
    "drive": ["https://www.googleapis.com/auth/drive"],
    "contacts": ["https://www.googleapis.com/auth/contacts.readonly"],
    "sheets": ["https://www.googleapis.com/auth/spreadsheets"],
    "docs": ["https://www.googleapis.com/auth/documents"],
}

ALL_SERVICES = list(SCOPE_GROUPS)

# Names users naturally reach for that map onto a scope group.
SERVICE_ALIASES = {
    "email": "gmail",
    "mail": "gmail",
    "gws": "all",
    "workspace": "all",
    "full": "all",
}

# Order used by --check-live when picking a service to probe. gmail first:
# it is the most commonly granted group and its probe is one cheap call.
PROBE_ORDER = ["gmail", "calendar", "drive", "contacts", "sheets", "docs"]

# Services with a cheap read-only endpoint we can actually call. sheets and
# docs have no list call, so they are verified by scope presence only.
PROBEABLE_SERVICES = {"gmail", "calendar", "drive", "contacts"}

# Backwards-compatible flat list of every scope (equivalent to "all").
SCOPES = [scope for group in SCOPE_GROUPS.values() for scope in group]

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
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_services(services: str | None) -> list[str]:
    """Turn a --services string into a validated, de-duplicated service list."""
    if not services:
        return list(ALL_SERVICES)
    resolved: list[str] = []
    unknown: list[str] = []
    for raw in services.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        name = SERVICE_ALIASES.get(name, name)
        if name == "all":
            return list(ALL_SERVICES)
        if name in SCOPE_GROUPS:
            if name not in resolved:
                resolved.append(name)
        else:
            unknown.append(raw.strip())
    if unknown:
        raise ValueError(
            f"Unknown service(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(ALL_SERVICES)}, all"
        )
    if not resolved:
        raise ValueError("--services was given but resolved to no known service.")
    return resolved


def _scopes_for_services(services: str | None) -> list[str]:
    """Flat scope list for a --services string (all services when omitted)."""
    return [scope for name in _resolve_services(services) for scope in SCOPE_GROUPS[name]]


def _granted_services(payload: dict) -> list[str]:
    """Which service groups the stored token actually covers."""
    raw = payload.get("scopes") or payload.get("scope")
    if not raw:
        return []
    granted = {s.strip() for s in (raw.split() if isinstance(raw, str) else raw) if s.strip()}
    return [name for name in ALL_SERVICES if granted.intersection(SCOPE_GROUPS[name])]


def _missing_scopes_from_payload(payload: dict, requested: list[str] | None = None) -> list[str]:
    raw = payload.get("scopes") or payload.get("scope")
    if not raw:
        return []
    granted = {s.strip() for s in (raw.split() if isinstance(raw, str) else raw) if s.strip()}
    return sorted(scope for scope in (requested or SCOPES) if scope not in granted)


def _format_missing_scopes(missing_scopes: list[str]) -> str:
    bullets = "\n".join(f"  - {scope}" for scope in missing_scopes)
    return (
        "Token is valid but missing requested Google Workspace scopes:\n"
        f"{bullets}\n"
        "Re-run setup with the services you need to refresh consent."
    )


def install_deps():
    """Sync Hermes' declared Google extra, ready for the next process."""
    if pm is None:
        print("ERROR: Run this script in the Hermes environment; use hermes setup first.")
        return False
    try:
        pm.sync_venv(["google"], explicit=True)
    except Exception as exc:
        print(f"ERROR: Failed to install Google dependencies: {exc}")
        return False
    print("Google dependencies synced. Restart Hermes, then rerun setup to continue OAuth.")
    return True


def _ensure_deps():
    """Let PM check imports and stop if activation needs a new process."""
    if pm is None:
        print("ERROR: Run this script in the Hermes environment; use hermes setup first.")
        sys.exit(1)
    try:
        pm.ensure_import("google")
    except Exception as exc:
        print(f"ERROR: Google dependencies unavailable: {exc}")
        sys.exit(1)


def _probe_service(service: str):
    """Make one cheap read-only call for a service. Raises on failure."""
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    # Omit scopes: the stored token may legitimately hold fewer than we
    # request, and passing scopes makes google-auth validate them on refresh.
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH))

    if service == "gmail":
        build("gmail", "v1", credentials=creds).users().getProfile(userId="me").execute()
    elif service == "calendar":
        build("calendar", "v3", credentials=creds).calendarList().list(maxResults=1).execute()
    elif service == "drive":
        build("drive", "v3", credentials=creds).files().list(pageSize=1, fields="files(id)").execute()
    elif service == "contacts":
        build("people", "v1", credentials=creds).people().connections().list(
            pageSize=1, personFields="names"
        ).execute()


def check_auth_live():
    """Check auth with a real API call to detect disabled_client/account issues.

    Probes the first granted service we can actually call, so a deliberately
    narrow grant (for example Gmail only) reports OK rather than failing on a
    service the user never authorized.
    """
    # Judge scope completeness against what the token actually granted, not the
    # full set: --check-live reports the live-call outcome, and a deliberate
    # narrow grant must not print a wall of "missing" services above it.
    payload = _load_token_payload(TOKEN_PATH)
    granted = _granted_services(payload)
    granted_arg = ",".join(granted) if granted else None

    # quiet=True suppresses the "AUTHENTICATED" print from check_auth so the
    # final status line reflects the live-call outcome (OK or FAILED).
    if not check_auth(quiet=True, requested_services=granted_arg):
        return False

    probeable = [s for s in PROBE_ORDER if s in granted and s in PROBEABLE_SERVICES]

    if not probeable:
        if granted:
            print(
                "LIVE_CHECK_OK: Token valid; granted services "
                f"({', '.join(granted)}) have no probe endpoint, verified by scope only."
            )
            return True
        print("LIVE_CHECK_FAILED: Token carries no recognized service scopes.")
        return False

    service = probeable[0]
    try:
        _probe_service(service)
        extra = f" (+{', '.join(probeable[1:])})" if len(probeable) > 1 else ""
        print(f"LIVE_CHECK_OK: Real {service} API call succeeded{extra}.")
        return True
    except Exception as e:
        err_str = str(e).lower()
        if "disabled_client" in err_str or "invalid_client" in err_str:
            print(f"LIVE_CHECK_FAILED: OAuth client or account disabled: {e}")
            print("  1. Check Google Cloud Console for disabled OAuth client")
            print("  2. Check myaccount.google.com for account status")
            print("  3. Do NOT retry with a disabled account")
        elif "insufficient" in err_str and "scope" in err_str:
            print(f"LIVE_CHECK_FAILED: {service} call rejected for insufficient scopes: {e}")
            print(f"  Granted: {', '.join(granted) or 'none'}")
            print("  Re-run setup with --services including that service to refresh consent.")
        else:
            print(f"LIVE_CHECK_FAILED: {e}")
        return False


def check_auth(quiet: bool = False, requested_services: str | None = None):
    """Check if stored credentials are valid. Prints status, exits 0 or 1.

    Only scopes for `requested_services` (default: everything) count as
    missing, so a narrow grant reports AUTHENTICATED rather than a wall of
    scope warnings for services the user never asked for.
    """
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

    expected_scopes = _scopes_for_services(requested_services)
    payload = _load_token_payload(TOKEN_PATH)
    if creds.valid:
        missing_scopes = _missing_scopes_from_payload(payload, expected_scopes)
        if missing_scopes:
            print(f"AUTHENTICATED (partial): Token valid but missing {len(missing_scopes)} requested scopes:")
            for s in missing_scopes:
                print(f"  - {s}")
        if not quiet:
            granted = _granted_services(payload)
            suffix = f" (services: {', '.join(granted)})" if granted else ""
            print(f"AUTHENTICATED: Token valid at {TOKEN_PATH}{suffix}")
        return True

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            TOKEN_PATH.write_text(
                json.dumps(
                    _normalize_authorized_user_payload(json.loads(creds.to_json())),
                    indent=2,
                ), encoding="utf-8"
            )
            payload = _load_token_payload(TOKEN_PATH)
            missing_scopes = _missing_scopes_from_payload(payload, expected_scopes)
            if missing_scopes:
                print(f"AUTHENTICATED (partial): Token refreshed but missing {len(missing_scopes)} requested scopes:")
                for s in missing_scopes:
                    print(f"  - {s}")
            if not quiet:
                granted = _granted_services(payload)
                suffix = f" (services: {', '.join(granted)})" if granted else ""
                print(f"AUTHENTICATED: Token refreshed at {TOKEN_PATH}{suffix}")
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

    CLIENT_SECRET_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"OK: Client secret saved to {CLIENT_SECRET_PATH}")


def _save_pending_auth(*, state: str, code_verifier: str, services: list[str] | None = None):
    """Persist the OAuth session bits needed for a later token exchange."""
    PENDING_AUTH_PATH.write_text(
        json.dumps(
            {
                "state": state,
                "code_verifier": code_verifier,
                "redirect_uri": REDIRECT_URI,
                # Remember what we asked for so the missing-scope warning after
                # the exchange is judged against the intended grant.
                "services": services or list(ALL_SERVICES),
            },
            indent=2,
        ), encoding="utf-8"
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


def get_auth_url(services: str | None = None):
    """Print the OAuth authorization URL. User visits this in a browser."""
    if not CLIENT_SECRET_PATH.exists():
        print("ERROR: No client secret stored. Run --client-secret first.")
        sys.exit(1)

    resolved = _resolve_services(services)
    _ensure_deps()
    from google_auth_oauthlib.flow import Flow

    flow = Flow.from_client_secrets_file(
        str(CLIENT_SECRET_PATH),
        scopes=[scope for name in resolved for scope in SCOPE_GROUPS[name]],
        redirect_uri=REDIRECT_URI,
        autogenerate_code_verifier=True,
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
    )
    _save_pending_auth(state=state, code_verifier=flow.code_verifier, services=resolved)
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
    requested = [scope for name in (pending_auth.get("services") or ALL_SERVICES)
                 for scope in SCOPE_GROUPS.get(name, [])] or list(SCOPES)
    granted_scopes = list(requested)
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
    elif granted_scopes != requested:
        # granted_scopes was extracted from the callback URL
        token_payload["scopes"] = granted_scopes

    # Judge the grant against what this session actually asked for, so a
    # deliberate Gmail-only run does not warn about Drive/Calendar.
    missing_scopes = _missing_scopes_from_payload(token_payload, requested)
    if missing_scopes:
        print(f"WARNING: Token missing some requested scopes: {', '.join(missing_scopes)}")
        print("Those services will not be available.")

    TOKEN_PATH.write_text(json.dumps(token_payload, indent=2), encoding="utf-8")
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
        # Omit scopes: the stored token may hold fewer scopes than the full
        # set, and passing scopes makes google-auth validate them on refresh.
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
    group.add_argument("--revoke", action="store_true", help="Revoke and delete stored token")
    group.add_argument("--install-deps", action="store_true", help="Install Python dependencies")
    parser.add_argument(
        "--services",
        metavar="LIST",
        help=(
            "Comma-separated services to authorize/check: "
            f"{', '.join(ALL_SERVICES)}, or all (default: all). "
            "Aliases: email/mail=gmail, workspace/full/gws=all"
        ),
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for --check, --check-live and --auth-url (default: text)",
    )
    args = parser.parse_args()

    # Validate --services up front so a typo fails before any network work.
    if args.services:
        try:
            _resolve_services(args.services)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(2)

    if args.check:
        ok = check_auth(requested_services=args.services)
        if args.format == "json":
            print(json.dumps({
                "status": "authenticated" if ok else "unauthenticated",
                "token_path": str(TOKEN_PATH),
                "services": _granted_services(_load_token_payload(TOKEN_PATH)),
            }))
        sys.exit(0 if ok else 1)
    if getattr(args, "check_live", False):
        ok = check_auth_live()
        if args.format == "json":
            print(json.dumps({
                "status": "ok" if ok else "failed",
                "token_path": str(TOKEN_PATH),
                "services": _granted_services(_load_token_payload(TOKEN_PATH)),
            }))
        sys.exit(0 if ok else 1)
    elif args.client_secret:
        store_client_secret(args.client_secret)
    elif args.auth_url:
        get_auth_url(args.services)
    elif args.auth_code:
        exchange_auth_code(args.auth_code)
    elif args.revoke:
        revoke()
    elif args.install_deps:
        sys.exit(0 if install_deps() else 1)


if __name__ == "__main__":
    main()

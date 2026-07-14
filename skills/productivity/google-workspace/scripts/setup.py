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
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure sibling modules (_hermes_home) are importable when run standalone.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from _hermes_home import display_hermes_home, get_hermes_home
import auth_contexts as gauth

HERMES_HOME = get_hermes_home()
TOKEN_PATH = HERMES_HOME / "google_token.json"
CLIENT_SECRET_PATH = HERMES_HOME / "google_client_secret.json"
PENDING_AUTH_PATH = HERMES_HOME / "google_oauth_pending.json"
SCOPES = gauth.SCOPES

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


def _load_token_payload(path: Path = TOKEN_PATH, auth_context: str = "default") -> dict:
    if path == TOKEN_PATH:
        return _token_payload(auth_context)
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _missing_scopes_from_payload(payload: dict, required_scopes: list[str] | None = None) -> list[str]:
    return gauth.missing_scopes(payload, required_scopes)


def _format_missing_scopes(missing_scopes: list[str]) -> str:
    bullets = "\n".join(f"  - {scope}" for scope in missing_scopes)
    return (
        "Token is valid but missing required Google Workspace scopes:\n"
        f"{bullets}\n"
        "Run the Google Workspace setup again from this same Hermes profile to refresh consent."
    )


def _client_secret_payload(auth_context: str = "default") -> dict:
    if auth_context == "default" and CLIENT_SECRET_PATH.exists() and not gauth.store_path().exists():
        try:
            return json.loads(CLIENT_SECRET_PATH.read_text())
        except Exception:
            return {}
    return gauth.get_client_secret(auth_context)


def _client_secret_file(auth_context: str = "default") -> Path:
    if auth_context == "default" and CLIENT_SECRET_PATH.exists() and not gauth.store_path().exists():
        return CLIENT_SECRET_PATH
    return gauth.materialize_client_secret_file(auth_context)


def _token_payload(auth_context: str = "default") -> dict:
    if auth_context == "default" and TOKEN_PATH.exists() and not gauth.store_path().exists():
        try:
            return json.loads(TOKEN_PATH.read_text())
        except Exception:
            return {}
    return gauth.get_token_payload(auth_context)


def _token_file(auth_context: str = "default") -> Path:
    if auth_context == "default" and TOKEN_PATH.exists() and not gauth.store_path().exists():
        return TOKEN_PATH
    return gauth.materialize_token_file(auth_context)


def _set_token_payload(auth_context: str, payload: dict, **kwargs):
    if auth_context == "default" and not gauth.store_path().exists():
        TOKEN_PATH.write_text(json.dumps(payload, indent=2))
        try:
            TOKEN_PATH.chmod(0o600)
        except OSError:
            pass
        return
    gauth.set_token_payload(auth_context, payload, **kwargs)


def _pending_payload(auth_context: str = "default") -> dict:
    if auth_context == "default" and PENDING_AUTH_PATH.exists() and not gauth.store_path().exists():
        try:
            return json.loads(PENDING_AUTH_PATH.read_text())
        except Exception:
            return {}
    return gauth.get_pending_auth(auth_context)


def _set_pending_payload(auth_context: str, payload: dict):
    if auth_context == "default" and not gauth.store_path().exists():
        PENDING_AUTH_PATH.write_text(json.dumps(payload, indent=2))
        try:
            PENDING_AUTH_PATH.chmod(0o600)
        except OSError:
            pass
        return
    gauth.set_pending_auth(auth_context, payload)


def _clear_pending(auth_context: str = "default"):
    if auth_context == "default":
        PENDING_AUTH_PATH.unlink(missing_ok=True)
    gauth.clear_pending_auth(auth_context)


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


def check_auth_live(auth_context: str = "default"):
    """Force a real OAuth refresh to detect disabled clients/accounts.

    Refreshing is service-neutral, so least-privilege Gmail-, Drive-, Sheets-,
    or Docs-only contexts do not need an unrelated Calendar scope merely to
    validate their OAuth client and account.
    """
    # quiet=True suppresses the "AUTHENTICATED" print from check_auth so the
    # final status line reflects the live-call outcome (OK or FAILED).
    if not check_auth(quiet=True, auth_context=auth_context):
        return False
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        payload = _token_payload(auth_context)
        creds = Credentials.from_authorized_user_file(str(_token_file(auth_context)))
        if not creds.refresh_token:
            print("LIVE_CHECK_FAILED: Stored credentials have no refresh token.")
            return False
        creds.refresh(Request())
        refreshed_payload = _normalize_authorized_user_payload(
            json.loads(creds.to_json())
        )
        existing_scopes = gauth.scope_list(
            payload.get("scopes") or payload.get("scope")
        )
        refreshed_scopes = gauth.scope_list(
            refreshed_payload.get("scopes") or refreshed_payload.get("scope")
        )
        if existing_scopes and not refreshed_scopes:
            refreshed_payload["scopes"] = existing_scopes
        _set_token_payload(auth_context, refreshed_payload)
        print("LIVE_CHECK_OK: OAuth token refresh succeeded.")
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


def check_auth(quiet: bool = False, auth_context: str = "default", required_scopes: list[str] | None = None):
    """Check if stored credentials are valid. Prints status, exits 0 or 1."""
    token_payload = _token_payload(auth_context)
    if not token_payload:
        print(f"NOT_AUTHENTICATED: No token for Google auth context {auth_context!r}")
        return False

    _ensure_deps()
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    try:
        # Don't pass scopes — user may have authorized only a subset.
        # Passing scopes forces google-auth to validate them on refresh,
        # which fails with invalid_scope if the token has fewer scopes
        # than requested.
        creds = Credentials.from_authorized_user_file(str(_token_file(auth_context)))
    except Exception as e:
        print(f"TOKEN_CORRUPT: {e}")
        return False

    payload = _token_payload(auth_context)
    if creds.valid:
        missing_scopes = _missing_scopes_from_payload(payload, required_scopes)
        if missing_scopes:
            print(f"AUTHENTICATED (partial): Token valid but missing {len(missing_scopes)} scopes:")
            for s in missing_scopes:
                print(f"  - {s}")
        if not quiet:
            print(f"AUTHENTICATED: Token valid for Google auth context {auth_context!r}")
        return True

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            refreshed_payload = _normalize_authorized_user_payload(json.loads(creds.to_json()))
            existing_scopes = gauth.scope_list(payload.get("scopes") or payload.get("scope"))
            if existing_scopes and not gauth.scope_list(refreshed_payload.get("scopes") or refreshed_payload.get("scope")):
                refreshed_payload["scopes"] = existing_scopes
            _set_token_payload(auth_context, refreshed_payload)
            missing_scopes = _missing_scopes_from_payload(_token_payload(auth_context), required_scopes)
            if missing_scopes:
                print(f"AUTHENTICATED (partial): Token refreshed but missing {len(missing_scopes)} scopes:")
                for s in missing_scopes:
                    print(f"  - {s}")
            if not quiet:
                print(f"AUTHENTICATED: Token refreshed for Google auth context {auth_context!r}")
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


def store_client_secret(path: str, auth_context: str = "default", account_hint: str = ""):
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

    gauth.set_client_secret(auth_context, data, account_hint=account_hint)
    print(f"OK: Client secret saved for Google auth context {auth_context!r}")


def _save_pending_auth(*, state: str, code_verifier: str, auth_context: str = "default", requested_scopes: list[str] | None = None, services: list[str] | None = None):
    """Persist the OAuth session bits needed for a later token exchange."""
    payload = {"state": state, "code_verifier": code_verifier, "redirect_uri": REDIRECT_URI}
    if requested_scopes is not None:
        payload["requested_scopes"] = requested_scopes
    if services is not None:
        payload["services"] = services
    _set_pending_payload(auth_context, payload)


def _load_pending_auth(auth_context: str = "default") -> dict:
    """Load the pending OAuth session created by get_auth_url()."""
    data = _pending_payload(auth_context)
    if not data:
        print("ERROR: No pending OAuth session found. Run --auth-url first.")
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


def get_auth_url(auth_context: str = "default", services: str = "all", scopes: str = ""):
    """Print the OAuth authorization URL. User visits this in a browser."""
    if not _client_secret_payload(auth_context):
        print(f"ERROR: No client secret stored for Google auth context {auth_context!r}. Run --client-secret first.")
        sys.exit(1)

    _ensure_deps()
    from google_auth_oauthlib.flow import Flow

    flow = Flow.from_client_secrets_file(
        str(_client_secret_file(auth_context)),
        scopes=gauth.resolve_scopes(services=services, scopes=scopes),
        redirect_uri=REDIRECT_URI,
        autogenerate_code_verifier=True,
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
    )
    requested_scopes = gauth.resolve_scopes(services=services, scopes=scopes)
    service_list = [s.strip().lower() for s in (services or "all").split(",") if s.strip()]
    _save_pending_auth(state=state, code_verifier=flow.code_verifier, auth_context=auth_context, requested_scopes=requested_scopes, services=service_list)
    # Print just the URL so the agent can extract it cleanly
    print(auth_url)


def exchange_auth_code(code: str, auth_context: str = "default", strict_scopes: bool = False):
    """Exchange the authorization code for a token and save it."""
    if not _client_secret_payload(auth_context):
        print(f"ERROR: No client secret stored for Google auth context {auth_context!r}. Run --client-secret first.")
        sys.exit(1)

    pending_auth = _load_pending_auth(auth_context)
    raw_callback = code
    code, returned_state = _extract_code_and_state(code)
    if returned_state and returned_state != pending_auth["state"]:
        print("ERROR: OAuth state mismatch. Run --auth-url again to start a fresh session.")
        sys.exit(1)

    _ensure_deps()
    from google_auth_oauthlib.flow import Flow
    from urllib.parse import parse_qs, urlparse

    # Extract granted scopes from the callback URL if the user pasted the full redirect URL.
    granted_scopes = list(pending_auth.get("requested_scopes") or SCOPES)
    if isinstance(raw_callback, str) and raw_callback.startswith("http"):
        params = parse_qs(urlparse(raw_callback).query)
        scope_val = (params.get("scope") or [""])[0].strip()
        if scope_val:
            granted_scopes = scope_val.split()

    flow = Flow.from_client_secrets_file(
        str(_client_secret_file(auth_context)),
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
    elif granted_scopes != SCOPES:
        # granted_scopes was extracted from the callback URL
        token_payload["scopes"] = granted_scopes

    requested_scopes = pending_auth.get("requested_scopes") or list(SCOPES)
    missing_scopes = _missing_scopes_from_payload(token_payload, requested_scopes)
    if missing_scopes:
        msg = f"Token missing requested Google Workspace scopes: {', '.join(missing_scopes)}"
        if strict_scopes:
            print(f"ERROR: {msg}")
            sys.exit(1)
        print(f"WARNING: {msg}")
        print("Some services may not be available.")

    _set_token_payload(auth_context, token_payload, services=pending_auth.get("services"), requested_scopes=requested_scopes)
    _clear_pending(auth_context)
    print(f"OK: Authenticated Google auth context {auth_context!r}.")
    print(f"Profile-scoped context store: {display_hermes_home()}/{gauth.AUTH_CONTEXTS_NAME}")


def revoke(auth_context: str = "default"):
    """Revoke stored token and delete it."""
    if not _token_payload(auth_context):
        print("No token to revoke.")
        return

    _ensure_deps()
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    try:
        creds = Credentials.from_authorized_user_file(str(_token_file(auth_context)), gauth.granted_scopes_for_context(auth_context))
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

    gauth.delete_token(auth_context)
    print(f"Deleted token for Google auth context {auth_context!r}")


def main():
    parser = argparse.ArgumentParser(description="Google Workspace OAuth setup for Hermes")
    parser.add_argument("--auth-context", default="default", help="Named Google OAuth auth context (default: legacy/default)")
    parser.add_argument("--services", default="all", help="Comma-separated service presets (e.g. gmail-readonly,calendar,drive) for --auth-url/--check")
    parser.add_argument("--scopes", default="", help="Comma- or space-separated explicit OAuth scopes; overrides --services")
    parser.add_argument("--account-hint", default="", help="Optional human account label/email saved with the context")
    parser.add_argument("--strict-scopes", action="store_true", help="Fail auth-code exchange if granted scopes miss requested scopes")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Check if auth is valid (exit 0=yes, 1=no)")
    group.add_argument("--check-live", action="store_true", help="Force a live OAuth refresh (detects disabled_client without requiring a service scope)")
    group.add_argument("--client-secret", metavar="PATH", help="Store OAuth client_secret.json")
    group.add_argument("--auth-url", action="store_true", help="Print OAuth URL for user to visit")
    group.add_argument("--auth-code", metavar="CODE", help="Exchange auth code for token")
    group.add_argument("--revoke", action="store_true", help="Revoke and delete stored token")
    group.add_argument("--install-deps", action="store_true", help="Install Python dependencies")
    group.add_argument("--print-scopes", action="store_true", help="Print scopes resolved from --services/--scopes and exit")
    group.add_argument("--list-contexts", action="store_true", help="List configured Google auth contexts")
    group.add_argument("--set-default-for", metavar="SERVICES", help="Comma-separated API services to route to --auth-context by default")
    args = parser.parse_args()

    try:
        requested_scopes = gauth.resolve_scopes(services=args.services, scopes=args.scopes)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    if args.set_default_for:
        try:
            gauth.set_default_for_services(args.auth_context, args.set_default_for)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        print(f"OK: Google auth context {args.auth_context!r} is default for {args.set_default_for}")
    if args.check:
        sys.exit(0 if check_auth(auth_context=args.auth_context, required_scopes=requested_scopes if args.services != "all" or args.scopes else None) else 1)
    if getattr(args, "check_live", False):
        sys.exit(0 if check_auth_live(auth_context=args.auth_context) else 1)
    elif args.client_secret:
        store_client_secret(args.client_secret, auth_context=args.auth_context, account_hint=args.account_hint)
    elif args.auth_url:
        get_auth_url(auth_context=args.auth_context, services=args.services, scopes=args.scopes)
    elif args.auth_code:
        exchange_auth_code(args.auth_code, auth_context=args.auth_context, strict_scopes=args.strict_scopes)
    elif args.revoke:
        revoke(auth_context=args.auth_context)
    elif args.install_deps:
        sys.exit(0 if install_deps() else 1)
    elif args.print_scopes:
        print("\n".join(requested_scopes))
    elif args.list_contexts:
        print(json.dumps({"contexts": gauth.list_contexts(), "default_contexts": gauth.load_store().get("default_contexts", {})}, indent=2))


if __name__ == "__main__":
    main()

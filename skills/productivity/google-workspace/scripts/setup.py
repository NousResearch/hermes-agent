#!/usr/bin/env python3
"""Google Workspace OAuth2 setup for Hermes Agent.

Fully non-interactive — designed to be driven by the agent via terminal commands.
The agent mediates between this script and the user (works on CLI, Telegram, Discord, etc.)

--identity is required on every invocation (e.g. 'jid', 'zarkash') — there
is no default identity, and each identity's credentials are fully isolated
from every other's (see _google_identities.py).

Commands:
  setup.py --identity jid --check                       # Is auth valid?
  setup.py --identity jid --client-secret /path/to.json  # Store OAuth client credentials
  setup.py --identity jid --auth-url                     # Print the OAuth URL for user to visit
  setup.py --identity jid --auth-code CODE               # Exchange auth code for token
  setup.py --identity jid --revoke                       # Revoke and delete stored token
  setup.py --install-deps                                # Install Python dependencies only (identity-agnostic)

Agent workflow (repeat per identity — swap --identity for a different person):
  1. Run --identity <name> --check. If exit 0, auth is good — skip setup.
  2. Ask that person for their client_secret.json path. Run --identity <name> --client-secret PATH.
  3. Run --identity <name> --auth-url. Send the printed URL to them.
  4. They open the URL, authorize, get redirected to a page with a code. If
     this is a new identity sharing a device/browser with an existing
     Google-authenticated identity, remind them to sign out of any other
     active Google sessions first, so the consent screen can't accidentally
     authorize the wrong account.
  5. They paste the code. Agent runs --identity <name> --auth-code CODE.
  6. Run --identity <name> --check to verify. Done.
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
from _google_identities import get_google_credentials, UnknownGoogleIdentityError

HERMES_HOME = get_hermes_home()

# Resolved per-identity at CLI dispatch time (see _resolve_identity / main()).
# Defaulted to identity="jid" so a direct module import (as tests do) behaves
# exactly as it always has. Real CLI usage always goes through main(), where
# --identity is a required argparse argument with no default.
TOKEN_PATH, CLIENT_SECRET_PATH, SCOPES = get_google_credentials("jid")
PENDING_AUTH_PATH = TOKEN_PATH.parent / "google_oauth_pending.json"
# Sidecar written only on a successful --auth-code token EXCHANGE (a real
# full re-auth), never on a routine access-token refresh. TOKEN_PATH's own
# mtime is NOT a reliable re-auth signal — google-auth rewrites the token
# file on every refresh() call too (see check_auth()/check_auth_live()
# above), so a script that inferred "last re-auth" from TOKEN_PATH.stat()
# would be fooled by ordinary background refreshes. This sidecar is the
# only reliable anchor for estimating Google's ~7-day Testing-mode refresh
# token expiry window (see the hermes-oauth-expiry-check cron job).
# Re-pointed alongside TOKEN_PATH for every identity in _resolve_identity()
# below — this is purely path-derived from whatever identity was resolved,
# so it works for any identity in _google_identities.py's registry without
# any code change here.
REAUTH_SIDECAR_PATH = TOKEN_PATH.parent / "google_token_reauth_at.json"
# Set by _resolve_identity() below; used only to label the sidecar payload
# with the identity it belongs to (defense-in-depth — the sidecar's own
# directory already scopes it, this is just for a human/script reading the
# file directly to confirm whose record it is without cross-referencing
# paths).
CURRENT_IDENTITY: str | None = "jid"


def _resolve_identity(identity: str | None) -> None:
    """Re-point TOKEN_PATH/CLIENT_SECRET_PATH/SCOPES/PENDING_AUTH_PATH/
    REAUTH_SIDECAR_PATH at the given identity. FAIL-CLOSED: raises for a
    missing/unregistered identity.
    """
    global TOKEN_PATH, CLIENT_SECRET_PATH, SCOPES, PENDING_AUTH_PATH
    global REAUTH_SIDECAR_PATH, CURRENT_IDENTITY
    TOKEN_PATH, CLIENT_SECRET_PATH, SCOPES = get_google_credentials(identity)
    PENDING_AUTH_PATH = TOKEN_PATH.parent / "google_oauth_pending.json"
    REAUTH_SIDECAR_PATH = TOKEN_PATH.parent / "google_token_reauth_at.json"
    CURRENT_IDENTITY = identity
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        TOKEN_PATH.parent.chmod(0o700)
    except OSError:
        pass


def _record_reauth_timestamp() -> None:
    """Write/overwrite the re-auth sidecar for the currently-resolved identity.

    Called only from the successful-exchange path in exchange_auth_code(),
    i.e. only when a *full* re-auth actually happened (a fresh --auth-url +
    --auth-code round trip), never on a token refresh. This is the only
    place that writes REAUTH_SIDECAR_PATH — keep it that way, or the "only
    a real re-auth advances this" guarantee breaks.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    payload = {
        "identity": CURRENT_IDENTITY,
        "recorded_at": now.isoformat(),
        "recorded_at_epoch": now.timestamp(),
    }
    REAUTH_SIDECAR_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        REAUTH_SIDECAR_PATH.chmod(0o600)
    except OSError:
        pass

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


def _load_token_payload(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _missing_scopes_from_payload(payload: dict) -> list[str]:
    raw = payload.get("scopes") or payload.get("scope")
    if not raw:
        return []
    granted = {s.strip() for s in (raw.split() if isinstance(raw, str) else raw) if s.strip()}
    return sorted(scope for scope in SCOPES if scope not in granted)


def _format_missing_scopes(missing_scopes: list[str]) -> str:
    bullets = "\n".join(f"  - {scope}" for scope in missing_scopes)
    return (
        "Token is valid but missing required Google Workspace scopes:\n"
        f"{bullets}\n"
        "Run the Google Workspace setup again from this same Hermes profile to refresh consent."
    )


_DEDICATED_VENV = Path(__file__).resolve().parent / ".venv"


def _reexec_under_dedicated_venv_if_needed():
    """Transparently switch to a dedicated venv if the interpreter that actually launched us
    (typically the bare system `python3` a terminal-tool invocation resolves to, not the Hermes
    managed venv) is missing required packages and can't install them -- e.g. no `pip` and
    PEP 668 "externally managed" blocking a direct install (see install_deps()'s third tier).
    A no-op once already running under the dedicated venv, or when it doesn't exist yet."""
    if sys.prefix != sys.base_prefix:
        return  # already inside some venv -- never re-exec again, avoids any possible loop
    venv_python = _DEDICATED_VENV / "bin" / "python3"
    if not venv_python.exists() or not _missing_required_packages():
        return
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)


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
            uv_error = e

        # Third choice: sys.executable itself can't take an install (no pip, and PEP 668
        # "externally managed" blocks a direct write into it -- the common case when this
        # script is launched as a bare `python3 setup.py ...` and that resolves to the
        # system interpreter rather than the Hermes managed venv). Create a small dedicated
        # venv next to this script instead of fighting the OS protection, install there, and
        # let _reexec_under_dedicated_venv_if_needed() route future runs (this one included,
        # via install_deps()'s own caller re-running _missing_required_packages after this
        # returns) through it transparently.
        try:
            if not _DEDICATED_VENV.exists():
                subprocess.check_call(
                    [uv, "venv", "--python", "3.11", str(_DEDICATED_VENV)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            venv_python = _DEDICATED_VENV / "bin" / "python3"
            subprocess.check_call(
                [uv, "pip", "install", "--python", str(venv_python), "--quiet"] + missing,
                stdout=subprocess.DEVNULL,
            )
            print("Dependencies installed into a dedicated venv (system Python has no pip / "
                  "is externally managed).")
            print(f"Re-run this command -- it will now transparently use {venv_python}.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install dependencies via uv (direct): {uv_error}")
            print(f"ERROR: Dedicated-venv fallback also failed: {e}")
            print(f"Manually: {uv} venv --python 3.11 {_DEDICATED_VENV} && "
                  f"{uv} pip install --python {_DEDICATED_VENV}/bin/python3 {' '.join(REQUIRED_PACKAGES)}")
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
                ), encoding="utf-8"
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
        data = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("ERROR: File is not valid JSON.")
        sys.exit(1)

    if "installed" not in data and "web" not in data:
        print("ERROR: Not a Google OAuth client secret file (missing 'installed' key).")
        print("Download the correct file from: https://console.cloud.google.com/apis/credentials")
        sys.exit(1)

    CLIENT_SECRET_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        CLIENT_SECRET_PATH.chmod(0o600)
    except OSError:
        pass
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


def get_auth_url():
    """Print the OAuth authorization URL. User visits this in a browser."""
    if not CLIENT_SECRET_PATH.exists():
        print("ERROR: No client secret stored. Run --client-secret first.")
        sys.exit(1)

    _ensure_deps()
    from google_auth_oauthlib.flow import Flow

    flow = Flow.from_client_secrets_file(
        str(CLIENT_SECRET_PATH),
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
        autogenerate_code_verifier=True,
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
    )
    _save_pending_auth(state=state, code_verifier=flow.code_verifier)
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
    granted_scopes = list(SCOPES)
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
    elif granted_scopes != SCOPES:
        # granted_scopes was extracted from the callback URL
        token_payload["scopes"] = granted_scopes

    missing_scopes = _missing_scopes_from_payload(token_payload)
    if missing_scopes:
        print(f"WARNING: Token missing some Google Workspace scopes: {', '.join(missing_scopes)}")
        print("Some services may not be available.")

    TOKEN_PATH.write_text(json.dumps(token_payload, indent=2), encoding="utf-8")
    try:
        TOKEN_PATH.chmod(0o600)
    except OSError:
        pass
    PENDING_AUTH_PATH.unlink(missing_ok=True)
    # Record the real re-auth timestamp NOW, only on a successful token
    # exchange (see REAUTH_SIDECAR_PATH's docstring above) — this is the
    # anchor the daily hermes-oauth-expiry-check job estimates the 7-day
    # window from. Best-effort: a sidecar write failure must never make an
    # otherwise-successful re-auth look like it failed to the caller.
    try:
        _record_reauth_timestamp()
    except Exception as e:
        print(f"WARNING: Could not record re-auth timestamp sidecar: {e}")
    print(f"OK: Authenticated. Token saved to {TOKEN_PATH}")


def revoke():
    """Revoke stored token and delete it."""
    if not TOKEN_PATH.exists():
        print("No token to revoke.")
        return

    _ensure_deps()
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    try:
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
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
    _reexec_under_dedicated_venv_if_needed()
    parser = argparse.ArgumentParser(description="Google Workspace OAuth setup for Hermes")
    parser.add_argument(
        "--identity",
        required=True,
        help=(
            "Which person this setup is for (e.g. 'jid', 'zarkash'). "
            "Required — there is no default identity."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Check if auth is valid (exit 0=yes, 1=no)")
    group.add_argument("--check-live", action="store_true", help="Check auth with a real API call (detects disabled_client)")
    group.add_argument("--client-secret", metavar="PATH", help="Store OAuth client_secret.json")
    group.add_argument("--auth-url", action="store_true", help="Print OAuth URL for user to visit")
    group.add_argument("--auth-code", metavar="CODE", help="Exchange auth code for token")
    group.add_argument("--revoke", action="store_true", help="Revoke and delete stored token")
    group.add_argument("--install-deps", action="store_true", help="Install Python dependencies")
    args = parser.parse_args()
    try:
        _resolve_identity(args.identity)
    except UnknownGoogleIdentityError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.check:
        sys.exit(0 if check_auth() else 1)
    if getattr(args, "check_live", False):
        sys.exit(0 if check_auth_live() else 1)
    elif args.client_secret:
        store_client_secret(args.client_secret)
    elif args.auth_url:
        get_auth_url()
    elif args.auth_code:
        exchange_auth_code(args.auth_code)
    elif args.revoke:
        revoke()
    elif args.install_deps:
        sys.exit(0 if install_deps() else 1)


if __name__ == "__main__":
    main()

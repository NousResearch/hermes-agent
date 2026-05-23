#!/usr/bin/env python3
"""Google Workspace OAuth2 setup for Hermes Agent.

Fully non-interactive — designed to be driven by the agent via terminal commands.
The agent mediates between this script and the user (works on CLI, Telegram, Discord, etc.)

Single-account quickstart (default):
  setup.py --check                          # Is auth valid? Exit 0 = yes, 1 = no
  setup.py --client-secret /path/to.json    # Store OAuth client credentials
  setup.py --auth-url                       # Print the OAuth URL for user to visit
  setup.py --auth-code CODE                 # Exchange auth code for token
  setup.py --revoke                         # Revoke and delete stored token

Multi-account commands:
  setup.py --account user@example.com --auth-url           # Start OAuth for a specific account
  setup.py --account user@example.com --auth-code CODE     # Token saved per-account
  setup.py --account user@example.com --check              # Check a specific account
  setup.py --list-accounts                                 # Show known accounts + default
  setup.py --set-default user@example.com                  # Pick the default account
  setup.py --migrate-legacy                                # Move legacy single-account token into the new layout
  setup.py --remove-account user@example.com               # Forget an account (revokes too)

Resolution precedence when --account is omitted:
  1. ``HERMES_GOOGLE_ACCOUNT`` environment variable
  2. Default account pointer (set with ``--set-default``)
  3. Legacy single-account file (``~/.hermes/google_token.json``) if present

Backward compatibility:
  After ``--migrate-legacy``, the legacy ``google_token.json`` becomes a symlink
  pointing at the default account's per-account token. Cron jobs and any code
  that reads ``google_token.json`` directly continue to work unchanged.

Agent workflow (single account, fresh install):
  1. Run --check. If exit 0, auth is good — skip setup.
  2. Ask user for client_secret.json path. Run --client-secret PATH.
  3. Run --auth-url. Send the printed URL to the user.
  4. User opens URL, authorizes, gets redirected to a page with a code.
  5. User pastes the code. Agent runs --auth-code CODE.
  6. Run --check to verify. Done.

Agent workflow (adding a second account):
  1. Optionally run --migrate-legacy if there's an unmigrated legacy token.
  2. Run --account second@example.com --auth-url
  3. User authorizes; paste code into --account second@... --auth-code CODE.
  4. Pick a default with --set-default <whichever-they-use-most>.
"""

from __future__ import annotations  # allow PEP 604 `X | None` on Python 3.9+

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Ensure sibling modules (_hermes_home, google_account) are importable when
# run standalone.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from _hermes_home import display_hermes_home, get_hermes_home
import google_account

HERMES_HOME = get_hermes_home()
# CLIENT_SECRET is shared across accounts — same OAuth client can authorize
# multiple Google accounts. PENDING_AUTH is per-session (separate state for
# concurrent OAuth flows).
CLIENT_SECRET_PATH = HERMES_HOME / "google_client_secret.json"

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

REQUIRED_PACKAGES = ["google-api-python-client", "google-auth-oauthlib", "google-auth-httplib2"]

# OAuth redirect for "out of band" manual code copy flow.
# Google deprecated OOB, so we use a localhost redirect and tell the user to
# copy the code from the browser's URL bar (or the page body).
REDIRECT_URI = "http://localhost:1"


def _resolved_token_path(account: str | None) -> Path:
    """Token path for the resolved account.

    When ``account`` is None, this falls back to ``HERMES_GOOGLE_ACCOUNT`` →
    default pointer → legacy single-account path. See google_account module
    for full details.
    """
    return google_account.resolve_token_path(account)


def _pending_auth_path(account: str | None) -> Path:
    """Per-account pending OAuth state. We key it on the resolved account so
    two concurrent ``--auth-url`` flows for different accounts don't stomp
    on each other. For the legacy unscoped flow we keep the original
    filename."""
    if account is None:
        return HERMES_HOME / "google_oauth_pending.json"
    norm = google_account.normalize_email(account)
    return HERMES_HOME / "google_tokens" / f".pending-{norm}.json"


def _normalize_authorized_user_payload(payload: dict) -> dict:
    normalized = dict(payload)
    if not normalized.get("type"):
        normalized["type"] = "authorized_user"
    return normalized


def _load_token_payload(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
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


def check_auth_live(account: str | None = None):
    """Check auth with a real API call to detect disabled_client/account issues."""
    # quiet=True suppresses the "AUTHENTICATED" print from check_auth so the
    # final status line reflects the live-call outcome (OK or FAILED).
    if not check_auth(account, quiet=True):
        return False
    try:
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials
        token_path = _resolved_token_path(account)
        creds = Credentials.from_authorized_user_file(str(token_path))
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


def check_auth(account: str | None = None, quiet: bool = False):
    """Check if stored credentials are valid. Prints status, exits 0 or 1."""
    token_path = _resolved_token_path(account)
    if not token_path.exists():
        print(f"NOT_AUTHENTICATED: No token at {token_path}")
        return False

    _ensure_deps()
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    try:
        # Don't pass scopes — user may have authorized only a subset.
        # Passing scopes forces google-auth to validate them on refresh,
        # which fails with invalid_scope if the token has fewer scopes
        # than requested.
        creds = Credentials.from_authorized_user_file(str(token_path))
    except Exception as e:
        print(f"TOKEN_CORRUPT: {e}")
        return False

    payload = _load_token_payload(token_path)
    if creds.valid:
        missing_scopes = _missing_scopes_from_payload(payload)
        if missing_scopes:
            print(f"AUTHENTICATED (partial): Token valid but missing {len(missing_scopes)} scopes:")
            for s in missing_scopes:
                print(f"  - {s}")
        if not quiet:
            print(f"AUTHENTICATED: Token valid at {token_path}")
        return True

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            token_path.write_text(
                json.dumps(
                    _normalize_authorized_user_payload(json.loads(creds.to_json())),
                    indent=2,
                )
            )
            missing_scopes = _missing_scopes_from_payload(_load_token_payload(token_path))
            if missing_scopes:
                print(f"AUTHENTICATED (partial): Token refreshed but missing {len(missing_scopes)} scopes:")
                for s in missing_scopes:
                    print(f"  - {s}")
            if not quiet:
                print(f"AUTHENTICATED: Token refreshed at {token_path}")
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


def _save_pending_auth(*, account: str | None, state: str, code_verifier: str):
    """Persist the OAuth session bits needed for a later token exchange."""
    pending_path = _pending_auth_path(account)
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_text(
        json.dumps(
            {
                "state": state,
                "code_verifier": code_verifier,
                "redirect_uri": REDIRECT_URI,
                "account": google_account.normalize_email(account) if account else None,
            },
            indent=2,
        )
    )


def _load_pending_auth(account: str | None) -> dict:
    """Load the pending OAuth session created by get_auth_url()."""
    pending_path = _pending_auth_path(account)
    if not pending_path.exists():
        print("ERROR: No pending OAuth session found. Run --auth-url first.")
        sys.exit(1)

    try:
        data = json.loads(pending_path.read_text())
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


def get_auth_url(account: str | None = None):
    """Print the OAuth authorization URL. User visits this in a browser.

    When ``account`` is provided, we hint Google to pre-select that account
    via ``login_hint``. This means the consent screen suggests the right
    account but the user still controls which one they ultimately authorize
    (you can't force a specific account without admin policy)."""
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
    auth_kwargs = {
        "access_type": "offline",
        "prompt": "consent",
    }
    if account:
        # login_hint = pre-fill the email picker. User can override.
        auth_kwargs["login_hint"] = google_account.normalize_email(account)
    auth_url, state = flow.authorization_url(**auth_kwargs)
    _save_pending_auth(account=account, state=state, code_verifier=flow.code_verifier)
    # Print just the URL so the agent can extract it cleanly
    print(auth_url)


def exchange_auth_code(code: str, account: str | None = None):
    """Exchange the authorization code for a token and save it."""
    if not CLIENT_SECRET_PATH.exists():
        print("ERROR: No client secret stored. Run --client-secret first.")
        sys.exit(1)

    pending_auth = _load_pending_auth(account)
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

    # Determine where to save:
    #  * If --account was passed, derive the email from the token and refuse
    #    if it doesn't match (authorization went to a different account than
    #    the user said they wanted).
    #  * Otherwise, derive the email and store per-account; only fall back to
    #    the legacy file if derivation fails entirely.
    derived = google_account.derive_email_from_token(token_payload)
    requested = google_account.normalize_email(account) if account else None

    if requested and derived and requested != derived:
        print(
            f"ERROR: --account was {requested} but the OAuth flow authorized {derived}."
        )
        print("  The user picked a different Google account in the consent screen.")
        print(f"  Re-run --account {derived} --auth-url, or pick the right account in the browser.")
        sys.exit(1)

    final_account = requested or derived

    # Embed the email on the token payload (helps downstream tooling & makes
    # accounts self-identifying without a network call).
    if final_account:
        token_payload["email"] = final_account

    if final_account:
        target = google_account.store_token_for_account(final_account, token_payload)
        # If this is the first account, mark it as the default so backward-compat
        # readers (legacy google_token.json symlink) just work.
        if google_account.get_default_account() is None:
            try:
                google_account.set_default_account(final_account)
            except Exception:
                pass
        print(f"OK: Authenticated as {final_account}. Token saved to {target}")
    else:
        # Couldn't determine the email and no --account was specified — fall
        # back to the legacy single-account file so first-time setups still work.
        legacy = google_account.legacy_token_path()
        legacy.write_text(json.dumps(token_payload, indent=2))
        print("WARNING: Could not determine the Google account email for this token.")
        print(f"OK: Authenticated. Token saved to {legacy}")
        print("Run --migrate-legacy after setup to move it into the multi-account layout.")

    _pending_auth_path(account).unlink(missing_ok=True)
    print(f"Profile-scoped token location: {display_hermes_home()}/google_tokens/")


def revoke(account: str | None = None):
    """Revoke stored token and delete it.

    When ``--account EMAIL`` is given we revoke that account's token only.
    Without it we revoke whichever token resolves via the precedence chain
    (env → default → legacy) — i.e. the one a plain ``setup.py --check``
    would use.
    """
    token_path = _resolved_token_path(account)
    if not token_path.exists():
        print("No token to revoke.")
        return

    _ensure_deps()
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    try:
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())

        import urllib.request
        urllib.request.urlopen(
            urllib.request.Request(
                "https://oauth2.googleapis.com/revoke?token=" + creds.token,
                method="POST",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            ),
            timeout=15,
        )
        print("Token revoked with Google.")
    except Exception as e:
        print(f"Remote revocation failed (token may already be invalid): {e}")

    # If we're deleting the legacy file directly (un-migrated install), also
    # remove the legacy pending session file.
    if token_path == google_account.legacy_token_path():
        (HERMES_HOME / "google_oauth_pending.json").unlink(missing_ok=True)

    # If this token was the default-account target, refresh the legacy
    # symlink so we don't leave a broken link behind.
    legacy = google_account.legacy_token_path()
    legacy_resolves_to_us = (
        legacy.is_symlink()
        and Path(os.readlink(legacy)) == token_path
    )

    token_path.unlink(missing_ok=True)
    print(f"Deleted {token_path}")

    if legacy_resolves_to_us:
        legacy.unlink(missing_ok=True)
        print(f"Removed legacy symlink {legacy}")
        # Pick a new default if any accounts remain.
        remaining = google_account.list_accounts()
        if remaining:
            try:
                google_account.set_default_account(remaining[0])
                print(f"Set new default account: {remaining[0]}")
            except Exception:
                pass


def list_accounts_cmd():
    """Print known accounts and the default."""
    accounts = google_account.list_accounts()
    default = google_account.get_default_account()
    legacy = google_account.legacy_token_path()
    legacy_unmigrated = legacy.exists() and not legacy.is_symlink()

    if not accounts and not legacy_unmigrated:
        print("No Google accounts configured.")
        print("Run --client-secret then --auth-url to set up your first account.")
        return

    print("Configured accounts:")
    for email in accounts:
        marker = "  * " if email == default else "    "
        print(f"{marker}{email}")
    if default:
        print(f"\nDefault: {default}")
    if legacy_unmigrated:
        print(
            "\nUnmigrated legacy token at "
            f"{legacy}.\n"
            "Run --migrate-legacy to move it into the multi-account layout."
        )


def set_default_cmd(email: str):
    """Mark ``email`` as the default account."""
    try:
        google_account.set_default_account(email)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    norm = google_account.normalize_email(email)
    print(f"OK: Default account set to {norm}.")
    print(f"Legacy symlink {google_account.legacy_token_path()} now points at this account.")


def migrate_legacy_cmd():
    """Move the legacy single-account token into the multi-account layout."""
    legacy = google_account.legacy_token_path()
    if legacy.is_symlink():
        target = legacy.resolve()
        print(f"Already migrated: {legacy} -> {target}")
        return
    if not legacy.exists():
        print("Nothing to migrate (no legacy google_token.json).")
        return

    payload = _load_token_payload(legacy)
    if not payload:
        print("ERROR: Legacy token file is unreadable. Re-run setup from scratch.")
        sys.exit(1)

    email = google_account.derive_email_from_token(payload)
    if not email:
        print(
            "ERROR: Could not determine which Google account this legacy token belongs to.\n"
            "  Network call to Google's tokeninfo/userinfo endpoints failed.\n"
            "  Either re-authenticate with --auth-url, or check your network and retry."
        )
        sys.exit(1)

    payload["email"] = email
    target = google_account.store_token_for_account(email, payload)
    legacy.unlink()
    google_account.set_default_account(email)  # rebuilds the symlink
    print(f"OK: Migrated legacy token -> {target}")
    print(f"Default account set to {email}.")
    print(f"Legacy symlink {legacy} now points at the new location.")


def remove_account_cmd(email: str):
    """Forget an account: revoke its token and delete the per-account file."""
    norm = google_account.normalize_email(email)
    target = google_account.token_path_for(norm)
    if not target.exists():
        print(f"No token for {norm}.")
        return
    revoke(norm)


def main():
    parser = argparse.ArgumentParser(description="Google Workspace OAuth setup for Hermes")
    parser.add_argument(
        "--account",
        metavar="EMAIL",
        default=None,
        help="Operate on a specific Google account (multi-account setups). "
        "If omitted, falls back to HERMES_GOOGLE_ACCOUNT env, then the "
        "default-account pointer, then the legacy single-account token.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Check if auth is valid (exit 0=yes, 1=no)")
    group.add_argument("--check-live", action="store_true", help="Check auth with a real API call (detects disabled_client)")
    group.add_argument("--client-secret", metavar="PATH", help="Store OAuth client_secret.json")
    group.add_argument("--auth-url", action="store_true", help="Print OAuth URL for user to visit")
    group.add_argument("--auth-code", metavar="CODE", help="Exchange auth code for token")
    group.add_argument("--revoke", action="store_true", help="Revoke and delete stored token")
    group.add_argument("--install-deps", action="store_true", help="Install Python dependencies")
    group.add_argument("--list-accounts", action="store_true", help="List configured Google accounts and the default")
    group.add_argument("--set-default", metavar="EMAIL", help="Set the default Google account")
    group.add_argument("--migrate-legacy", action="store_true", help="Migrate legacy single-account token into the multi-account layout")
    group.add_argument("--remove-account", metavar="EMAIL", help="Revoke and forget an account")
    args = parser.parse_args()

    # Validate --account early so weird strings fail fast.
    account = args.account
    if account is not None:
        try:
            account = google_account.normalize_email(account)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    if args.check:
        sys.exit(0 if check_auth(account) else 1)
    if getattr(args, "check_live", False):
        sys.exit(0 if check_auth_live(account) else 1)
    elif args.client_secret:
        store_client_secret(args.client_secret)
    elif args.auth_url:
        get_auth_url(account)
    elif args.auth_code:
        exchange_auth_code(args.auth_code, account)
    elif args.revoke:
        revoke(account)
    elif args.install_deps:
        sys.exit(0 if install_deps() else 1)
    elif args.list_accounts:
        list_accounts_cmd()
    elif args.set_default:
        set_default_cmd(args.set_default)
    elif args.migrate_legacy:
        migrate_legacy_cmd()
    elif args.remove_account:
        remove_account_cmd(args.remove_account)


if __name__ == "__main__":
    main()

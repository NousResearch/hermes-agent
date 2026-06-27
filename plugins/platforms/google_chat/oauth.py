"""User OAuth helper for the Google Chat gateway adapter.

Google Chat's ``media.upload`` REST endpoint hard-rejects service-account
authentication:

    "This method doesn't support app authentication with a service
     account. Authenticate with a user account."

(See https://developers.google.com/workspace/chat/api/reference/rest/v1/media/upload
and https://developers.google.com/chat/api/guides/auth/users.)

For the bot to deliver native file attachments — the same drag-and-drop
file widget the user gets when they upload manually — each user must
grant the bot the ``chat.messages.create`` scope ONCE in their own DM.
The bot stores per-user refresh tokens and calls ``media.upload`` plus
the subsequent ``messages.create`` *as the requesting user* whenever a
file needs sending.

This module is BOTH a CLI tool (driven by the agent via slash commands or
terminal commands) AND a library imported by ``google_chat.py``:

    Library functions (called from the adapter at runtime):
        load_user_credentials(email=None) -> Credentials | None
        refresh_or_none(creds, email=None) -> Credentials | None
        build_user_chat_service(creds) -> chat_v1.Resource
        list_authorized_emails() -> List[str]

    CLI commands (driven by the agent through the /setup-files slash
    command, modeled on skills/productivity/google-workspace/scripts/setup.py):
        --check                          Exit 0 if auth is valid, else 1
        --client-secret /path/to.json    Persist OAuth client credentials
        --auth-url                       Print the OAuth URL for the user
        --auth-code CODE                 Exchange auth code for token
        --revoke                         Revoke and delete stored token
        --install-deps                   Install Python dependencies
        --email EMAIL                    Scope CLI ops to a specific user
                                         (defaults to legacy single-user
                                         mode when omitted)

The flow mirrors the existing google-workspace skill exactly so anyone
familiar with that flow can read this without surprises.

Token storage layout
--------------------
- Per-user tokens (keyed by sender email):
    ``${HERMES_HOME}/google_chat_user_tokens/<sanitized_email>.json``
- Legacy single-user token (fallback, untouched for backward compat):
    ``${HERMES_HOME}/google_chat_user_token.json``
- Per-user pending OAuth state during /setup-files start → exchange:
    ``${HERMES_HOME}/google_chat_user_oauth_pending/<sanitized_email>.json``
- Legacy pending state:
    ``${HERMES_HOME}/google_chat_user_oauth_pending.json``
- Shared OAuth client (one per host):
    ``${HERMES_HOME}/google_chat_user_client_secret.json``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Pin the legacy logger name so operator-side log filters keep matching
# after the in-tree → plugin migration. See adapter.py for context.
logger = logging.getLogger("gateway.platforms.google_chat_user_oauth")

# Use the project's HERMES_HOME helper so the token follows the user's
# profile (e.g. tests can override via HERMES_HOME=/tmp/...).
try:
    from hermes_constants import display_hermes_home, get_hermes_home
except (ModuleNotFoundError, ImportError):
    # Fallback for environments where hermes_constants isn't importable
    # (mirrors the same fallback used by the google-workspace skill's
    # _hermes_home.py shim).
    def get_hermes_home() -> Path:
        val = os.environ.get("HERMES_HOME", "").strip()
        return Path(val) if val else Path.home() / ".hermes"

    def display_hermes_home() -> str:
        home = get_hermes_home()
        try:
            return "~/" + str(home.relative_to(Path.home()))
        except ValueError:
            return str(home)


def _hermes_home() -> Path:
    """Resolve HERMES_HOME at call time (NOT module import).

    Tests and ``HERMES_HOME=...`` env overrides need this to be late-
    binding. If we cached the path at import time, switching profiles
    or tweaking env vars in tests would silently keep using the old
    path."""
    return get_hermes_home()


# Filesystem-safe key: lowercase, allow ``[a-z0-9._-@]``, replace anything
# else with ``_``. ``ramon.fernandez@nttdata.com`` stays human-readable
# (``ramon.fernandez@nttdata.com.json``) which makes admin debugging by
# ``ls ~/.hermes/google_chat_user_tokens/`` trivial.
_EMAIL_FS_RE = re.compile(r"[^a-z0-9._@-]+")


def _sanitize_email(email: str) -> str:
    cleaned = _EMAIL_FS_RE.sub("_", (email or "").strip().lower())
    return cleaned or "_unknown_"


def _legacy_token_path() -> Path:
    return _hermes_home() / "google_chat_user_token.json"


def _user_tokens_dir() -> Path:
    return _hermes_home() / "google_chat_user_tokens"


def _legacy_pending_path() -> Path:
    return _hermes_home() / "google_chat_user_oauth_pending.json"


def _user_pending_dir() -> Path:
    return _hermes_home() / "google_chat_user_oauth_pending"


def _token_path(email: Optional[str] = None) -> Path:
    """Return the on-disk token path for ``email`` or the legacy path."""
    if email:
        return _user_tokens_dir() / f"{_sanitize_email(email)}.json"
    return _legacy_token_path()


def _client_secret_path() -> Path:
    return _hermes_home() / "google_chat_user_client_secret.json"


def _pending_auth_path(email: Optional[str] = None) -> Path:
    if email:
        return _user_pending_dir() / f"{_sanitize_email(email)}.json"
    return _legacy_pending_path()


# Minimum scope for native Chat attachment delivery.
# `chat.messages.create` covers BOTH `media.upload` and the subsequent
# `messages.create` that references the attachmentDataRef. We deliberately
# do NOT request drive.file or other scopes — least privilege.
SCOPES: List[str] = [
    "https://www.googleapis.com/auth/chat.messages.create",
]

# Pip packages required for the OAuth flow.
_REQUIRED_PACKAGES = [
    "google-api-python-client",
    "google-auth-oauthlib",
    "google-auth-httplib2",
]

# Out-of-band redirect: Google deprecated the ``urn:ietf:wg:oauth:2.0:oob``
# flow, so we use a localhost redirect that's expected to FAIL. The user
# copies the auth code from the failed browser URL bar back into chat.
# Same trick used by skills/productivity/google-workspace/scripts/setup.py.
_REDIRECT_URI = "http://localhost:1"


# =============================================================================
# Library API — called from the adapter at runtime
# =============================================================================


def load_user_credentials(email: Optional[str] = None) -> Optional[Any]:
    """Load + validate persisted user OAuth credentials.

    ``email`` selects the per-user token file; ``None`` falls back to the
    legacy single-user path (left in place for installs that ran the
    pre-multi-user flow). Returns a ``google.oauth2.credentials.Credentials``
    instance ready for use, or ``None`` if no token is stored, the token
    is corrupt, or refresh fails. Adapter callers should treat ``None``
    as "user has not run /setup-files yet" and surface the setup-instructions
    fallback to the user.

    Does NOT raise on the no-token case — that's expected.
    """
    token_path = _token_path(email)
    if not token_path.exists():
        return None

    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
    except ImportError:
        logger.warning(
            "[google_chat_user_oauth] google-auth not installed; user-OAuth "
            "attachment delivery is disabled. Install hermes-agent[google_chat]."
        )
        return None

    try:
        # Don't pass scopes — user may have authorized only a subset, and
        # passing scopes makes refresh validate them strictly. Same logic
        # as the google-workspace skill.
        creds = Credentials.from_authorized_user_file(str(token_path))
    except Exception as exc:
        logger.warning(
            "[google_chat_user_oauth] token at %s is corrupt: %s",
            token_path, exc,
        )
        return None

    if creds.valid:
        return creds

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception as exc:
            logger.warning(
                "[google_chat_user_oauth] token refresh failed (user "
                "should re-run /setup-files): %s", exc,
            )
            return None
        # Persist refreshed token so next start picks up the new access
        # token without an unnecessary refresh round-trip.
        _persist_credentials(creds, token_path)
        return creds

    # Token exists but is unusable (e.g. revoked, no refresh token).
    return None


def refresh_or_none(creds: Any, email: Optional[str] = None) -> Optional[Any]:
    """Refresh ``creds`` if expired. Returns the credentials or ``None``.

    Used by the adapter just before calling media.upload to ensure the
    token is current. Returns ``None`` if refresh fails — caller falls
    back to the text-notice path. ``email`` controls where the refreshed
    token is written back; ``None`` keeps the legacy single-file path.
    """
    if creds is None:
        return None

    if creds.valid:
        return creds

    try:
        from google.auth.transport.requests import Request
    except ImportError:
        return None

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _persist_credentials(creds, _token_path(email))
            return creds
        except Exception as exc:
            logger.warning(
                "[google_chat_user_oauth] refresh failed: %s", exc,
            )
            return None

    return None


def build_user_chat_service(creds: Any) -> Any:
    """Build a Google Chat API client authenticated as the user.

    Used for media.upload + the subsequent messages.create that
    references the attachmentDataRef. The bot's separate SA-authed
    client (``self._chat_api`` in the adapter) is for everything else.
    """
    from googleapiclient.discovery import build as build_service
    return build_service("chat", "v1", credentials=creds, cache_discovery=False)


def list_authorized_emails() -> List[str]:
    """Return the set of user emails that have stored per-user tokens.

    Lists files in the per-user tokens dir; does NOT include the legacy
    single-user token (its owner is unknown). Sanitized filenames lose
    the ``+suffix`` part of plus-addressed emails — accept that and use
    this list only for admin display, not for trust decisions.
    """
    d = _user_tokens_dir()
    if not d.exists():
        return []
    out: List[str] = []
    for f in d.iterdir():
        if f.is_file() and f.suffix == ".json":
            out.append(f.stem)
    out.sort()
    return out


def _persist_credentials(creds: Any, token_path: Path) -> None:
    """Atomic-ish JSON write of refreshed credentials."""
    try:
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(
            json.dumps(
                _normalize_authorized_user_payload(json.loads(creds.to_json())),
                indent=2,
            )
        )
    except Exception:
        logger.debug(
            "[google_chat_user_oauth] failed to persist credentials at %s",
            token_path, exc_info=True,
        )


# =============================================================================
# CLI commands — driven by the agent via /setup-files
# =============================================================================


def _normalize_authorized_user_payload(payload: dict) -> dict:
    """Ensure the persisted token JSON has the type field google-auth expects."""
    normalized = dict(payload)
    if not normalized.get("type"):
        normalized["type"] = "authorized_user"
    return normalized


def _ensure_deps() -> None:
    """Check deps available; install if not; exit on failure."""
    try:
        import googleapiclient  # noqa: F401
        import google_auth_oauthlib  # noqa: F401
    except ImportError:
        if not install_deps():
            sys.exit(1)


def install_deps() -> bool:
    try:
        import googleapiclient  # noqa: F401
        import google_auth_oauthlib  # noqa: F401
        print("Dependencies already installed.")
        return True
    except ImportError:
        pass

    print("Installing Google Chat OAuth dependencies...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + _REQUIRED_PACKAGES,
            stdout=subprocess.DEVNULL,
        )
        print("Dependencies installed.")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: Failed to install dependencies: {exc}")
        print("Or install via the optional extra:")
        print("  pip install 'hermes-agent[google_chat]'")
        return False


def check_auth(email: Optional[str] = None) -> bool:
    """Print status; return True if creds are usable.

    Per-user when ``email`` given, legacy single-user when omitted.
    """
    token_path = _token_path(email)
    if not token_path.exists():
        print(f"NOT_AUTHENTICATED: No token at {token_path}")
        return False

    creds = load_user_credentials(email)
    if creds is None:
        print(f"TOKEN_INVALID: Re-run /setup-files (path: {token_path})")
        return False

    print(f"AUTHENTICATED: Token valid at {token_path}")
    return True


def store_client_secret(path: str) -> None:
    """Validate and copy the user's OAuth client_secret.json into HERMES_HOME."""
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
        print(
            "ERROR: Not a Google OAuth client secret file (missing "
            "'installed' or 'web' key)."
        )
        print(
            "Download from: https://console.cloud.google.com/apis/credentials"
        )
        sys.exit(1)

    target = _client_secret_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2))
    print(f"OK: Client secret saved to {target}")


def _save_pending_auth(*, state: str, code_verifier: str,
                      email: Optional[str] = None) -> None:
    pending = _pending_auth_path(email)
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text(
        json.dumps(
            {
                "state": state,
                "code_verifier": code_verifier,
                "redirect_uri": _REDIRECT_URI,
                "email": email or "",
            },
            indent=2,
        )
    )


def _load_pending_auth(email: Optional[str] = None) -> dict:
    pending = _pending_auth_path(email)
    if not pending.exists():
        print("ERROR: No pending OAuth session found. Run --auth-url first.")
        sys.exit(1)
    try:
        data = json.loads(pending.read_text())
    except Exception as exc:
        print(f"ERROR: Could not read pending OAuth session: {exc}")
        print("Run --auth-url again to start a fresh session.")
        sys.exit(1)
    if not data.get("state") or not data.get("code_verifier"):
        print("ERROR: Pending OAuth session is missing PKCE data.")
        print("Run --auth-url again.")
        sys.exit(1)
    return data


def _extract_code_and_state(code_or_url: str) -> Tuple[str, Optional[str]]:
    """Accept a raw auth code OR the full failed-redirect URL the user pastes."""
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


def get_auth_url(email: Optional[str] = None) -> None:
    """Print the OAuth URL for the user to visit. Persists PKCE state.

    ``email`` namespaces the pending state so two users can be mid-flow
    in parallel without trampling each other's PKCE verifier.
    """
    if not _client_secret_path().exists():
        print("ERROR: No client secret stored. Run --client-secret first.")
        sys.exit(1)

    _ensure_deps()
    from google_auth_oauthlib.flow import Flow

    flow = Flow.from_client_secrets_file(
        str(_client_secret_path()),
        scopes=SCOPES,
        redirect_uri=_REDIRECT_URI,
        autogenerate_code_verifier=True,
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
    )
    _save_pending_auth(state=state, code_verifier=flow.code_verifier, email=email)
    print(auth_url)


def exchange_auth_code(code: str, email: Optional[str] = None) -> None:
    """Exchange an auth code (or pasted redirect URL) for a refresh token.

    ``email`` selects the destination token path. ``None`` writes to the
    legacy single-user path (kept for the existing CLI entrypoint and for
    pre-multi-user installs).
    """
    if not _client_secret_path().exists():
        print("ERROR: No client secret stored. Run --client-secret first.")
        sys.exit(1)

    pending_auth = _load_pending_auth(email)
    raw_callback = code
    code, returned_state = _extract_code_and_state(code)
    if returned_state and returned_state != pending_auth["state"]:
        print(
            "ERROR: OAuth state mismatch. Run --auth-url again to start a "
            "fresh session."
        )
        sys.exit(1)

    _ensure_deps()
    from google_auth_oauthlib.flow import Flow
    from urllib.parse import parse_qs, urlparse

    granted_scopes = list(SCOPES)
    if isinstance(raw_callback, str) and raw_callback.startswith("http"):
        params = parse_qs(urlparse(raw_callback).query)
        scope_val = (params.get("scope") or [""])[0].strip()
        if scope_val:
            granted_scopes = scope_val.split()

    flow = Flow.from_client_secrets_file(
        str(_client_secret_path()),
        scopes=granted_scopes,
        redirect_uri=pending_auth.get("redirect_uri", _REDIRECT_URI),
        state=pending_auth["state"],
        code_verifier=pending_auth["code_verifier"],
    )

    try:
        # Accept partial scopes — user may deselect items in the consent screen.
        os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
        flow.fetch_token(code=code)
    except Exception as exc:
        print(f"ERROR: Token exchange failed: {exc}")
        print("The code may have expired. Run --auth-url to get a fresh URL.")
        sys.exit(1)

    creds = flow.credentials
    token_payload = _normalize_authorized_user_payload(json.loads(creds.to_json()))

    actually_granted = (
        list(creds.granted_scopes or [])
        if hasattr(creds, "granted_scopes") and creds.granted_scopes
        else []
    )
    if actually_granted:
        token_payload["scopes"] = actually_granted
    elif granted_scopes != SCOPES:
        token_payload["scopes"] = granted_scopes

    token_path = _token_path(email)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(json.dumps(token_payload, indent=2))
    _pending_auth_path(email).unlink(missing_ok=True)

    print(f"OK: Authenticated. Token saved to {token_path}")
    rel_label = (
        f"{display_hermes_home()}/google_chat_user_tokens/{_sanitize_email(email)}.json"
        if email
        else f"{display_hermes_home()}/google_chat_user_token.json"
    )
    print(f"Profile path: {rel_label}")


def revoke(email: Optional[str] = None) -> None:
    """Revoke the stored token with Google and delete it locally.

    Per-user when ``email`` given, legacy single-user when omitted.
    """
    token_path = _token_path(email)
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
                f"https://oauth2.googleapis.com/revoke?token={creds.token}",
                method="POST",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        )
        print("Token revoked with Google.")
    except Exception as exc:
        print(f"Remote revocation failed (token may already be invalid): {exc}")

    token_path.unlink(missing_ok=True)
    _pending_auth_path(email).unlink(missing_ok=True)
    print(f"Deleted {token_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Google Chat user-OAuth setup for Hermes (native attachment delivery)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true",
                       help="Check if auth is valid (exit 0=yes, 1=no)")
    group.add_argument("--client-secret", metavar="PATH",
                       help="Store OAuth client_secret.json")
    group.add_argument("--auth-url", action="store_true",
                       help="Print OAuth URL for user to visit")
    group.add_argument("--auth-code", metavar="CODE",
                       help="Exchange auth code for token")
    group.add_argument("--revoke", action="store_true",
                       help="Revoke and delete stored token")
    group.add_argument("--install-deps", action="store_true",
                       help="Install Python dependencies")
    parser.add_argument("--email", metavar="EMAIL", default=None,
                       help="Scope operation to a specific user's token "
                            "(default: legacy single-user path)")
    args = parser.parse_args()

    email = args.email or None
    if args.check:
        sys.exit(0 if check_auth(email) else 1)
    elif args.client_secret:
        store_client_secret(args.client_secret)
    elif args.auth_url:
        get_auth_url(email)
    elif args.auth_code:
        exchange_auth_code(args.auth_code, email)
    elif args.revoke:
        revoke(email)
    elif args.install_deps:
        sys.exit(0 if install_deps() else 1)


if __name__ == "__main__":
    main()

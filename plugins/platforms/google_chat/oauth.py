"""User OAuth helper for the Google Chat gateway adapter.

Google Chat's ``media.upload`` rejects service-account auth, so for native file
attachments each user grants the bot ``chat.messages.create`` ONCE in their own DM;
the bot stores per-user refresh tokens and uploads *as the user*
(https://developers.google.com/chat/api/guides/auth/users). Library API for the
adapter plus a CLI driven by ``/setup-files`` (``--help``; ``--email`` omitted ==
legacy single-user mode). Files under ``${HERMES_HOME}``: ``google_chat_user_tokens/
<email>.json`` (per-user) / ``google_chat_user_token.json`` (legacy); pending PKCE state
in ``google_chat_user_oauth_pending[/<email>].json``; ``google_chat_user_client_secret.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import stat
import sys
from importlib.metadata import version as _distribution_version
from pathlib import Path
from typing import Any, List, NoReturn, Optional, Tuple

from packaging.requirements import Requirement

from hermes_constants import display_hermes_home, get_hermes_home
from utils import atomic_write_text

# Pinned legacy logger name so operator log filters keep matching (see adapter.py).
logger = logging.getLogger("gateway.platforms.google_chat_user_oauth")

# Filesystem-safe key: lowercase, keep ``[a-z0-9._-@]`` so token files stay
# human-readable under ``ls ~/.hermes/google_chat_user_tokens/``.
_EMAIL_FS_RE = re.compile(r"[^a-z0-9._@-]+")

# Least privilege: chat.messages.create covers BOTH media.upload and the
# subsequent messages.create; no drive.file or other scopes.
SCOPES: List[str] = ["https://www.googleapis.com/auth/chat.messages.create"]

# Pip packages required by the Google Chat adapter and its OAuth flow.
_REQUIRED_PACKAGES = [
    "google-cloud-pubsub==2.39.0",
    "google-api-python-client==2.194.0",
    "google-auth==2.55.1",
    "google-auth-oauthlib==1.3.1",
    "google-auth-httplib2==0.3.1",
    "httplib2==0.32.0",
    "pyasn1==0.6.4",
]

# Google deprecated the ``oob`` flow: use a localhost redirect that is expected
# to FAIL; the user pastes the code from the failed browser URL back into chat.
_REDIRECT_URI = "http://localhost:1"


def _sanitize_email(email: str) -> str:
    cleaned = _EMAIL_FS_RE.sub("_", (email or "").strip().lower())
    return cleaned or "_unknown_"


def _token_rel(email: Optional[str]) -> str:
    """HERMES_HOME-relative token file: per-user under the tokens dir, else the legacy path."""
    return f"google_chat_user_tokens/{_sanitize_email(email)}.json" if email else "google_chat_user_token.json"


def _user_tokens_dir() -> Path:
    return get_hermes_home() / "google_chat_user_tokens"


def _token_path(email: Optional[str] = None) -> Path:
    """Per-user token path for ``email``, or the legacy single-user path."""
    return get_hermes_home() / _token_rel(email)


def _client_secret_path() -> Path:
    return get_hermes_home() / "google_chat_user_client_secret.json"


def _pending_auth_path(email: Optional[str] = None) -> Path:
    if email:
        return get_hermes_home() / "google_chat_user_oauth_pending" / f"{_sanitize_email(email)}.json"
    return get_hermes_home() / "google_chat_user_oauth_pending.json"


# -- Library API — called from the adapter at runtime -------------------------


def _refresh_and_persist(creds: Any, token_path: Path, request_cls: Any, *, failure_msg: str) -> Optional[Any]:
    """Refresh expired creds and write them back; None when unusable or refresh fails."""
    if creds.valid:
        return creds
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(request_cls())
        except Exception as exc:
            logger.warning("[google_chat_user_oauth] %s: %s", failure_msg, exc)
            return None
        _persist_credentials(creds, token_path)
        return creds
    # Token exists but is unusable (e.g. revoked, no refresh token).
    return None


def load_user_credentials(email: Optional[str] = None) -> Optional[Any]:
    """Load + validate persisted user OAuth credentials.

    ``None`` email → legacy single-user path. Returns ``None`` (never raises) when
    no token is stored, the token is corrupt, or refresh fails — callers treat
    that as "user has not run /setup-files yet".
    """
    token_path = _token_path(email)
    if not token_path.exists():
        return None
    # Hand-provisioned / legacy token files commonly end up 0o644; warn the owner.
    from utils import warn_if_credential_file_broadly_readable

    warn_if_credential_file_broadly_readable(token_path, label="[google_chat_user_oauth]", log=logger)
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
    except ImportError:
        logger.warning(
            "[google_chat_user_oauth] google-auth not installed; user-OAuth "
            "attachment delivery is disabled. Run `hermes setup` to install Google Chat support."
        )
        return None
    try:
        # No scopes: the user may have authorized a subset, and passing scopes
        # makes refresh validate them strictly.
        creds = Credentials.from_authorized_user_file(str(token_path))
    except Exception as exc:
        logger.warning("[google_chat_user_oauth] token at %s is corrupt: %s", token_path, exc)
        return None
    return _refresh_and_persist(
        creds, token_path, Request, failure_msg="token refresh failed (user should re-run /setup-files)",
    )


def refresh_or_none(creds: Any, email: Optional[str] = None) -> Optional[Any]:
    """Refresh ``creds`` if expired; ``None`` on failure (caller falls back to the
    text-notice path). ``email`` selects where the refreshed token is written."""
    if creds is None:
        return None
    if creds.valid:
        return creds
    try:
        from google.auth.transport.requests import Request
    except ImportError:
        return None
    return _refresh_and_persist(creds, _token_path(email), Request, failure_msg="refresh failed")


def build_user_chat_service(creds: Any) -> Any:
    """Chat API client authenticated as the user (for media.upload + messages.create)."""
    from googleapiclient.discovery import build as build_service
    return build_service("chat", "v1", credentials=creds, cache_discovery=False)


def list_authorized_emails() -> List[str]:
    """Sanitized emails with stored per-user tokens (admin display only, not trust;
    excludes the legacy single-user token whose owner is unknown)."""
    d = _user_tokens_dir()
    if not d.exists():
        return []
    return sorted(f.stem for f in d.iterdir() if f.is_file() and f.suffix == ".json")


def _persist_credentials(creds: Any, token_path: Path) -> None:
    """Persist refreshed credentials atomically with private permissions."""
    try:
        _write_private_json(token_path, _normalize_authorized_user_payload(json.loads(creds.to_json())))
    except Exception:
        logger.debug("[google_chat_user_oauth] failed to persist credentials at %s", token_path, exc_info=True)


# -- CLI commands — driven by the agent via /setup-files ----------------------


def _normalize_authorized_user_payload(payload: dict) -> dict:
    """Ensure the persisted token JSON has the type field google-auth expects."""
    normalized = dict(payload)
    if not normalized.get("type"):
        normalized["type"] = "authorized_user"
    return normalized


def _chmod_quiet(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _write_private_json(path: Path, data: Any) -> None:
    """Atomically write JSON with 0o600 permissions (0o700 parent) where supported."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _chmod_quiet(path.parent, 0o700)
    # mkstemp's 0o600 temp + atomic rename never exposes the token at process umask.
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False), create_mode=0o600)
    _chmod_quiet(path, stat.S_IRUSR | stat.S_IWUSR)


def _fail(*lines: str) -> NoReturn:
    """Print CLI error lines and exit 1."""
    for line in lines:
        print(line)
    sys.exit(1)


def _ensure_deps() -> None:
    """Check exact dependency versions; install if stale; exit on failure."""
    if _missing_required_packages() and not install_deps():
        sys.exit(1)


def _missing_required_packages() -> List[str]:
    """Return exact requirements absent or stale in this interpreter."""
    missing = []
    for spec in _REQUIRED_PACKAGES:
        requirement = Requirement(spec)
        try:
            installed = _distribution_version(requirement.name)
            satisfied = requirement.specifier.contains(installed, prereleases=True)
        except Exception:
            satisfied = False
        if not satisfied:
            missing.append(spec)
    return missing


def install_deps() -> bool:
    missing = _missing_required_packages()
    if not missing:
        print("Dependencies already installed.")
        return True
    print("Installing Google Chat dependencies...")
    try:
        from hermes_cli.tools_config import _pip_install

        result = _pip_install(["--quiet"] + missing)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or "install failed").strip()[:300])
        remaining = _missing_required_packages()
        if remaining:
            raise RuntimeError("dependencies remain stale after install: " + " ".join(remaining))
        print("Dependencies installed.")
        return True
    except Exception as exc:
        print(f"ERROR: Failed to install dependencies: {exc}")
        print("Run `hermes setup` to repair the managed installation, then retry.")
        return False


def check_auth(email: Optional[str] = None) -> bool:
    """Print status; return True if creds are usable."""
    token_path = _token_path(email)
    if not token_path.exists():
        print(f"NOT_AUTHENTICATED: No token at {token_path}")
        return False
    if load_user_credentials(email) is None:
        print(f"TOKEN_INVALID: Re-run /setup-files (path: {token_path})")
        return False
    print(f"AUTHENTICATED: Token valid at {token_path}")
    return True


def store_client_secret(path: str) -> None:
    """Validate and copy the user's OAuth client_secret.json into HERMES_HOME."""
    src = Path(path).expanduser().resolve()
    if not src.exists():
        _fail(f"ERROR: File not found: {src}")
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _fail("ERROR: File is not valid JSON.")
    if "installed" not in data and "web" not in data:
        _fail(
            "ERROR: Not a Google OAuth client secret file (missing 'installed' or 'web' key).",
            "Download from: https://console.cloud.google.com/apis/credentials")
    target = _client_secret_path()
    _write_private_json(target, data)
    print(f"OK: Client secret saved to {target}")


def _save_pending_auth(*, state: str, code_verifier: str, email: Optional[str] = None) -> None:
    _write_private_json(
        _pending_auth_path(email),
        {"state": state, "code_verifier": code_verifier, "redirect_uri": _REDIRECT_URI, "email": email or ""},
    )


def _load_pending_auth(email: Optional[str] = None) -> dict:
    pending = _pending_auth_path(email)
    if not pending.exists():
        _fail("ERROR: No pending OAuth session found. Run --auth-url first.")
    try:
        data = json.loads(pending.read_text(encoding="utf-8"))
    except Exception as exc:
        _fail(f"ERROR: Could not read pending OAuth session: {exc}", "Run --auth-url again to start a fresh session.")
    if not data.get("state") or not data.get("code_verifier"):
        _fail("ERROR: Pending OAuth session is missing PKCE data.", "Run --auth-url again.")
    return data


def _callback_params(code_or_url: str) -> Optional[dict]:
    """Query params of a pasted failed-redirect URL; ``None`` for a raw auth code."""
    if not code_or_url.startswith("http"):
        return None
    from urllib.parse import parse_qs, urlparse

    return parse_qs(urlparse(code_or_url).query)


def _extract_code_and_state(code_or_url: str) -> Tuple[str, Optional[str]]:
    """Accept a raw auth code OR the full failed-redirect URL the user pastes."""
    params = _callback_params(code_or_url)
    if params is None:
        return code_or_url, None
    if "code" not in params:
        _fail("ERROR: No 'code' parameter found in URL.")
    return params["code"][0], params.get("state", [None])[0]


def _require_client_secret() -> None:
    if not _client_secret_path().exists():
        _fail("ERROR: No client secret stored. Run --client-secret first.")


def get_auth_url(email: Optional[str] = None) -> None:
    """Print the OAuth URL for the user to visit; persists PKCE state under ``email``
    so two users can be mid-flow in parallel."""
    _require_client_secret()
    _ensure_deps()
    from google_auth_oauthlib.flow import Flow

    flow = Flow.from_client_secrets_file(
        str(_client_secret_path()), scopes=SCOPES, redirect_uri=_REDIRECT_URI, autogenerate_code_verifier=True,
    )
    auth_url, state = flow.authorization_url(access_type="offline", prompt="consent")
    _save_pending_auth(state=state, code_verifier=flow.code_verifier, email=email)
    print(auth_url)


def exchange_auth_code(code: str, email: Optional[str] = None) -> None:
    """Exchange an auth code (or pasted redirect URL) for a refresh token stored
    at the per-user path for ``email`` (legacy single-user path when None)."""
    _require_client_secret()
    pending_auth = _load_pending_auth(email)
    raw_callback = code
    code, returned_state = _extract_code_and_state(code)
    if returned_state and returned_state != pending_auth["state"]:
        _fail("ERROR: OAuth state mismatch. Run --auth-url again to start a fresh session.")
    _ensure_deps()
    from google_auth_oauthlib.flow import Flow

    granted_scopes = list(SCOPES)
    params = _callback_params(raw_callback) if isinstance(raw_callback, str) else None
    scope_val = (params.get("scope") or [""])[0].strip() if params is not None else ""
    if scope_val:
        granted_scopes = scope_val.split()
    flow = Flow.from_client_secrets_file(
        str(_client_secret_path()), scopes=granted_scopes,
        redirect_uri=pending_auth.get("redirect_uri", _REDIRECT_URI), state=pending_auth["state"],
        code_verifier=pending_auth["code_verifier"])
    try:
        # Accept partial scopes — user may deselect items in the consent screen.
        os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
        flow.fetch_token(code=code)
    except Exception as exc:
        _fail(f"ERROR: Token exchange failed: {exc}", "The code may have expired. Run --auth-url to get a fresh URL.")
    creds = flow.credentials
    token_payload = _normalize_authorized_user_payload(json.loads(creds.to_json()))
    actually_granted = list(getattr(creds, "granted_scopes", None) or [])
    if actually_granted:
        token_payload["scopes"] = actually_granted
    elif granted_scopes != SCOPES:
        token_payload["scopes"] = granted_scopes
    token_path = _token_path(email)
    _write_private_json(token_path, token_payload)
    _pending_auth_path(email).unlink(missing_ok=True)
    print(f"OK: Authenticated. Token saved to {token_path}")
    print(f"Profile path: {display_hermes_home()}/{_token_rel(email)}")


def revoke(email: Optional[str] = None) -> None:
    """Revoke the stored token with Google and delete it locally."""
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
                headers={"Content-Type": "application/x-www-form-urlencoded"}),
            timeout=15)
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
    group.add_argument("--check", action="store_true", help="Check if auth is valid (exit 0=yes, 1=no)")
    group.add_argument("--client-secret", metavar="PATH", help="Store OAuth client_secret.json")
    group.add_argument("--auth-url", action="store_true", help="Print OAuth URL for user to visit")
    group.add_argument("--auth-code", metavar="CODE", help="Exchange auth code for token")
    group.add_argument("--revoke", action="store_true", help="Revoke and delete stored token")
    group.add_argument("--install-deps", action="store_true", help="Install Python dependencies")
    parser.add_argument(
        "--email", metavar="EMAIL", default=None,
        help="Scope operation to a specific user's token (default: legacy single-user path)")
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


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import secrets  # noqa: F401,E402
import subprocess  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'atomic_replace': ('utils', 'atomic_replace'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----

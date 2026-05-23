#!/usr/bin/env python3
"""Multi-account token resolution for the Google Workspace skill.

Accounts are identified by Gmail address (lower-cased, normalized).
Tokens for each account live at ``~/.hermes/google_tokens/<email>.json``.

Backward compatibility
----------------------
The legacy single-account layout used ``~/.hermes/google_token.json``.
After ``--migrate-legacy``:

* The legacy token is moved to ``google_tokens/<derived-email>.json``
* ``google_token.json`` becomes a symlink to that file
* The migrated account is marked as the default

Anything that read ``~/.hermes/google_token.json`` directly (cron jobs,
``GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE``, third-party scripts) continues
to work unchanged because the symlink resolves to the default account.

Resolution precedence (most to least specific):

1. Explicit ``--account EMAIL`` CLI flag
2. ``HERMES_GOOGLE_ACCOUNT`` environment variable
3. Default account pointer at ``google_tokens/default``
4. Legacy file ``google_token.json`` (un-migrated single-account installs)
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

# Ensure sibling modules (_hermes_home) are importable when run standalone.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from _hermes_home import get_hermes_home

ACCOUNT_ENV_VAR = "HERMES_GOOGLE_ACCOUNT"

# Conservative — Gmail addresses, GSuite domain addresses, anything with @.
# We reject anything else so weird strings can't escape into filenames.
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def hermes_home() -> Path:
    return get_hermes_home()


def tokens_dir() -> Path:
    return hermes_home() / "google_tokens"


def legacy_token_path() -> Path:
    return hermes_home() / "google_token.json"


def default_pointer_path() -> Path:
    return tokens_dir() / "default"


def normalize_email(email: str) -> str:
    """Normalize an email address for use as an account key.

    Lowercases and strips whitespace. Raises ValueError on anything that
    doesn't look like an email address — protects against directory
    traversal and weird filenames.
    """
    if not email:
        raise ValueError("Account email is empty.")
    cleaned = email.strip().lower()
    if not _EMAIL_RE.match(cleaned):
        raise ValueError(f"Not a valid email address: {email!r}")
    if "/" in cleaned or "\\" in cleaned or ".." in cleaned:
        # Belt and suspenders — _EMAIL_RE already excludes these.
        raise ValueError(f"Email contains illegal characters: {email!r}")
    return cleaned


def token_path_for(email: str) -> Path:
    """Return the on-disk token path for a given account email."""
    return tokens_dir() / f"{normalize_email(email)}.json"


def _read_default_pointer() -> Optional[str]:
    pointer = default_pointer_path()
    if not pointer.exists():
        return None
    try:
        value = pointer.read_text().strip()
    except OSError:
        return None
    if not value:
        return None
    try:
        return normalize_email(value)
    except ValueError:
        return None


def list_accounts() -> List[str]:
    """List all known accounts (sorted).

    An account is "known" if it has a valid token JSON in ``google_tokens/``.
    The legacy ``google_token.json`` is not listed here unless it has been
    migrated.
    """
    d = tokens_dir()
    if not d.is_dir():
        return []
    out: List[str] = []
    for entry in sorted(d.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix != ".json":
            continue
        # Strip ".json" and validate as email — protects against junk files.
        stem = entry.stem
        try:
            normalize_email(stem)
        except ValueError:
            continue
        out.append(stem)
    return out


def get_default_account() -> Optional[str]:
    """Return the default account email, or None if not configured.

    Resolution: pointer file → if missing AND only one account exists,
    that one wins.
    """
    pointed = _read_default_pointer()
    if pointed:
        # Verify the pointer's target actually exists.
        if token_path_for(pointed).exists():
            return pointed
        return None
    accounts = list_accounts()
    if len(accounts) == 1:
        return accounts[0]
    return None


def set_default_account(email: str) -> None:
    """Set the default account and refresh the legacy ``google_token.json``
    symlink so backward-compat consumers keep working."""
    norm = normalize_email(email)
    target = token_path_for(norm)
    if not target.exists():
        raise FileNotFoundError(
            f"No token for {norm} at {target}. Run --account {norm} setup first."
        )
    tokens_dir().mkdir(parents=True, exist_ok=True)
    pointer = default_pointer_path()
    pointer.write_text(norm + "\n")

    # Refresh the legacy symlink. We do this carefully:
    #   - If the legacy path is a symlink (already managed by us), replace it.
    #   - If it's a regular file, leave it alone — the user hasn't migrated
    #     yet and we don't want to clobber their real token. They'd hit this
    #     case if they set a default before running --migrate-legacy, which
    #     is unusual but possible.
    legacy = legacy_token_path()
    if legacy.is_symlink() or not legacy.exists():
        try:
            if legacy.is_symlink() or legacy.exists():
                legacy.unlink()
            legacy.symlink_to(target)
        except OSError:
            # Symlinks unsupported (Windows without admin etc.) — fall back
            # to copying the JSON contents instead.
            legacy.write_text(target.read_text())


def resolve_account(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve which account to use given the precedence chain.

    Returns the normalized email, or None if we should fall back to the
    pre-migration legacy single-account file.
    """
    if explicit:
        return normalize_email(explicit)
    env = os.environ.get(ACCOUNT_ENV_VAR, "").strip()
    if env:
        return normalize_email(env)
    default = get_default_account()
    if default:
        return default
    return None


def resolve_token_path(explicit_account: Optional[str] = None) -> Path:
    """Return the token path to use for this invocation.

    If an account is resolved, returns its dedicated token file
    (``google_tokens/<email>.json``). Otherwise returns the legacy
    single-account path (``google_token.json``) for backward compatibility.

    The returned path may not exist — callers should check.
    """
    account = resolve_account(explicit_account)
    if account is not None:
        return token_path_for(account)
    return legacy_token_path()


def _userinfo_email_via_tokeninfo(access_token: str) -> Optional[str]:
    """Fetch the email associated with an access token via Google's
    tokeninfo endpoint. Returns None on any error (network, scope, etc.)."""
    import urllib.error
    import urllib.parse
    import urllib.request

    url = (
        "https://oauth2.googleapis.com/tokeninfo?"
        + urllib.parse.urlencode({"access_token": access_token})
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return None
    email = data.get("email")
    if not email:
        return None
    try:
        return normalize_email(email)
    except ValueError:
        return None


def _userinfo_email_via_userinfo(access_token: str) -> Optional[str]:
    """Fallback: Google userinfo endpoint."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return None
    email = data.get("email")
    if not email:
        return None
    try:
        return normalize_email(email)
    except ValueError:
        return None


def _email_via_gmail_profile(access_token: str) -> Optional[str]:
    """Last-resort fallback: Gmail API users.getProfile returns emailAddress.

    Useful when the token has gmail scopes but neither userinfo.email nor
    openid scopes (legacy tokens often lack them)."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        "https://gmail.googleapis.com/gmail/v1/users/me/profile",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return None
    email = data.get("emailAddress")
    if not email:
        return None
    try:
        return normalize_email(email)
    except ValueError:
        return None


def derive_email_from_token(token_payload: dict) -> Optional[str]:
    """Best-effort: figure out the Gmail address that owns a token.

    Tries (in order):

    1. ``email`` field on the token payload (some flows write it)
    2. ID token claims if present
    3. ``oauth2.googleapis.com/tokeninfo`` lookup using the access token
    4. ``oauth2/v2/userinfo`` lookup as a fallback

    Returns the normalized email or None. Network calls have a 15s timeout.
    """
    direct = token_payload.get("email")
    if isinstance(direct, str) and direct.strip():
        try:
            return normalize_email(direct)
        except ValueError:
            pass

    # Some payloads carry an id_token with "email" in the JWT body.
    id_token = token_payload.get("id_token")
    if isinstance(id_token, str) and id_token.count(".") == 2:
        import base64

        try:
            body = id_token.split(".")[1]
            # JWT base64url, may need padding
            padding = 4 - (len(body) % 4)
            if padding != 4:
                body = body + ("=" * padding)
            claims = json.loads(base64.urlsafe_b64decode(body))
            claim_email = claims.get("email")
            if isinstance(claim_email, str) and claim_email.strip():
                try:
                    return normalize_email(claim_email)
                except ValueError:
                    pass
        except Exception:
            pass

    access = token_payload.get("token") or token_payload.get("access_token")
    if isinstance(access, str) and access:
        via_tokeninfo = _userinfo_email_via_tokeninfo(access)
        if via_tokeninfo:
            return via_tokeninfo
        via_userinfo = _userinfo_email_via_userinfo(access)
        if via_userinfo:
            return via_userinfo
        via_gmail = _email_via_gmail_profile(access)
        if via_gmail:
            return via_gmail

    return None


def store_token_for_account(email: str, token_payload: dict) -> Path:
    """Write a token JSON to ``google_tokens/<email>.json`` with 0600 perms.
    Returns path."""
    norm = normalize_email(email)
    d = tokens_dir()
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    path = token_path_for(norm)
    path.write_text(json.dumps(token_payload, indent=2))
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path

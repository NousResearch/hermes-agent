"""Google OAuth helpers for Torben's EA scope."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

REDIRECT_URI = "http://localhost:1"
ALLOWED_SCOPES = {
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
    "https://www.googleapis.com/auth/calendar.events",
}


@dataclass(frozen=True)
class GoogleAccount:
    alias: str
    email: str
    role: str
    enabled: bool
    token_path: Path
    client_secret_path: Path
    scopes: tuple[str, ...]


@dataclass
class GoogleAuthResult:
    status: str
    account: str
    email: str
    token_path: str | None = None
    url: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "account": self.account,
            "email": self.email,
            "token_path": self.token_path,
            "url": self.url,
            "reason": self.reason,
        }


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_expiry(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_google_accounts(path: str | Path) -> dict[str, GoogleAccount]:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw_accounts = data.get("accounts") or {}
    if not isinstance(raw_accounts, dict):
        raise ValueError(f"Google account config must contain an accounts object: {config_path}")

    accounts: dict[str, GoogleAccount] = {}
    for alias, raw in raw_accounts.items():
        if not isinstance(raw, dict):
            continue
        account_alias = str(raw.get("alias") or alias)
        accounts[account_alias] = GoogleAccount(
            alias=account_alias,
            email=str(raw.get("email") or ""),
            role=str(raw.get("role") or ""),
            enabled=bool(raw.get("enabled", True)),
            token_path=Path(str(raw["token_path"])).expanduser(),
            client_secret_path=Path(str(raw["client_secret_path"])).expanduser(),
            scopes=tuple(str(scope) for scope in (raw.get("scopes") or [])),
        )
    return accounts


def account_for_alias(path: str | Path, alias: str) -> GoogleAccount:
    accounts = load_google_accounts(path)
    if alias not in accounts:
        known = ", ".join(sorted(accounts))
        raise ValueError(f"Unknown Google account {alias!r}. Known accounts: {known}")
    account = accounts[alias]
    if not account.enabled:
        raise ValueError(f"Google account {alias!r} is disabled.")
    return account


def account_home(account: GoogleAccount) -> Path:
    return account.token_path.parent


def pending_path(account: GoogleAccount) -> Path:
    return account_home(account) / "google_oauth_pending.json"


def configured_scopes(account: GoogleAccount) -> list[str]:
    scopes = list(account.scopes)
    if not scopes:
        raise ValueError(f"Google account {account.alias!r} has no configured OAuth scopes.")
    forbidden = sorted(set(scopes) - ALLOWED_SCOPES)
    if forbidden:
        blocked = ", ".join(forbidden)
        raise ValueError(f"Refusing unsupported Google scopes for {account.alias}: {blocked}")
    return scopes


def read_only_scopes(account: GoogleAccount) -> list[str]:
    """Backward-compatible alias for tests and older profile scripts."""

    return configured_scopes(account)


def load_client_secret(account: GoogleAccount) -> dict[str, Any]:
    if not account.client_secret_path.exists():
        raise FileNotFoundError(f"Missing client secret for {account.alias}: {account.client_secret_path}")
    data = json.loads(account.client_secret_path.read_text(encoding="utf-8"))
    app = data.get("installed") or data.get("web") or {}
    required = ("client_id", "client_secret", "auth_uri", "token_uri")
    missing = [key for key in required if not app.get(key)]
    if missing:
        raise ValueError(f"Client secret for {account.alias} is missing fields: {', '.join(missing)}")
    return app


def make_code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def extract_code_and_state(code_or_url: str) -> tuple[str, str | None, list[str]]:
    if not code_or_url.startswith("http"):
        return code_or_url, None, []
    parsed = urlparse(code_or_url)
    params = parse_qs(parsed.query)
    if "code" not in params:
        raise ValueError("No code= parameter found in pasted redirect URL.")
    scope_value = (params.get("scope") or [""])[0].strip()
    scopes = scope_value.split() if scope_value else []
    return params["code"][0], (params.get("state") or [None])[0], scopes


def build_auth_url(account: GoogleAccount) -> GoogleAuthResult:
    scopes = configured_scopes(account)
    client = load_client_secret(account)
    account_home(account).mkdir(parents=True, exist_ok=True)
    state = secrets.token_urlsafe(24)
    code_verifier = secrets.token_urlsafe(64)
    params = urllib.parse.urlencode(
        {
            "client_id": client["client_id"],
            "redirect_uri": REDIRECT_URI,
            "response_type": "code",
            "scope": " ".join(scopes),
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
            "code_challenge": make_code_challenge(code_verifier),
            "code_challenge_method": "S256",
        }
    )
    url = f"{client['auth_uri']}?{params}"
    pending_path(account).write_text(
        json.dumps(
            {
                "account": account.alias,
                "state": state,
                "code_verifier": code_verifier,
                "redirect_uri": REDIRECT_URI,
                "scopes": scopes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pending_path(account).chmod(0o600)
    return GoogleAuthResult(status="auth_url", account=account.alias, email=account.email, url=url)


def exchange_auth_code(account: GoogleAccount, code_or_url: str) -> GoogleAuthResult:
    pending = pending_path(account)
    if not pending.exists():
        raise FileNotFoundError(f"No pending OAuth session for {account.alias}; run auth-url first.")
    data = json.loads(pending.read_text(encoding="utf-8"))
    code, returned_state, callback_scopes = extract_code_and_state(code_or_url)
    if returned_state and returned_state != data.get("state"):
        raise ValueError("OAuth state mismatch. Re-run auth-url and use the newest redirect URL.")
    scopes = callback_scopes or data.get("scopes") or configured_scopes(account)
    forbidden = sorted(set(scopes) - ALLOWED_SCOPES)
    if forbidden:
        blocked = ", ".join(forbidden)
        raise ValueError(f"Google returned non-Torben scopes; refusing token: {blocked}")
    missing = sorted(set(configured_scopes(account)) - set(scopes))
    if missing:
        blocked = ", ".join(missing)
        raise ValueError(f"Google did not grant required scopes for {account.alias}: {blocked}")

    client = load_client_secret(account)
    params = urllib.parse.urlencode(
        {
            "code": code,
            "client_id": client["client_id"],
            "client_secret": client["client_secret"],
            "redirect_uri": data.get("redirect_uri", REDIRECT_URI),
            "grant_type": "authorization_code",
            "code_verifier": data["code_verifier"],
        }
    ).encode("utf-8")
    request = urllib.request.Request(client["token_uri"], data=params, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            token_response = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Token exchange failed for {account.alias}: HTTP {exc.code}. "
            "Run auth-url again if the code expired."
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Token exchange failed for {account.alias}: network error.") from exc

    if not token_response.get("refresh_token"):
        raise RuntimeError(
            f"Token exchange for {account.alias} returned no refresh token. "
            "Confirm the consent prompt completed, then run auth-url again."
        )
    expires_in = int(token_response.get("expires_in") or 3600)
    payload = {
        "type": "authorized_user",
        "token": token_response.get("access_token"),
        "refresh_token": token_response.get("refresh_token"),
        "token_uri": client["token_uri"],
        "client_id": client["client_id"],
        "client_secret": client["client_secret"],
        "scopes": scopes,
        "expiry": (utc_now() + timedelta(seconds=expires_in)).isoformat(),
    }
    account.token_path.parent.mkdir(parents=True, exist_ok=True)
    account.token_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    account.token_path.chmod(0o600)
    pending.unlink(missing_ok=True)
    return GoogleAuthResult(
        status="authenticated",
        account=account.alias,
        email=account.email,
        token_path=str(account.token_path),
    )


def check_account(account: GoogleAccount) -> GoogleAuthResult:
    if not account.token_path.exists():
        return GoogleAuthResult(
            status="not_authenticated",
            account=account.alias,
            email=account.email,
            token_path=str(account.token_path),
            reason="no token file",
        )
    try:
        payload = json.loads(account.token_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return GoogleAuthResult(
            status="token_invalid",
            account=account.alias,
            email=account.email,
            token_path=str(account.token_path),
            reason=str(exc),
        )
    expiry = parse_expiry(str(payload.get("expiry") or ""))
    granted_scopes = set(payload.get("scopes") or [])
    required_scopes = set(configured_scopes(account))
    missing_scopes = sorted(required_scopes - granted_scopes)
    if missing_scopes:
        return GoogleAuthResult(
            status="scope_upgrade_required",
            account=account.alias,
            email=account.email,
            token_path=str(account.token_path),
            reason=", ".join(missing_scopes),
        )
    if payload.get("token") and expiry and expiry > utc_now():
        return GoogleAuthResult(
            status="authenticated",
            account=account.alias,
            email=account.email,
            token_path=str(account.token_path),
        )
    required = ("client_id", "client_secret", "refresh_token", "token_uri")
    missing = [key for key in required if not payload.get(key)]
    if missing:
        return GoogleAuthResult(
            status="token_invalid",
            account=account.alias,
            email=account.email,
            token_path=str(account.token_path),
            reason=f"missing {', '.join(missing)}",
        )

    params = urllib.parse.urlencode(
        {
            "client_id": payload["client_id"],
            "client_secret": payload["client_secret"],
            "refresh_token": payload["refresh_token"],
            "grant_type": "refresh_token",
        }
    ).encode("utf-8")
    request = urllib.request.Request(str(payload["token_uri"]), data=params, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            refreshed = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return GoogleAuthResult(
            status="refresh_failed",
            account=account.alias,
            email=account.email,
            token_path=str(account.token_path),
            reason=f"HTTP {exc.code}",
        )
    except OSError:
        return GoogleAuthResult(
            status="refresh_failed",
            account=account.alias,
            email=account.email,
            token_path=str(account.token_path),
            reason="network error",
        )
    expires_in = int(refreshed.get("expires_in") or 3600)
    payload["token"] = refreshed.get("access_token")
    payload["expiry"] = (utc_now() + timedelta(seconds=expires_in)).isoformat()
    payload["scopes"] = configured_scopes(account)
    account.token_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    account.token_path.chmod(0o600)
    return GoogleAuthResult(
        status="authenticated_refreshed",
        account=account.alias,
        email=account.email,
        token_path=str(account.token_path),
    )

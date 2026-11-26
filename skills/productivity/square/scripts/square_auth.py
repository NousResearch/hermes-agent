#!/usr/bin/env python3
"""Shared Square OAuth token loading and refresh support."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from hermes_constants import get_hermes_home
except ModuleNotFoundError:
    HERMES_AGENT_ROOT = Path(__file__).resolve().parents[4]
    if HERMES_AGENT_ROOT.exists():
        sys.path.insert(0, str(HERMES_AGENT_ROOT))
    from hermes_constants import get_hermes_home


HERMES_HOME = get_hermes_home()
TOKEN_PATH = HERMES_HOME / "square_token.json"
CLIENT_SECRET_PATH = HERMES_HOME / "square_client_secret.json"
TOKEN_URL = "https://connect.squareup.com/oauth2/token"
REFRESH_MARGIN = timedelta(minutes=5)
REQUEST_TIMEOUT_SECONDS = 30


class SquareAuthError(RuntimeError):
    """Raised when Square credentials cannot produce a valid access token."""


def _load_json(path: Path, label: str) -> dict:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SquareAuthError(f"Missing {label} at {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SquareAuthError(f"Could not read {label} at {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise SquareAuthError(f"Invalid {label} at {path}: expected a JSON object")
    return data


def _parse_expiry(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _token_needs_refresh(token_data: dict, now: datetime | None = None) -> bool:
    """Refresh tokens with missing/invalid expiry or less than five minutes left."""
    expiry = _parse_expiry(token_data.get("expires_at"))
    if expiry is None:
        return True
    current = now or datetime.now(timezone.utc)
    return expiry <= current + REFRESH_MARGIN


def request_token(payload: dict) -> dict:
    request = urllib.request.Request(
        TOKEN_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            raw_body = response.read()
    except urllib.error.HTTPError as exc:
        raw_body = exc.read()
        try:
            detail = json.loads(raw_body).get("error_description")
        except (AttributeError, json.JSONDecodeError):
            detail = raw_body.decode("utf-8", errors="replace")
        raise SquareAuthError(
            f"Square token request failed with HTTP {exc.code}: {detail or 'unknown error'}"
        ) from exc
    except (OSError, urllib.error.URLError) as exc:
        raise SquareAuthError(f"Square token request failed: {exc}") from exc

    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise SquareAuthError("Square token response was not valid JSON") from exc
    if status != 200:
        detail = body.get("error_description", body.get("error", "unknown error"))
        raise SquareAuthError(f"Square token request failed with HTTP {status}: {detail}")
    if not isinstance(body, dict) or not body.get("access_token"):
        raise SquareAuthError("Square token response did not include an access_token")
    return body


def refresh_access_token(token_data: dict | None = None) -> str:
    """Refresh and atomically persist the Square OAuth token."""
    current = token_data or _load_json(TOKEN_PATH, "Square token")
    refresh_token = current.get("refresh_token")
    if not refresh_token:
        raise SquareAuthError("Square token has expired and has no refresh_token; authorize again")

    credentials = _load_json(CLIENT_SECRET_PATH, "Square client credentials")
    client_id = credentials.get("clientId") or credentials.get("client_id")
    client_secret = credentials.get("clientSecret") or credentials.get("client_secret")
    if not client_id or not client_secret:
        raise SquareAuthError("Square client credentials are missing clientId or clientSecret")

    refreshed = request_token(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
    )
    if not refreshed.get("refresh_token"):
        refreshed["refresh_token"] = refresh_token

    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = TOKEN_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(refreshed, indent=2))
    os.replace(temporary_path, TOKEN_PATH)
    return refreshed["access_token"]


def get_valid_access_token(*, force_refresh: bool = False) -> str:
    """Return a usable token, refreshing it before a request when necessary."""
    token_data = _load_json(TOKEN_PATH, "Square token")
    access_token = token_data.get("access_token")
    if not access_token:
        raise SquareAuthError("Square token is missing access_token; authorize again")
    if force_refresh or _token_needs_refresh(token_data):
        return refresh_access_token(token_data)
    return access_token

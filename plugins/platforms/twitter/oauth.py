from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode

import httpx

from hermes_constants import get_hermes_home
from utils import atomic_json_write

AUTHORIZE_URL = "https://x.com/i/oauth2/authorize"
TOKEN_URL = "https://api.x.com/2/oauth2/token"
SCOPES = (
    "tweet.read",
    "tweet.write",
    "users.read",
    "offline.access",
    "dm.read",
    "dm.write",
    "bookmark.read",
    "bookmark.write",
    "media.write",
)


@dataclass(frozen=True)
class OAuthTokens:
    access_token: str
    refresh_token: str = ""
    expires_at: float = 0
    scopes: tuple[str, ...] = ()
    client_id: str = ""
    user_id: str = ""
    username: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OAuthTokens":
        access_token = value.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise ValueError("invalid Twitter OAuth token record")
        raw_scopes = value.get("scopes") or ()
        if isinstance(raw_scopes, str):
            raw_scopes = raw_scopes.split()
        return cls(
            access_token=access_token,
            refresh_token=str(value.get("refresh_token") or ""),
            expires_at=float(value.get("expires_at") or 0),
            scopes=tuple(map(str, raw_scopes)),
            client_id=str(value.get("client_id") or ""),
            user_id=str(value.get("user_id") or ""),
            username=str(value.get("username") or ""),
        )

    def expired(self, *, leeway: float = 60) -> bool:
        return bool(self.expires_at and self.expires_at <= time.time() + leeway)


def token_path() -> Path:
    return get_hermes_home() / "twitter" / "oauth2.json"


def create_s256_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def create_pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    return verifier, create_s256_challenge(verifier)


def build_authorization_url(
    *, client_id: str, redirect_uri: str, challenge: str, state: str
) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(SCOPES),
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def load_tokens() -> OAuthTokens | None:
    path = token_path()
    try:
        return OAuthTokens.from_mapping(json.loads(path.read_text()))
    except FileNotFoundError:
        return None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_tokens(tokens: OAuthTokens | Mapping[str, Any]) -> OAuthTokens:
    record = tokens if isinstance(tokens, OAuthTokens) else OAuthTokens.from_mapping(tokens)
    payload = asdict(record)
    payload["scopes"] = list(record.scopes)
    atomic_json_write(token_path(), payload, mode=0o600)
    return record


class OAuthClient:
    def __init__(
        self,
        client_id: str,
        redirect_uri: str,
        *,
        client: httpx.AsyncClient | None = None,
    ):
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self.client = client or httpx.AsyncClient(timeout=30)
        self._owns_client = client is None
        self._refresh_lock = asyncio.Lock()

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def exchange_code(self, code: str, verifier: str) -> OAuthTokens:
        response = await self.client.post(
            TOKEN_URL,
            data={
                "code": code,
                "grant_type": "authorization_code",
                "client_id": self.client_id,
                "redirect_uri": self.redirect_uri,
                "code_verifier": verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        return self._persist_response(response.json())

    async def refresh(self, tokens: OAuthTokens) -> OAuthTokens:
        async with self._refresh_lock:
            current = load_tokens() or tokens
            if not current.expired():
                return current
            if not current.refresh_token:
                raise RuntimeError("Twitter OAuth refresh token is missing")
            response = await self.client.post(
                TOKEN_URL,
                data={
                    "refresh_token": current.refresh_token,
                    "grant_type": "refresh_token",
                    "client_id": self.client_id,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            return self._persist_response(response.json(), current=current)

    def _persist_response(
        self, payload: Mapping[str, Any], *, current: OAuthTokens | None = None
    ) -> OAuthTokens:
        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError("Twitter OAuth response omitted access_token")
        expires_in = float(payload.get("expires_in") or 0)
        scopes = str(payload.get("scope") or "").split()
        tokens = OAuthTokens(
            access_token=access_token,
            refresh_token=str(
                payload.get("refresh_token")
                or (current.refresh_token if current else "")
            ),
            expires_at=time.time() + expires_in if expires_in else 0,
            scopes=tuple(scopes),
            client_id=self.client_id,
            user_id=current.user_id if current else "",
            username=current.username if current else "",
        )
        return save_tokens(tokens)

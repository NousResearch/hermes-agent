from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import time
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from hermes_constants import get_hermes_home
from utils import atomic_json_write

AUTHORIZE_URL = "https://x.com/i/oauth2/authorize"
TOKEN_URL = "https://api.x.com/2/oauth2/token"
logger = logging.getLogger(__name__)
_refresh_locks: dict[str, asyncio.Lock] = {}
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


def _lock_refresh_file(path: Path):
    lock_path = Path(f"{path}.refresh.lock")
    handle = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+b")
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return handle
    except Exception:
        logger.debug(
            "Twitter OAuth cross-process lock unavailable; in-process only",
            exc_info=True,
        )
        if handle is not None:
            handle.close()
        return None


def _unlock_refresh_file(handle) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    finally:
        handle.close()


@asynccontextmanager
async def _profile_refresh_lock(path: Path):
    key = str(path.resolve())
    lock = _refresh_locks.setdefault(key, asyncio.Lock())
    async with lock:
        acquire = asyncio.create_task(asyncio.to_thread(_lock_refresh_file, path))
        try:
            try:
                handle = await asyncio.shield(acquire)
            except asyncio.CancelledError:
                handle = await acquire
                if handle is not None:
                    await asyncio.to_thread(_unlock_refresh_file, handle)
                raise
            try:
                yield
            finally:
                if handle is not None:
                    await asyncio.shield(
                        asyncio.to_thread(_unlock_refresh_file, handle)
                    )
        finally:
            if not acquire.done():
                acquire.cancel()


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


def _loopback_redirect(redirect_uri: str) -> tuple[str, int, str]:
    parsed = urlparse(redirect_uri)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.port is None
        or not parsed.path.startswith("/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Twitter redirect_uri must be an HTTP loopback URL with a port")
    return parsed.hostname, parsed.port, parsed.path


async def wait_for_callback(
    redirect_uri: str,
    expected_state: str,
    *,
    timeout: float = 180,
    on_ready: Callable[[], Any] | None = None,
) -> str:
    host, port, expected_path = _loopback_redirect(redirect_uri)
    loop = asyncio.get_running_loop()
    result: asyncio.Future[str] = loop.create_future()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        status = "400 Bad Request"
        body = "Authorization failed. You may close this window."
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=5)
            parts = line.decode("ascii", "replace").strip().split()
            if len(parts) != 3 or parts[0] != "GET":
                raise ValueError("malformed Twitter OAuth callback")
            target = urlparse(parts[1])
            if target.path != expected_path:
                raise ValueError("unexpected Twitter OAuth callback path")
            query = parse_qs(target.query)
            state = (query.get("state") or [""])[0]
            if not secrets.compare_digest(state, expected_state):
                raise ValueError("Twitter OAuth callback state mismatch")
            if query.get("error"):
                raise RuntimeError("Twitter OAuth authorization was denied")
            code = (query.get("code") or [""])[0]
            if not code:
                raise ValueError("Twitter OAuth callback omitted code")
            if not result.done():
                result.set_result(code)
            status = "200 OK"
            body = "Twitter authorization completed. You may close this window."
        except Exception as exc:
            if not result.done():
                result.set_exception(exc)
        finally:
            response = (
                f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\n"
                f"Content-Length: {len(body.encode())}\r\nConnection: close\r\n\r\n{body}"
            )
            writer.write(response.encode())
            try:
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

    server = await asyncio.start_server(handle, host, port)
    try:
        if on_ready is not None:
            on_ready()
        async with asyncio.timeout(timeout):
            return await result
    finally:
        server.close()
        await server.wait_closed()


async def authorize(
    client_id: str,
    redirect_uri: str,
    *,
    timeout: float = 180,
    open_url: Callable[[str], Any] = webbrowser.open,
) -> OAuthTokens:
    verifier, challenge = create_pkce_pair()
    state = secrets.token_urlsafe(32)
    url = build_authorization_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        challenge=challenge,
        state=state,
    )
    code = await wait_for_callback(
        redirect_uri, state, timeout=timeout, on_ready=lambda: open_url(url)
    )
    oauth = OAuthClient(client_id, redirect_uri)
    try:
        tokens = await oauth.exchange_code(code, verifier)
        from .client import XClient

        client = XClient(token=tokens.access_token)
        try:
            identity = (await client.identity()).get("data") or {}
        finally:
            await client.close()
        user_id = str(identity.get("id") or "")
        if not user_id:
            raise RuntimeError("X did not return the authenticated user ID")
        return save_tokens(
            OAuthTokens(
                **{
                    **asdict(tokens),
                    "user_id": user_id,
                    "username": str(identity.get("username") or ""),
                }
            )
        )
    finally:
        await oauth.close()


async def refresh_if_needed(client_id: str, redirect_uri: str) -> OAuthTokens:
    tokens = load_tokens()
    if tokens is None:
        raise RuntimeError("Twitter OAuth is not configured")
    if not tokens.expired():
        return tokens
    oauth = OAuthClient(client_id or tokens.client_id, redirect_uri)
    try:
        return await oauth.refresh(tokens)
    finally:
        await oauth.close()


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
        async with _profile_refresh_lock(token_path()):
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
        missing = set(SCOPES) - set(scopes)
        if missing:
            raise RuntimeError(
                "Twitter OAuth response omitted required scopes: "
                + ", ".join(sorted(missing))
            )
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

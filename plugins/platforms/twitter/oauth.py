from __future__ import annotations

import asyncio
import base64
import hashlib
import json
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

from agent.secret_scope import get_secret
from hermes_constants import get_hermes_home
from utils import atomic_json_write

AUTHORIZE_URL = "https://x.com/i/oauth2/authorize"
TOKEN_URL = "https://api.x.com/2/oauth2/token"
_scoped_locks: dict[str, asyncio.Lock] = {}
_LOCK_TIMEOUT_SECONDS = 30
_LOCK_RETRY_SECONDS = 0.1
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
    client_type: str = "public"
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
        client_type = str(value.get("client_type") or "public")
        if client_type not in {"public", "confidential"}:
            raise ValueError("invalid Twitter OAuth client type")
        return cls(
            access_token=access_token,
            refresh_token=str(value.get("refresh_token") or ""),
            expires_at=float(value.get("expires_at") or 0),
            scopes=tuple(map(str, raw_scopes)),
            client_id=str(value.get("client_id") or ""),
            client_type=client_type,
            user_id=str(value.get("user_id") or ""),
            username=str(value.get("username") or ""),
        )

    def expired(self, *, leeway: float = 60) -> bool:
        return bool(self.expires_at and self.expires_at <= time.time() + leeway)


def token_path() -> Path:
    return get_hermes_home() / "twitter" / "oauth2.json"


@asynccontextmanager
async def _twitter_scoped_lock(scope: str, profile_key: str, _account_id: str):
    """Serialize one profile's Twitter file mutations across processes."""
    identity = str(Path(profile_key).resolve())
    key = f"{scope}:{identity}"
    lock = _scoped_locks.setdefault(key, asyncio.Lock())
    lock_path = Path(identity) / "twitter" / f".{scope}.lock"
    async with lock:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt

                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out waiting for Twitter {scope} lock")
                    await asyncio.sleep(_LOCK_RETRY_SECONDS)
            try:
                yield
            finally:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def active_profile_key() -> str:
    return str(get_hermes_home().resolve())


@asynccontextmanager
async def token_refresh_lock(profile_key: str, account_id: str):
    async with _twitter_scoped_lock(
        "twitter-oauth-refresh", profile_key, account_id
    ):
        yield


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
    client_type: str = "public",
    client_secret: str = "",
    timeout: float = 180,
    open_url: Callable[[str], Any] = webbrowser.open,
    transport: httpx.AsyncBaseTransport | None = None,
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
    oauth = OAuthClient(
        client_id,
        redirect_uri,
        client_type=client_type,
        client_secret=client_secret,
        transport=transport,
    )
    try:
        tokens = await oauth.exchange_code(code, verifier)
        from .client import XClient

        client = XClient(token=tokens.access_token, transport=transport)
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


async def refresh_if_needed(
    client_id: str,
    redirect_uri: str,
    *,
    client_type: str | None = None,
    client_secret: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> OAuthTokens:
    tokens = load_tokens()
    if tokens is None:
        raise RuntimeError(
            "Twitter OAuth is missing or invalid; run setup to re-authorize"
        )
    if tokens.client_id and client_id and tokens.client_id != client_id:
        raise RuntimeError("Twitter OAuth client changed; run setup to re-authorize")
    if client_type is not None and tokens.client_type != client_type:
        raise RuntimeError("Twitter OAuth client type changed; run setup to re-authorize")
    if not tokens.expired():
        return tokens
    resolved_type = client_type or tokens.client_type
    resolved_secret = (
        get_secret("TWITTER_CLIENT_SECRET", "") or ""
        if client_secret is None
        else client_secret
    )
    oauth = OAuthClient(
        client_id or tokens.client_id,
        redirect_uri,
        client_type=resolved_type,
        client_secret=resolved_secret,
        transport=transport,
    )
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
        client_type: str = "public",
        client_secret: str = "",
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if client_type not in {"public", "confidential"}:
            raise ValueError("Twitter OAuth client type must be public or confidential")
        if client_type == "confidential" and not client_secret:
            raise ValueError("Twitter confidential OAuth client secret is required")
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self.client_type = client_type
        self.client_secret = client_secret
        self.client = client or httpx.AsyncClient(timeout=30, transport=transport)
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def exchange_code(self, code: str, verifier: str) -> OAuthTokens:
        response = await self._post_token(
            {
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": self.redirect_uri,
                "code_verifier": verifier,
            },
        )
        return self._tokens_from_response(response.json())

    async def _post_token(self, data: dict[str, str]) -> httpx.Response:
        auth = None
        if self.client_type == "public":
            data["client_id"] = self.client_id
        else:
            auth = httpx.BasicAuth(self.client_id, self.client_secret)
        response = await self.client.post(
            TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            auth=auth,
        )
        response.raise_for_status()
        return response

    async def refresh(self, tokens: OAuthTokens) -> OAuthTokens:
        account_id = tokens.user_id or tokens.client_id or self.client_id
        async with token_refresh_lock(active_profile_key(), account_id):
            current = load_tokens() or tokens
            if current.client_id and current.client_id != self.client_id:
                raise RuntimeError(
                    "Twitter OAuth client changed; run setup to re-authorize"
                )
            if current.client_type != self.client_type:
                raise RuntimeError(
                    "Twitter OAuth client type changed; run setup to re-authorize"
                )
            if not current.expired():
                return current
            if not current.refresh_token:
                raise RuntimeError("Twitter OAuth refresh token is missing")
            response = await self._post_token(
                {
                    "refresh_token": current.refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            return save_tokens(
                self._tokens_from_response(response.json(), current=current)
            )

    def _tokens_from_response(
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
            client_type=self.client_type,
            user_id=current.user_id if current else "",
            username=current.username if current else "",
        )
        return tokens

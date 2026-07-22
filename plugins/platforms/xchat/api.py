"""Async X API client for the X Chat platform adapter.

A thin httpx-based client for the handful of X API v2 endpoints the adapter
needs. The official ``xdk`` Python client is synchronous (requests-based),
which doesn't fit the async gateway — the Chat endpoints are plain
OAuth2-bearer REST, so direct calls are wire-identical. Only the E2EE layer
needs a real SDK (``chatxdk``, see ``crypto.py``).

Also owns OAuth2 token refresh: X user access tokens expire (~2h). When a
refresh token + client id are configured, :meth:`XChatApi.ensure_token`
renews the access token through ``POST /2/oauth2/token`` and persists the
rotated pair via a caller-supplied callback (X rotates refresh tokens on
every use).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover - httpx is a core Hermes dependency
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

BASE_URL = "https://api.x.com"

# Fields we always request on the events endpoint — the decrypt path needs
# encoded_event; sender_id/conversation_id drive session routing.
_EVENT_FIELDS = (
    "conversation_id,created_at_msec,encoded_event,id,sender_id"
)

# Refresh the access token this many seconds before its reported expiry.
_TOKEN_REFRESH_SLACK = 300


class XChatApiError(Exception):
    """Raised for non-2xx responses from the X API."""

    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"X API HTTP {status}: {detail}")


class XChatRateLimited(XChatApiError):
    """HTTP 429 — carries the reset epoch when the API reports one."""

    def __init__(self, detail: str, reset_epoch: Optional[int]) -> None:
        super().__init__(429, detail)
        self.reset_epoch = reset_epoch


class XChatApi:
    """Async client bound to one bot account's OAuth2 user token."""

    def __init__(
        self,
        access_token: str,
        *,
        refresh_token: str = "",
        client_id: str = "",
        client_secret: str = "",
        token_expires_at: float = 0.0,
        on_token_refresh: Optional[Callable[[str, str], Awaitable[None]]] = None,
        base_url: str = BASE_URL,
        client: Optional["httpx.AsyncClient"] = None,
    ) -> None:
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._client_id = client_id
        self._client_secret = client_secret
        # 0 = unknown expiry; refresh only reactively on 401.
        self._token_expires_at = token_expires_at
        self._on_token_refresh = on_token_refresh
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._refresh_lock = asyncio.Lock()

    # -- lifecycle -----------------------------------------------------------

    def _http(self) -> "httpx.AsyncClient":
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    # -- auth ----------------------------------------------------------------

    @property
    def can_refresh(self) -> bool:
        return bool(self._refresh_token and self._client_id)

    async def ensure_token(self) -> None:
        """Proactively refresh the access token when close to expiry."""
        if not self.can_refresh or not self._token_expires_at:
            return
        if time.time() < self._token_expires_at - _TOKEN_REFRESH_SLACK:
            return
        await self._refresh_access_token()

    async def _refresh_access_token(self) -> None:
        """POST /2/oauth2/token (refresh_token grant). Rotates both tokens."""
        async with self._refresh_lock:
            # Another task may have refreshed while we waited on the lock.
            if self._token_expires_at and time.time() < self._token_expires_at - _TOKEN_REFRESH_SLACK:
                return
            data = {
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
                "client_id": self._client_id,
            }
            auth = None
            if self._client_secret:
                auth = (self._client_id, self._client_secret)
            resp = await self._http().post(
                f"{self._base_url}/2/oauth2/token", data=data, auth=auth
            )
            if resp.status_code >= 300:
                raise XChatApiError(resp.status_code, resp.text[:300])
            tok = resp.json()
            self._access_token = tok.get("access_token") or self._access_token
            # X rotates refresh tokens on every use — always adopt the new one.
            new_refresh = tok.get("refresh_token")
            if new_refresh:
                self._refresh_token = new_refresh
            expires_in = tok.get("expires_in")
            if expires_in:
                self._token_expires_at = time.time() + float(expires_in)
            logger.info("[xchat] OAuth2 access token refreshed")
            if self._on_token_refresh is not None:
                try:
                    await self._on_token_refresh(self._access_token, self._refresh_token)
                except Exception:
                    logger.warning("[xchat] token persist callback failed", exc_info=True)

    # -- request core ----------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
        _retried_auth: bool = False,
    ) -> dict[str, Any]:
        await self.ensure_token()
        headers = {"Authorization": f"Bearer {self._access_token}"}
        resp = await self._http().request(
            method,
            f"{self._base_url}{path}",
            params=params,
            json=json_body,
            headers=headers,
        )
        if resp.status_code == 401 and self.can_refresh and not _retried_auth:
            # Reactive refresh — covers the no-known-expiry case.
            await self._refresh_access_token()
            return await self._request(
                method, path, params=params, json_body=json_body, _retried_auth=True
            )
        if resp.status_code == 429:
            reset = resp.headers.get("x-user-limit-24hour-reset") or resp.headers.get(
                "x-rate-limit-reset"
            )
            raise XChatRateLimited(
                resp.text[:300], int(reset) if reset and reset.isdigit() else None
            )
        if resp.status_code >= 300:
            raise XChatApiError(resp.status_code, resp.text[:300])
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError:
            return {}

    # -- identity ------------------------------------------------------------

    async def get_my_user(self) -> dict[str, Any]:
        """GET /2/users/me — the bot account's own id/username."""
        out = await self._request("GET", "/2/users/me")
        return out.get("data") or {}

    async def get_public_keys(self, user_id: str) -> list[dict[str, Any]]:
        """GET /2/users/{id}/public_keys — a user's registered E2EE keys."""
        out = await self._request("GET", f"/2/users/{user_id}/public_keys")
        data = out.get("data") or []
        return data if isinstance(data, list) else [data]

    async def add_public_key(self, user_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST /2/users/{id}/public_keys — register the bot's public keys.

        Rate limited to a handful of writes per 24h; raises
        :class:`XChatRateLimited` on 429 so callers stop instead of burning
        the daily budget.
        """
        return await self._request("POST", f"/2/users/{user_id}/public_keys", json_body=body)

    # -- conversations ---------------------------------------------------------

    @staticmethod
    def _conv_path_id(conversation_id: str) -> str:
        # Events embed the colon form; URL paths take the hyphen form.
        return str(conversation_id).replace(":", "-")

    async def get_conversations(
        self, *, max_results: int = 100, pagination_token: Optional[str] = None
    ) -> dict[str, Any]:
        """GET /2/chat/conversations — list the bot's conversations."""
        params: dict[str, Any] = {"max_results": max_results}
        if pagination_token:
            params["pagination_token"] = pagination_token
        return await self._request("GET", "/2/chat/conversations", params=params)

    async def get_events(
        self,
        conversation_id: str,
        *,
        max_results: int = 50,
        pagination_token: Optional[str] = None,
    ) -> dict[str, Any]:
        """GET /2/chat/conversations/{id}/events — raw (encrypted) events."""
        params: dict[str, Any] = {
            "max_results": max_results,
            "chat_message_event.fields": _EVENT_FIELDS,
        }
        if pagination_token:
            params["pagination_token"] = pagination_token
        return await self._request(
            "GET",
            f"/2/chat/conversations/{self._conv_path_id(conversation_id)}/events",
            params=params,
        )

    async def send_message(self, conversation_id: str, body: dict[str, str]) -> dict[str, Any]:
        """POST /2/chat/conversations/{id}/messages — send encrypted ciphertext.

        ``body`` is the dict produced by ``XChatCrypto.encrypt_text``. For a
        1:1 conversation ``conversation_id`` may be the recipient's bare user
        id; the server derives the canonical conversation id.
        """
        return await self._request(
            "POST",
            f"/2/chat/conversations/{self._conv_path_id(conversation_id)}/messages",
            json_body=body,
        )

    async def send_typing(self, conversation_id: str) -> None:
        """POST /2/chat/conversations/{id}/typing — best-effort typing indicator."""
        await self._request(
            "POST",
            f"/2/chat/conversations/{self._conv_path_id(conversation_id)}/typing",
        )

    async def add_conversation_keys(
        self, conversation_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST /2/chat/conversations/{id}/keys — initialize/rotate a conversation key."""
        return await self._request(
            "POST",
            f"/2/chat/conversations/{self._conv_path_id(conversation_id)}/keys",
            json_body=body,
        )

"""
Thin async client for the Linq Blue partner API (v3).

Unlike Photon — which has no public HTTP send endpoint and therefore needs a
Node ``spectrum-ts`` sidecar — Linq exposes a first-class REST API, so all
outbound traffic (text, media-by-URL, typing, read receipts, reactions) is a
direct ``httpx`` call from this process.  No sidecar, no Node dependency.

Endpoints (base: ``https://api.linqapp.com/api/partner/v3``)::

    POST /chats/{chat_id}/messages      send a message (text and/or media parts)
    POST /chats/{chat_id}/typing        start a typing indicator
    POST /chats/{chat_id}/read          mark the chat read
    POST /messages/{message_id}/reactions   add/remove a tapback
    GET  /phonenumbers                  list the account's Linq numbers (probe)

Auth is a bearer token (``LINQ_API_TOKEN``).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover - httpx ships with Hermes
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api.linqapp.com/api/partner/v3"
USER_AGENT = "Hermes-Linq/0.1"
_MAX_RETRIES = 2
_RETRY_DELAY_MS = 500

# Reactions Linq's tapback API understands.
REACTION_TYPES = ("love", "like", "dislike", "laugh", "emphasize", "question")


class LinqApiError(RuntimeError):
    """Raised for non-2xx responses from the Linq API."""

    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"Linq API error {status}: {body[:200]}")


class LinqClient:
    """Async Linq partner API client.

    Holds its own ``httpx.AsyncClient`` so a single client can be shared
    across an adapter's lifetime; callers must ``await close()`` (or use it
    as an async context manager) when done.
    """

    def __init__(
        self,
        token: str,
        *,
        api_base: str = DEFAULT_API_BASE,
        timeout: float = 30.0,
        client: "Optional[httpx.AsyncClient]" = None,
    ) -> None:
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for the Linq client. Run: pip install httpx")
        self._token = token
        self._api_base = api_base.rstrip("/")
        self._timeout = timeout
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def __aenter__(self) -> "LinqClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client:
            try:
                await self._client.aclose()
            except Exception:
                pass

    # -- internals --------------------------------------------------------

    def _headers(self, *, json_body: bool = True) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._token}",
            "User-Agent": USER_AGENT,
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> "httpx.Response":
        url = f"{self._api_base}{path}"
        for attempt in range(_MAX_RETRIES + 1):
            resp = await self._client.request(
                method,
                url,
                headers=self._headers(json_body=json_body is not None),
                json=json_body,
            )
            # Honour 429 back-pressure with a bounded exponential backoff.
            if resp.status_code == 429 and attempt < _MAX_RETRIES:
                retry_after = 0.0
                try:
                    retry_after = float(resp.headers.get("retry-after") or 0)
                except ValueError:
                    retry_after = 0.0
                delay = retry_after if retry_after > 0 else (_RETRY_DELAY_MS / 1000) * (2 ** attempt)
                await asyncio.sleep(min(delay, 10.0))
                continue
            return resp
        return resp  # pragma: no cover - unreachable, last response returned above

    # -- outbound ---------------------------------------------------------

    async def send_message(
        self,
        chat_id: str,
        *,
        text: Optional[str] = None,
        media_url: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Send a message to a Linq chat. Returns ``{"message_id", "chat_id"}``."""
        parts: List[Dict[str, Any]] = []
        if text:
            parts.append({"type": "text", "value": text})
        if media_url:
            parts.append({"type": "media", "url": media_url})
        if not parts:
            raise ValueError("send_message requires text or media_url")

        message: Dict[str, Any] = {"parts": parts}
        if reply_to_message_id:
            message["reply_to"] = {"message_id": reply_to_message_id}

        resp = await self._request(
            "POST",
            f"/chats/{_quote(chat_id)}/messages",
            json_body={"message": message},
        )
        if not resp.is_success:
            raise LinqApiError(resp.status_code, _safe_text(resp))
        data = _safe_json(resp)
        msg = data.get("message") if isinstance(data, dict) else None
        return {
            "message_id": (msg or {}).get("id", "unknown") if isinstance(msg, dict) else "unknown",
            "chat_id": data.get("chat_id", chat_id) if isinstance(data, dict) else chat_id,
        }

    async def start_typing(self, chat_id: str) -> bool:
        return await self._fire_and_forget("POST", f"/chats/{_quote(chat_id)}/typing")

    async def stop_typing(self, chat_id: str) -> bool:
        return await self._fire_and_forget("DELETE", f"/chats/{_quote(chat_id)}/typing")

    async def mark_read(self, chat_id: str) -> bool:
        return await self._fire_and_forget("POST", f"/chats/{_quote(chat_id)}/read")

    async def send_reaction(
        self,
        message_id: str,
        reaction_type: str,
        *,
        operation: str = "add",
    ) -> bool:
        if reaction_type not in REACTION_TYPES:
            raise ValueError(f"unknown reaction type: {reaction_type}")
        return await self._fire_and_forget(
            "POST",
            f"/messages/{_quote(message_id)}/reactions",
            json_body={"operation": operation, "type": reaction_type},
        )

    async def list_phone_numbers(self, *, timeout: Optional[float] = None) -> List[str]:
        """Return the account's Linq phone numbers (used by the connectivity probe)."""
        resp = await self._client.request(
            "GET",
            f"{self._api_base}/phonenumbers",
            headers=self._headers(json_body=False),
            timeout=timeout or self._timeout,
        )
        if not resp.is_success:
            raise LinqApiError(resp.status_code, _safe_text(resp))
        data = _safe_json(resp)
        numbers = (data or {}).get("phone_numbers") or []
        return [str(n.get("phone_number")) for n in numbers if isinstance(n, dict) and n.get("phone_number")]

    # -- side-effect helper ----------------------------------------------

    async def _fire_and_forget(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """POST/DELETE a side-effect call (typing/read/reaction); never raises."""
        try:
            resp = await self._request(method, path, json_body=json_body)
            return resp.is_success
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("[linq] %s %s failed: %s", method, path, exc)
            return False


def _quote(value: str) -> str:
    from urllib.parse import quote

    return quote(str(value), safe="")


def _safe_text(resp: "httpx.Response") -> str:
    try:
        return resp.text
    except Exception:
        return ""


def _safe_json(resp: "httpx.Response") -> Dict[str, Any]:
    try:
        data = resp.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

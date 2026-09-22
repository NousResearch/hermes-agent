"""Threema Gateway REST client (https://msgapi.threema.ch).

One thin ``httpx`` wrapper. Every endpoint authenticates with ``from`` + ``secret``
as request parameters, so this module is also the only place that must never let a
URL reach a log line: ``_safe_url`` strips the secret before anything is logged.

Status codes are mapped to messages the agent can act on, because the API returns
them as bare text: 402 means the account is out of credits (every send and every
blob upload costs one), 413 means the box exceeded 7812 bytes, and 429 is
rate-limiting.
"""

from __future__ import annotations

import binascii
import logging
import re
import time
from typing import Any, Dict, Optional, Tuple

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover - checked by check_requirements()
    httpx = None  # type: ignore[assignment]
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://msgapi.threema.ch"
MAX_BOX_BYTES = 7812  # hex-encoded cap enforced by /send_e2e
MAX_BLOB_BYTES = 50 * 1024 * 1024
PUBKEY_CACHE_TTL = 24 * 3600
IDENTITY_RE = re.compile(r"^[0-9A-Z*][0-9A-Z]{7}$")

_STATUS_HINTS = {
    400: "Bad request — the recipient ID is invalid, or this Gateway ID is not set up for end-to-end mode.",
    401: "Unauthorized — the Gateway ID or API secret is wrong.",
    402: "Out of credits — every message and every blob upload costs one credit. Top up in the Gateway panel.",
    404: "Not found — no Threema ID matches that lookup, or the blob has expired.",
    413: "Too large — the encrypted message exceeds 7812 bytes, or the blob exceeds 50 MB.",
    429: "Rate-limited by Threema. Back off and retry.",
    500: "Threema reported a temporary internal error. Retry shortly.",
}


class ThreemaAPIError(Exception):
    """An API call failed. ``status`` is the HTTP code when there was a response."""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def out_of_credits(self) -> bool:
        return self.status == 402

    @property
    def rate_limited(self) -> bool:
        return self.status == 429


def valid_identity(identity: str) -> bool:
    """8 characters; Gateway IDs start with ``*``, user IDs are alphanumeric uppercase."""
    return bool(IDENTITY_RE.match((identity or "").strip().upper()))


def _safe_url(url: str) -> str:
    """The API secret travels in the query string — never let it reach a log."""
    return re.sub(r"(secret=)[^&]*", r"\1***", str(url))


class ThreemaClient:
    """``transport`` is injectable so tests exercise the real request shapes without network."""

    def __init__(
        self,
        identity: str,
        secret: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        client: Optional[Any] = None,
    ) -> None:
        self.identity = (identity or "").strip()
        self._secret = (secret or "").strip()
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None
        self._pubkeys: Dict[str, Tuple[bytes, float]] = {}

    # -- lifecycle ----------------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is None:
            if not HTTPX_AVAILABLE:
                raise ThreemaAPIError("httpx is not installed")
            try:
                from gateway.platforms._http_client_limits import platform_httpx_limits
                limits = platform_httpx_limits()
            except Exception:  # pragma: no cover - defensive, plugin may run standalone
                limits = None
            self._client = httpx.AsyncClient(timeout=self._timeout, limits=limits) if limits else httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None

    # -- transport ----------------------------------------------------------

    @property
    def _auth(self) -> Dict[str, str]:
        return {"from": self.identity, "secret": self._secret}

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        client = self._ensure_client()
        url = f"{self.base_url}{path}"
        try:
            response = await client.request(method, url, **kwargs)
        except Exception as exc:  # httpx.TransportError and friends
            raise ThreemaAPIError(f"Cannot reach {_safe_url(url)}: {exc}") from exc
        if response.status_code != 200:
            hint = _STATUS_HINTS.get(response.status_code, "")
            body = (response.text or "").strip()[:200]
            detail = f"HTTP {response.status_code} for {_safe_url(url)}"
            raise ThreemaAPIError(f"{detail}. {hint} {body}".strip(), status=response.status_code)
        return response

    # -- endpoints ----------------------------------------------------------

    async def send_e2e(
        self,
        to: str,
        nonce: bytes,
        box: bytes,
        *,
        no_delivery_receipts: bool = False,
        no_push: bool = False,
    ) -> str:
        """POST /send_e2e — nonce and box go up hex-encoded. Returns the message id."""
        if len(box) > MAX_BOX_BYTES:
            raise ThreemaAPIError(f"Encrypted message is {len(box)} bytes, over the {MAX_BOX_BYTES}-byte limit", status=413)
        data = dict(self._auth, to=to.strip().upper(),
                    nonce=binascii.hexlify(nonce).decode("ascii"),
                    box=binascii.hexlify(box).decode("ascii"))
        if no_delivery_receipts:
            data["noDeliveryReceipts"] = "1"
        if no_push:
            data["noPush"] = "1"
        response = await self._request("POST", "/send_e2e", data=data)
        return (response.text or "").strip()

    async def upload_blob(self, data: bytes) -> str:
        """POST /upload_blob — auth in the query string, blob as multipart. Costs one credit."""
        if len(data) > MAX_BLOB_BYTES:
            raise ThreemaAPIError(f"Blob is {len(data)} bytes, over the 50 MB limit", status=413)
        response = await self._request(
            "POST", "/upload_blob", params=self._auth, files={"blob": ("blob", data, "application/octet-stream")},
        )
        return (response.text or "").strip()

    async def download_blob(self, blob_id: str) -> bytes:
        """GET /blobs/{id}. Threema may delete the blob within an hour of the first fetch."""
        response = await self._request("GET", f"/blobs/{blob_id}", params=self._auth)
        return response.content

    async def public_key(self, identity: str, *, refresh: bool = False) -> bytes:
        """GET /pubkeys/{id}, cached for a day — Threema explicitly asks callers to cache."""
        key = identity.strip().upper()
        cached = self._pubkeys.get(key)
        if cached and not refresh and cached[1] > time.time():
            return cached[0]
        response = await self._request("GET", f"/pubkeys/{key}", params=self._auth)
        try:
            raw = binascii.unhexlify((response.text or "").strip())
        except (binascii.Error, ValueError) as exc:
            raise ThreemaAPIError(f"Public key for {key} is not valid hex") from exc
        self._pubkeys[key] = (raw, time.time() + PUBKEY_CACHE_TTL)
        return raw

    def cache_public_key(self, identity: str, key: bytes) -> None:
        self._pubkeys[identity.strip().upper()] = (key, time.time() + PUBKEY_CACHE_TTL)

    async def capabilities(self, identity: str) -> set:
        """GET /capabilities/{id} → {'text', 'image', 'file', ...}."""
        response = await self._request("GET", f"/capabilities/{identity.strip().upper()}", params=self._auth)
        return {part.strip() for part in (response.text or "").split(",") if part.strip()}

    async def credits(self) -> Optional[int]:
        """GET /credits. Returns None when the balance is not an integer (unlimited plans)."""
        response = await self._request("GET", "/credits", params=self._auth)
        text = (response.text or "").strip()
        try:
            return int(text)
        except ValueError:
            logger.debug("[Threema] Non-numeric credit balance %r", text)
            return None

    async def lookup_phone(self, phone: str) -> str:
        """GET /lookup/phone/{e164} — digits only, no leading +."""
        digits = re.sub(r"\D", "", phone or "")
        response = await self._request("GET", f"/lookup/phone/{digits}", params=self._auth)
        return (response.text or "").strip()

    async def lookup_email(self, email: str) -> str:
        response = await self._request("GET", f"/lookup/email/{(email or '').strip()}", params=self._auth)
        return (response.text or "").strip()

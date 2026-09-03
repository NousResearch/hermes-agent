"""Bounded, SSRF-safe outbound image fetching for Discord."""

from __future__ import annotations

import inspect
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple
from urllib.parse import urljoin

from tools.url_safety import async_is_safe_url, create_ssrf_safe_async_client


_DISCORD_IMAGE_DOWNLOAD_MAX_BYTES = 50 * 1024 * 1024  # generous limit for images/animations
_DISCORD_IMAGE_BATCH_DOWNLOAD_MAX_BYTES = 100 * 1024 * 1024
_DISCORD_IMAGE_DECODED_READ_CHUNK_MAX_BYTES = 64 * 1024
_DISCORD_IMAGE_DOWNLOAD_BUDGET_CONTEXT = ContextVar(
    "discord_image_download_budget",
    default=None,
)
_DISCORD_IMAGE_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_DISCORD_IMAGE_MAX_REDIRECTS = 10


class _DiscordImageDownloadBudget:
    """Track remote response bytes across one outbound image batch."""

    def __init__(self, limit_bytes: int):
        self.limit_bytes = limit_bytes
        self.bytes_read = 0

    def _raise_exhausted(self, detail: str = "") -> None:
        suffix = f" ({detail})" if detail else ""
        raise ValueError(
            f"Cumulative image response body exceeded {self.limit_bytes} bytes{suffix}"
        )

    def check_response(self, declared_length: Optional[int]) -> None:
        """Reject a response that cannot fit without consuming its iterator."""
        if self.bytes_read >= self.limit_bytes:
            self._raise_exhausted()
        if (
            declared_length is not None
            and declared_length > self.limit_bytes - self.bytes_read
        ):
            # The body was not read, so leave cumulative accounting unchanged.
            self._raise_exhausted(f"{declared_length} bytes declared")

    def account(self, byte_count: int) -> None:
        """Account a streamed chunk before callers can retain or upload it."""
        if byte_count <= 0:
            return
        if self.bytes_read >= self.limit_bytes:
            self._raise_exhausted()
        self.bytes_read += byte_count
        if self.bytes_read > self.limit_bytes:
            self._raise_exhausted(f"{self.bytes_read} bytes read")


async def _read_response_bytes_bounded(
    resp: Any,
    limit_bytes: int,
    *,
    download_budget: Any = None,
) -> bytes:
    """Read an httpx streaming response body with an aggregate byte limit."""
    headers = getattr(resp, "headers", {}) or {}
    declared_length = None
    for header_name in ("Content-Length", "content-length"):
        try:
            raw_length = headers.get(header_name)
        except (AttributeError, TypeError):
            raw_length = None
        if raw_length is not None:
            try:
                if isinstance(raw_length, (str, bytes, bytearray, int)):
                    declared_length = int(raw_length)
            except (TypeError, ValueError):
                pass
            break

    async def close_response() -> None:
        for method_name in ("aclose", "close", "release"):
            method = getattr(resp, method_name, None)
            if not callable(method):
                continue
            try:
                result = method()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                continue
            break

    if declared_length is not None and declared_length > limit_bytes:
        await close_response()
        raise ValueError(
            f"Response body exceeded {limit_bytes} bytes "
            f"({declared_length} bytes declared)"
        )

    if download_budget is not None:
        try:
            download_budget.check_response(declared_length)
        except ValueError:
            await close_response()
            raise

    chunks = []
    total_bytes = 0
    decoded_read_chunk_size = min(
        _DISCORD_IMAGE_DECODED_READ_CHUNK_MAX_BYTES,
        limit_bytes,
    )
    if download_budget is not None:
        remaining_budget = download_budget.limit_bytes - download_budget.bytes_read
        decoded_read_chunk_size = min(decoded_read_chunk_size, remaining_budget)

    async for chunk in resp.aiter_bytes(chunk_size=decoded_read_chunk_size):
        if download_budget is not None:
            try:
                download_budget.account(len(chunk))
            except ValueError:
                await close_response()
                raise
        total_bytes += len(chunk)
        if total_bytes > limit_bytes:
            await close_response()
            raise ValueError(
                f"Response body exceeded {limit_bytes} bytes "
                f"({total_bytes} bytes read)"
            )
        chunks.append(chunk)

    return b"".join(chunks)


async def _read_url_image_with_redirect_guard(
    client: Any,
    url: str,
    *,
    timeout: Any,
    request_kwargs: Dict[str, Any],
    download_budget: Any = None,
    async_is_safe_url_fn: Optional[Callable[[str], Awaitable[bool]]] = None,
    max_bytes: Optional[int] = None,
    read_response_bytes_fn: Optional[Callable[..., Any]] = None,
    redirect_statuses: Optional[set[int]] = None,
    max_redirects: Optional[int] = None,
) -> Tuple[int, bytes, Dict[str, str]]:
    """Read an image URL while re-checking every redirect target for SSRF."""
    if async_is_safe_url_fn is None:
        async_is_safe_url_fn = async_is_safe_url
    if max_bytes is None:
        max_bytes = _DISCORD_IMAGE_DOWNLOAD_MAX_BYTES
    if read_response_bytes_fn is None:
        read_response_bytes_fn = _read_response_bytes_bounded
    if redirect_statuses is None:
        redirect_statuses = _DISCORD_IMAGE_REDIRECT_STATUSES
    if max_redirects is None:
        max_redirects = _DISCORD_IMAGE_MAX_REDIRECTS

    current_url = url
    for _ in range(max_redirects + 1):
        if not await async_is_safe_url_fn(current_url):
            raise ValueError("Blocked unsafe image URL redirect")

        async with client.stream(
            "GET",
            current_url,
            timeout=timeout,
            follow_redirects=False,
            **request_kwargs,
        ) as resp:
            raw_headers = getattr(resp, "headers", {}) or {}
            headers = {str(key).lower(): value for key, value in dict(raw_headers).items()}
            status = int(getattr(resp, "status_code", getattr(resp, "status", 0)))
            if status in redirect_statuses:
                # Redirect bodies are intentionally not drained; the stream context closes them,
                # and the budget counts only bytes actually read.
                location = headers.get("location")
                if not location:
                    return status, b"", headers
                next_url = urljoin(current_url, str(location))
                if not await async_is_safe_url_fn(next_url):
                    raise ValueError("Blocked redirect to private/internal address")
                current_url = next_url
                continue

            return status, await read_response_bytes_fn(
                resp,
                max_bytes,
                download_budget=download_budget,
            ), headers

    raise ValueError("Too many image URL redirects")


def _discord_image_extension_from_bytes(data: bytes) -> Optional[str]:
    """Return the supported image extension based on the body magic bytes."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def _create_discord_image_http_client(
    proxy_url: Optional[str] = None,
    *,
    client_factory: Optional[Callable[..., Any]] = None,
) -> Any:
    """Create the SSRF-safe client used for outbound Discord image fetches."""
    if client_factory is None:
        client_factory = create_ssrf_safe_async_client

    client_kwargs: Dict[str, Any] = {
        "timeout": 30.0,
        "follow_redirects": False,
        # ``resolve_proxy_url`` explicitly applies DISCORD_PROXY, generic
        # proxy variables, and the macOS system proxy.  Avoid httpx selecting
        # a second proxy policy from the environment behind its back.
        "trust_env": False,
    }
    if proxy_url:
        client_kwargs["proxy"] = proxy_url
    return client_factory(**client_kwargs)

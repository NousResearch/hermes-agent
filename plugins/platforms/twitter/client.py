from __future__ import annotations

import base64
import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from .queue import OperationNotStartedError, RateQueue

API_BASE = "https://api.x.com"
MAX_METRIC_POST_IDS = 20


class XApiError(RuntimeError):
    def __init__(
        self, status: int, endpoint: str, detail: str, *, retry_after: float = 0
    ):
        endpoint = _normalized_endpoint(endpoint)
        suffix = f": {detail[:200]}" if detail else ""
        super().__init__(f"X API {status} on {endpoint}{suffix}")
        self.status = status
        self.endpoint = endpoint
        self.retry_after = retry_after


class AmbiguousWriteError(RuntimeError):
    pass


def _normalized_endpoint(endpoint: str) -> str:
    path = str(endpoint).split("?", 1)[0].split("#", 1)[0]
    if not path.startswith("/") or not re.fullmatch(r"/[A-Za-z0-9_./:-]*", path):
        return "/unknown"
    parts = path.split("/")
    for index in range(2, len(parts)):
        if re.fullmatch(r"[0-9]+(?:-[0-9]+)*", parts[index]):
            parts[index] = ":id"
    return "/".join(parts)[:200]


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    errors = payload.get("errors")
    error = errors[0] if isinstance(errors, list) and errors else None
    source = error if isinstance(error, dict) else payload
    fields = []
    code = source.get("code")
    if isinstance(code, (str, int)) and not isinstance(code, bool):
        code = str(code).strip()
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", code):
            fields.append(f"code={code}")
    title = source.get("title")
    if isinstance(title, str):
        title = " ".join(title.split())
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._:/()'-]{0,79}", title):
            fields.append(f"title={title}")
    return "; ".join(fields)


def _write_id(payload: dict[str, Any], field: str, endpoint: str) -> str:
    data = payload.get("data")
    value = data.get(field) if isinstance(data, dict) else ""
    value = str(value or "")
    if not value.isascii() or not value.isdigit():
        raise AmbiguousWriteError(
            f"X delivery outcome is uncertain because {endpoint} omitted a valid {field}"
        )
    return value


class XClient:
    def __init__(
        self,
        *,
        token: str,
        token_provider: Callable[[], Awaitable[str]] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        max_pending: int = 100,
        max_wait_seconds: float = 900,
    ):
        self._client = httpx.AsyncClient(
            base_url=API_BASE,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
            transport=transport,
        )
        self._token_provider = token_provider
        self._queue = RateQueue(
            max_pending=max_pending, max_wait_seconds=max_wait_seconds
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        bucket: str = "read",
        ambiguous_write: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        async def execute() -> dict[str, Any]:
            for attempt in range(2):
                if self._token_provider is not None:
                    try:
                        token = await self._token_provider()
                    except asyncio.CancelledError as exc:
                        if ambiguous_write:
                            raise OperationNotStartedError() from exc
                        raise
                    self._client.headers["Authorization"] = f"Bearer {token}"
                try:
                    response = await self._client.request(method, path, **kwargs)
                except httpx.RequestError as exc:
                    if ambiguous_write:
                        raise AmbiguousWriteError(
                            "X delivery outcome is uncertain for "
                            f"{_normalized_endpoint(path)}"
                        ) from exc
                    if method.upper() == "GET" and attempt == 0:
                        await asyncio.sleep(0.1)
                        continue
                    raise
                retry_after = _retry_delay(response)
                if response.status_code == 429 and attempt == 0:
                    await asyncio.sleep(retry_after)
                    continue
                if response.is_error:
                    if ambiguous_write and (
                        response.status_code == 408 or response.status_code >= 500
                    ):
                        raise AmbiguousWriteError(
                            "X delivery outcome is uncertain for "
                            f"{_normalized_endpoint(path)}"
                        )
                    raise XApiError(
                        response.status_code,
                        path,
                        _error_detail(response),
                        retry_after=retry_after,
                    )
                if not response.content:
                    return {}
                try:
                    payload = response.json()
                except ValueError as exc:
                    if ambiguous_write:
                        raise AmbiguousWriteError(
                            "X delivery outcome is uncertain for "
                            f"{_normalized_endpoint(path)}"
                        ) from exc
                    raise
                return payload if isinstance(payload, dict) else {}
            raise XApiError(429, path, "rate limit retry exhausted")

        return await self._queue.run(bucket, execute)

    async def identity(self) -> dict[str, Any]:
        return await self.request("GET", "/2/users/me", params={"user.fields": "username"})

    async def mentions(
        self, user_id: str, *, since_id: str = "", pagination_token: str = ""
    ) -> dict[str, Any]:
        params = {
            "max_results": "100",
            "tweet.fields": (
                "attachments,author_id,conversation_id,created_at,entities,"
                "in_reply_to_user_id,public_metrics,referenced_tweets"
            ),
            "expansions": (
                "attachments.media_keys,author_id,entities.mentions.username,"
                "referenced_tweets.id,referenced_tweets.id.author_id"
            ),
            "media.fields": "alt_text,height,media_key,preview_image_url,type,url,width",
            "user.fields": "created_at,description,id,location,name,public_metrics,username,verified",
        }
        if since_id:
            params["since_id"] = str(since_id)
        if pagination_token:
            params["pagination_token"] = pagination_token
        return await self.request(
            "GET",
            f"/2/users/{quote(str(user_id), safe='')}/mentions",
            bucket="mentions",
            params=params,
        )

    async def dm_events(self, *, pagination_token: str = "") -> dict[str, Any]:
        params = {
            "max_results": "100",
            "event_types": "MessageCreate",
            "dm_event.fields": (
                "id,sender_id,text,created_at,dm_conversation_id,attachments,"
                "participant_ids"
            ),
            "expansions": "sender_id,attachments.media_keys",
            "user.fields": "id,name,username",
            "media.fields": "alt_text,height,media_key,preview_image_url,type,url,width",
        }
        if pagination_token:
            params["pagination_token"] = pagination_token
        return await self.request(
            "GET", "/2/dm_events", bucket="direct_messages", params=params
        )

    async def create_post(
        self,
        text: str,
        *,
        reply_to: str | None = None,
        media_ids: list[str] | None = None,
    ) -> str:
        body: dict[str, Any] = {"text": text}
        if reply_to:
            body["reply"] = {"in_reply_to_tweet_id": str(reply_to)}
        if media_ids:
            body["media"] = {"media_ids": list(map(str, media_ids))}
        payload = await self.request(
            "POST",
            "/2/tweets",
            bucket="public_writes",
            ambiguous_write=True,
            json=body,
        )
        return _write_id(payload, "id", "the post response")

    async def send_dm(
        self, conversation_id: str, text: str, *, media_id: str | None = None
    ) -> str:
        body: dict[str, Any] = {"text": text}
        if media_id:
            body["attachments"] = [{"media_id": str(media_id)}]
        path = f"/2/dm_conversations/{quote(str(conversation_id), safe='')}/messages"
        payload = await self.request(
            "POST", path, bucket="dm_writes", ambiguous_write=True, json=body
        )
        return _write_id(payload, "dm_event_id", "the DM response")

    async def conversation_posts(self, conversation_id: str) -> dict[str, Any]:
        return await self.request(
            "GET",
            "/2/tweets/search/recent",
            bucket="enrichment",
            params={
                "query": f"conversation_id:{conversation_id}",
                "max_results": "100",
                "tweet.fields": "author_id,conversation_id,created_at,public_metrics,referenced_tweets",
                "expansions": "author_id",
                "user.fields": "id,name,username",
            },
        )

    async def quote_posts(self, post_id: str, *, limit: int = 5) -> dict[str, Any]:
        limit = max(1, min(int(limit), 100))
        return await self.request(
            "GET",
            f"/2/tweets/{quote(str(post_id), safe='')}/quote_tweets",
            bucket="enrichment",
            params={
                "max_results": str(max(10, limit)),
                "tweet.fields": "author_id,conversation_id,created_at,entities,referenced_tweets",
                "expansions": "author_id",
                "user.fields": "id,name,username",
            },
        )

    async def lookup_posts(self, ids: list[str]) -> dict[str, Any]:
        return await self.request(
            "GET",
            "/2/tweets",
            bucket="enrichment",
            params={
                "ids": ",".join(map(str, ids)),
                "tweet.fields": "author_id,conversation_id,created_at,entities,referenced_tweets",
                "expansions": "author_id",
                "user.fields": "id,name,username",
            },
        )

    async def bookmarks(
        self, user_id: str, operation: str, *, post_id: str = ""
    ) -> dict[str, Any]:
        path = f"/2/users/{quote(str(user_id), safe='')}/bookmarks"
        if operation == "list":
            return await self.request("GET", path, bucket="tools")
        if operation == "add":
            return await self.request(
                "POST", path, bucket="tools", json={"tweet_id": str(post_id)}
            )
        if operation == "remove":
            return await self.request(
                "DELETE",
                f"{path}/{quote(str(post_id), safe='')}",
                bucket="tools",
            )
        raise ValueError("bookmark operation must be list, add, or remove")

    async def post_metrics(self, ids: list[str]) -> dict[str, Any]:
        if not all(isinstance(item, str) for item in ids):
            raise ValueError("post metrics accepts string X post IDs")
        normalized = list(ids)
        if not 1 <= len(normalized) <= MAX_METRIC_POST_IDS or any(
            not item.isascii() or not item.isdigit() for item in normalized
        ):
            raise ValueError(
                f"post metrics accepts 1 to {MAX_METRIC_POST_IDS} numeric X post IDs"
            )
        return await self.request(
            "GET",
            "/2/tweets",
            bucket="tools",
            params={"ids": ",".join(normalized), "tweet.fields": "public_metrics,non_public_metrics"},
        )

    async def upload_image(self, path: Path, *, for_dm: bool = False) -> str:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        payload = await self.request(
            "POST",
            "/2/media/upload",
            bucket="media_writes",
            json={
                "media": encoded,
                "media_category": "dm_image" if for_dm else "tweet_image",
                "media_type": _media_type(path),
                "shared": False,
            },
        )
        data = payload.get("data")
        media_id = str(data.get("id") or "") if isinstance(data, dict) else ""
        if not media_id.isascii() or not media_id.isdigit():
            raise XApiError(502, "/2/media/upload", "response omitted media id")
        return media_id


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    try:
        return types[suffix]
    except KeyError as exc:
        raise ValueError("Twitter supports JPG, PNG, and WEBP images") from exc


def _retry_delay(response: httpx.Response) -> float:
    raw_retry = response.headers.get("Retry-After")
    if raw_retry:
        try:
            return max(0.0, min(float(raw_retry), 900.0))
        except ValueError:
            pass
    raw_reset = response.headers.get("x-rate-limit-reset")
    if raw_reset:
        try:
            return max(0.0, min(float(raw_reset) - time.time(), 900.0))
        except ValueError:
            pass
    return 1.0 if response.status_code == 429 else 0.0

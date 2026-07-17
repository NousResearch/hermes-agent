from __future__ import annotations

import base64
import asyncio
import time
from pathlib import Path
from typing import Any

import httpx

from .queue import RateQueue

API_BASE = "https://api.x.com"


class XApiError(RuntimeError):
    def __init__(
        self, status: int, endpoint: str, detail: str, *, retry_after: float = 0
    ):
        super().__init__(f"X API {status} on {endpoint}: {detail[:200]}")
        self.status = status
        self.endpoint = endpoint
        self.retry_after = retry_after


class AmbiguousWriteError(RuntimeError):
    pass


class XClient:
    def __init__(
        self,
        *,
        token: str,
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
                try:
                    response = await self._client.request(method, path, **kwargs)
                except httpx.RequestError as exc:
                    if ambiguous_write:
                        raise AmbiguousWriteError(
                            f"X delivery outcome is uncertain for {path}"
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
                    detail = ""
                    try:
                        payload = response.json()
                        error = (payload.get("errors") or [{}])[0]
                        detail = str(
                            error.get("detail")
                            or error.get("title")
                            or payload.get("detail")
                            or ""
                        )
                    except (TypeError, ValueError):
                        detail = response.text
                    if ambiguous_write and response.status_code >= 500:
                        raise AmbiguousWriteError(
                            f"X delivery outcome is uncertain for {path}"
                        )
                    raise XApiError(
                        response.status_code,
                        path,
                        detail,
                        retry_after=retry_after,
                    )
                if not response.content:
                    return {}
                payload = response.json()
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
        return await self.request("GET", f"/2/users/{user_id}/mentions", params=params)

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
        return await self.request("GET", "/2/dm_events", params=params)

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
            bucket="write_post",
            ambiguous_write=True,
            json=body,
        )
        post_id = (payload.get("data") or {}).get("id")
        if not post_id:
            raise XApiError(502, "/2/tweets", "response omitted post id")
        return str(post_id)

    async def send_dm(
        self, conversation_id: str, text: str, *, media_id: str | None = None
    ) -> str:
        body: dict[str, Any] = {"text": text}
        if media_id:
            body["attachments"] = [{"media_id": str(media_id)}]
        path = f"/2/dm_conversations/{conversation_id}/messages"
        payload = await self.request(
            "POST", path, bucket="write_dm", ambiguous_write=True, json=body
        )
        event_id = (payload.get("data") or {}).get("dm_event_id")
        if not event_id:
            raise XApiError(502, path, "response omitted DM event id")
        return str(event_id)

    async def conversation_posts(self, conversation_id: str) -> dict[str, Any]:
        return await self.request(
            "GET",
            "/2/tweets/search/recent",
            bucket="optional_enrichment",
            params={
                "query": f"conversation_id:{conversation_id}",
                "max_results": "100",
                "tweet.fields": "author_id,conversation_id,created_at,public_metrics,referenced_tweets",
                "expansions": "author_id",
                "user.fields": "id,name,username",
            },
        )

    async def lookup_posts(self, ids: list[str]) -> dict[str, Any]:
        return await self.request(
            "GET", "/2/tweets", params={"ids": ",".join(map(str, ids))}
        )

    async def bookmarks(
        self, user_id: str, operation: str, *, post_id: str = ""
    ) -> dict[str, Any]:
        path = f"/2/users/{user_id}/bookmarks"
        if operation == "list":
            return await self.request("GET", path)
        if operation == "add":
            return await self.request("POST", path, json={"tweet_id": str(post_id)})
        if operation == "remove":
            return await self.request("DELETE", f"{path}/{post_id}")
        raise ValueError("bookmark operation must be list, add, or remove")

    async def post_metrics(self, ids: list[str]) -> dict[str, Any]:
        return await self.request(
            "GET",
            "/2/tweets",
            params={"ids": ",".join(map(str, ids)), "tweet.fields": "public_metrics,non_public_metrics"},
        )

    async def upload_image(self, path: Path, *, for_dm: bool = False) -> str:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        payload = await self.request(
            "POST",
            "/2/media/upload",
            bucket="media",
            json={
                "media": encoded,
                "media_category": "dm_image" if for_dm else "tweet_image",
                "media_type": _media_type(path),
                "shared": False,
            },
        )
        media_id = (payload.get("data") or {}).get("id")
        if not media_id:
            raise XApiError(502, "/2/media/upload", "response omitted media id")
        return str(media_id)


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

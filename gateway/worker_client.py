"""Front→worker turn dispatch over the worker's api_server HTTP+SSE (Tier-2).

Reuses the existing api_server contract (design §6): ``POST /v1/runs`` starts a
run; ``GET /v1/runs/{run_id}/events`` streams ``message.delta`` / ``approval.request``
/ ``run.completed``; ``POST /v1/runs/{run_id}/approval`` resolves an approval.
Text deltas feed the caller's stream consumer; approvals round-trip through the
caller's handler. All requests bind loopback and carry the per-worker bearer.

HTTP (post + SSE) is injected so the relay logic is tested without a server.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Awaitable, Callable, Optional, Protocol

logger = logging.getLogger(__name__)


class StreamConsumer(Protocol):
    def on_delta(self, text: str) -> None: ...


class WorkerRunError(RuntimeError):
    """The worker reported run.failed."""


class WorkerClient:
    def __init__(
        self,
        base_url: str,
        key: str,
        *,
        post: Callable[[str, dict], Awaitable[dict]] | None = None,
        sse: Callable[[str], AsyncIterator[dict]] | None = None,
        delete: Callable[[str], Awaitable[None]] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.key = key
        self._post = post or self._aiohttp_post
        self._sse = sse or self._aiohttp_sse
        self._delete = delete or self._aiohttp_delete

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.key}"}

    async def dispatch(
        self,
        *,
        input: str,
        consumer: StreamConsumer,
        instructions: Optional[str] = None,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        media_refs: list[dict] | None = None,
        approval_handler: Callable[[dict], Awaitable[str]] | None = None,
        media_handler: Callable[[dict], Awaitable[None]] | None = None,
        continue_session: bool = False,
    ) -> dict:
        """Run one turn on the worker; relay deltas, handle approvals, return the terminal event.

        ``continue_session`` asks the worker to rehydrate its own transcript for
        ``session_id`` from its state.db, so a routed conversation keeps memory
        across turns (the front never holds the worker's history).
        """
        body = {k: v for k, v in {
            "input": input,
            "instructions": instructions,
            "session_id": session_id,
            "model": model,
            "media_refs": media_refs or None,
            "continue_session": True if (continue_session and session_id) else None,
        }.items() if v is not None}
        started = await self._post(f"{self.base_url}/v1/runs", body)
        run_id = started.get("run_id")

        async for event in self._sse(f"{self.base_url}/v1/runs/{run_id}/events"):
            name = event.get("event")
            if name == "message.delta":
                consumer.on_delta(event.get("delta", ""))
            elif name == "response.media":
                if media_handler:
                    await media_handler(event)
            elif name == "approval.request":
                choice = await approval_handler(event) if approval_handler else "deny"
                await self._post(f"{self.base_url}/v1/runs/{run_id}/approval", {"choice": choice})
            elif name == "run.completed":
                return event
            elif name == "run.cancelled":
                return event
            elif name == "run.failed":
                raise WorkerRunError(event.get("error") or "worker run failed")
        raise WorkerRunError("worker event stream ended without a terminal event")

    async def reset_session(self, session_id: str) -> None:
        """Clear the worker's session (forwarded /new or /reset).

        Scopes the reset to the routed profile's own state.db — the host
        profile's sessions are never touched.  Idempotent: a 404 (no session
        yet — e.g. /new before the first routed turn) is treated as success so
        the user sees a clean reset, not a scary "could not reach profile".
        """
        try:
            await self._delete(f"{self.base_url}/api/sessions/{session_id}")
        except Exception as e:
            if getattr(e, "status", None) == 404:
                return
            raise

    async def _aiohttp_post(self, url: str, body: dict) -> dict:
        import aiohttp

        async with aiohttp.ClientSession(headers=self._headers) as s:
            async with s.post(url, json=body) as r:
                r.raise_for_status()
                return await r.json()

    async def _aiohttp_delete(self, url: str) -> None:
        import aiohttp

        async with aiohttp.ClientSession(headers=self._headers) as s:
            async with s.delete(url) as r:
                if r.status == 404:
                    return  # idempotent delete — already absent
                r.raise_for_status()

    async def _aiohttp_sse(self, url: str) -> AsyncIterator[dict]:
        import aiohttp

        async with aiohttp.ClientSession(headers=self._headers) as s:
            async with s.get(url) as r:
                r.raise_for_status()
                async for raw in r.content:
                    line = raw.decode("utf-8").strip()
                    if line.startswith("data:"):
                        try:
                            yield json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            logger.debug("skipping non-JSON SSE data line")

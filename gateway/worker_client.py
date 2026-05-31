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
    ):
        self.base_url = base_url.rstrip("/")
        self.key = key
        self._post = post or self._aiohttp_post
        self._sse = sse or self._aiohttp_sse

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
        approval_handler: Callable[[dict], Awaitable[str]] | None = None,
    ) -> dict:
        """Run one turn on the worker; relay deltas, handle approvals, return the terminal event."""
        body = {k: v for k, v in {
            "input": input,
            "instructions": instructions,
            "session_id": session_id,
            "model": model,
        }.items() if v is not None}
        started = await self._post(f"{self.base_url}/v1/runs", body)
        run_id = started.get("run_id")

        async for event in self._sse(f"{self.base_url}/v1/runs/{run_id}/events"):
            name = event.get("event")
            if name == "message.delta":
                consumer.on_delta(event.get("delta", ""))
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

    async def _aiohttp_post(self, url: str, body: dict) -> dict:
        import aiohttp

        async with aiohttp.ClientSession(headers=self._headers) as s:
            async with s.post(url, json=body) as r:
                r.raise_for_status()
                return await r.json()

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

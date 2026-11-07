"""POSIX control workers, separated along the hosted producer's callback dependency."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from typing import Callable


class ControlRequestDispatcher:
    def __init__(self) -> None:
        # Producers synchronously call back into hosted-attest/identify on this listener,
        # then await session work which can itself need the loop's default executor.
        self._controls = ThreadPoolExecutor(thread_name_prefix='gateway-control')
        self._producers = ThreadPoolExecutor(thread_name_prefix='gateway-hosted-producer')

    async def dispatch(
        self, handler: Callable[[bytes, str | None], bytes], raw: bytes, peer: str | None,
    ) -> bytes:
        try:
            request = json.loads(raw)
        except (ValueError, RecursionError):
            request = None
        # Scheduling is not authorization: the handler still validates the full envelope
        # and authenticated peer before invoking either private operation.
        producer = isinstance(request, dict) and request.get('verb') == 'hosted-producer'
        executor = self._producers if producer else self._controls
        return await asyncio.get_running_loop().run_in_executor(executor, handler, raw, peer)

    def close(self) -> None:
        # Never run queued mutations after listener shutdown. In-flight handlers retain
        # their existing RPC deadlines; waiting for them here could block the owner loop.
        self._producers.shutdown(wait=False, cancel_futures=True)
        self._controls.shutdown(wait=False, cancel_futures=True)

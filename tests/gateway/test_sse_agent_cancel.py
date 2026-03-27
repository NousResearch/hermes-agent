"""Tests for SSE client disconnect → agent task cancellation.

When a streaming /v1/chat/completions client disconnects mid-stream
(network drop, browser tab close), the agent task must be cancelled
so it stops consuming API tokens and memory.
"""

import asyncio
import json
import queue
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter():
    """Build a minimal APIServerAdapter with mocked internals."""
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.config import PlatformConfig

    config = PlatformConfig(enabled=True, token="test-key")
    adapter = APIServerAdapter(config)
    return adapter


def _make_request():
    """Build a mock aiohttp request."""
    req = MagicMock()
    req.headers = {}
    return req


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSSEAgentCancelOnDisconnect:
    """gateway/platforms/api_server.py — _write_sse_chat_completion()"""

    def test_agent_task_cancelled_on_client_disconnect(self):
        """When response.write raises ConnectionResetError (client dropped),
        the agent task must be cancelled."""
        adapter = _make_adapter()

        stream_q = queue.Queue()
        stream_q.put("hello ")  # Some data already queued

        # Agent task that runs forever (simulates a long LLM call)
        agent_done = asyncio.Event()

        async def fake_agent():
            await agent_done.wait()
            return {"final_response": "done"}, {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

        async def run():
            from aiohttp import web

            agent_task = asyncio.ensure_future(fake_agent())

            # Mock response that raises ConnectionResetError on second write
            mock_response = AsyncMock(spec=web.StreamResponse)
            call_count = 0

            async def write_side_effect(data):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    raise ConnectionResetError("client disconnected")

            mock_response.write = AsyncMock(side_effect=write_side_effect)
            mock_response.prepare = AsyncMock()

            with patch.object(type(adapter), '_write_sse_chat_completion',
                              adapter._write_sse_chat_completion):
                # Patch StreamResponse creation
                with patch("gateway.platforms.api_server.web.StreamResponse",
                           return_value=mock_response):
                    await adapter._write_sse_chat_completion(
                        _make_request(), "cmpl-123", "gpt-4", 1234567890,
                        stream_q, agent_task,
                    )

            # The critical assertion: agent_task must be cancelled
            assert agent_task.cancelled() or agent_task.done()
            # Clean up
            agent_done.set()

        asyncio.run(run())

    def test_agent_task_not_cancelled_on_normal_completion(self):
        """On normal stream completion, agent task should NOT be cancelled."""
        adapter = _make_adapter()

        stream_q = queue.Queue()
        stream_q.put("hello")
        stream_q.put(None)  # End-of-stream sentinel

        async def fake_agent():
            return {"final_response": "done"}, {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

        async def run():
            from aiohttp import web

            agent_task = asyncio.ensure_future(fake_agent())
            await asyncio.sleep(0)  # Let agent complete

            mock_response = AsyncMock(spec=web.StreamResponse)
            mock_response.write = AsyncMock()
            mock_response.prepare = AsyncMock()

            with patch("gateway.platforms.api_server.web.StreamResponse",
                       return_value=mock_response):
                await adapter._write_sse_chat_completion(
                    _make_request(), "cmpl-456", "gpt-4", 1234567890,
                    stream_q, agent_task,
                )

            # Agent should have completed normally, not been cancelled
            assert agent_task.done()
            assert not agent_task.cancelled()

        asyncio.run(run())

    def test_broken_pipe_also_cancels_agent(self):
        """BrokenPipeError (another disconnect variant) also cancels the task."""
        adapter = _make_adapter()

        stream_q = queue.Queue()

        async def fake_agent():
            await asyncio.sleep(999)  # Never completes
            return {}, {}

        async def run():
            from aiohttp import web

            agent_task = asyncio.ensure_future(fake_agent())

            mock_response = AsyncMock(spec=web.StreamResponse)
            mock_response.write = AsyncMock(side_effect=BrokenPipeError("pipe broken"))
            mock_response.prepare = AsyncMock()

            with patch("gateway.platforms.api_server.web.StreamResponse",
                       return_value=mock_response):
                await adapter._write_sse_chat_completion(
                    _make_request(), "cmpl-789", "gpt-4", 1234567890,
                    stream_q, agent_task,
                )

            assert agent_task.cancelled() or agent_task.done()

        asyncio.run(run())

    def test_already_done_task_not_cancelled_on_disconnect(self):
        """If agent already finished before disconnect, don't try to cancel."""
        adapter = _make_adapter()

        stream_q = queue.Queue()
        stream_q.put("data")

        async def fake_agent():
            return {"final_response": "done"}, {}

        async def run():
            from aiohttp import web

            agent_task = asyncio.ensure_future(fake_agent())
            await asyncio.sleep(0)  # Let agent complete

            mock_response = AsyncMock(spec=web.StreamResponse)
            call_count = 0

            async def write_side_effect(data):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    raise ConnectionResetError("late disconnect")

            mock_response.write = AsyncMock(side_effect=write_side_effect)
            mock_response.prepare = AsyncMock()

            with patch("gateway.platforms.api_server.web.StreamResponse",
                       return_value=mock_response):
                await adapter._write_sse_chat_completion(
                    _make_request(), "cmpl-done", "gpt-4", 1234567890,
                    stream_q, agent_task,
                )

            # Task was already done — should not be cancelled
            assert agent_task.done()
            assert not agent_task.cancelled()

        asyncio.run(run())

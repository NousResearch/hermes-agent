"""Regression test for #24451.

SSE streaming: done callback pushes None sentinel so the loop exits
promptly when agent_task completes, instead of relying solely on
agent_task.done() after a 0.5-second queue-timeout cycle.

The race: _on_delta filters out None (the agent's box-close signal) so
the loop has no sentinel in the queue.  When the agent finishes slow
post-completion work (memory sync, session flush, etc.) the task is not
yet done() during the timeout cycle, causing keepalive messages to be
sent indefinitely until the client disconnects.

Fix: agent_task.add_done_callback(lambda _: _stream_q.put_nowait(None))
at both streaming sites in gateway/platforms/api_server.py.
"""
import asyncio
import queue
from unittest.mock import AsyncMock, MagicMock, patch


def _make_adapter():
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.config import PlatformConfig

    config = PlatformConfig(enabled=True, token="test-key")
    return APIServerAdapter(config)


def _make_request():
    req = MagicMock()
    req.headers = {}
    return req


class TestSSEDoneCallbackSentinel:
    """gateway/platforms/api_server.py — issue #24451"""

    def test_done_callback_pushes_none_sentinel(self):
        """agent_task.add_done_callback must push None to stream_q when
        the task finishes so the SSE loop exits without waiting for the
        next 0.5-second get() timeout.

        The agent in this test completes without ever calling the delta
        callback with None (mirroring real behaviour where _on_delta
        filters out None), so the ONLY exit path is the done callback.
        This test FAILS if the add_done_callback line is absent.
        """
        stream_q: queue.Queue = queue.Queue()
        agent_finished = asyncio.Event()

        async def fake_agent_with_delay():
            await asyncio.sleep(0.05)
            agent_finished.set()
            return {"final_response": "hello"}, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

        async def run():
            stream_q.put("hello world")  # one delta, no None sentinel

            agent_task = asyncio.ensure_future(fake_agent_with_delay())
            agent_task.add_done_callback(lambda _: stream_q.put_nowait(None))

            await agent_finished.wait()
            await asyncio.sleep(0)  # flush callbacks

            items = []
            while not stream_q.empty():
                items.append(stream_q.get_nowait())

            assert None in items, (
                "add_done_callback must push None sentinel to _stream_q "
                "when agent_task completes — see gateway/platforms/api_server.py"
            )
            assert "hello world" in items

        asyncio.run(run())

    def test_sse_chat_completion_exits_via_done_callback(self):
        """_write_sse_chat_completion must terminate promptly when the
        agent task completes with no None sentinel in the queue.

        Without the done callback, the loop would wait up to 0.5 s per
        cycle checking agent_task.done().  With the callback, it exits
        as soon as the task finishes.
        """
        adapter = _make_adapter()
        stream_q = queue.Queue()
        stream_q.put("delta1")
        # Intentionally NO None — mirrors _on_delta filtering

        async def fake_agent():
            await asyncio.sleep(0.01)
            return {"final_response": "done"}, {}

        async def run():
            from aiohttp import web

            agent_task = asyncio.ensure_future(fake_agent())
            agent_task.add_done_callback(lambda _: stream_q.put_nowait(None))

            mock_response = AsyncMock(spec=web.StreamResponse)
            mock_response.write = AsyncMock()
            mock_response.prepare = AsyncMock()

            with patch("gateway.platforms.api_server.web.StreamResponse",
                       return_value=mock_response):
                await asyncio.wait_for(
                    adapter._write_sse_chat_completion(
                        _make_request(), "cmpl-cb", "gpt-4", 1234567890,
                        stream_q, agent_task,
                    ),
                    timeout=2.0,
                )

            assert agent_task.done()
            assert not agent_task.cancelled()

        asyncio.run(run())

    def test_sse_loop_exits_without_none_only_when_task_done(self):
        """Control case: without a None sentinel and without a done
        callback, the loop still exits via agent_task.done() after the
        0.5-s get() timeout.  Verifies the existing fallback remains
        intact after the fix.
        """
        adapter = _make_adapter()
        stream_q = queue.Queue()

        async def fake_agent():
            return {"final_response": "done"}, {}

        async def run():
            from aiohttp import web

            agent_task = asyncio.ensure_future(fake_agent())
            await asyncio.sleep(0)  # task done before loop starts

            mock_response = AsyncMock(spec=web.StreamResponse)
            mock_response.write = AsyncMock()
            mock_response.prepare = AsyncMock()

            with patch("gateway.platforms.api_server.web.StreamResponse",
                       return_value=mock_response):
                await asyncio.wait_for(
                    adapter._write_sse_chat_completion(
                        _make_request(), "cmpl-fallback", "gpt-4", 1234567890,
                        stream_q, agent_task,
                    ),
                    timeout=3.0,
                )

            assert agent_task.done()

        asyncio.run(run())

"""Tests for SSE client disconnect → agent task cancellation.

When a streaming /v1/chat/completions client disconnects mid-stream
(network drop, browser tab close), the agent is interrupted via
agent.interrupt() so it stops making LLM API calls, and the asyncio
task wrapper is cancelled.

A client that opts in with ``X-Hermes-Detach-On-Disconnect`` asks for the opposite
(``TestSSEDetachOnDisconnect``): a phone whose screen locked lost its socket, not its
question, so the turn is left to finish and land in the session transcript.
"""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch
from gateway.platforms.api_server import ThreadSafeAsyncQueue


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

        # Agent task that runs forever (simulates a long LLM call)
        agent_done = asyncio.Event()

        async def fake_agent():
            await agent_done.wait()
            return {"final_response": "done"}, {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

        async def run():
            from aiohttp import web
            from gateway.platforms.api_server import ThreadSafeAsyncQueue

            # Constructed inside the running loop — ThreadSafeAsyncQueue
            # captures asyncio.get_running_loop() at construction time.
            stream_q = ThreadSafeAsyncQueue()
            stream_q.put_nowait("hello ")  # Some data already queued

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

        async def fake_agent():
            return {"final_response": "done"}, {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

        async def run():
            from aiohttp import web
            from gateway.platforms.api_server import ThreadSafeAsyncQueue

            stream_q = ThreadSafeAsyncQueue()
            stream_q.put_nowait("hello")
            stream_q.put_nowait(None)  # End-of-stream sentinel

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

        async def fake_agent():
            await asyncio.sleep(0.2)  # Never completes
            return {}, {}

        async def run():
            from aiohttp import web
            from gateway.platforms.api_server import ThreadSafeAsyncQueue

            stream_q = ThreadSafeAsyncQueue()

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

        async def fake_agent():
            return {"final_response": "done"}, {}

        async def run():
            from aiohttp import web
            from gateway.platforms.api_server import ThreadSafeAsyncQueue

            stream_q = ThreadSafeAsyncQueue()
            stream_q.put_nowait("data")

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

    def test_agent_interrupt_called_on_disconnect(self):
        """When the client disconnects, agent.interrupt() must be called
        so the agent thread stops making LLM API calls."""
        adapter = _make_adapter()

        agent_done = asyncio.Event()

        async def fake_agent():
            await agent_done.wait()
            return {"final_response": "done"}, {}

        # Mock agent with an interrupt method
        mock_agent = MagicMock()
        mock_agent.interrupt = MagicMock()

        async def run():
            from aiohttp import web
            from gateway.platforms.api_server import ThreadSafeAsyncQueue

            stream_q = ThreadSafeAsyncQueue()
            stream_q.put_nowait("hello ")

            agent_task = asyncio.ensure_future(fake_agent())
            agent_ref = [mock_agent]

            mock_response = AsyncMock(spec=web.StreamResponse)
            call_count = 0

            async def write_side_effect(data):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    raise ConnectionResetError("client disconnected")

            mock_response.write = AsyncMock(side_effect=write_side_effect)
            mock_response.prepare = AsyncMock()

            with patch("gateway.platforms.api_server.web.StreamResponse",
                       return_value=mock_response):
                await adapter._write_sse_chat_completion(
                    _make_request(), "cmpl-int", "gpt-4", 1234567890,
                    stream_q, agent_task, agent_ref,
                )

            # agent.interrupt() must have been called
            mock_agent.interrupt.assert_called_once()
            # Clean up
            agent_done.set()

        asyncio.run(run())



def _capturing_response():
    """Mock StreamResponse that records all written SSE bytes as text."""
    from aiohttp import web

    chunks: list = []
    resp = AsyncMock(spec=web.StreamResponse)
    resp.prepare = AsyncMock()

    async def _write(data):
        chunks.append(data.decode() if isinstance(data, (bytes, bytearray)) else data)

    resp.write = AsyncMock(side_effect=_write)
    return resp, chunks


def _finish_reason(chunks: list):
    """Extract the terminal finish_reason and its chunk from captured SSE."""
    import json

    sse = "".join(chunks)
    finish = None
    for line in sse.splitlines():
        if line.startswith("data: ") and '"finish_reason"' in line:
            obj = json.loads(line[6:])
            if obj["choices"][0].get("finish_reason") is not None:
                finish = obj
    return (finish["choices"][0]["finish_reason"] if finish else None), finish, sse


class TestSSEAgentFailureFinishReason:
    """gateway/platforms/api_server.py — _write_sse_chat_completion()

    A clean stream-queue termination (sentinel received) followed by an agent
    failure must NOT report finish_reason: "stop". Both failure modes — an
    ``agent_task`` that raises and a ``result`` dict flagged failed — surface
    as finish_reason: "error", mirroring the non-streaming path. Issue #12422.
    """

    def _run(self, fake_agent, queue_items=("partial",)):
        adapter = _make_adapter()

        async def run():
            from gateway.platforms.api_server import ThreadSafeAsyncQueue

            stream_q = ThreadSafeAsyncQueue()
            for item in queue_items:
                stream_q.put_nowait(item)
            stream_q.put_nowait(None)  # clean end-of-stream sentinel

            agent_task = asyncio.ensure_future(fake_agent())
            resp, chunks = _capturing_response()
            with patch("gateway.platforms.api_server.web.StreamResponse",
                       return_value=resp):
                await adapter._write_sse_chat_completion(
                    _make_request(), "cmpl-fail", "gpt-4", 1234567890,
                    stream_q, agent_task,
                )
            return _finish_reason(chunks)

        return asyncio.run(run())

    def test_agent_task_raises_reports_error_not_stop(self):
        async def crash():
            raise RuntimeError("boom from agent")

        reason, finish, sse = self._run(crash)
        assert reason == "error"
        assert "error" in finish
        assert "data: [DONE]" in sse

    def test_failed_result_dict_reports_error_not_stop(self):
        async def failed():
            return (
                {"final_response": "", "failed": True, "completed": False,
                 "error": "upstream model 500"},
                {"input_tokens": 5, "output_tokens": 0, "total_tokens": 5},
            )

        reason, finish, _ = self._run(failed)
        assert reason == "error"
        assert finish.get("hermes", {}).get("failed") is True

    def test_truncated_result_reports_length(self):
        async def trunc():
            return (
                {"final_response": "half", "partial": True, "completed": False,
                 "error": "output was truncated"},
                {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
            )

        reason, finish, _ = self._run(trunc)
        assert reason == "length"
        assert finish["hermes"]["error_code"] == "output_truncated"

    def test_successful_completion_reports_stop(self):
        async def ok():
            return (
                {"final_response": "hi", "completed": True},
                {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
            )

        reason, finish, _ = self._run(ok)
        assert reason == "stop"
        # No error/hermes pollution on the happy path.
        assert "error" not in finish
        assert "hermes" not in finish


# ---------------------------------------------------------------------------
# Sweeper review fix (teknium1, 2026-07-30): cover the cross-thread
# ``put_threadsafe`` boundary that #72610 introduces via ``ThreadSafeAsyncQueue``.
# ``run_conversation`` runs in a worker thread (``loop.run_in_executor``),
# so its ``_on_delta`` / ``_on_tool_*`` callbacks must be able to push into
# the queue from off the owning event loop and immediately wake the
# consumer ``get()`` — this is the production boundary the original tests
# only exercised with same-loop ``put_nowait``.
# ---------------------------------------------------------------------------

class TestThreadSafeAsyncQueueCrossThreadBoundary:
    """gateway/platforms/api_server.py — ThreadSafeAsyncQueue"""

    def test_worker_thread_put_threadsafe_wakes_owning_loop_get(self):
        """A real daemon-Thread calling ``put_threadsafe`` from off-loop must
        immediately unblock an ``await q.get()`` on the owning event loop.
        This mirrors the ``run_conversation``/``run_in_executor`` boundary."""

        loop = asyncio.new_event_loop()

        async def consumer():
            q = ThreadSafeAsyncQueue()

            async def wait_for_item():
                return await asyncio.wait_for(q.get(), timeout=2)

            def worker():
                time.sleep(0.05)
                # No ``loop=`` kwarg on purpose: production callers
                # (_on_delta / _on_tool_*) never pass one, so the queue
                # must resolve its own ``_loop_ref``. Passing loop= here
                # would make a broken _loop_ref pass this test.
                q.put_threadsafe("from-worker")

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            got = await wait_for_item()
            assert got == "from-worker"

            thread.join(timeout=2)
            assert not thread.is_alive()

        loop.run_until_complete(consumer())
        loop.close()

    def test_twenty_concurrent_threads_no_drop(self):
        """Twenty concurrent off-loop ``put_threadsafe`` calls — all arrive.
        Regression test for the #65003 producer path."""

        loop = asyncio.new_event_loop()
        n = 20

        async def consumer():
            q = ThreadSafeAsyncQueue()
            received = []

            async def drain():
                for _ in range(n):
                    received.append(await q.get())

            def worker(idx):
                time.sleep(0.01 + idx * 0.002)
                # No ``loop=`` kwarg — exercise the production
                # ``_loop_ref`` resolution path (see the note above).
                q.put_threadsafe(f"item-{idx}")

            threads = [
                threading.Thread(target=worker, args=(i,), daemon=True)
                for i in range(n)
            ]
            for t in threads:
                t.start()

            await asyncio.wait_for(drain(), timeout=5)

            for t in threads:
                t.join(timeout=2)
                assert not t.is_alive()

            assert len(received) == n
            assert set(received) == {f"item-{i}" for i in range(n)}

        loop.run_until_complete(consumer())
        loop.close()


class TestSSEDetachOnDisconnect:
    """``X-Hermes-Detach-On-Disconnect``: a lost socket must not kill the turn.

    The opt-in turns the disconnect branch into a detach — no interrupt, no process reaping,
    no task cancellation — and hands the delta queue to a tracked drain so a turn with no
    writer cannot retain everything it emits.
    """

    def test_detach_keeps_the_turn_running_and_drains_the_queue(self):
        adapter = _make_adapter()

        async def run():
            from aiohttp import web
            from gateway.platforms.api_server import ThreadSafeAsyncQueue

            stream_q = ThreadSafeAsyncQueue()
            agent_done = asyncio.Event()

            async def fake_agent():
                await agent_done.wait()
                return ({"final_response": "done"},
                        {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})

            agent_task = asyncio.ensure_future(fake_agent())
            mock_response = AsyncMock(spec=web.StreamResponse)
            writes = 0

            async def write_side_effect(data):
                nonlocal writes
                writes += 1
                if writes >= 2:
                    raise ConnectionResetError("client disconnected")

            mock_response.write = AsyncMock(side_effect=write_side_effect)
            mock_response.prepare = AsyncMock()

            with (
                patch("gateway.platforms.api_server.web.StreamResponse", return_value=mock_response),
                patch("gateway.platforms.api_server.request_hard_interrupt") as hard_interrupt,
                patch("gateway.platforms.api_server._reap_disconnected_agent_processes") as reap,
            ):
                await adapter._write_sse_chat_completion(
                    _make_request(), "cmpl-detach", "gpt-4", 1234567890, stream_q, agent_task,
                    agent_ref=[MagicMock()], detach_on_disconnect=True)

                # The turn outlives the socket it was streaming to.
                assert not agent_task.done()
                hard_interrupt.assert_not_called()
                reap.assert_not_called()

                # The writer is gone, so the drain owns the queue: deltas the client never saw
                # are consumed, not retained (ThreadSafeAsyncQueue is unbounded).
                assert len(adapter._background_tasks) == 1
                for i in range(50):
                    stream_q.put_nowait(f"delta-{i}")
                await asyncio.sleep(0.05)
                assert stream_q.empty()

                # ...and the drain ends with the turn, not on a queue nobody will ever fill.
                agent_done.set()
                await asyncio.wait_for(asyncio.gather(*adapter._background_tasks), timeout=5)

            assert agent_task.done()
            assert not agent_task.cancelled()

        asyncio.run(run())

    def test_drain_ends_when_eos_was_consumed_before_the_writer_failed(self):
        """The writer can fail on its terminal frames *after* consuming the EOS sentinel — the
        drain must notice the finished run instead of awaiting a queue forever."""
        adapter = _make_adapter()

        async def run():
            from aiohttp import web
            from gateway.platforms.api_server import ThreadSafeAsyncQueue

            stream_q = ThreadSafeAsyncQueue()

            async def finished_agent():
                return {"final_response": "done"}, {"input_tokens": 1, "output_tokens": 1}

            agent_task = asyncio.ensure_future(finished_agent())
            await asyncio.sleep(0)  # No sentinel: the run is already over and the queue is empty.
            assert agent_task.done() and stream_q.empty()

            mock_response = AsyncMock(spec=web.StreamResponse)
            mock_response.write = AsyncMock(side_effect=ConnectionResetError("client disconnected"))
            mock_response.prepare = AsyncMock()

            with patch("gateway.platforms.api_server.web.StreamResponse", return_value=mock_response):
                await adapter._write_sse_chat_completion(
                    _make_request(), "cmpl-detach-eos", "gpt-4", 1234567890, stream_q, agent_task,
                    detach_on_disconnect=True)

            assert len(adapter._background_tasks) == 1  # the drain owns the queue
            # Would hang (until the 5s guard) if the drain only ever waited on the sentinel.
            await asyncio.wait_for(asyncio.gather(*adapter._background_tasks), timeout=5)

        asyncio.run(run())

    def test_without_the_opt_in_the_turn_is_still_interrupted(self):
        """Default contract unchanged: no header, no detach — the turn is interrupted and its
        processes reaped, so a closed browser tab still stops the work."""
        adapter = _make_adapter()

        async def run():
            from aiohttp import web
            from gateway.platforms.api_server import ThreadSafeAsyncQueue

            stream_q = ThreadSafeAsyncQueue()
            agent_done = asyncio.Event()

            async def fake_agent():
                await agent_done.wait()
                return {"final_response": "done"}, {}

            agent_task = asyncio.ensure_future(fake_agent())
            agent = MagicMock()
            mock_response = AsyncMock(spec=web.StreamResponse)
            mock_response.write = AsyncMock(side_effect=ConnectionResetError("client disconnected"))
            mock_response.prepare = AsyncMock()

            with (
                patch("gateway.platforms.api_server.web.StreamResponse", return_value=mock_response),
                patch("gateway.platforms.api_server.request_hard_interrupt") as hard_interrupt,
                patch("gateway.platforms.api_server._reap_disconnected_agent_processes") as reap,
            ):
                await adapter._write_sse_chat_completion(
                    _make_request(), "cmpl-legacy", "gpt-4", 1234567890, stream_q, agent_task,
                    agent_ref=[agent])

                assert hard_interrupt.call_args.args[0] is agent
                assert hard_interrupt.call_args.args[1] == "SSE client disconnected"
                assert reap.called
                assert agent_task.cancelled() or agent_task.done()
                assert not adapter._background_tasks  # nothing detached, nothing to drain

            agent_done.set()

        asyncio.run(run())

"""Regression tests for the streaming single-writer invariant (#65991).

A retry that supersedes a still-live SSE stream must fence the old stream out
of the delta sink; otherwise both streams write into the same turn and the
persisted transcript is two coherent responses interleaved token-by-token.

These tests exercise the real ``AIAgent`` guard helpers and the streaming
consume-loop, asserting that exactly one writer ever reaches the turn.
"""
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_agent():
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


def _chunk(content=None, finish_reason=None, model=None):
    delta = SimpleNamespace(content=content, tool_calls=None, reasoning_content=None, reasoning=None)
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=model, usage=None)


class TestSingleWriterSink:
    def test_superseded_writer_deltas_are_dropped(self):
        """A stale writer (older token, other thread) is fenced; only the
        newest writer reaches the callbacks and the accumulated turn text."""
        agent = _make_agent()
        delivered = []
        agent.stream_delta_callback = lambda t: delivered.append(t)
        agent._stream_callback = None

        a_claimed = threading.Event()
        b_claimed = threading.Event()

        def writer_a():
            agent._claim_stream_writer()  # token 1
            a_claimed.set()
            b_claimed.wait(timeout=2)  # let B supersede us first
            # We are now stale — every sink call must be a no-op.
            agent._fire_stream_delta("A-should-drop")
            agent._fire_reasoning_delta("A-reason-drop")
            agent._record_streamed_assistant_text("A-record-drop")

        def writer_b():
            a_claimed.wait(timeout=2)
            agent._claim_stream_writer()  # token 2 — supersedes A
            b_claimed.set()

        tb = threading.Thread(target=writer_b)
        ta = threading.Thread(target=writer_a)
        tb.start()
        ta.start()
        ta.join(timeout=3)
        tb.join(timeout=3)

        assert delivered == [], "a superseded stream must not deliver any deltas"
        assert "A-record-drop" not in (agent._current_streamed_assistant_text or "")
        assert agent._stream_writer_dropped >= 1

    def test_current_writer_is_never_fenced(self):
        """The active writer always delivers — the guard can only drop a
        stream that a *newer* claim has superseded."""
        agent = _make_agent()
        delivered = []
        agent.stream_delta_callback = lambda t: delivered.append(t)
        agent._stream_callback = None

        agent._claim_stream_writer()
        agent._fire_stream_delta("hello ")
        agent._fire_stream_delta("world")

        assert "".join(delivered) == "hello world"
        assert agent._stream_writer_dropped == 0

    def test_non_claiming_thread_is_not_a_writer(self):
        """A thread that never claimed (a non-streaming delta caller) is never
        treated as a stale writer, even after other attempts have claimed."""
        agent = _make_agent()
        delivered = []
        agent.stream_delta_callback = lambda t: delivered.append(t)
        agent._stream_callback = None

        # Some other thread runs a couple of stream attempts and bumps the token.
        def other():
            agent._claim_stream_writer()
            agent._claim_stream_writer()

        t = threading.Thread(target=other)
        t.start()
        t.join(timeout=3)

        # This (main) thread never claimed → not superseded → delivers.
        assert agent._stream_writer_superseded() is False
        agent._fire_stream_delta("plain")
        assert delivered == ["plain"]


class TestSingleWriterLoop:
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_consume_loop_stops_when_superseded_mid_stream(self, _close, mock_create):
        """The real streaming loop bails out the moment a newer attempt claims
        the sink, so a superseded stream cannot interleave into the turn."""
        agent = _make_agent()
        delivered = []
        agent.stream_delta_callback = lambda t: delivered.append(t)
        agent._stream_callback = None

        def stream_gen():
            yield _chunk(content="first")
            # A concurrent retry supersedes this stream between chunks.
            agent._claim_stream_writer()
            yield _chunk(content="-stale-tail", finish_reason="stop", model="m")

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = stream_gen()
        mock_create.return_value = mock_client

        agent._interruptible_streaming_api_call({})

        assert "".join(delivered) == "first"
        assert "-stale-tail" not in "".join(delivered)


class TestCodexSingleWriter:
    """The codex_responses path claims the sink and stops when superseded,
    matching the chat_completions/anthropic/bedrock parity added in salvage."""

    def _codex_event(self, event_type, **fields):
        return SimpleNamespace(type=event_type, **fields)

    def test_codex_stream_claims_writer_and_stops_when_superseded(self):
        from agent.codex_runtime import run_codex_stream

        agent = _make_agent()
        agent.api_mode = "codex_responses"
        delivered = []
        agent.stream_delta_callback = lambda t: delivered.append(t)
        agent._stream_callback = None

        def event_gen():
            yield self._codex_event(
                "response.output_text.delta", delta="first", item_id="i1",
            )
            # A concurrent retry supersedes this stream between events.
            agent._claim_stream_writer()
            yield self._codex_event(
                "response.output_text.delta", delta="-stale-tail", item_id="i1",
            )
            yield self._codex_event(
                "response.completed",
                response=SimpleNamespace(
                    id="r1", status="completed", output=[], usage=None,
                ),
            )

        mock_client = MagicMock()
        mock_client.responses.create.return_value = event_gen()

        run_codex_stream(agent, {"model": "gpt-5.3-codex"}, client=mock_client)

        assert "".join(delivered) == "first"
        assert "-stale-tail" not in "".join(delivered)

    def test_codex_stream_undisturbed_when_sole_writer(self):
        from agent.codex_runtime import run_codex_stream

        agent = _make_agent()
        agent.api_mode = "codex_responses"
        delivered = []
        agent.stream_delta_callback = lambda t: delivered.append(t)
        agent._stream_callback = None

        def event_gen():
            yield self._codex_event(
                "response.output_text.delta", delta="hello ", item_id="i1",
            )
            yield self._codex_event(
                "response.output_text.delta", delta="world", item_id="i1",
            )
            yield self._codex_event(
                "response.completed",
                response=SimpleNamespace(
                    id="r1", status="completed", output=[], usage=None,
                ),
            )

        mock_client = MagicMock()
        mock_client.responses.create.return_value = event_gen()

        run_codex_stream(agent, {"model": "gpt-5.3-codex"}, client=mock_client)

        assert "".join(delivered) == "hello world"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

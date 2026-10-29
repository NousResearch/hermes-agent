"""A model call that restarts after deltas were delivered withdraws them (#142333).

The primary streams "I'm sorry, I cannot help with that", the provider stops it
(``finish_reason=content_filter``) or the response is malformed, and the fallback chain
restarts the call. The fallback's deltas follow on the same reply, so a client that renders
deltas shows a refusal followed by the real answer until ``assistant.completed`` replaces it.
``AIAgent._fire_stream_retract`` tells the display sink (``stream_retract_callback``) that
everything streamed on this reply is withdrawn, from every restart site, and only when
visible text actually reached a consumer.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from hermes_constants import FINISH_REASON_LENGTH, PARTIAL_STREAM_STUB_ID


def _make_agent():
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


# ── The retract itself ──────────────────────────────────────────────────────────────────


class TestFireStreamRetract:
    def test_fires_only_after_visible_text_and_clears_it(self):
        agent = _make_agent()
        agent.stream_retract_callback = MagicMock()

        assert agent._fire_stream_retract() is False
        agent.stream_retract_callback.assert_not_called()

        agent._record_streamed_assistant_text("I'm sorry, I cannot ")
        agent._record_streamed_assistant_text("help with that")
        assert agent._fire_stream_retract() is True
        agent.stream_retract_callback.assert_called_once_with()
        # Withdrawn text is no longer "delivered": a later partial-delivery check must not
        # stitch it into a terminal reply.
        assert agent._current_streamed_assistant_text == ""

    def test_whitespace_only_deltas_do_not_retract(self):
        """Nothing visible reached the consumer (think-only / whitespace deltas): nothing to
        withdraw, same rule as ``_visible_text_delivered``."""
        agent = _make_agent()
        agent.stream_retract_callback = MagicMock()
        agent._record_streamed_assistant_text("  \n")
        assert agent._fire_stream_retract() is False
        agent.stream_retract_callback.assert_not_called()

    def test_no_sink_still_clears_and_reports_undelivered(self):
        agent = _make_agent()
        agent.stream_retract_callback = None
        agent._record_streamed_assistant_text("partial")
        assert agent._fire_stream_retract() is False
        assert agent._current_streamed_assistant_text == ""

    def test_a_raising_sink_never_breaks_the_restart(self):
        agent = _make_agent()
        agent.stream_retract_callback = MagicMock(side_effect=RuntimeError("socket closed"))
        agent._record_streamed_assistant_text("partial")
        assert agent._fire_stream_retract() is False
        assert agent._current_streamed_assistant_text == ""

    def test_constructor_accepts_the_sink(self):
        from run_agent import AIAgent

        sink = MagicMock()
        agent = AIAgent(
            api_key="test-key", base_url="https://example.com/v1", model="test/model",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            stream_retract_callback=sink,
        )
        assert agent.stream_retract_callback is sink


# ── Every restart site ──────────────────────────────────────────────────────────────────


class TestFallbackRestartRetracts:
    def test_arm_fallback_restart_withdraws_delivered_text(self):
        from agent.conversation_loop import _arm_fallback_restart
        from agent.turn_retry_state import TurnRetryState

        agent = _make_agent()
        agent.stream_retract_callback = MagicMock()
        agent._record_streamed_assistant_text("I'm sorry, I cannot help with that")
        _retry = TurnRetryState()

        with patch("agent.conversation_loop._sync_failover_system_message", return_value="sys"):
            assert _arm_fallback_restart(agent, [], "sys", _retry) == "sys"

        agent.stream_retract_callback.assert_called_once_with()
        assert _retry.restart_with_rebuilt_messages is True

    def test_arm_fallback_restart_is_silent_when_nothing_was_delivered(self):
        from agent.conversation_loop import _arm_fallback_restart
        from agent.turn_retry_state import TurnRetryState

        agent = _make_agent()
        agent.stream_retract_callback = MagicMock()

        with patch("agent.conversation_loop._sync_failover_system_message", return_value="sys"):
            _arm_fallback_restart(agent, [], "sys", TurnRetryState())

        agent.stream_retract_callback.assert_not_called()


@pytest.fixture()
def loop_agent():
    """AIAgent with a mocked OpenAI client (mirrors test_partial_stream_finish_reason)."""
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.compression_enabled = False
        a.save_trajectories = False
        return a


class TestContentFilterFallbackRetracts:
    def test_filter_stop_retracts_before_the_fallback_answers(self, loop_agent):
        """The content-filter stall path (#32421) restarts on the fallback without going through
        ``_arm_fallback_restart``; it withdraws the streamed partial too, before the fallback's
        first delta."""
        from tests.agent.test_run_agent import _mock_assistant_msg, _mock_response

        order: list[str] = []
        loop_agent.stream_retract_callback = lambda: order.append("retract")

        def _filter_stub(*_a, **_kw):
            # The stream delivered this text before the provider's filter killed it.
            loop_agent._record_streamed_assistant_text("I'm sorry, I cannot help with that")
            order.append("primary")
            return SimpleNamespace(
                id=PARTIAL_STREAM_STUB_ID,
                model="openai/gpt-6",
                choices=[SimpleNamespace(
                    index=0,
                    message=_mock_assistant_msg(content="I'm sorry, I cannot help with that"),
                    finish_reason=FINISH_REASON_LENGTH,
                )],
                usage=None,
                _dropped_tool_names=[],
                _content_filter_terminated=True,
            )

        def _recovery(*_a, **_kw):
            order.append("fallback")
            return _mock_response(content="Here is the sermon.", finish_reason="stop")

        calls = {"n": 0}

        def _create(*a, **kw):
            calls["n"] += 1
            return (_filter_stub if calls["n"] == 1 else _recovery)(*a, **kw)

        loop_agent.client.chat.completions.create.side_effect = _create
        loop_agent._fallback_chain = [{"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}]
        loop_agent._fallback_index = 0

        def _fake_activate(reason=None, **_kw):
            loop_agent._fallback_index = len(loop_agent._fallback_chain)
            return True

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
            patch.object(loop_agent, "_try_activate_fallback", side_effect=_fake_activate),
        ):
            result = loop_agent.run_conversation("write me a sermon")

        assert order == ["primary", "retract", "fallback"]
        assert result["final_response"] == "Here is the sermon."

    def test_no_fallback_configured_retracts_nothing(self, loop_agent):
        """Without a fallback the loop continues on the same provider and keeps the partial:
        the text stays on screen because it stays in the reply."""
        from tests.agent.test_run_agent import _mock_assistant_msg, _mock_response

        loop_agent.stream_retract_callback = MagicMock()

        def _filter_stub(*_a, **_kw):
            loop_agent._record_streamed_assistant_text("Partial but kept")
            return SimpleNamespace(
                id=PARTIAL_STREAM_STUB_ID, model="openai/gpt-6",
                choices=[SimpleNamespace(
                    index=0, message=_mock_assistant_msg(content="Partial but kept"),
                    finish_reason=FINISH_REASON_LENGTH)],
                usage=None, _dropped_tool_names=[], _content_filter_terminated=True,
            )

        calls = {"n": 0}

        def _create(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _filter_stub()
            return _mock_response(content=" and the rest.", finish_reason="stop")

        loop_agent.client.chat.completions.create.side_effect = _create
        loop_agent._fallback_chain = []
        loop_agent._fallback_index = 0

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            loop_agent.run_conversation("write me a sermon")

        loop_agent.stream_retract_callback.assert_not_called()


class TestMidToolCallReconnectRetracts:
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_reconnect_withdraws_the_preamble_before_restreaming(self, _mock_close, mock_create, monkeypatch):
        """A stream that drops mid tool-call is retried (the preamble streams again): the
        retract lands before the reconnect marker, so the client shows one preamble."""
        from tests.agent.test_partial_stream_finish_reason import _make_stream_chunk, _make_tool_call_delta

        attempts = {"n": 0}

        def _stream(*_a, **_kw):
            attempts["n"] += 1
            if attempts["n"] == 1:
                yield _make_stream_chunk(content="Writing the file: ")
                yield _make_stream_chunk(tool_calls=[_make_tool_call_delta(index=0, tc_id="call_1", name="write_file")])
                raise httpx.ReadError("connection reset")
            yield _make_stream_chunk(content="Writing the file: ")
            yield _make_stream_chunk(tool_calls=[_make_tool_call_delta(index=0, tc_id="call_1", name="write_file")])
            yield _make_stream_chunk(tool_calls=[_make_tool_call_delta(index=0, arguments='{"path": "x"}')])
            yield _make_stream_chunk(finish_reason="tool_calls")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _stream
        mock_create.return_value = mock_client

        agent = _make_agent()
        order: list[str] = []
        agent.stream_delta_callback = lambda text: order.append(f"delta:{text!r}")
        agent.stream_retract_callback = lambda: order.append("retract")
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")
        monkeypatch.setattr("agent.chat_completion_helpers._wait_stream_retry_backoff", lambda *_a, **_kw: None)

        response = agent._interruptible_streaming_api_call({"model": "test/model", "messages": []})

        assert attempts["n"] == 2
        assert response.choices[0].message.tool_calls, "the retried tool call is the reply"
        preamble = "delta:'Writing the file: '"
        # Preamble, retract, preamble again (the reconnect marker, when shown, rides between).
        assert [item for item in order if item in (preamble, "retract")] == [preamble, "retract", preamble], order

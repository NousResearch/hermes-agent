"""Streaming-pipeline regression: tool-call XML must never reach delta consumers.

Repro from the review on NousResearch/hermes-agent#114136: Responses-wire deltas flow
through ``agent/stream_delivery.py::_fire_stream_delta`` into ``stream_delta_callback`` /
``_stream_callback`` / the ``on_stream_delta`` hooks, where only reasoning tags were
scrubbed — so a serialized ``<atem:function_calls>`` block reached the CLI/gateway/TTS
consumers raw before final cleanup. This drives the real pipeline (real ``AIAgent``
defaults from ``agent_init``, real callback fan-out) rather than the scrubber unit, so it
fails if the wiring is missing even when the scrubber class is correct.
"""

from __future__ import annotations


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


BLOCK = (
    "Checking the queue.\n"
    "<atem:function_calls>\n"
    '<atem:invoke name="default.terminal">\n'
    '<atem:parameter name="command">echo hi</atem:parameter>\n'
    "</atem:invoke>\n"
    "</atem:function_calls>"
)


def _stream(agent, deltas) -> str:
    """Push deltas through the production stream path; return what consumers saw."""
    visible: list[str] = []
    agent.stream_delta_callback = visible.append
    for delta in deltas:
        agent._fire_stream_delta(delta)
    agent._reset_stream_delivery_tracking()  # end-of-stream flush
    return "".join(visible)


class TestStreamedToolCallXmlSuppression:
    def test_block_never_reaches_delta_consumers(self):
        out = _stream(_make_agent(), [BLOCK])
        assert "atem:" not in out
        assert "function_calls" not in out
        assert out == "Checking the queue.\n"

    def test_tags_split_across_deltas_never_leak(self):
        deltas = [
            "Checking the queue.\n<atem:function_",
            'calls>\n<atem:invoke name="default.terminal">\n',
            "<atem:parameter",
            ' name="command">echo hi</atem:parameter>\n</atem:invoke>\n',
            "</atem:function_calls>",
        ]
        out = _stream(_make_agent(), deltas)
        assert "atem:" not in out
        assert out == "Checking the queue.\n"

    def test_cut_serialization_drops_the_unterminated_tail(self):
        deltas = ['Waiting.\n<atem:function_calls>\n<atem:invoke name="x">']
        out = _stream(_make_agent(), deltas)
        assert "atem:" not in out
        assert out == "Waiting.\n"

    def test_plain_prose_is_untouched(self):
        out = _stream(_make_agent(), ["No tags ", "here, just ", "prose."])
        assert out == "No tags here, just prose."


class TestCutInsideCloserSuppression:
    """A stream ending inside an anchored block's closer delivers no fragment (#114136)."""

    def test_closer_tail_split_across_deltas_is_suppressed(self):
        out = _stream(_make_agent(), ["Waiting.\n<atem:function_calls>{}\n</atem:function", "_"])
        assert "atem:" not in out
        assert "function_" not in out
        assert out == "Waiting.\n"


class TestNamedFunctionBlockSuppression:
    """Named <function name=…> blocks never reach delta consumers, whole or split."""

    def test_named_block_never_reaches_delta_consumers(self):
        out = _stream(_make_agent(), ['Hello\n<function name="search">query</function> done'])
        assert "function name" not in out
        assert out == "Hello\n done"

    def test_named_block_split_across_deltas_never_leaks(self):
        deltas = [
            'Hello\n<function na',
            'me="search">qu',
            "ery</func",
            "tion> done",
        ]
        out = _stream(_make_agent(), deltas)
        assert "function name" not in out
        assert out == "Hello\n done"


class TestArgMarkupSuppression:
    """Line-anchored GLM <arg_key>/<arg_value> markup never reaches delta consumers."""

    def test_arg_markup_never_reaches_delta_consumers(self):
        out = _stream(_make_agent(), ["Hello\n<arg_key>name</arg_key> done"])
        assert "arg_key" not in out
        assert out == "Hello\n"

    def test_arg_markup_split_across_deltas_never_leaks(self):
        out = _stream(_make_agent(), ["Hello\n<arg_", "key>name</arg_", "value> done"])
        assert "arg_key" not in out
        assert "arg_value" not in out
        assert out == "Hello\n"


class TestEndOfStreamFlush:
    """A response ending in a benign partial tag is delivered in full at stream end."""

    def test_safe_tail_is_flushed_before_terminal_delivery(self):
        from types import SimpleNamespace

        from agent.chat_completion_helpers import _with_stream_emitters

        agent = _make_agent()
        visible: list[str] = []
        agent.stream_delta_callback = visible.append
        agent._fire_stream_delta("ordinary suffix <foo")
        assert "".join(visible) == "ordinary suffix "
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ordinary suffix <foo"))],
        )
        _with_stream_emitters(agent, lambda: response)
        assert "".join(visible) == "ordinary suffix <foo"
        assert agent._current_streamed_assistant_text == "ordinary suffix <foo"

    def test_anchored_block_is_still_dropped_at_stream_end(self):
        from types import SimpleNamespace

        from agent.chat_completion_helpers import _with_stream_emitters

        agent = _make_agent()
        visible: list[str] = []
        agent.stream_delta_callback = visible.append
        agent._fire_stream_delta("Waiting.\n<atem:function_calls>{}\n</atem:invoke>")
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="x"))])
        _with_stream_emitters(agent, lambda: response)
        assert "".join(visible) == "Waiting.\n"


class TestCodexRetryStartsWithFreshScrubbers:
    """A transport retry inside run_codex_stream must not inherit the failed attempt's
    anchored block: attempt one's unterminated serialization would otherwise swallow
    attempt two's complete answer (#114136)."""

    def test_retry_delivers_successful_attempt_in_full(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import httpx

        agent = _make_agent()
        visible: list[str] = []
        agent.stream_delta_callback = visible.append
        agent._interrupt_requested = False

        def _delta(text):
            return SimpleNamespace(type="response.output_text.delta", delta=text)

        def _completed(text):
            return SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    status="completed",
                    id="r1",
                    usage=None,
                    output=[
                        SimpleNamespace(
                            type="message",
                            content=[SimpleNamespace(type="output_text", text=text)],
                        )
                    ],
                ),
            )

        class _FailingStream:
            def __iter__(self_inner):
                yield _delta("Visible intro. ")
                yield _delta("\n<atem:function_calls>{}\n<atem:invoke>")
                raise httpx.RemoteProtocolError("peer closed connection mid-iteration")

            def close(self_inner):
                return None

        class _GoodStream:
            def __iter__(self_inner):
                yield _delta("Full recovered answer.")
                yield _completed("Full recovered answer.")

            def close(self_inner):
                return None

        calls = {"n": 0}

        def _create(**kwargs):
            calls["n"] += 1
            return _FailingStream() if calls["n"] == 1 else _GoodStream()

        mock_client = MagicMock()
        mock_client.responses.create.side_effect = _create

        agent._run_codex_stream({"model": "test/model"}, client=mock_client)

        assert calls["n"] == 2
        delivered = "".join(visible)
        assert delivered.count("Visible intro. ") == 1
        assert "Full recovered answer." in delivered
        assert "atem:" not in delivered

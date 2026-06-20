"""Regression tests for streamed surrogate-pair splits.

Providers can deliver JSON-escaped UTF-16 surrogate halves in adjacent stream
text deltas (e.g. ``"\ud83d"`` then ``"\ude00"`` for 😀).  Python keeps those
as lone surrogate code points until we actively recombine them.  Passing a lone
surrogate to the gateway/CLI callback crashes on UTF-8 encode and used to make
Hermes return a 0-char partial-stream stub and fall back to codex.
"""
from __future__ import annotations

import random
from types import SimpleNamespace

from agent.chat_completion_helpers import (
    _repair_anthropic_message_surrogates,
    interruptible_streaming_api_call,
)
from agent.message_sanitization import _SurrogateSplicer, _splice_surrogates
from run_agent import AIAgent


def _decompose_as_utf16_surrogates(text: str) -> str:
    """Represent astral characters as surrogate-pair code points."""
    return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")


def _collect(splicer: _SurrogateSplicer, pieces: list[str]) -> str:
    return "".join(splicer.feed(piece) for piece in pieces) + splicer.flush()


class TestSurrogateSplicer:
    def test_recombines_surrogate_pair_split_across_deltas(self):
        splicer = _SurrogateSplicer()

        assert splicer.feed("a\ud83d") == "a"
        assert splicer.feed("\ude00b") == "😀b"
        assert splicer.flush() == ""

    def test_recombines_adjacent_surrogate_pair_inside_one_delta(self):
        splicer = _SurrogateSplicer()

        assert splicer.feed("x\ud83d\ude00y") == "x😀y"
        assert splicer.flush() == ""

    def test_orphan_surrogates_become_replacement_characters(self):
        splicer = _SurrogateSplicer()

        assert splicer.feed("a\ude00b") == "a�b"
        assert splicer.feed("c\ud83d") == "c"
        assert splicer.flush() == "�"

    def test_seeded_random_splits_preserve_astral_text_exactly(self):
        corpus = "alpha 😀 beta 🧪 gamma 𝄞 delta 🚀 end"
        decomposed = _decompose_as_utf16_surrogates(corpus)
        rng = random.Random(20260620)

        for _ in range(1000):
            cuts = sorted(rng.sample(range(len(decomposed) + 1), k=rng.randint(1, 8)))
            pieces: list[str] = []
            prev = 0
            for cut in cuts:
                pieces.append(decomposed[prev:cut])
                prev = cut
            pieces.append(decomposed[prev:])

            assert _collect(_SurrogateSplicer(), pieces) == corpus

    def test_stateless_splice_repairs_final_message_text(self):
        assert _splice_surrogates("a\ud83d\ude00b\ud83d") == "a😀b�"

    def test_anthropic_final_message_repair_preserves_sdk_shape(self):
        text_block = SimpleNamespace(type="text", text="a\ud83d\ude00b")
        thinking_block = SimpleNamespace(type="thinking", thinking="x\ud83e\uddd0y")
        message = SimpleNamespace(content=[text_block, thinking_block])

        assert _repair_anthropic_message_surrogates(message) is message
        assert text_block.text == "a😀b"
        assert thinking_block.thinking == "x🧐y"
        text_block.text.encode("utf-8")
        thinking_block.thinking.encode("utf-8")


class TestStreamSurrogateFloor:
    def test_fire_stream_delta_sanitizes_before_callback(self):
        agent = AIAgent.__new__(AIAgent)
        captured: list[str] = []

        def cb(text: str) -> None:
            captured.append(text)
            text.encode("utf-8")  # must not raise inside the callback path

        agent.stream_delta_callback = cb
        agent._stream_callback = None
        agent._stream_needs_break = False
        agent._current_streamed_assistant_text = ""
        agent._stream_think_scrubber = None
        agent._stream_context_scrubber = None

        agent._fire_stream_delta("x\ud83dy")

        assert captured == ["x�y"]
        assert agent._current_streamed_assistant_text == "x�y"

    def test_fire_reasoning_delta_sanitizes_before_callback(self):
        agent = AIAgent.__new__(AIAgent)
        captured: list[str] = []
        agent.reasoning_callback = lambda text: captured.append(text)

        agent._fire_reasoning_delta("think \ud83d")

        assert captured == ["think �"]


class TestChatCompletionsStreamPath:
    def test_streaming_call_recombines_content_and_reasoning_deltas(self):
        class FakeStream:
            response = None

            def __iter__(self):
                return iter([
                    SimpleNamespace(
                        model="fake-model",
                        choices=[SimpleNamespace(
                            delta=SimpleNamespace(
                                content="a\ud83d",
                                reasoning_content=None,
                                reasoning=None,
                                tool_calls=None,
                            ),
                            finish_reason=None,
                        )],
                        usage=None,
                    ),
                    SimpleNamespace(
                        model="fake-model",
                        choices=[SimpleNamespace(
                            delta=SimpleNamespace(
                                content="\ude00b",
                                reasoning_content=None,
                                reasoning=None,
                                tool_calls=None,
                            ),
                            finish_reason=None,
                        )],
                        usage=None,
                    ),
                    SimpleNamespace(
                        model="fake-model",
                        choices=[SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                reasoning_content="think \ud83e",
                                reasoning=None,
                                tool_calls=None,
                            ),
                            finish_reason=None,
                        )],
                        usage=None,
                    ),
                    SimpleNamespace(
                        model="fake-model",
                        choices=[SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                reasoning_content="\uddd0",
                                reasoning=None,
                                tool_calls=None,
                            ),
                            finish_reason="stop",
                        )],
                        usage=None,
                    ),
                ])

        class FakeCompletions:
            def create(self, **_kwargs):
                return FakeStream()

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            chat = FakeChat()

        agent = AIAgent.__new__(AIAgent)
        streamed: list[str] = []
        reasoning: list[str] = []

        def stream_cb(text: str) -> None:
            streamed.append(text)
            text.encode("utf-8")

        def reasoning_cb(text: str) -> None:
            reasoning.append(text)
            text.encode("utf-8")

        agent.api_mode = "chat_completions"
        agent.provider = "unit-test-provider"
        agent.model = "unit-test-model"
        agent.base_url = None
        agent._interrupt_requested = False
        agent.stream_delta_callback = stream_cb
        agent._stream_callback = None
        agent.reasoning_callback = reasoning_cb
        agent._stream_needs_break = False
        agent._current_streamed_assistant_text = ""
        agent._stream_think_scrubber = None
        agent._stream_context_scrubber = None
        agent._create_request_openai_client = lambda **_kwargs: FakeClient()
        agent._capture_rate_limits = lambda *_args, **_kwargs: None
        agent._capture_credits = lambda *_args, **_kwargs: None
        agent._stream_diag_init = lambda: {}
        agent._stream_diag_capture_response = lambda *_args, **_kwargs: None
        agent._check_openrouter_cache_status = lambda *_args, **_kwargs: None
        agent._touch_activity = lambda *_args, **_kwargs: None
        agent._abort_request_openai_client = lambda *_args, **_kwargs: None
        agent._close_request_openai_client = lambda *_args, **_kwargs: None
        agent._buffer_status = lambda *_args, **_kwargs: None
        agent._emit_stream_drop = lambda *_args, **_kwargs: None
        agent._log_stream_retry = lambda *_args, **_kwargs: None
        agent._replace_primary_openai_client = lambda *_args, **_kwargs: None

        response = interruptible_streaming_api_call(
            agent,
            {"model": "unit-test-model", "messages": []},
        )

        assert streamed == ["a", "😀b"]
        assert reasoning == ["think ", "🧐"]
        assert response.choices[0].message.content == "a😀b"
        assert response.choices[0].message.reasoning_content == "think 🧐"
        response.choices[0].message.content.encode("utf-8")
        response.choices[0].message.reasoning_content.encode("utf-8")


class TestAnthropicMessagesStreamPath:
    def test_streaming_call_recombines_text_thinking_and_final_message(self):
        class FakeAnthropicStream:
            response = None

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                return iter([
                    SimpleNamespace(
                        type="content_block_delta",
                        delta=SimpleNamespace(type="text_delta", text="a\ud83d"),
                    ),
                    SimpleNamespace(
                        type="content_block_delta",
                        delta=SimpleNamespace(type="text_delta", text="\ude00b"),
                    ),
                    SimpleNamespace(
                        type="content_block_delta",
                        delta=SimpleNamespace(type="thinking_delta", thinking="think \ud83e"),
                    ),
                    SimpleNamespace(
                        type="content_block_delta",
                        delta=SimpleNamespace(type="thinking_delta", thinking="\uddd0"),
                    ),
                ])

            def get_final_message(self):
                return SimpleNamespace(content=[
                    SimpleNamespace(type="text", text="a\ud83d\ude00b"),
                    SimpleNamespace(type="thinking", thinking="think \ud83e\uddd0"),
                ])

        agent = AIAgent.__new__(AIAgent)
        streamed: list[str] = []
        reasoning: list[str] = []

        def stream_cb(text: str) -> None:
            streamed.append(text)
            text.encode("utf-8")

        def reasoning_cb(text: str) -> None:
            reasoning.append(text)
            text.encode("utf-8")

        agent.api_mode = "anthropic_messages"
        agent.provider = "unit-test-provider"
        agent.model = "unit-test-model"
        agent.base_url = None
        agent._interrupt_requested = False
        agent.stream_delta_callback = stream_cb
        agent._stream_callback = None
        agent.reasoning_callback = reasoning_cb
        agent._stream_needs_break = False
        agent._current_streamed_assistant_text = ""
        agent._stream_think_scrubber = None
        agent._stream_context_scrubber = None
        agent._try_refresh_anthropic_client_credentials = lambda: None
        agent._anthropic_client = SimpleNamespace(
            messages=SimpleNamespace(stream=lambda **_kwargs: FakeAnthropicStream())
        )
        agent._stream_diag_init = lambda: {}
        agent._stream_diag_capture_response = lambda *_args, **_kwargs: None
        agent._touch_activity = lambda *_args, **_kwargs: None
        agent._rebuild_anthropic_client = lambda: None
        agent._buffer_status = lambda *_args, **_kwargs: None
        agent._emit_stream_drop = lambda *_args, **_kwargs: None
        agent._log_stream_retry = lambda *_args, **_kwargs: None
        agent._replace_primary_openai_client = lambda *_args, **_kwargs: None

        response = interruptible_streaming_api_call(
            agent,
            {"model": "unit-test-model", "messages": []},
        )

        assert streamed == ["a", "😀b"]
        assert reasoning == ["think ", "🧐"]
        assert response.content[0].text == "a😀b"
        assert response.content[1].thinking == "think 🧐"
        response.content[0].text.encode("utf-8")
        response.content[1].thinking.encode("utf-8")


class TestSurrogateUnicodeEncodeClassifier:
    def test_surrogate_unicode_encode_error_is_provider_stream_parse_error(self):
        dummy = SimpleNamespace(api_mode="anthropic_messages")
        try:
            "\ud83d".encode("utf-8")
        except UnicodeEncodeError as exc:
            err = exc
        else:  # pragma: no cover
            raise AssertionError("encoding a lone surrogate should raise")

        assert AIAgent._is_provider_stream_parse_error(dummy, err)

    def test_non_surrogate_unicode_encode_error_is_not_stream_parse_error(self):
        dummy = SimpleNamespace(api_mode="anthropic_messages")
        try:
            "é".encode("ascii")
        except UnicodeEncodeError as exc:
            err = exc
        else:  # pragma: no cover
            raise AssertionError("encoding non-ascii as ascii should raise")

        assert not AIAgent._is_provider_stream_parse_error(dummy, err)

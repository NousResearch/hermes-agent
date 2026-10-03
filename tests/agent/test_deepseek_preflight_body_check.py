"""Regression tests for DeepSeek preflight body-size check (#30771 / #30809).

DeepSeek silently returns HTTP 400 on request bodies over ~880 KB.
The check must:

1. Fire from BOTH streaming and non-streaming callers (shared dispatch
   + each interruptible entry).
2. Raise a 413-classifiable error (``DeepSeekPayloadTooLargeError``) so
   ``conversation_loop`` routes through ``FailoverReason.payload_too_large``
   recovery, NOT bare ``ValueError`` local-validation abort.
3. Be a no-op for every non-DeepSeek provider.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.chat_completion_helpers import (
    DeepSeekPayloadTooLargeError,
    _DEEPSEEK_BODY_LIMIT_BYTES,
    _deepseek_preflight_body_check,
    interruptible_api_call,
    interruptible_streaming_api_call,
)
from agent.error_classifier import FailoverReason, classify_api_error
from run_agent import AIAgent
import run_agent


def _make_agent(base_url: str) -> MagicMock:
    agent = MagicMock()
    agent.base_url = base_url
    return agent


def _oversized_kwargs() -> dict:
    """Build api_kwargs whose serialised body exceeds the DeepSeek limit."""
    big_message = "x" * (_DEEPSEEK_BODY_LIMIT_BYTES + 1_000)
    return {
        "messages": [{"role": "user", "content": big_message}],
        "model": "deepseek-v3",
    }


def _undersized_kwargs() -> dict:
    return {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "deepseek-v3",
    }


def _mock_response(content="Hello", finish_reason="stop"):
    msg = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="deepseek-chat")
    resp.usage = None
    return resp


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0)


class TestDeepSeekPreflightBodyCheck:
    def test_raises_on_oversized_body_for_deepseek(self):
        agent = _make_agent("https://api.deepseek.com/v1")
        with pytest.raises(DeepSeekPayloadTooLargeError, match="exceeds DeepSeek"):
            _deepseek_preflight_body_check(agent, _oversized_kwargs())

    def test_error_has_status_code_413(self):
        agent = _make_agent("https://api.deepseek.com/v1")
        with pytest.raises(DeepSeekPayloadTooLargeError) as exc_info:
            _deepseek_preflight_body_check(agent, _oversized_kwargs())
        assert exc_info.value.status_code == 413
        assert exc_info.value.body_bytes >= _DEEPSEEK_BODY_LIMIT_BYTES

    def test_classifies_as_payload_too_large(self):
        """Must enter FailoverReason.payload_too_large recovery, not local validation."""
        agent = _make_agent("https://api.deepseek.com/v1")
        with pytest.raises(DeepSeekPayloadTooLargeError) as exc_info:
            _deepseek_preflight_body_check(agent, _oversized_kwargs())
        classified = classify_api_error(exc_info.value)
        assert classified.reason == FailoverReason.payload_too_large
        assert classified.should_compress is True
        assert classified.retryable is True

    def test_no_raise_on_undersized_body_for_deepseek(self):
        agent = _make_agent("https://api.deepseek.com/v1")
        _deepseek_preflight_body_check(agent, _undersized_kwargs())

    def test_no_raise_for_openai(self):
        agent = _make_agent("https://api.openai.com/v1")
        _deepseek_preflight_body_check(agent, _oversized_kwargs())

    def test_no_raise_for_openrouter(self):
        agent = _make_agent("https://openrouter.ai/api/v1")
        _deepseek_preflight_body_check(agent, _oversized_kwargs())

    def test_no_raise_for_anthropic(self):
        agent = _make_agent("https://api.anthropic.com/v1")
        _deepseek_preflight_body_check(agent, _oversized_kwargs())

    def test_no_raise_for_local_endpoint(self):
        agent = _make_agent("http://127.0.0.1:11434/v1")
        _deepseek_preflight_body_check(agent, _oversized_kwargs())

    def test_no_raise_when_base_url_is_none(self):
        agent = _make_agent(None)
        _deepseek_preflight_body_check(agent, _oversized_kwargs())

    def test_error_message_includes_byte_count_and_remedy(self):
        agent = _make_agent("https://api.deepseek.com/v1")
        with pytest.raises(DeepSeekPayloadTooLargeError) as exc_info:
            _deepseek_preflight_body_check(agent, _oversized_kwargs())
        msg = str(exc_info.value)
        assert "bytes" in msg
        assert "compress" in msg.lower() or "Compress" in msg
        # Classifier message-pattern surface
        assert "payload too large" in msg.lower()
        assert "request entity too large" in msg.lower()

    def test_limit_constant_is_sensible(self):
        assert 800_000 <= _DEEPSEEK_BODY_LIMIT_BYTES <= 920_000


class TestDeepSeekPreflightEntryPoints:
    """Both streaming and non-streaming entries must run the preflight."""

    def _mock_entry_agent(self, *, streaming_capable=True):
        agent = MagicMock()
        agent.base_url = "https://api.deepseek.com/v1"
        agent._interrupt_requested = False
        agent.api_mode = "chat_completions"
        agent.verbose_logging = False
        agent.provider = "deepseek"
        # Force streaming path past should_use_direct_api_call
        return agent

    def test_interruptible_api_call_no_http_on_oversized(self):
        """Non-streaming entry: preflight raises before any HTTP."""
        agent = self._mock_entry_agent()
        # If preflight is skipped, thread/worker would call into make_client
        with (
            patch(
                "agent.chat_completion_helpers.should_use_direct_api_call",
                return_value=True,
            ),
            patch(
                "agent.chat_completion_helpers.direct_api_call",
                side_effect=AssertionError("HTTP must not be attempted"),
            ) as mock_direct,
        ):
            with pytest.raises(DeepSeekPayloadTooLargeError):
                interruptible_api_call(agent, _oversized_kwargs())
        mock_direct.assert_not_called()

    def test_interruptible_streaming_api_call_no_http_on_oversized(self):
        """Streaming entry: preflight raises before any HTTP (issue #30809)."""
        agent = self._mock_entry_agent()
        with (
            patch(
                "agent.chat_completion_helpers.should_use_direct_api_call",
                return_value=False,
            ),
            patch.object(
                agent,
                "_interruptible_api_call",
                side_effect=AssertionError("must not fall back to non-stream HTTP"),
            ),
        ):
            with pytest.raises(DeepSeekPayloadTooLargeError):
                interruptible_streaming_api_call(agent, _oversized_kwargs())


class TestDeepSeekPreflightEndToEnd:
    """End-to-end: oversized DeepSeek request compresses via payload_too_large."""

    def _build_agent(self, *, compression_enabled=True):
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            a = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://api.deepseek.com/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = compression_enabled
        a.save_trajectories = False
        a.provider = "deepseek"
        a.model = "deepseek-chat"
        return a

    def test_non_streaming_oversized_triggers_compression_no_http(self):
        """Non-streaming path: preflight → payload_too_large → compress → retry."""
        agent = self._build_agent(compression_enabled=True)
        http_calls = {"n": 0}
        ok_resp = _mock_response("ok after compress")

        def _create(**kwargs):
            http_calls["n"] += 1
            return ok_resp

        agent.client.chat.completions.create.side_effect = _create

        # First build returns oversized kwargs; after compress, undersized.
        oversized = _oversized_kwargs()
        undersized = _undersized_kwargs()
        build_n = {"n": 0}

        def _build(*_a, **_k):
            build_n["n"] += 1
            return oversized if build_n["n"] == 1 else undersized

        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_build_api_kwargs", side_effect=_build),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            # Force non-streaming (Mock client already does, but be explicit)
            patch.object(agent, "_has_stream_consumers", return_value=False),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called()
        # HTTP only on the post-compression undersized attempt
        assert http_calls["n"] == 1
        assert result.get("failed") is not True
        assert result["completed"] is True

    def test_streaming_oversized_triggers_compression_no_http(self):
        """Streaming path: same recovery contract; first attempt never hits HTTP."""
        agent = self._build_agent(compression_enabled=True)
        http_calls = {"n": 0}
        create_calls = {"n": 0}
        ok_resp = _mock_response("ok after compress stream")

        def _create(**kwargs):
            create_calls["n"] += 1
            http_calls["n"] += 1
            return ok_resp

        agent.client.chat.completions.create.side_effect = _create

        oversized = _oversized_kwargs()
        undersized = _undersized_kwargs()
        build_n = {"n": 0}

        def _build(*_a, **_k):
            build_n["n"] += 1
            return oversized if build_n["n"] == 1 else undersized

        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        stream_entry_hits = {"n": 0}
        orig_stream = agent._interruptible_streaming_api_call

        def _stream_wrapper(api_kwargs, *, on_first_delta=None):
            stream_entry_hits["n"] += 1
            # First hit must oversize-preflight before any HTTP; after compress
            # re-dispatch through non-streaming success path for a clean result.
            if stream_entry_hits["n"] == 1:
                return orig_stream(api_kwargs, on_first_delta=on_first_delta)
            return agent._interruptible_api_call(api_kwargs)

        with (
            patch.object(agent, "_build_api_kwargs", side_effect=_build),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            # Force streaming branch at _perform_api_call
            patch.object(agent, "_has_stream_consumers", return_value=True),
            patch.object(
                agent,
                "_interruptible_streaming_api_call",
                side_effect=_stream_wrapper,
            ),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        # Streaming entry must have been taken at least for the oversized turn.
        assert stream_entry_hits["n"] >= 1
        mock_compress.assert_called()
        assert http_calls["n"] == 1
        assert result.get("failed") is not True
        assert result["completed"] is True

    def test_compression_disabled_does_not_loop_http(self):
        """With compression off, oversized preflight aborts without HTTP calls."""
        agent = self._build_agent(compression_enabled=False)
        http_calls = {"n": 0}

        def _create(**kwargs):
            http_calls["n"] += 1
            raise AssertionError("HTTP must not fire for oversized DeepSeek body")

        agent.client.chat.completions.create.side_effect = _create

        with (
            patch.object(agent, "_build_api_kwargs", return_value=_oversized_kwargs()),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent, "_has_stream_consumers", return_value=False),
        ):
            result = agent.run_conversation(
                "hello",
                conversation_history=[
                    {"role": "user", "content": "previous question"},
                    {"role": "assistant", "content": "previous answer"},
                ],
            )

        mock_compress.assert_not_called()
        assert http_calls["n"] == 0
        # Terminal failure / incomplete is acceptable; must not hang or HTTP
        assert result.get("completed") is not True or result.get("failed") is True

"""Tests for payload/context-length → compression retry logic in AIAgent.

Verifies that:
- HTTP 413 errors trigger history compression and retry
- HTTP 400 context-length errors trigger compression (not generic 4xx abort)
- Preflight compression proactively compresses oversized sessions before API calls
"""

import pytest
#pytestmark = pytest.mark.skip(reason="Hangs in non-interactive environments")



import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import SUMMARY_PREFIX
from run_agent import AIAgent
import run_agent


# ---------------------------------------------------------------------------
# Fast backoff for compression retry tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_compression_sleep(monkeypatch):
    """Short-circuit the 2s time.sleep between compression retries.

    Production code has ``time.sleep(2)`` in multiple places after a 413/context
    compression, for rate-limit smoothing. Tests assert behavior, not timing.
    """
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None, usage=None):
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = SimpleNamespace(**usage) if usage else None
    return resp


def _make_413_error(*, use_status_code=True, message="Request entity too large"):
    """Create an exception that mimics a 413 HTTP error."""
    err = Exception(message)
    if use_status_code:
        err.status_code = 413
    return err


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
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
        a.tool_delay = 0
        a.compression_enabled = False
        a.save_trajectories = False
        return a


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHTTP413Compression:
    """413 errors should trigger compression, not abort as generic 4xx."""

    def test_413_triggers_compression(self, agent):
        """A 413 error should call _compress_context and retry, not abort."""
        # First call raises 413; second call succeeds after compression.
        err_413 = _make_413_error()
        ok_resp = _mock_response(content="Success after compression", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_413, ok_resp]

        # Prefill so there are multiple messages for compression to reduce
        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            # Compression reduces 3 messages down to 1
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert result["completed"] is True
        assert result["final_response"] == "Success after compression"

    def test_413_not_treated_as_generic_4xx(self, agent):
        """413 must NOT hit the generic 4xx abort path; it should attempt compression."""
        err_413 = _make_413_error()
        ok_resp = _mock_response(content="Recovered", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_413, ok_resp]

        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "compressed",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        # If 413 were treated as generic 4xx, result would have "failed": True
        assert result.get("failed") is not True
        assert result["completed"] is True

    def test_413_error_message_detection(self, agent):
        """413 detected via error message string (no status_code attr)."""
        err = _make_413_error(use_status_code=False, message="error code: 413")
        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err, ok_resp]

        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "compressed",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert result["completed"] is True

    def test_413_clears_conversation_history_on_persist(self, agent):
        """After 413-triggered compression, _persist_session must receive None history.

        Bug: _compress_context() creates a new session and resets _last_flushed_db_idx=0,
        but if conversation_history still holds the original (pre-compression) list,
        _flush_messages_to_session_db computes flush_from = max(len(history), 0) which
        exceeds len(compressed_messages), so messages[flush_from:] is empty and nothing
        is written to the new session → "Session found but has no messages" on resume.
        """
        err_413 = _make_413_error()
        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_413, ok_resp]

        big_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(200)
        ]

        persist_calls = []

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(
                agent, "_persist_session",
                side_effect=lambda msgs, hist: persist_calls.append(hist),
            ),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "summary"}],
                "compressed prompt",
            )
            agent.run_conversation("hello", conversation_history=big_history)

        assert len(persist_calls) >= 1, "Expected at least one _persist_session call"
        for hist in persist_calls:
            assert hist is None, (
                f"conversation_history should be None after mid-loop compression, "
                f"got list with {len(hist)} items"
            )

    def test_context_overflow_clears_conversation_history_on_persist(self, agent):
        """After context-overflow compression, _persist_session must receive None history."""
        err_400 = Exception(
            "Error code: 400 - This endpoint's maximum context length is 128000 tokens. "
            "However, you requested about 270460 tokens."
        )
        err_400.status_code = 400
        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_400, ok_resp]

        big_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(200)
        ]

        persist_calls = []

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(
                agent, "_persist_session",
                side_effect=lambda msgs, hist: persist_calls.append(hist),
            ),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "summary"}],
                "compressed prompt",
            )
            agent.run_conversation("hello", conversation_history=big_history)

        assert len(persist_calls) >= 1
        for hist in persist_calls:
            assert hist is None, (
                f"conversation_history should be None after context-overflow compression, "
                f"got list with {len(hist)} items"
            )

    def test_400_context_length_triggers_compression(self, agent):
        """A 400 with 'maximum context length' should trigger compression, not abort as generic 4xx.

        OpenRouter returns HTTP 400 (not 413) for context-length errors. Before
        the fix, this was caught by the generic 4xx handler which aborted
        immediately — now it correctly triggers compression+retry.
        """
        err_400 = Exception(
            "Error code: 400 - {'error': {'message': "
            "\"This endpoint's maximum context length is 204800 tokens. "
            "However, you requested about 270460 tokens.\", 'code': 400}}"
        )
        err_400.status_code = 400
        ok_resp = _mock_response(content="Recovered after compression", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_400, ok_resp]

        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        # Must NOT have "failed": True (which would mean the generic 4xx handler caught it)
        assert result.get("failed") is not True
        assert result["completed"] is True
        assert result["final_response"] == "Recovered after compression"

    def test_400_reduce_length_triggers_compression(self, agent):
        """A 400 with 'reduce the length' should trigger compression."""
        err_400 = Exception(
            "Error code: 400 - Please reduce the length of the messages"
        )
        err_400.status_code = 400
        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_400, ok_resp]

        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "compressed",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert result["completed"] is True

    def test_context_length_retry_rebuilds_request_after_compression(self, agent):
        """Retry must send the compressed transcript, not the stale oversized payload."""
        err_400 = Exception(
            "Error code: 400 - {'error': {'message': "
            "\"This endpoint's maximum context length is 128000 tokens. "
            "Please reduce the length of the messages.\"}}"
        )
        err_400.status_code = 400
        ok_resp = _mock_response(content="Recovered after real compression", finish_reason="stop")

        request_payloads = []

        def _side_effect(**kwargs):
            request_payloads.append(kwargs)
            if len(request_payloads) == 1:
                raise err_400
            return ok_resp

        agent.client.chat.completions.create.side_effect = _side_effect

        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "compressed summary"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        assert result["completed"] is True
        assert len(request_payloads) == 2
        assert len(request_payloads[1]["messages"]) < len(request_payloads[0]["messages"])
        assert request_payloads[1]["messages"][0] == {
            "role": "system",
            "content": "compressed prompt",
        }
        assert request_payloads[1]["messages"][1] == {
            "role": "user",
            "content": "compressed summary",
        }


    def test_large_generic_provider_error_compresses_before_retry(self, agent):
        """Large generic provider errors should compress once before retrying."""
        agent.compression_enabled = True
        agent.compression_soft_request_limit = 220_000
        agent.retry_compress_on_provider_error = True
        err = Exception(
            "An error occurred while processing your request. "
            "You can retry your request. Please include the request ID req_123."
        )
        ok_resp = _mock_response(content="Recovered after provider-error compression", finish_reason="stop")
        request_payloads = []

        def _side_effect(**kwargs):
            request_payloads.append(kwargs)
            if len(request_payloads) == 1:
                raise err
            return ok_resp

        agent.client.chat.completions.create.side_effect = _side_effect
        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch("agent.conversation_loop.estimate_request_tokens_rough", return_value=230_000),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "compressed summary"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert result["completed"] is True
        assert result["final_response"] == "Recovered after provider-error compression"
        assert len(request_payloads) == 2
        assert len(request_payloads[1]["messages"]) < len(request_payloads[0]["messages"])
        assert request_payloads[1]["messages"][0] == {
            "role": "system",
            "content": "compressed prompt",
        }
        assert request_payloads[1]["messages"][1] == {
            "role": "user",
            "content": "compressed summary",
        }

    def test_small_generic_provider_error_does_not_compress(self, agent):
        """Below soft_request_limit, generic provider errors use normal retry."""
        agent.compression_enabled = True
        agent.compression_soft_request_limit = 220_000
        agent.retry_compress_on_provider_error = True
        err = Exception("An error occurred while processing your request. You can retry your request.")
        ok_resp = _mock_response(content="Recovered by normal retry", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err, ok_resp]

        with (
            patch("agent.conversation_loop.estimate_request_tokens_rough", return_value=50_000),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        mock_compress.assert_not_called()
        assert agent.client.chat.completions.create.call_count == 2
        assert result["completed"] is True
        assert result["final_response"] == "Recovered by normal retry"

    @pytest.mark.parametrize(
        ("message", "status_code"),
        [
            ("auth_unavailable: incorrect api key", 401),
            ("Unauthorized: permission denied", 403),
            ("HTTP 429 usage limit / rate limit / 429 cooling down", 429),
            ("model not found", 404),
            ("quota exceeded", 402),
        ],
    )
    def test_auth_quota_rate_and_model_errors_do_not_provider_compress(
        self, agent, message, status_code
    ):
        """Auth/quota/rate/model errors must not trigger compression retry."""
        agent.compression_enabled = True
        agent.compression_soft_request_limit = 220_000
        agent.retry_compress_on_provider_error = True
        agent._api_max_retries = 1
        err = Exception(message)
        err.status_code = status_code
        agent.client.chat.completions.create.side_effect = [err]

        with (
            patch("agent.conversation_loop.estimate_request_tokens_rough", return_value=230_000),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        mock_compress.assert_not_called()
        assert result["completed"] is False

    @pytest.mark.parametrize(
        "message",
        [
            "not authorized — an error occurred while processing your request; please include the request id",
            "not permitted — you can retry your request with the request id",
            "permission denied — an error occurred while processing your request",
            "insufficient credits / quota exceeded — please include the request id",
            "model not found — you can retry your request",
        ],
    )
    def test_provider_error_excludes_sensitive_errors_without_status_code(
        self, agent, message
    ):
        """Sensitive/provider-account errors are excluded even without HTTP status."""
        agent.compression_enabled = True
        agent.compression_soft_request_limit = 220_000
        agent.retry_compress_on_provider_error = True
        agent._api_max_retries = 1
        err = Exception(message)
        agent.client.chat.completions.create.side_effect = [err]

        with (
            patch("agent.conversation_loop.estimate_request_tokens_rough", return_value=230_000),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        mock_compress.assert_not_called()
        assert result["completed"] is False

    def test_provider_error_compression_disabled_by_default_knobs(self, agent):
        """Default conservative knobs keep generic provider errors on normal retry."""
        agent.compression_enabled = True
        agent.compression_soft_request_limit = 0
        agent.retry_compress_on_provider_error = False
        err = Exception("An error occurred while processing your request. You can retry your request.")
        ok_resp = _mock_response(content="Recovered without provider-error compression", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err, ok_resp]

        with (
            patch("agent.conversation_loop.estimate_request_tokens_rough", return_value=230_000),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        mock_compress.assert_not_called()
        assert agent.client.chat.completions.create.call_count == 2
        assert result["completed"] is True
        assert result["final_response"] == "Recovered without provider-error compression"

    def test_provider_error_compression_same_messages_is_bounded(self, agent):
        """If compression cannot shrink messages, fall through without looping."""
        agent.compression_enabled = True
        agent.compression_soft_request_limit = 220_000
        agent.retry_compress_on_provider_error = True
        agent._api_max_retries = 1
        err = Exception("An error occurred while processing your request. Please include the request ID.")
        agent.client.chat.completions.create.side_effect = [err]

        with (
            patch("agent.conversation_loop.estimate_request_tokens_rough", return_value=230_000),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            same_messages = [{"role": "user", "content": "hello"}]
            mock_compress.return_value = (same_messages, "same prompt")
            result = agent.run_conversation("hello")

        mock_compress.assert_called_once()
        assert agent.client.chat.completions.create.call_count == 1
        assert result["completed"] is False

    def test_gateway_provider_error_compression_triggers_circuit_breaker(self, agent):
        """Gateway turns should halt after provider-error compression recovery."""
        agent.platform = "qqbot"
        agent.compression_enabled = True
        agent.compression_soft_request_limit = 220_000
        agent.retry_compress_on_provider_error = True
        err = Exception("An error occurred while processing your request. You can retry your request.")
        agent.client.chat.completions.create.side_effect = [err]
        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch("agent.conversation_loop.estimate_request_tokens_rough", return_value=230_000),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "compressed summary"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        agent.client.chat.completions.create.assert_called_once()
        assert result["compression_circuit_breaker"] is True
        assert result["completed"] is False
        assert "context 刚压缩恢复" in result["final_response"]

    def test_413_cannot_compress_further(self, agent):
        """When compression can't reduce messages, return partial result."""
        err_413 = _make_413_error()
        agent.client.chat.completions.create.side_effect = [err_413]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            # Compression returns same number of messages → can't compress further
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "same prompt",
            )
            result = agent.run_conversation("hello")

        assert result["completed"] is False
        assert result.get("partial") is True
        assert "413" in result["error"]

    def test_gateway_413_compression_halts_instead_of_retrying_heavy_turn(self, agent):
        """Gateway turns should stop after automatic compression recovery."""
        agent.platform = "qqbot"
        err_413 = _make_413_error()
        agent.client.chat.completions.create.side_effect = [err_413]

        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "compressed summary"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert agent.client.chat.completions.create.call_count == 1
        assert result["compression_circuit_breaker"] is True
        assert result["completed"] is False
        assert "context 刚压缩恢复" in result["final_response"]
        assert "Codex" in result["final_response"]


class TestPreflightCompression:
    """Preflight compression should compress history before the first API call."""

    def test_compress_context_emits_lifecycle_status_before_work(self, agent):
        """Direct context compression should tell gateway users why the turn paused."""
        events = []
        agent.status_callback = lambda ev, msg: events.append((ev, msg))

        def _fake_compress(messages, current_tokens=None, focus_topic=None):
            events.append(("compress", "started"))
            return [{"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"}]

        with (
            patch.object(agent.context_compressor, "compress", side_effect=_fake_compress),
            patch.object(agent, "_build_system_prompt", return_value="new system prompt"),
            patch("run_agent.estimate_request_tokens_rough", return_value=42),
        ):
            compressed, new_system_prompt = agent._compress_context(
                [{"role": "user", "content": "hello"}],
                "system prompt",
                approx_tokens=1234,
            )

        assert compressed == [{"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"}]
        assert new_system_prompt == "new system prompt"
        assert events[0][0] == "lifecycle"
        assert "正在压缩 context" in events[0][1]
        assert events[1] == ("compress", "started")

    def test_preflight_compresses_oversized_history(self, agent):
        """When loaded history exceeds the model's context threshold, compress before API call."""
        agent.compression_enabled = True
        # Set a small context so the history is "oversized", but large enough
        # that the compressed result (2 short messages) fits in a single pass.
        agent.context_compressor.context_length = 2000
        agent.context_compressor.threshold_tokens = 200

        # Build a history that will be large enough to trigger preflight
        # (each message ~50 chars ≈ 13 tokens, 40 messages ≈ 520 tokens > 200 threshold)
        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message number {i} with some extra text padding"})
            big_history.append({"role": "assistant", "content": f"Response number {i} with extra padding here"})

        ok_resp = _mock_response(content="After preflight", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]
        status_messages = []
        agent.status_callback = lambda ev, msg: status_messages.append((ev, msg))

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            # Simulate compression reducing messages to a small set that fits
            mock_compress.return_value = (
                [
                    {"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"},
                    {"role": "user", "content": "hello"},
                ],
                "new system prompt",
            )
            result = agent.run_conversation("hello", conversation_history=big_history)

        # Preflight compression is a multi-pass loop (up to 3 passes for very
        # large sessions, breaking when no further reduction is possible).
        # First pass must have received the full oversized history.
        assert mock_compress.call_count >= 1, "Preflight compression never ran"
        first_call_messages = mock_compress.call_args_list[0].args[0]
        assert len(first_call_messages) >= 40, (
            f"First preflight pass should see the full history, got "
            f"{len(first_call_messages)} messages"
        )
        assert result["completed"] is True
        assert result["final_response"] == "After preflight"
        assert any(
            ev == "lifecycle" and "预先压缩 context" in msg
            for ev, msg in status_messages
        )

    def test_gateway_preflight_compression_halts_before_first_api_call(self, agent):
        """Gateway preflight compression should ask for a new stage, not continue."""
        agent.platform = "qqbot"
        agent.compression_enabled = True
        agent.context_compressor.context_length = 2000
        agent.context_compressor.threshold_tokens = 200

        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message number {i} with padding"})
            big_history.append({"role": "assistant", "content": f"Response number {i} with padding"})

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [
                    {"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"},
                    {"role": "user", "content": "hello"},
                ],
                "new system prompt",
            )
            result = agent.run_conversation("hello", conversation_history=big_history)

        mock_compress.assert_called_once()
        agent.client.chat.completions.create.assert_not_called()
        assert result["compression_circuit_breaker"] is True
        assert "继续下一阶段" in result["final_response"]

    def test_gateway_preflight_uses_session_context_platform(self, agent):
        """Gateway ContextVar platform should trigger the breaker even without agent.platform."""
        from gateway.session_context import clear_session_vars, set_session_vars

        agent.platform = None
        agent.compression_enabled = True
        agent.context_compressor.context_length = 2000
        agent.context_compressor.threshold_tokens = 200
        big_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i} padding"}
            for i in range(40)
        ]

        tokens = set_session_vars(platform="qqbot")
        try:
            with (
                patch.object(agent, "_compress_context") as mock_compress,
                patch.object(agent, "_persist_session"),
                patch.object(agent, "_save_trajectory"),
                patch.object(agent, "_cleanup_task_resources"),
            ):
                mock_compress.return_value = (
                    [
                        {"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"},
                        {"role": "user", "content": "hello"},
                    ],
                    "new system prompt",
                )
                result = agent.run_conversation("hello", conversation_history=big_history)
        finally:
            clear_session_vars(tokens)

        mock_compress.assert_called_once()
        agent.client.chat.completions.create.assert_not_called()
        assert result["compression_circuit_breaker"] is True

    def test_gateway_preflight_breaker_can_be_disabled_by_env(self, agent, monkeypatch):
        """Operators can opt out when they need transparent gateway compression."""
        agent.platform = "qqbot"
        agent.compression_enabled = True
        agent.context_compressor.context_length = 2000
        agent.context_compressor.threshold_tokens = 300
        monkeypatch.setenv("HERMES_DISABLE_COMPRESSION_CIRCUIT_BREAKER", "true")
        big_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i} padding"}
            for i in range(40)
        ]
        ok_resp = _mock_response(content="continued", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [
                    {"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"},
                    {"role": "user", "content": "hello"},
                ],
                "new system prompt",
            )
            result = agent.run_conversation("hello", conversation_history=big_history)

        mock_compress.assert_called_once()
        agent.client.chat.completions.create.assert_called_once()
        assert result["completed"] is True
        assert result["final_response"] == "continued"

    def test_compression_breaker_resolves_env_platform_sources(self, agent, monkeypatch):
        """Platform resolver should cover env fallbacks and trim whitespace."""
        from agent.conversation_loop import _compression_circuit_breaker_enabled

        agent.platform = None
        monkeypatch.delenv("HERMES_DISABLE_COMPRESSION_CIRCUIT_BREAKER", raising=False)
        monkeypatch.delenv("HERMES_PLATFORM", raising=False)
        monkeypatch.setenv("HERMES_SESSION_SOURCE", " qqbot ")
        assert _compression_circuit_breaker_enabled(agent) is True

        monkeypatch.setenv("HERMES_PLATFORM", " webui ")
        assert _compression_circuit_breaker_enabled(agent) is False

        monkeypatch.setenv("HERMES_PLATFORM", " telegram ")
        assert _compression_circuit_breaker_enabled(agent) is True

    @pytest.mark.parametrize("platform", ["cli", "webui", "local"])
    def test_preflight_compression_continues_for_direct_platforms(self, agent, platform):
        """Direct interfaces keep the historical transparent compression behavior."""
        agent.platform = platform
        agent.compression_enabled = True
        agent.context_compressor.context_length = 2000
        agent.context_compressor.threshold_tokens = 300
        big_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i} padding"}
            for i in range(40)
        ]
        ok_resp = _mock_response(content="continued", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [
                    {"role": "user", "content": f"{SUMMARY_PREFIX}\nPrevious conversation"},
                    {"role": "user", "content": "hello"},
                ],
                "new system prompt",
            )
            result = agent.run_conversation("hello", conversation_history=big_history)

        mock_compress.assert_called_once()
        agent.client.chat.completions.create.assert_called_once()
        assert result["completed"] is True
        assert result["final_response"] == "continued"

    def test_no_preflight_when_under_threshold(self, agent):
        """When history fits within context, no preflight compression needed."""
        agent.compression_enabled = True
        # Large context — history easily fits
        agent.context_compressor.context_length = 1000000
        agent.context_compressor.threshold_tokens = 850000

        small_history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

        ok_resp = _mock_response(content="No compression needed", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=small_history)

        mock_compress.assert_not_called()
        assert result["completed"] is True

    def test_no_preflight_when_compression_disabled(self, agent):
        """Preflight should not run when compression is disabled."""
        agent.compression_enabled = False
        agent.context_compressor.context_length = 100
        agent.context_compressor.threshold_tokens = 85

        big_history = [
            {"role": "user", "content": "x" * 1000},
            {"role": "assistant", "content": "y" * 1000},
        ] * 10

        ok_resp = _mock_response(content="OK", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=big_history)

        mock_compress.assert_not_called()

    def test_preflight_respects_anti_thrash(self, agent):
        """Preflight must call ``should_compress()`` so anti-thrash applies.

        Regression for #29335 — preflight used to bypass ``should_compress()``
        and re-trigger every turn even when the prior two passes each saved
        <10% (the canonical infinite-compression-loop signal).
        """
        agent.compression_enabled = True
        agent.context_compressor.context_length = 2000
        agent.context_compressor.threshold_tokens = 200

        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message {i} padded"})
            big_history.append({"role": "assistant", "content": f"Response {i} padded"})

        ok_resp = _mock_response(content="No preflight", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch.object(agent.context_compressor, "should_compress", return_value=False) as mock_should,
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=big_history)

        # The gate consulted should_compress — anti-thrash had a chance to vote.
        mock_should.assert_called()
        # And vetoed: even though tokens >= threshold, no compression ran.
        mock_compress.assert_not_called()
        assert result["completed"] is True


class TestToolResultPreflightCompression:
    """Compression should trigger when tool results push context past the threshold."""

    def test_large_tool_results_trigger_compression(self, agent):
        """When tool results push estimated tokens past threshold, compress before next call."""
        agent.compression_enabled = True
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 130_000  # below the 135k reported usage
        agent.context_compressor.last_prompt_tokens = 130_000
        agent.context_compressor.last_completion_tokens = 5_000

        tc = SimpleNamespace(
            id="tc1", type="function",
            function=SimpleNamespace(name="web_search", arguments='{"query":"test"}'),
        )
        tool_resp = _mock_response(
            content=None, finish_reason="stop", tool_calls=[tc],
            usage={"prompt_tokens": 130_000, "completion_tokens": 5_000, "total_tokens": 135_000},
        )
        ok_resp = _mock_response(
            content="Done after compression", finish_reason="stop",
            usage={"prompt_tokens": 50_000, "completion_tokens": 100, "total_tokens": 50_100},
        )
        agent.client.chat.completions.create.side_effect = [tool_resp, ok_resp]
        large_result = "x" * 100_000

        with (
            patch("run_agent.handle_function_call", return_value=large_result),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}], "compressed prompt",
            )
            result = agent.run_conversation("hello")

        mock_compress.assert_called_once()
        assert result["completed"] is True

    def test_anthropic_prompt_too_long_safety_net(self, agent):
        """Anthropic 'prompt is too long' error triggers compression as safety net."""
        err_400 = Exception(
            "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
            "'message': 'prompt is too long: 233153 tokens > 200000 maximum'}}"
        )
        err_400.status_code = 400
        ok_resp = _mock_response(content="Recovered", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_400, ok_resp]
        prefill = [
            {"role": "user", "content": "previous"},
            {"role": "assistant", "content": "answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}], "compressed",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert result["completed"] is True

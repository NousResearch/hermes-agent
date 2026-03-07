"""Tests for payload/context-length → compression retry logic in AIAgent.

Verifies that:
- HTTP 413 errors trigger history compression and retry
- HTTP 400 context-length errors trigger compression (not generic 4xx abort)
- Preflight compression proactively compresses oversized sessions before API calls
"""

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


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


class TestPreflightCompression:
    """Preflight compression should compress history before the first API call."""

    def test_preflight_compresses_oversized_history(self, agent):
        """When loaded history exceeds the model's context threshold, compress before API call."""
        agent.compression_enabled = True
        # Set a very small context so the history is "oversized"
        agent.context_compressor.context_length = 100
        agent.context_compressor.threshold_tokens = 85  # 85% of 100

        # Build a history that will be large enough to trigger preflight
        # (each message ~20 chars = ~5 tokens, 20 messages = ~100 tokens > 85 threshold)
        big_history = []
        for i in range(20):
            big_history.append({"role": "user", "content": f"Message number {i} with some extra text padding"})
            big_history.append({"role": "assistant", "content": f"Response number {i} with extra padding here"})

        ok_resp = _mock_response(content="After preflight", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [ok_resp]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            # Simulate compression reducing messages
            mock_compress.return_value = (
                [
                    {"role": "user", "content": "[CONTEXT SUMMARY]: Previous conversation"},
                    {"role": "user", "content": "hello"},
                ],
                "new system prompt",
            )
            result = agent.run_conversation("hello", conversation_history=big_history)

        # Preflight compression should have been called BEFORE the API call
        mock_compress.assert_called_once()
        assert result["completed"] is True
        assert result["final_response"] == "After preflight"

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

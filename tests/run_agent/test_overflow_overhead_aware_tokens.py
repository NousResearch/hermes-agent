"""Regression tests: overflow recovery handlers must pass overhead-aware token estimates.

PR fix (LCM issue 441): 413, context-overflow, and long-context-tier recovery handlers
were passing a messages-only token estimate to _compress_context instead of the
overhead-aware estimate_request_tokens_rough(api_messages, tools=agent.tools or None),
which includes tool schemas and system prompt overhead.

These tests assert that each recovery handler:
1. Calls estimate_request_tokens_rough with a non-None `tools` argument.
2. Passes the resulting value as approx_tokens to _compress_context.

The sentinel pattern (return_value=987654) makes the assertion unambiguous: if
approx_tokens==987654 the overhead-aware path was taken; any other value means the
handler used a different (likely messages-only) estimate.
"""

import pytest

from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

from run_agent import AIAgent
import run_agent


# ---------------------------------------------------------------------------
# Shared fixtures / helpers (mirrored from test_413_compression.py)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Short-circuit all time.sleep and jittered_backoff calls."""
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    from agent import retry_utils as _retry_utils
    monkeypatch.setattr(_retry_utils, "jittered_backoff", lambda *a, **k: 0.0)


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


@pytest.fixture()
def agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
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
        a.compression_enabled = True
        a.save_trajectories = False
        return a


def _prefill():
    return [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]


# ---------------------------------------------------------------------------
# Sentinel: any value that could not coincidentally appear from a messages-only
# estimate during these tests.
# ---------------------------------------------------------------------------
_SENTINEL_TOKENS = 987_654


# ---------------------------------------------------------------------------
# 1. 413 / payload-too-large handler
# ---------------------------------------------------------------------------


class TestHTTP413OverheadAwareTokens:
    """The 413 recovery handler must call estimate_request_tokens_rough with
    tools=agent.tools (non-None) and pass the result as approx_tokens."""

    def test_413_passes_overhead_aware_tokens_to_compress(self, agent):
        """approx_tokens passed to _compress_context equals the overhead-aware estimate."""
        err = Exception("Request entity too large")
        err.status_code = 413
        ok_resp = _mock_response(content="Success", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err, ok_resp]

        with (
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                return_value=_SENTINEL_TOKENS,
            ) as mock_estimate,
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "compressed"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=_prefill())

        # _compress_context must have been called at least once for compression
        mock_compress.assert_called()

        # Find the call that came from the 413 handler (approx_tokens=sentinel)
        compress_kwargs_list = [c.kwargs for c in mock_compress.call_args_list]
        sentinel_call = next(
            (kw for kw in compress_kwargs_list if kw.get("approx_tokens") == _SENTINEL_TOKENS),
            None,
        )
        assert sentinel_call is not None, (
            f"No _compress_context call received approx_tokens={_SENTINEL_TOKENS}. "
            f"Calls received approx_tokens values: "
            f"{[kw.get('approx_tokens') for kw in compress_kwargs_list]}"
        )

    def test_413_estimate_called_with_non_none_tools(self, agent):
        """estimate_request_tokens_rough must receive tools=<non-None> in the 413 handler."""
        err = Exception("Request entity too large")
        err.status_code = 413
        ok_resp = _mock_response(content="Success", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err, ok_resp]

        estimate_calls = []

        def _capture_estimate(messages, tools=None):
            estimate_calls.append({"messages": messages, "tools": tools})
            return _SENTINEL_TOKENS

        with (
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                side_effect=_capture_estimate,
            ),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "compressed"}],
                "compressed prompt",
            )
            agent.run_conversation("hello", conversation_history=_prefill())

        # At least one estimate call from the 413 handler must have non-None tools
        handler_calls_with_tools = [c for c in estimate_calls if c["tools"] is not None]
        assert handler_calls_with_tools, (
            "estimate_request_tokens_rough was never called with non-None tools "
            "during 413 recovery. All calls: "
            + str([c["tools"] for c in estimate_calls])
        )


# ---------------------------------------------------------------------------
# 2. Context-overflow / input-too-large handler
# ---------------------------------------------------------------------------


class TestContextOverflowOverheadAwareTokens:
    """The context-overflow (input overflow) recovery handler must call
    estimate_request_tokens_rough with tools=agent.tools and pass the result
    as approx_tokens to _compress_context."""

    @staticmethod
    def _make_context_overflow_error():
        """Build a 400 error that the classifier routes to context_overflow."""
        err = Exception(
            "Error code: 400 - {'error': {'message': "
            "\"This endpoint's maximum context length is 128000 tokens. "
            "However, you requested about 200000 tokens. "
            "Please reduce the length of the messages.\"}}"
        )
        err.status_code = 400
        return err

    def test_context_overflow_passes_overhead_aware_tokens_to_compress(self, agent):
        """approx_tokens passed to _compress_context equals the overhead-aware estimate."""
        err = self._make_context_overflow_error()
        ok_resp = _mock_response(content="Recovered", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err, ok_resp]

        with (
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                return_value=_SENTINEL_TOKENS,
            ) as mock_estimate,
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "compressed"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=_prefill())

        mock_compress.assert_called()

        compress_kwargs_list = [c.kwargs for c in mock_compress.call_args_list]
        sentinel_call = next(
            (kw for kw in compress_kwargs_list if kw.get("approx_tokens") == _SENTINEL_TOKENS),
            None,
        )
        assert sentinel_call is not None, (
            f"No _compress_context call received approx_tokens={_SENTINEL_TOKENS}. "
            f"Calls received approx_tokens values: "
            f"{[kw.get('approx_tokens') for kw in compress_kwargs_list]}"
        )

    def test_context_overflow_estimate_called_with_non_none_tools(self, agent):
        """estimate_request_tokens_rough must receive tools=<non-None> in the context-overflow handler."""
        err = self._make_context_overflow_error()
        ok_resp = _mock_response(content="Recovered", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err, ok_resp]

        estimate_calls = []

        def _capture_estimate(messages, tools=None):
            estimate_calls.append({"messages": messages, "tools": tools})
            return _SENTINEL_TOKENS

        with (
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                side_effect=_capture_estimate,
            ),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "compressed"}],
                "compressed prompt",
            )
            agent.run_conversation("hello", conversation_history=_prefill())

        handler_calls_with_tools = [c for c in estimate_calls if c["tools"] is not None]
        assert handler_calls_with_tools, (
            "estimate_request_tokens_rough was never called with non-None tools "
            "during context-overflow recovery. All calls: "
            + str([c["tools"] for c in estimate_calls])
        )

    def test_prompt_too_long_variant_passes_overhead_aware_tokens(self, agent):
        """Anthropic 'prompt is too long' error also routes to context_overflow handler."""
        err = Exception(
            "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
            "'message': 'prompt is too long: 233153 tokens > 200000 maximum'}}"
        )
        err.status_code = 400
        ok_resp = _mock_response(content="Recovered", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err, ok_resp]

        with (
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                return_value=_SENTINEL_TOKENS,
            ),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "compressed"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=_prefill())

        mock_compress.assert_called()
        compress_kwargs_list = [c.kwargs for c in mock_compress.call_args_list]
        sentinel_call = next(
            (kw for kw in compress_kwargs_list if kw.get("approx_tokens") == _SENTINEL_TOKENS),
            None,
        )
        assert sentinel_call is not None, (
            f"'prompt is too long' path did not pass overhead-aware approx_tokens. "
            f"Got: {[kw.get('approx_tokens') for kw in compress_kwargs_list]}"
        )


# ---------------------------------------------------------------------------
# 3. Anthropic long-context tier (429) handler
# ---------------------------------------------------------------------------


class TestLongContextTierOverheadAwareTokens:
    """The Anthropic long-context-tier 429 handler must call
    estimate_request_tokens_rough with tools=agent.tools and pass the result
    as approx_tokens to _compress_context."""

    @staticmethod
    def _make_long_context_tier_error():
        """Build a 429 'extra usage required for long context requests' error."""
        err = Exception(
            "Error code: 429 - {'error': {'type': 'rate_limit_error', "
            "'message': 'Extra usage is required for long context requests. "
            "Please enable extra usage in your account settings.'}}"
        )
        err.status_code = 429
        return err

    def test_long_context_tier_passes_overhead_aware_tokens_to_compress(self, agent):
        """approx_tokens passed to _compress_context equals the overhead-aware estimate."""
        err = self._make_long_context_tier_error()
        ok_resp = _mock_response(content="Recovered after context tier", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err, ok_resp]

        with (
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                return_value=_SENTINEL_TOKENS,
            ),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "compressed"}],
                "compressed prompt",
            )
            result = agent.run_conversation("hello", conversation_history=_prefill())

        mock_compress.assert_called()
        compress_kwargs_list = [c.kwargs for c in mock_compress.call_args_list]
        sentinel_call = next(
            (kw for kw in compress_kwargs_list if kw.get("approx_tokens") == _SENTINEL_TOKENS),
            None,
        )
        assert sentinel_call is not None, (
            f"Long-context-tier handler did not pass overhead-aware approx_tokens. "
            f"Got: {[kw.get('approx_tokens') for kw in compress_kwargs_list]}"
        )

    def test_long_context_tier_estimate_called_with_non_none_tools(self, agent):
        """estimate_request_tokens_rough must receive tools=<non-None> in the long-context handler."""
        err = self._make_long_context_tier_error()
        ok_resp = _mock_response(content="Recovered", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err, ok_resp]

        estimate_calls = []

        def _capture_estimate(messages, tools=None):
            estimate_calls.append({"messages": messages, "tools": tools})
            return _SENTINEL_TOKENS

        with (
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                side_effect=_capture_estimate,
            ),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "compressed"}],
                "compressed prompt",
            )
            agent.run_conversation("hello", conversation_history=_prefill())

        handler_calls_with_tools = [c for c in estimate_calls if c["tools"] is not None]
        assert handler_calls_with_tools, (
            "estimate_request_tokens_rough was never called with non-None tools "
            "during long-context-tier recovery. All calls: "
            + str([c["tools"] for c in estimate_calls])
        )

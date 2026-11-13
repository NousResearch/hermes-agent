"""Regression tests for the provider response id in the post_api_request hook.

Observability plugins join a gateway's spend log on the provider's own response
id (``chatcmpl-...`` / ``gen-...``). The hook payload therefore carries it as
``response_id`` (and inside the sanitised ``response`` payload as ``id``), taken
from the response object the provider returned.

Contract under test:

- A non-streamed response reports its ``id`` as ``response_id``.
- A streamed response reports the id its chunks carried.
- A response without an id reports ``None`` — never a fabricated value.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from tests.agent.test_run_agent import (
    _make_tool_defs,
    _mock_response,
)


def _make_stream_chunk(content=None, finish_reason=None, model=None, usage=None, chunk_id=None):
    """Mock streaming chunk matching OpenAI's ChatCompletionChunk shape."""
    delta = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=model, usage=usage, id=chunk_id)


@pytest.fixture()
def agent():
    """Minimal AIAgent with mocked OpenAI client and tool loading."""
    with (
        patch(
            "model_tools.get_tool_definitions",
            return_value=_make_tool_defs("web_search"),
        ),
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


def _run_with_hooks(agent, message="hi"):
    """Run a conversation with pre/post api hooks recorded; return post calls."""
    hook_calls = []

    def _record_hook(name, **kwargs):
        hook_calls.append((name, kwargs))
        return []

    with (
        patch(
            "hermes_cli.lifecycle.has_hook",
            side_effect=lambda name: name in {"pre_api_request", "post_api_request"},
        ),
        patch("hermes_cli.lifecycle.invoke_hook", side_effect=_record_hook),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(message)

    post = [kw for name, kw in hook_calls if name == "post_api_request"]
    return result, post


class TestPostApiRequestResponseId:
    """The response_id kwarg observed by post_api_request subscribers."""

    def test_non_streamed_response_reports_provider_id(self, agent):
        response = _mock_response(content="Done", finish_reason="stop")
        response.id = "chatcmpl-abc123"
        agent.client.chat.completions.create.return_value = response

        result, post = _run_with_hooks(agent)

        assert result["final_response"] == "Done"
        assert len(post) == 1
        assert post[0]["response_id"] == "chatcmpl-abc123"
        # The sanitised response payload carries the same id.
        assert post[0]["response"]["id"] == "chatcmpl-abc123"

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_streamed_response_reports_chunk_id(self, _mock_close, mock_create, agent):
        chunks = [
            _make_stream_chunk(content="Hello", chunk_id="chatcmpl-stream42"),
            _make_stream_chunk(
                content=" world",
                finish_reason="stop",
                model="test-model",
                chunk_id="chatcmpl-stream42",
            ),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_create.return_value = mock_client
        agent.stream_delta_callback = lambda _text: None

        result, post = _run_with_hooks(agent)

        assert result["final_response"] == "Hello world"
        assert len(post) == 1
        assert post[0]["response_id"] == "chatcmpl-stream42"

    def test_response_without_id_reports_none(self, agent):
        """No provider id → None, never a fabricated value."""
        agent.client.chat.completions.create.return_value = _mock_response(
            content="Done", finish_reason="stop"
        )

        _, post = _run_with_hooks(agent)

        assert len(post) == 1
        assert post[0]["response_id"] is None
        assert post[0]["response"]["id"] is None

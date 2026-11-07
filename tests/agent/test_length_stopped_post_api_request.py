"""A length-stopped attempt still closes with ``post_api_request``.

``finish_reason='length'`` sends the response to truncation recovery before
``normalize_model_response``, which is where ``post_api_request`` fires. The attempt is a
finished provider call, so an observer that opened a span on ``pre_api_request`` must see
it end (observer-hooks.md: every request ends with ``post_api_request`` or
``api_request_error``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import FINISH_REASON_LENGTH


@pytest.fixture()
def loop_agent():
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
        a.context_compressor.context_length = 200_000
        return a


def _response(content: str, finish_reason: str):
    from tests.agent.test_run_agent import _mock_assistant_msg
    return SimpleNamespace(
        id="resp",
        model="test/model",
        choices=[SimpleNamespace(index=0, message=_mock_assistant_msg(content=content), finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=40, total_tokens=140),
    )


def test_length_stopped_attempt_fires_post_api_request(loop_agent):
    loop_agent.client.chat.completions.create.side_effect = [
        _response("This answer is cut off mid", FINISH_REASON_LENGTH),
        _response(" sentence and continues here.", "stop"),
    ]
    hook_calls = []

    def _record_hook(name, **kwargs):
        hook_calls.append((name, kwargs))
        return []

    with (
        patch("hermes_cli.lifecycle.has_hook",
              side_effect=lambda name: name in {"pre_api_request", "post_api_request", "api_request_error"}),
        patch("hermes_cli.lifecycle.invoke_hook", side_effect=_record_hook),
        patch.object(loop_agent, "_persist_session"),
        patch.object(loop_agent, "_save_trajectory"),
        patch.object(loop_agent, "_cleanup_task_resources"),
    ):
        loop_agent.run_conversation("write a long answer")

    pre = [kw["api_request_id"] for name, kw in hook_calls if name == "pre_api_request"]
    closed = [kw["api_request_id"] for name, kw in hook_calls if name in {"post_api_request", "api_request_error"}]
    assert len(pre) == 2
    assert closed == pre
    first_post = next(kw for name, kw in hook_calls if name == "post_api_request")
    assert first_post["finish_reason"] == FINISH_REASON_LENGTH

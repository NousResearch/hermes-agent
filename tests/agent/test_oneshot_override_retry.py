"""One-shot request overrides survive a retry of the same call (#99897).

The continuation reasoning-off flag and the ephemeral max_tokens (continuation boost, output-cap clamp) were
cleared while the request was BUILT. The retry loop rebuilds the request on every attempt, so after one 429/5xx
the retried continuation went out with full thinking and the default cap again, and a clamped request hit the
same 400 and clamped (and compressed) a second time. Real AIAgent and retry loop; only the client is faked.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import FINISH_REASON_LENGTH


class _ProviderError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = {}
        self.response = SimpleNamespace(headers={})


@pytest.fixture()
def agent():
    from run_agent import AIAgent

    with patch("model_tools.get_tool_definitions", return_value=[]), \
            patch("model_tools.check_toolset_requirements", return_value={}), \
            patch("agent.process_bootstrap.OpenAI"):
        a = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                    skip_context_files=True, skip_memory=True)
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a.compression_enabled = False
    a.save_trajectories = False
    return a


def _run(a, message):
    with patch.object(a, "_persist_session"), patch.object(a, "_save_trajectory"), \
            patch.object(a, "_cleanup_task_resources"), \
            patch("agent.turn_api_error.compute_error_backoff", lambda *x, **k: 0), \
            patch("agent.turn_api_error.interruptible_backoff_sleep", lambda *x, **k: None), \
            patch("time.sleep", lambda *_: None):
        return a.run_conversation(message)


def _sent(a):
    return [(c.kwargs.get("max_tokens"), (c.kwargs.get("extra_body") or {}).get("reasoning"))
            for c in a.client.chat.completions.create.call_args_list]


def test_a_retried_continuation_keeps_its_overrides_and_the_next_turn_drops_them(agent):
    from tests.agent.test_run_agent import _mock_assistant_msg, _mock_response

    thinking_only = SimpleNamespace(id="c1", model="test/model", usage=None, choices=[SimpleNamespace(
        index=0, message=_mock_assistant_msg(content=""), finish_reason=FINISH_REASON_LENGTH)])
    agent.reasoning_config = {"enabled": True, "effort": "high"}
    agent._supports_reasoning_extra_body = lambda: True
    agent.client.chat.completions.create.side_effect = [
        thinking_only, _ProviderError("Service Unavailable", status_code=503), _mock_response(content="answer"),
        _mock_response(content="next turn")]

    assert _run(agent, "long report")["completed"] is True
    first, continuation, retry = _sent(agent)
    assert continuation[1] == {"enabled": False, "effort": "none"} and continuation[0] is not None
    assert retry == continuation

    _run(agent, "and now?")
    assert _sent(agent)[3] == first


def test_a_retried_clamped_request_keeps_its_clamp(agent):
    from tests.agent.test_run_agent import _mock_response

    agent.max_tokens = 100000
    agent.compression_enabled = True
    overflow = ("This endpoint's maximum context length is 200000 tokens. However, you requested about 250000 "
                "tokens (150000 of text input, 0 of tool input, 100000 in the output). Please reduce the length "
                "of either one, or use the \"middle-out\" transform.")
    agent.client.chat.completions.create.side_effect = [
        _ProviderError(overflow, status_code=400), _ProviderError("Service Unavailable", status_code=503),
        _mock_response(content="fine")]

    assert _run(agent, "hi")["completed"] is True
    caps = [cap for cap, _reasoning in _sent(agent)]
    assert caps[1] < 100000 and caps[2] == caps[1]

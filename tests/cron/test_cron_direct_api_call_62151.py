"""Regression guard for #62151 — gateway cron must not wedge on 2nd+ API call.

Gateway-fired cron jobs hung forever on the 2nd+ non-streaming API call when
``interruptible_api_call`` spawned a daemon worker inside nested cron thread
pools. The worker logged client creation but never opened a TCP connection.
The same job succeeded via ``hermes cron tick``. Cron has no interactive
interrupt surface, so the fix routes cron through a synchronous direct call on
the conversation thread instead of the interrupt worker.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.chat_completion_helpers import (
    direct_api_call,
    should_use_direct_api_call,
)


def _make_agent(*, platform="cron"):
    agent = MagicMock()
    agent.platform = platform
    agent.api_mode = "chat_completions"
    agent._touch_activity = MagicMock()
    agent._create_request_openai_client = MagicMock()
    agent._close_request_openai_client = MagicMock()
    return agent


def test_should_use_direct_api_call_only_for_cron_platform():
    assert should_use_direct_api_call(_make_agent(platform="cron")) is True
    assert should_use_direct_api_call(_make_agent(platform="cli")) is False
    assert should_use_direct_api_call(_make_agent(platform="telegram")) is False
    assert should_use_direct_api_call(_make_agent(platform=None)) is False


def test_direct_api_call_runs_two_sequential_requests_on_same_thread():
    """Mirror the 2nd+ call failure mode: two back-to-back completions.create."""
    agent = _make_agent()
    calls = {"n": 0}
    fake_client = MagicMock()

    def _create(**_kwargs):
        calls["n"] += 1
        return fake_client

    fake_client.chat.completions.create.side_effect = [
        SimpleNamespace(id="first"),
        SimpleNamespace(id="second"),
    ]
    agent._create_request_openai_client.side_effect = _create

    first = direct_api_call(agent, {"model": "m", "messages": []})
    second = direct_api_call(agent, {"model": "m", "messages": []})

    assert first.id == "first"
    assert second.id == "second"
    assert calls["n"] == 2
    assert fake_client.chat.completions.create.call_count == 2
    assert agent._close_request_openai_client.call_count == 2

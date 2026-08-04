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
    agent.provider = "openrouter"
    agent._interrupt_requested = False
    agent._touch_activity = MagicMock()
    agent._create_request_openai_client = MagicMock()
    agent._close_request_openai_client = MagicMock()
    return agent


def test_should_use_direct_api_call_only_for_cron_openai_wire():
    assert should_use_direct_api_call(_make_agent(platform="cron")) is True
    assert should_use_direct_api_call(_make_agent(platform="cli")) is False
    assert should_use_direct_api_call(_make_agent(platform="telegram")) is False
    assert should_use_direct_api_call(_make_agent(platform=None)) is False

    for api_mode in ("codex_responses", "anthropic_messages", "bedrock_converse"):
        agent = _make_agent(platform="cron")
        agent.api_mode = api_mode
        assert should_use_direct_api_call(agent) is False

    moa = _make_agent(platform="cron")
    moa.provider = "moa"
    assert should_use_direct_api_call(moa) is False


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


def test_direct_api_call_keeps_activity_alive_during_slow_wait(monkeypatch):
    """Mid-wait activity heartbeats must tick while the inline request blocks.

    Subagents use direct_api_call (non-streaming). Without mid-call
    ``_touch_activity`` ticks, the async stall monitor treats a slow-but-
    healthy local model wait as frozen progress and interrupts around 450s
    (``Operation interrupted: waiting for model response``).
    """
    import threading
    import time

    from agent import chat_completion_helpers as helpers

    monkeypatch.setattr(helpers, "_DIRECT_API_ACTIVITY_HEARTBEAT_SECONDS", 0.05)

    agent = _make_agent(platform="subagent")
    started = threading.Event()
    release = threading.Event()
    fake_client = MagicMock()

    def _slow_create(**_kwargs):
        started.set()
        assert release.wait(timeout=2.0)
        return SimpleNamespace(id="slow")

    fake_client.chat.completions.create.side_effect = _slow_create
    agent._create_request_openai_client.return_value = fake_client

    result_box = {}

    def _runner():
        result_box["response"] = direct_api_call(
            agent, {"model": "m", "messages": []}
        )

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    assert started.wait(timeout=2.0)

    # Allow several heartbeat intervals while the request is still blocked.
    deadline = time.time() + 1.0
    while time.time() < deadline and agent._touch_activity.call_count < 3:
        time.sleep(0.05)

    touches_while_blocked = agent._touch_activity.call_count
    release.set()
    worker.join(timeout=2.0)

    assert worker.is_alive() is False
    assert result_box["response"].id == "slow"
    assert touches_while_blocked >= 3, (
        f"expected mid-wait activity heartbeats, got {touches_while_blocked}"
    )
    assert all(
        call.args[0] == "waiting for non-streaming API response"
        for call in agent._touch_activity.call_args_list
    )

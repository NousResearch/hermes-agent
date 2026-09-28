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

    codex = _make_agent(platform="cron")
    codex.api_mode = "codex_responses"
    assert should_use_direct_api_call(codex) is True  # #69734

    for api_mode in ("anthropic_messages", "bedrock_converse"):
        agent = _make_agent(platform="cron")
        agent.api_mode = api_mode
        assert should_use_direct_api_call(agent) is False

    moa = _make_agent(platform="cron")
    moa.provider = "moa"
    assert should_use_direct_api_call(moa) is False




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


def test_direct_api_call_heartbeat_stops_on_exception(monkeypatch):
    """The activity heartbeat thread must be joined on error paths so no
    stray _touch_activity fires after the call has failed.
    """
    import threading
    import time

    from agent import chat_completion_helpers as helpers

    monkeypatch.setattr(helpers, "_DIRECT_API_ACTIVITY_HEARTBEAT_SECONDS", 0.05)

    agent = _make_agent(platform="subagent")
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("provider down")
    agent._create_request_openai_client.return_value = fake_client

    raised = threading.Event()

    def _runner():
        try:
            direct_api_call(agent, {"model": "m", "messages": []})
        except RuntimeError:
            raised.set()

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    worker.join(timeout=3.0)

    assert raised.is_set(), "expected RuntimeError from direct_api_call"
    # Give any stray heartbeat a chance to fire after the call returned.
    time.sleep(0.2)
    touches_before = agent._touch_activity.call_count
    time.sleep(0.2)
    touches_after = agent._touch_activity.call_count
    assert touches_after == touches_before, (
        "heartbeat thread still firing after direct_api_call raised"
    )

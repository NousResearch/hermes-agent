"""The gateway tells the agent, per turn, when a bare silence marker may not stay silent.

A human turn that ends on a bare marker gets a visible "needed a reply" fallback (#110952). When
the marker follows a reply the model already wrote beside a housekeeping tool, the agent ends the
turn on that reply instead (``agent/turn_final_response.py``), before the output transform and
persistence. The agent cannot judge the turn's silence policy itself, so ``TurnRunner`` sets
``_silence_marker_needs_reply`` from the same ``silence_allowed`` verdict the outer shaping uses,
for the duration of that run only, including each queued follow-up's own verdict.

Driven through the real ``GatewayRunner._run_agent`` turn wiring; only the agent is a recording
double (the real agent half is covered in ``tests/agent/test_reply_line_before_silence.py`` and,
composed end to end, in ``test_silence_reply_composed*.py``).
"""

import importlib
import sys
import types

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.response_filters import INTERNAL_NOTIFICATION_DISPLAY_KIND
from gateway.session import SessionSource
from tests.gateway.test_run_progress_topics import _ACTIVE_ADAPTER, ProgressCaptureAdapter, _make_runner


class RecordingAgent:
    """Records the per-turn flag the gateway set while ``run_conversation`` runs."""

    seen: list = []
    instances: list = []

    def __init__(self, **kwargs):
        self.tools = []
        type(self).instances.append(self)

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        type(self).seen.append(getattr(self, "_silence_marker_needs_reply", None))
        return {"final_response": "ok", "messages": [], "api_calls": 1}


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="dm-1", chat_type="dm", user_id="u1")


async def _run(monkeypatch, tmp_path, *, queued_reply_expected=None, queued=False, **turn_kwargs):
    import hermes_yaml as yaml

    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"display": {}}), encoding="utf-8")
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = RecordingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(RecordingAgent, "seen", [])
    monkeypatch.setattr(RecordingAgent, "instances", [])

    adapter = ProgressCaptureAdapter(platform=Platform.TELEGRAM)
    monkeypatch.setitem(_ACTIVE_ADAPTER, "adapter", adapter)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    session_key = "agent:main:telegram:dm:dm-1"
    if queued:
        adapter._pending_messages[session_key] = MessageEvent(
            text="and tomorrow?", message_type=MessageType.TEXT, source=_source(), message_id="queued-1",
            reply_expected=queued_reply_expected)
    await runner._run_agent(
        message="when is my meeting?", context_prompt="", history=[], source=_source(),
        session_id="sess-reply-flag", session_key=session_key, **turn_kwargs,
    )
    return RecordingAgent.seen


@pytest.mark.asyncio
@pytest.mark.parametrize(("turn_kwargs", "expected"), [
    ({"reply_expected": True}, True),
    ({}, True),  # unknown addressing on a human turn: the marker would warn
    ({"reply_expected": False}, False),
    ({"reply_expected": True, "persist_user_display_kind": INTERNAL_NOTIFICATION_DISPLAY_KIND}, False),
], ids=["addressed", "unknown", "not-addressed", "machinery"])
async def test_flag_follows_the_turns_silence_verdict(monkeypatch, tmp_path, turn_kwargs, expected):
    seen = await _run(monkeypatch, tmp_path, **turn_kwargs)

    assert seen == [expected]
    # Scoped to the run: a cached agent never carries it into another host's turn.
    assert all(agent._silence_marker_needs_reply is False for agent in RecordingAgent.instances)


@pytest.mark.asyncio
async def test_a_queued_follow_up_gets_its_own_verdict(monkeypatch, tmp_path):
    seen = await _run(monkeypatch, tmp_path, reply_expected=True, queued=True, queued_reply_expected=False)

    assert seen == [True, False]

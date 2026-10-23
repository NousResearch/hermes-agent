"""Gateway-level regression: a real goal judge that ends in a terminal state must NOT enqueue a
continuation dispatch.

The runaway-completion bug surfaced as the gateway's post-turn hook ``_post_turn_goal_continuation``
dispatching continuation turns for a goal the user had stopped. Its unit tests pin the
manager-level decision, but the actual dispatch is a separate layer: only
``decision["should_continue"]`` truthy enqueues a synthetic user message into the adapter FIFO.

These tests drive the REAL ``_post_turn_goal_continuation`` with a real ``GoalManager`` +
``SessionDB`` and assert the FIFO stays empty when the judge ends the goal (done) or a user stop
lands while the judge runs (pause-during-judge) — proving no continuation is dispatched, not just
that the helper returns ``should_continue=False``.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, build_session_key
from hermes_cli.goals import GoalManager


class _ProbeAdapter(BasePlatformAdapter):
    """Minimal adapter recording FIFO pending slots + sent messages."""

    def __init__(self) -> None:
        super().__init__(PlatformConfig(enabled=True, token="x"), Platform.SLACK)
        self.sent: list[str] = []

    async def start(self):  # pragma: no cover - unused
        pass

    async def stop(self):  # pragma: no cover - unused
        pass

    async def connect(self):  # pragma: no cover - unused
        pass

    async def disconnect(self):  # pragma: no cover - unused
        pass

    async def get_chat_info(self, chat_id):  # pragma: no cover - unused
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)

        class _R:
            success = True
            message_id = "m1"

        return _R()

    async def send_typing(self, chat_id, metadata=None):
        pass


def _slack_source() -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK,
        user_id="U1",
        chat_id="C1",
        user_name="tester",
        chat_type="channel",
    )


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    goals._get_session_db()  # pre-warm the cache so the async path's first write isn't dropped
    yield home
    goals._DB_CACHE.clear()


@pytest.mark.asyncio
async def test_done_verdict_does_not_dispatch_a_continuation(hermes_home):
    """A real done verdict in _post_turn_goal_continuation must leave the adapter FIFO empty."""
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.SLACK: PlatformConfig(enabled=True, token="x")},
    )
    runner.adapters = {Platform.SLACK: _ProbeAdapter()}
    adapter = runner.adapters[Platform.SLACK]
    src = _slack_source()
    key = build_session_key(src)
    runner._queued_events = {}
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=key,
        session_id="goal-done-sess",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.SLACK,
        chat_type="channel",
    )

    GoalManager("goal-done-sess").set("count to 25", max_turns=20)

    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("done", "verified 25 reached", False, None, False),
    ):
        await runner._post_turn_goal_continuation(
            session_entry=runner.session_store.get_or_create_session.return_value,
            source=src,
            final_response="Done — goal complete.",
        )

    assert key not in adapter._pending_messages, (
        "a done goal must not enqueue another continuation turn"
    )


@pytest.mark.asyncio
async def test_pause_during_judge_does_not_dispatch_a_continuation(hermes_home):
    """A user pause landing while the judge runs must not enqueue a continuation in the gateway."""
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.SLACK: PlatformConfig(enabled=True, token="x")},
    )
    runner.adapters = {Platform.SLACK: _ProbeAdapter()}
    adapter = runner.adapters[Platform.SLACK]
    src = _slack_source()
    key = build_session_key(src)
    runner._queued_events = {}
    sess = SessionEntry(
        session_key=key,
        session_id="goal-paused-sess",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.SLACK,
        chat_type="channel",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = sess

    GoalManager("goal-paused-sess").set("count to 25", max_turns=20)

    def _judge_pauses(goal, last_response, **kwargs):
        GoalManager("goal-paused-sess").pause(reason="user-paused")
        return ("continue", "keep going", False, None, False)

    with patch("hermes_cli.goals.judge_goal", side_effect=_judge_pauses):
        await runner._post_turn_goal_continuation(
            session_entry=sess,
            source=src,
            final_response="worked toward it",
        )

    assert key not in adapter._pending_messages, (
        "a pause landing during the judge must not enqueue a continuation onto a stopped goal"
    )


@pytest.mark.asyncio
async def test_post_turn_hooks_forward_tool_evidence_to_the_goal_judge(hermes_home):
    """The real post-turn path hands the judge the turn's tool artifacts, not just the prose.

    The gateway hook is selected by an explicit flag in ``_run_post_turn_hooks`` (the tuple's
    label is log text only), and that flag is what carries ``agent_result`` down to
    ``evaluate_after_turn`` → ``judge_goal``. A judge that only sees final prose keeps answering
    CONTINUE on a goal whose proof lives in a tool result, so this pins the whole rail.
    """
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.SLACK: PlatformConfig(enabled=True, token="x")},
    )
    runner.adapters = {Platform.SLACK: _ProbeAdapter()}
    adapter = runner.adapters[Platform.SLACK]
    src = _slack_source()
    key = build_session_key(src)
    runner._queued_events = {}

    class _Store:
        def __init__(self, entry):
            self.entry = entry

        def get_or_create_session(self, source, *, touch_activity=True):
            return self.entry

        def _generate_session_key(self, source):
            return key

    sess = SessionEntry(
        session_key=key,
        session_id="goal-evidence-sess",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.SLACK,
        chat_type="channel",
    )
    runner.session_store = _Store(sess)
    GoalManager("goal-evidence-sess").set("write count.json with count 25", max_turns=20)

    seen: dict[str, object] = {}

    def _judge(goal, last_response, **kwargs):
        seen["evidence"] = kwargs.get("evidence")
        return ("done", "count.json holds count 25", False, None, False)

    agent_result = {
        "final_response": "Wrote it.",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path": "count.json", "content": "{\\"count\\": 25}"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": '{"bytes_written": 13}'},
            {"role": "assistant", "content": "Wrote it."},
        ],
    }

    with patch("hermes_cli.goals.judge_goal", side_effect=_judge):
        await runner._run_post_turn_hooks(agent_result=agent_result, source=src, is_internal=False)

    evidence = seen.get("evidence") or []
    assert any("write_file" in item and "count.json" in item for item in evidence), (
        f"the judge never received the tool evidence that proves the goal: {evidence!r}"
    )
    assert key not in adapter._pending_messages, "a done goal must not enqueue another turn"


@pytest.mark.asyncio
async def test_followup_after_goal_done_does_not_dispatch(hermes_home):
    """A continuation ticking after the goal is already done must be a no-op in the gateway."""
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.SLACK: PlatformConfig(enabled=True, token="x")},
    )
    runner.adapters = {Platform.SLACK: _ProbeAdapter()}
    adapter = runner.adapters[Platform.SLACK]
    src = _slack_source()
    key = build_session_key(src)
    runner._queued_events = {}
    sess = SessionEntry(
        session_key=key,
        session_id="goal-already-done-sess",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.SLACK,
        chat_type="channel",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = sess

    mgr = GoalManager("goal-already-done-sess")
    mgr.set("count to 25", max_turns=20)
    mgr.mark_done("achieved")  # the goal is already terminal before the hook ticks

    # A sloppy CONTINUE judge must not resurrect or dispatch onto the done goal.
    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("continue", "keep going", False, None, False),
    ):
        await runner._post_turn_goal_continuation(
            session_entry=sess,
            source=src,
            final_response="the goal is already done",
        )

    assert key not in adapter._pending_messages, (
        "a done goal must not be re-dispatched by a later continuation tick"
    )
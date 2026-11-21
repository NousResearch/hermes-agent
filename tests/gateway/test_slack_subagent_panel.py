"""Slack subagent panel: delegate_task children render as one in-place message with Stop / Result.

Drives the real ``TurnRunner.progress_callback`` (the parent agent's ``tool_progress_callback``) into the
real ``SlackAdapter`` with a capturing Web API client, and the Stop / Result presses through the
adapter's real Block Kit action handlers.
"""

import asyncio
import importlib.util
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_slack_mock():
    if "slack_bolt" in sys.modules:
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    sys.modules["slack_bolt"] = slack_bolt
    sys.modules["slack_bolt.async_app"] = slack_bolt.async_app
    handler_mod = MagicMock()
    handler_mod.AsyncSocketModeHandler = MagicMock
    sys.modules["slack_bolt.adapter"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode.async_handler"] = handler_mod
    sdk_mod = MagicMock()
    sdk_mod.web = MagicMock()
    sdk_mod.web.async_client = MagicMock()
    sdk_mod.web.async_client.AsyncWebClient = MagicMock
    sys.modules["slack_sdk"] = sdk_mod
    sys.modules["slack_sdk.web"] = sdk_mod.web
    sys.modules["slack_sdk.web.async_client"] = sdk_mod.web.async_client


_ensure_slack_mock()

from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from plugins.platforms.slack.adapter import SlackAdapter
from tests.waiting import async_wait_until


class _Client:
    """Captures the Web API calls the panel makes."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def _rec(self, name, result):
        async def call(**kwargs):
            self.calls.append((name, kwargs))
            return result
        return call

    def __getattr__(self, name):
        if name == "chat_postMessage":
            return self._rec(name, {"ok": True, "ts": "1700.0001", "channel": "C1"})
        return self._rec(name, {"ok": True})

    def named(self, name):
        return [kw for n, kw in self.calls if n == name]


def _adapter(client, extra):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test", extra=extra))
    adapter._app = MagicMock()
    adapter._app.client = client
    adapter._team_clients = {"T1": client}
    adapter._channel_team = {"C1": "T1"}
    adapter._is_interactive_user_authorized = lambda *a, **k: True
    return adapter


def _turn(adapter, loop):
    class _Runner:
        def _delivery_adapter_for(self, source):
            return adapter

        async def _deliver_platform_notice(self, source, content):
            return None

    source = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="group", user_id="U_OWNER",
                           user_name="owner", thread_id="1699.0001")
    ctx = TurnContext(source=source, _run_still_current=lambda: True, progress_queue=None, _loop_for_step=loop,
                      _progress_metadata={"thread_id": "1699.0001", "slack_team_id": "T1"})
    from gateway.run_turn_runner import TurnRunner
    return TurnRunner(_Runner(), ctx)


def _press(action_id, value, user):
    body = {"user": {"id": user, "name": user}, "channel": {"id": "C1"}, "team": {"id": "T1"},
            "trigger_id": "trig-1", "message": {"ts": "1700.0001", "thread_ts": "1699.0001"}}
    return body, {"action_id": action_id, "value": value}


@pytest.fixture
def no_throttle(monkeypatch):
    if importlib.util.find_spec("gateway.run_turn_subagent_panel") is not None:
        monkeypatch.setattr("gateway.run_turn_subagent_panel.PUBLISH_INTERVAL_S", 0.0)
    # approvals.approvers narrowed: only the turn's owner (and admins) may press the panel's buttons.
    monkeypatch.setattr("tools.approval_context._get_approval_config", lambda: {"approvers": []})


class _LiveChild:
    def __init__(self):
        self.stops = []

    def hard_interrupt(self, message=None):
        self.stops.append(message)
        return True


@pytest.fixture
def live_child():
    from tools.delegate_tool_registry import _register_subagent, _unregister_subagent
    child = _LiveChild()
    _register_subagent({"subagent_id": "sa-0-aaa", "agent": child, "goal": "audit the auth module"})
    yield child
    _unregister_subagent("sa-0-aaa")


async def _published(client, count):
    """Wait until the panel made *count* Web API calls in total."""
    await async_wait_until(lambda: len(client.calls) >= count, timeout=2.0, what=f"{count} panel calls")


def test_subagent_events_render_one_panel_with_stop_then_result(no_throttle, live_child):
    """Opted in: the children's events post ONE panel and edit it in place, a running row offers Stop
    (interrupting exactly that child, for the turn's owner only) and a finished row offers Result,
    whose modal carries the child's full result text, not the 500-char summary."""
    async def scenario():
        client = _Client()
        adapter = _adapter(client, {"subagent_panel": True})
        turn = _turn(adapter, asyncio.get_running_loop())
        cb = turn.progress_callback
        ident = {"subagent_id": "sa-0-aaa", "goal": "audit the auth module", "depth": 0}
        cb("subagent.start", preview="audit the auth module", **ident)
        await _published(client, 1)
        cb("subagent.tool", "read_file", "auth.py", None, **ident, tool_count=1)
        await _published(client, 2)
        posts, updates = client.named("chat_postMessage"), client.named("chat_update")
        assert len(posts) == 1 and posts[0]["thread_ts"] == "1699.0001"
        row = updates[-1]["blocks"][1]["text"]["text"]
        assert "audit the auth module" in row and "`read_file`" in row
        stop = updates[-1]["blocks"][2]["elements"][0]
        assert stop["action_id"] == "hermes_subagent_stop"

        body, action = _press(stop["action_id"], stop["value"], "U_OTHER")
        await adapter._handle_subagent_stop(AsyncMock(), body, action)
        assert live_child.stops == []  # someone else's turn: refused privately
        body, action = _press(stop["action_id"], stop["value"], "U_OWNER")
        await adapter._handle_subagent_stop(AsyncMock(), body, action)
        assert len(live_child.stops) == 1

        full = "finding " * 1500
        cb("subagent.complete", preview=full[:160], status="completed", summary=full[:500],
           result_text=full, duration_seconds=12.0, **ident)
        await async_wait_until(lambda: client.named("chat_update")[-1]["blocks"][2]["elements"][0]["action_id"]
                               == "hermes_subagent_result", timeout=2.0, what="finished row")
        last = client.named("chat_update")[-1]["blocks"]
        result_btn = last[2]["elements"][0]
        assert result_btn["action_id"] == "hermes_subagent_result"
        body, action = _press(result_btn["action_id"], result_btn["value"], "U_OWNER")
        await adapter._handle_subagent_result(AsyncMock(), body, action)
        view = client.named("views_open")[-1]["view"]
        shown = "".join(b["text"]["text"] for b in view["blocks"])
        assert len(shown) > 11_000 and full.startswith(shown)
        assert len(client.named("chat_postMessage")) == 1  # always the same message

    asyncio.run(scenario())


def test_panel_is_off_by_default(no_throttle):
    """Without the opt-in the same events post nothing (legacy behaviour)."""

    async def scenario():
        client = _Client()
        turn = _turn(_adapter(client, {}), asyncio.get_running_loop())
        turn.progress_callback("subagent.start", preview="g", subagent_id="sa-0-bbb", goal="g", depth=0)
        for _ in range(20):  # let any scheduled publish run
            await asyncio.sleep(0)
        assert client.calls == []

    asyncio.run(scenario())

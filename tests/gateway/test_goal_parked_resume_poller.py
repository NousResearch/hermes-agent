"""A parked /goal resumes in the gateway once its wait barrier lifts, with no other turn arriving.

Live case: the judge parked a Discord goal for the 600 s the agent announced for a subagent batch.
The batch ran 35 min; the timer lapsed at 10 min and nothing re-judged, so the thread stayed silent
past the promised ETA until the batch result happened to start a turn. The CLI idle tick
(``_maybe_resume_parked_goal``) covers this; the gateway had no counterpart.
"""

import asyncio
import time

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from hermes_cli import goals


class _Adapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="wire-1")


@pytest.fixture
def parked(monkeypatch):
    goals._DB_CACHE.clear()
    goals._get_session_db()
    source = SessionSource(platform=Platform.DISCORD, chat_id="7", user_id="7", chat_type="thread", message_id="m1")
    key = build_session_key(source)
    adapter = _Adapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.DISCORD)
    received = []

    async def handler(event):
        received.append(event)
        return None

    adapter.set_message_handler(handler)
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._delivery_adapter_for = lambda source: adapter
    runner._run_in_executor_with_context = asyncio.to_thread
    mgr = goals.GoalManager(session_id="parked-goal")
    mgr.set("fix the missing CRM messages")
    watch = {key: (source, "parked-goal")}
    yield runner, adapter, watch, key, received
    goals._DB_CACHE.clear()


def _expire_barrier():
    mgr = goals.GoalManager(session_id="parked-goal")
    mgr.state.waiting_until = time.time() - 1
    mgr._save()


async def _poll(runner, adapter, watch):
    await runner._parked_goal_poll_once(watch)
    await asyncio.gather(*adapter._background_tasks)


@pytest.mark.asyncio
async def test_lapsed_timer_resumes_an_idle_parked_goal_exactly_once(parked):
    runner, adapter, watch, key, received = parked
    goals.GoalManager(session_id="parked-goal").wait_for_seconds(600, reason="subagent batch, ETA 10 min")

    await _poll(runner, adapter, watch)
    assert received == [] and key in watch  # timer still running

    _expire_barrier()
    adapter._active_sessions[key] = asyncio.Event()
    await _poll(runner, adapter, watch)
    assert received == [] and key in watch  # a live turn re-judges on its own
    adapter._active_sessions.pop(key)

    await _poll(runner, adapter, watch)
    await _poll(runner, adapter, watch)
    assert len(received) == 1
    assert received[0].text.startswith("[Continuing toward your standing goal]")
    assert received[0].source.message_id is None
    assert key not in watch
    assert goals.GoalManager(session_id="parked-goal").state.waiting_until == 0.0


@pytest.mark.asyncio
async def test_returned_delegation_is_left_to_its_result_turn_until_the_deadline(parked, monkeypatch):
    runner, adapter, watch, key, received = parked
    monkeypatch.setattr(goals, "count_active_delegations", lambda sid: 1)
    goals.GoalManager(session_id="parked-goal").wait_for_seconds(600, reason="1 batch", on_delegations=1)
    monkeypatch.setattr(goals, "count_active_delegations", lambda sid: 0)  # the batch returned

    await _poll(runner, adapter, watch)
    assert received == [] and key in watch

    _expire_barrier()  # the result turn never came: the deadline is the fallback
    await _poll(runner, adapter, watch)
    assert len(received) == 1 and key not in watch


@pytest.mark.parametrize("cancel", ["unwait", "pause", "clear", "replace"])
@pytest.mark.asyncio
async def test_unwait_pause_clear_or_replace_drops_the_watch_without_a_turn(parked, cancel):
    runner, adapter, watch, key, received = parked
    mgr = goals.GoalManager(session_id="parked-goal")
    mgr.wait_for_seconds(600, reason="subagent batch")
    {"unwait": mgr.stop_waiting, "pause": mgr.pause, "clear": mgr.clear,
     "replace": lambda: mgr.set("a different goal")}[cancel]()

    await _poll(runner, adapter, watch)
    assert received == [] and key not in watch

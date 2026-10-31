"""Native approval responders must share the turn's real typing resume lifecycle."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway.run_turn_runner import TurnRunner
from gateway.session_pending_controls import PendingControls
from tests.gateway import test_discord_typing_pause as typing_tests
from tools import approval
from tools.approval_gateway_wait import _await_gateway_decision


typing_fixture = typing_tests.typing


class Events:
    def __init__(self):
        self.lock = threading.RLock()

    def publish(self, *args, **kwargs):
        pass


def turn(adapter, key, controls, published, release, awakened, current, *, failed_notify=False, decided=None):
    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        session_key=key, session_id=key, _status_adapter=adapter, _status_chat_id="77",
        source=SimpleNamespace(user_id="worker", user_name="Worker", is_bot=False),
        title_user_message=None, persist_user_display_kind=None, persist_user_display_metadata=None,
        moa_config=None, inbound_message_id=None, mute_notification_reply=False, reply_expected=True,
        _run_still_current=lambda: current[0],
    )
    runner._native_image_run_message = lambda: "request"
    loop = asyncio.get_running_loop()

    def notify(data):
        adapter.pause_typing_for_chat("77")
        controls.register(key, key, 1, data)
        loop.call_soon_threadsafe(published.set)
        if failed_notify:
            raise RuntimeError("fixture notification failed")

    runner._approval_notify_sync = notify

    def run_conversation(message, **kwargs):
        decision = _await_gateway_decision(key, notify, {"command": key, "pattern_key": key})
        awakened.append((decision, "77" in adapter._typing_paused))
        if decided is not None:
            loop.call_soon_threadsafe(decided.set)
        assert release.wait(5), "test did not release the settled agent"
        return decision

    agent = SimpleNamespace(run_conversation=run_conversation)
    return asyncio.create_task(asyncio.to_thread(
        runner._run_conversation_with_approval, agent, [], None, None, None))


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["once", "deny"])
async def test_native_decision_resumes_before_worker_continues(typing_fixture, choice):
    adapter, request, clock = typing_fixture
    controls = PendingControls(Events())
    published, release, awakened, current = asyncio.Event(), threading.Event(), [], [True]
    key = "native-typing-single"
    worker = turn(adapter, key, controls, published, release, awakened, current)
    try:
        await asyncio.wait_for(published.wait(), 5)
        prompt, = controls.snapshot(key, 1)
        result = controls.respond(key, 1, prompt["prompt_id"], {"choice": choice})
        assert result["status"] == "resolved"
        assert "77" not in adapter._typing_paused, "native resolution left the turn typing-paused"
        await adapter.send_typing("77")
        await clock.parked()
        assert request.await_count == 1
        release.set()
        assert (await asyncio.wait_for(worker, 5))["choice"] == choice
        assert awakened[0][1] is False, "resume happened after waking the worker"
    finally:
        release.set()
        approval.unregister_gateway_notify(key)
        await asyncio.wait_for(worker, 5)
        await adapter.stop_typing("77")


@pytest.mark.asyncio
async def test_native_decision_preserves_other_sessions_shared_chat_wait(typing_fixture):
    adapter, request, clock = typing_fixture
    controls = [PendingControls(Events()), PendingControls(Events())]
    keys = ["native-typing-first", "native-typing-second"]
    published = [asyncio.Event(), asyncio.Event()]
    releases = [threading.Event(), threading.Event()]
    awakened = [[], []]
    workers = [turn(adapter, key, control, event, release, state, [True])
               for key, control, event, release, state in zip(keys, controls, published, releases, awakened)]
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in published)), 5)
        prompts = [control.snapshot(key, 1)[0] for control, key in zip(controls, keys)]
        assert controls[0].respond(keys[0], 1, prompts[0]["prompt_id"], {"choice": "once"})["status"] == "resolved"
        releases[0].set()
        await asyncio.wait_for(workers[0], 5)
        assert approval.has_blocking_approval(keys[1])
        assert "77" in adapter._typing_paused
        await adapter.send_typing("77")
        request.assert_not_awaited()
        assert controls[0].respond(keys[0], 1, prompts[0]["prompt_id"], {"choice": "once"})["status"] == "already_resolved"
        assert "77" in adapter._typing_paused
        assert controls[1].respond(keys[1], 1, prompts[1]["prompt_id"], {"choice": "deny"})["status"] == "resolved"
        assert "77" not in adapter._typing_paused, "last native decision did not release shared chat"
        await adapter.send_typing("77")
        await clock.parked()
        assert request.await_count == 1
    finally:
        for release, key in zip(releases, keys):
            release.set()
            approval.unregister_gateway_notify(key)
        await asyncio.wait_for(asyncio.gather(*workers), 5)
        await adapter.stop_typing("77")


@pytest.mark.asyncio
async def test_retired_turn_callback_does_not_resume_later_pause(typing_fixture):
    adapter, request, _clock = typing_fixture
    controls = PendingControls(Events())
    published, release, awakened, current = asyncio.Event(), threading.Event(), [], [True]
    key = "native-typing-retired"
    worker = turn(adapter, key, controls, published, release, awakened, current)
    try:
        await asyncio.wait_for(published.wait(), 5)
        prompt, = controls.snapshot(key, 1)
        current[0] = False
        assert controls.respond(key, 1, "unknown-id", {"choice": "once"})["status"] == "already_resolved"
        assert "77" in adapter._typing_paused
        controls.respond(key, 1, prompt["prompt_id"], {"choice": "deny"})
        assert "77" in adapter._typing_paused
        await adapter.send_typing("77")
        request.assert_not_awaited()
    finally:
        release.set()
        approval.unregister_gateway_notify(key)
        await asyncio.wait_for(worker, 5)
        await adapter.stop_typing("77")


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["timeout", "notify_failed"])
async def test_unanswered_approval_releases_only_live_idle_surface(typing_fixture, monkeypatch, ending):
    adapter, request, clock = typing_fixture
    controls = PendingControls(Events())
    published, decided = asyncio.Event(), asyncio.Event()
    release, awakened = threading.Event(), []
    key = "native-typing-unanswered"
    if ending == "timeout":
        monkeypatch.setattr("tools.approval_gateway_wait._poll_event", lambda *a, **kw: "timeout")
    worker = turn(adapter, key, controls, published, release, awakened, [True],
                  failed_notify=ending == "notify_failed", decided=decided)
    try:
        await asyncio.wait_for(decided.wait(), 5)
        assert awakened[0][1] is False
        assert not approval.has_blocking_approval(key)
        await adapter.send_typing("77")
        await clock.parked()
        assert request.await_count == 1
    finally:
        release.set()
        approval.unregister_gateway_notify(key)
        await asyncio.wait_for(worker, 5)
        assert key not in approval._gateway_before_wake_cbs
        await adapter.stop_typing("77")

"""MCP sign-in completions wake only their still-pending originating chat."""

import asyncio
import contextvars
import sys
import threading
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import run as gateway_run, wake
from gateway.config import Platform
from gateway.session import SessionSource
from hermes_constants import get_hermes_home
from tools.connectors import mcp, mcp_oauth
from tools.mcp_dashboard_oauth import DashboardOAuthFlow


class PushAdapter:
    supports_async_delivery = True

    def __init__(self, home):
        self.home = home
        self.events = []

    async def handle_message(self, event):
        assert get_hermes_home() == self.home
        event._gateway_accepted = True
        self.events.append(event)


@pytest.fixture
def route_rig(tmp_path):
    home = tmp_path / "secondary"
    home.mkdir()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="fixture-chat", chat_type="dm",
                           thread_id="fixture-thread", profile="secondary")
    route = {"session_key": "agent:secondary:telegram:dm:fixture-chat",
             "parent_session_id": "fixture-parent"}
    adapter = PushAdapter(home)
    scoped = []

    async def classify(parent, key):
        assert get_hermes_home() == home
        assert (parent, key) == (route["parent_session_id"], route["session_key"])
        scoped.append(True)
        return "deliver"

    def resolve(platform, restored):
        assert get_hermes_home() == home
        assert platform == "telegram" and restored is source
        return adapter

    runner = SimpleNamespace(
        session_store=SimpleNamespace(_ensure_loaded=lambda: None,
                                      _entries={route["session_key"]: SimpleNamespace(origin=source)}),
        _restored_source=lambda entry: entry.origin,
        _resolve_profile_home_for_source=lambda _source: home,
        _classify_completion_target=AsyncMock(side_effect=classify),
        _resolve_injection_adapter=resolve,
    )
    for method in ("_build_process_event_source", "_completion_event_scope"):
        setattr(runner, method, MethodType(getattr(gateway_run.GatewayRunner, method), runner))
    return SimpleNamespace(runner=runner, route=route, source=source, adapter=adapter, scoped=scoped)


@pytest.mark.asyncio
async def test_wake_route_restores_source_and_enters_profile_scope(route_rig):
    rig = route_rig
    assert await wake.wake_route(rig.runner, rig.route, "reports connected", lambda: True,
                                producer="mcp-sign-in-wake")
    event, = rig.adapter.events
    assert event.internal and event.source is rig.source
    assert event.text == "reports connected"
    assert event.metadata["gateway_session_id"] == rig.route["parent_session_id"]
    assert rig.scoped == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["adopted", "terminal", "retry", "nonpush", "source", "adapter"])
async def test_wake_route_drops_unavailable_or_stale_targets(route_rig, reason):
    rig = route_rig
    changes = {
        "terminal": lambda: setattr(rig.runner, "_classify_completion_target", AsyncMock(return_value="terminal")),
        "retry": lambda: setattr(rig.runner, "_classify_completion_target", AsyncMock(return_value="retry")),
        "nonpush": lambda: setattr(rig.adapter, "supports_async_delivery", False),
        "source": lambda: setattr(rig.runner, "_build_process_event_source", lambda _route: None),
        "adapter": lambda: setattr(rig.runner, "_resolve_injection_adapter", lambda *_args: None),
    }
    changes.get(reason, lambda: None)()
    assert not await wake.wake_route(rig.runner, rig.route, "reports connected",
                                    lambda: reason != "adopted", producer="mcp-sign-in-wake")
    assert not rig.adapter.events


@pytest.mark.asyncio
async def test_wake_route_without_parent_skips_classification(route_rig):
    rig = route_rig
    rig.route["parent_session_id"] = ""
    assert await wake.wake_route(rig.runner, rig.route, "reports connected", lambda: True,
                                producer="mcp-sign-in-wake")
    rig.runner._classify_completion_target.assert_not_awaited()
    assert "gateway_session_id" not in rig.adapter.events[0].metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["unchanged", "new", "compression"])
async def test_wake_route_authority_preserves_captured_session(tmp_path, monkeypatch, caplog, transition):
    from gateway.platforms.event import MessageEvent
    from hermes_state_runtime import list_session_admissions
    from tests.gateway.test_automation_compression_media import _admit_human, _discord_owner

    runner, adapter, authority, source = await _discord_owner(tmp_path, monkeypatch)
    adapter.gateway_runner = runner
    try:
        receipt = await _admit_human(runner, authority, MessageEvent(
            text="human", source=source, message_id="human"))
        parent = receipt.ref.session_id
        key = runner.session_store._generate_session_key(source)
        route = {"session_key": key, "parent_session_id": parent}
        classify = runner._classify_completion_target

        async def classify_then_transition(captured_parent, captured_key):
            result = await classify(captured_parent, captured_key)
            assert result == "deliver"
            if transition == "new":
                runner.session_store.reset_session(key)
                # The replacement has provenance too: an unpinned wake can enter it.
                await _admit_human(runner, authority, MessageEvent(
                    text="new conversation", source=source, message_id="new-human"))
            elif transition == "compression":
                authority.db.append_message(parent, "user", "human")
                child = parent + "-tip"
                authority.db.publish_compression_child(
                    parent_session_id=parent, child_session_id=child, source="discord",
                    messages=[{"role": "user", "content": "summary"}], require_compression_lease=False)
                assert runner.session_store.advance_compression_session(key, parent, child) is not None
            return result

        monkeypatch.setattr(runner, "_classify_completion_target", classify_then_transition)
        accepted = await wake.wake_route(runner, route, "reports connected", lambda: True,
                                        producer="mcp-sign-in-wake")
        assert accepted is (transition != "new")
        owner = authority.logical_owner(runner.session_store.peek_session_id(key))
        rows = list_session_admissions(authority.db, session_id=owner, pending_only=False)
        if transition == "new":
            assert [row["payload"]["text"] for row in rows] == ["new conversation"]
            assert len(list_session_admissions(authority.db, session_id=parent, pending_only=False)) == 1
            assert "admission_conflict" in caplog.text
        else:
            assert owner == parent
            assert [row["payload"]["text"] for row in rows] == ["human", "reports connected"]
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [wake.WakeNotAccepted(), RuntimeError("fixture failure")])
async def test_wake_route_swallows_admission_errors(route_rig, monkeypatch, caplog, error):
    monkeypatch.setattr(wake, "deliver_wake", AsyncMock(side_effect=error))
    assert not await wake.wake_route(route_rig.runner, route_rig.route, "private notice text",
                                    lambda: True, producer="mcp-sign-in-wake")
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "private notice text" not in warnings[0].getMessage()


@pytest.mark.asyncio
@pytest.mark.parametrize("producer", [None, "mcp-sign-in-wake"])
@pytest.mark.parametrize("session_pin", [None, "fixture-parent"])
async def test_deliver_wake_preserves_default_and_accepts_producer(route_rig, monkeypatch, producer, session_pin):
    admit = AsyncMock(return_value=True)
    monkeypatch.setattr("gateway.session_automation.admit_route_wake", admit)
    rig = route_rig
    rig.adapter.gateway_runner = rig.runner
    kwargs = {} if producer is None else {"producer": producer}
    if session_pin is not None:
        kwargs["session_pin"] = session_pin
    await wake.deliver_wake(rig.adapter, source=rig.source, text="reports connected", **kwargs)
    assert admit.await_args.args[4] == (producer or "kanban-wake")
    assert admit.await_args.args[5] == ({"gateway_session_id": session_pin} if session_pin else {})
    assert not rig.adapter.events


@pytest.fixture
def live_wakes(route_rig, monkeypatch):
    from tools.connectors import mcp_sign_in_wake as sign_in

    rig = route_rig
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()
        loop.close()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()
    assert ready.wait(3)
    rig.runner._gateway_loop, rig.runner._running = loop, True
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: rig.runner)
    monkeypatch.setattr(sign_in, "_WAIT_SECONDS", 2)
    monkeypatch.setattr(sign_in, "async_delivery_supported", lambda: True)
    monkeypatch.setattr(sign_in, "session_is_messaging_surface", lambda: True)
    env = {"HERMES_SESSION_KEY": rig.route["session_key"], "HERMES_SESSION_ID": "fixture-parent",
           "HERMES_SESSION_PLATFORM": "telegram", "HERMES_SESSION_CHAT_ID": "fixture-chat",
           "HERMES_SESSION_CHAT_TYPE": "dm", "HERMES_SESSION_THREAD_ID": "fixture-thread",
           "HERMES_SESSION_USER_ID": "fixture-user", "HERMES_SESSION_USER_NAME": "fixture-name",
           "HERMES_SESSION_SCOPE_ID": "fixture-scope"}
    monkeypatch.setattr(sign_in, "get_session_env", lambda key: env.get(key, ""))
    threads = []

    def tracked_thread(**kwargs):
        thread = threading.Thread(**kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(sign_in, "Thread", tracked_thread)
    rig.sign_in, rig.env, rig.threads = sign_in, env, threads
    mcp._LATE_ATTEMPTS.clear()
    yield rig
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()
    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join(timeout=3)
    assert not loop_thread.is_alive()
    mcp._LATE_ATTEMPTS.clear()


def _attempt():
    flow = DashboardOAuthFlow("fixture-flow", "reports", None, "", "")
    flow.status = "authorization_required"
    return mcp_oauth.OAuthAttempt("https://idp.example/authorize", flow)


def _settle_watchers(rig):
    for thread in rig.threads:
        thread.join(timeout=3)
        assert not thread.is_alive()

    async def barrier():
        tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        if tasks:
            await asyncio.gather(*tasks)

    asyncio.run_coroutine_threadsafe(barrier(), rig.runner._gateway_loop).result(timeout=3)


def test_sign_in_worker_wakes_once_with_captured_route_and_empty_context(live_wakes, monkeypatch):
    rig = live_wakes
    attempt = _attempt()
    pending = {"reports": attempt}
    called = []
    ambient = contextvars.ContextVar("fixture_ambient", default="empty")
    ambient.set("tool-thread")
    delivered = threading.Event()
    real_wake = wake.wake_route

    async def observe(runner, route, text, still_pending, *, producer):
        called.append((dict(route), producer, ambient.get()))
        try:
            return await real_wake(runner, route, text, still_pending, producer=producer)
        finally:
            delivered.set()

    monkeypatch.setattr(wake, "wake_route", observe)
    captured = {"session_key": rig.env["HERMES_SESSION_KEY"], "parent_session_id": "fixture-parent",
                **{field: rig.env["HERMES_SESSION_" + field.upper()]
                   for field in ("platform", "chat_id", "chat_type", "thread_id", "user_id",
                                 "user_name", "scope_id")}}
    assert rig.sign_in.arm_sign_in_wakes(pending, lambda name, obj: pending.get(name) is obj)
    thread, = rig.threads
    assert thread.daemon and thread.name == "mcp-sign-in-wake"
    rig.env["HERMES_SESSION_CHAT_ID"] = "another-chat"
    attempt.flow.mark_approved()
    assert not attempt.flow.worker_done and not rig.adapter.events
    attempt.flow.mark_worker_done()
    assert delivered.wait(3)
    _settle_watchers(rig)
    assert called == [(captured, "mcp-sign-in-wake", "empty")]
    event, = rig.adapter.events
    assert event.internal and event.source is rig.source and "reports" in event.text
    assert event.text == (
        "[Hermes gateway notice, not typed by the user: the MCP server 'reports' finished signing in "
        "and its tools are connected now. If the user's earlier request in this conversation needed "
        "it and is not finished yet, continue it without asking them to confirm, and do not repeat "
        "anything already done. Otherwise tell them briefly that reports is connected.]"
    )


@pytest.mark.parametrize("outcome", ["error", "cancelled", "adopted", "discovery", "timeout", "shutdown"])
def test_sign_in_worker_does_not_wake_unsuccessful_or_adopted_attempt(live_wakes, outcome):
    rig = live_wakes
    attempt = _attempt()
    key = ("fixture-home", "fixture-session")
    mcp._LATE_ATTEMPTS[key] = {"reports": attempt}
    still_pending = lambda name, obj: mcp._LATE_ATTEMPTS.get(key, {}).get(name) is obj
    assert rig.sign_in.arm_sign_in_wakes({"reports": attempt}, still_pending)
    if outcome in {"error", "cancelled"}:
        attempt.flow.mark_error("fixture failure", cancelled=outcome == "cancelled")
    else:
        attempt.flow.mark_approved()
    if outcome == "adopted":
        mcp._LATE_ATTEMPTS[key].pop("reports")
    elif outcome == "discovery":
        attempt.flow.discovery_error = "fixture discovery failed"
    elif outcome == "shutdown":
        rig.runner._running = False
    if outcome != "timeout":
        attempt.flow.mark_worker_done()
    _settle_watchers(rig)
    assert not rig.adapter.events


@pytest.mark.parametrize("gate", ["async", "surface", "cron", "runner", "stopped", "loop", "key", "url", "approved"])
def test_sign_in_gates_start_no_thread(live_wakes, monkeypatch, gate):
    rig = live_wakes
    attempt = _attempt()
    changes = {
        "async": lambda: monkeypatch.setattr(rig.sign_in, "async_delivery_supported", lambda: False),
        "surface": lambda: monkeypatch.setattr(rig.sign_in, "session_is_messaging_surface", lambda: False),
        "cron": lambda: rig.env.__setitem__("HERMES_CRON_SESSION", "1"),
        "runner": lambda: monkeypatch.setitem(sys.modules, "gateway.run",
                                            SimpleNamespace(_gateway_runner_ref=lambda: None)),
        "stopped": lambda: setattr(rig.runner, "_running", False),
        "loop": lambda: setattr(rig.runner, "_gateway_loop", SimpleNamespace(is_running=lambda: False)),
        "key": lambda: rig.env.__setitem__("HERMES_SESSION_KEY", ""),
        "url": lambda: setattr(attempt, "auth_url", ""),
        "approved": attempt.flow.mark_approved,
    }
    changes[gate]()
    assert not rig.sign_in.arm_sign_in_wakes({"reports": attempt}, lambda *_args: True)
    assert not rig.threads


def test_sign_in_arming_failure_is_nonfatal(live_wakes, monkeypatch, caplog):
    def fail_thread(**_kwargs):
        raise RuntimeError("fixture thread unavailable")

    monkeypatch.setattr(live_wakes.sign_in, "Thread", fail_thread)
    assert not live_wakes.sign_in.arm_sign_in_wakes({"reports": _attempt()}, lambda *_args: True)
    assert "WARNING" in caplog.text

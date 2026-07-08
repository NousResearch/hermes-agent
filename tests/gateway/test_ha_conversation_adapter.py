"""HAConversationAdapter await-window bridge tests (stubbed agent)."""

import asyncio

import pytest
import pytest_asyncio
from wyoming.asr import Transcript
from wyoming.client import AsyncTcpClient
from wyoming.handle import Handled, NotHandled

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_mod = load_plugin_adapter("ha_conversation")


def make_config(**overrides):
    extra = {"bind_host": "127.0.0.1", "port": 0, "ack_after_seconds": 8.0,
             "announce_mode": "off", "announce_entity": "",
             "max_transcript_chars": 2000}
    extra.update(overrides)
    return PlatformConfig(enabled=True, extra=extra)


def test_validate_config():
    assert _mod.validate_config(make_config()) is True
    assert _mod.validate_config(make_config(port="not-a-port")) is False
    assert _mod.validate_config(make_config(announce_mode="bogus")) is False
    assert _mod.validate_config(
        make_config(announce_mode="default_device", announce_entity="")
    ) is False
    assert _mod.validate_config(
        make_config(announce_mode="default_device",
                    announce_entity="assist_satellite.kitchen")
    ) is True
    # empty extra: platform unconfigured
    assert _mod.validate_config(PlatformConfig(extra={})) is False


def test_apply_yaml_config_reads_platform_cfg_argument():
    section = {"enabled": True, "port": 10611, "announce_mode": "broadcast"}
    # nested-only style: top-level key absent, loader binds the block
    extra = _mod._apply_yaml_config({}, section)
    assert extra["port"] == 10611
    assert extra["announce_mode"] == "broadcast"
    assert _mod._apply_yaml_config({}, {}) is None


def test_register_declares_platform_entry():
    calls = {}

    class Ctx:
        def register_platform(self, **kwargs):
            calls.update(kwargs)

    _mod.register(Ctx())
    assert calls["name"] == "ha_conversation"
    assert callable(calls["check_fn"])
    assert calls["is_connected"] is _mod.validate_config  # config, not deps
    assert calls["apply_yaml_config_fn"] is _mod._apply_yaml_config
    hint = calls["platform_hint"].lower()
    assert "spoken" in hint or "aloud" in hint


@pytest_asyncio.fixture
async def rig(monkeypatch):
    """Connected adapter on an ephemeral port with a stubbed agent."""
    adapter = _mod.HAConversationAdapter(make_config())
    replies = {"handler": None}

    async def fake_handle_message(event):
        if replies["handler"] is not None:
            await replies["handler"](adapter, event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
    assert await adapter.connect() is True
    try:
        yield adapter, replies
    finally:
        await adapter.disconnect()


async def _ask(adapter, text, context=None, timeout=5):
    async with AsyncTcpClient("127.0.0.1", adapter.server_port) as client:
        await client.write_event(Transcript(text, context=context or {}).event())
        return await asyncio.wait_for(client.read_event(), timeout)


@pytest.mark.asyncio
async def test_reply_reaches_asking_socket(rig):
    adapter, replies = rig

    async def reply(a, event):
        await a.send("home", f"you said {event.text}")

    replies["handler"] = reply
    event = await _ask(adapter, "hello there")
    assert Handled.is_type(event.type)
    assert Handled.from_event(event).text == "you said hello there"


@pytest.mark.asyncio
async def test_multi_send_joins_into_one_handled(rig):
    adapter, replies = rig

    async def chunked(a, event):
        await a.send("home", "part one.")
        await a.send("home", "part two.")

    replies["handler"] = chunked
    event = await _ask(adapter, "long answer please")
    assert Handled.from_event(event).text == "part one.\npart two."


@pytest.mark.asyncio
async def test_no_send_turn_gets_fallback_text(rig):
    adapter, replies = rig
    replies["handler"] = None  # agent produced nothing
    event = await _ask(adapter, "silent treatment")
    assert Handled.is_type(event.type)
    assert Handled.from_event(event).text  # some non-empty fallback


@pytest.mark.asyncio
async def test_empty_and_oversized_transcripts_not_handled(rig):
    adapter, replies = rig
    replies["handler"] = None
    event = await _ask(adapter, "")
    assert NotHandled.is_type(event.type)
    event = await _ask(adapter, "x" * 5000)
    assert NotHandled.is_type(event.type)


@pytest.mark.asyncio
async def test_send_outside_window_routes_to_announce(rig):
    adapter, replies = rig
    announced = []

    async def fake_announce(text, *, entity):
        announced.append((text, entity))
        from gateway.platforms.base import SendResult
        return SendResult(success=True)

    adapter._announce = fake_announce
    result = await adapter.send("home", "cron job finished")
    assert result.success is True
    assert announced == [("cron job finished", None)]


@pytest.mark.asyncio
async def test_agent_exception_speaks_error_not_nothandled(rig):
    adapter, replies = rig

    async def boom(a, event):
        raise RuntimeError("tool exploded")

    replies["handler"] = boom
    event = await _ask(adapter, "break please")
    assert Handled.is_type(event.type)
    assert "sorry" in (Handled.from_event(event).text or "").lower()


@pytest.mark.asyncio
async def test_concurrent_utterances_serialize_and_both_answer(rig):
    adapter, replies = rig

    async def reply(a, event):
        await asyncio.sleep(0.05)
        await a.send("home", f"answer: {event.text}")

    replies["handler"] = reply
    a, b = await asyncio.gather(_ask(adapter, "first"), _ask(adapter, "second"))
    texts = {Handled.from_event(a).text, Handled.from_event(b).text}
    assert texts == {"answer: first", "answer: second"}


@pytest.mark.asyncio
async def test_run_turn_and_wait_bridges_real_session_task(monkeypatch):
    """Exercises the real handle_message -> _start_session_processing ->
    _session_tasks bridge that _run_turn_and_wait relies on, instead of the
    handle_message monkeypatch every other test in this file uses (which
    never populates _session_tasks and never runs the `await task` branch).

    Only _process_message_background is stubbed — the deepest practical
    seam that still lets the base class genuinely install the session
    guard, spawn the task via asyncio.create_task, and record it in
    _session_tasks the same way production does. The fake also mirrors the
    minimal cleanup that _process_message_background's own finally block
    performs (dropping _active_sessions/_session_tasks), so a second
    utterance proves the bridge doesn't wedge the session.
    """
    adapter = _mod.HAConversationAdapter(make_config())

    async def _unused_handler(event):
        return None

    adapter.set_message_handler(_unused_handler)

    async def fake_process_message_background(event, session_key):
        await asyncio.sleep(0.05)
        await adapter.send("home", "bridged reply")
        adapter._active_sessions.pop(session_key, None)
        adapter._session_tasks.pop(session_key, None)

    monkeypatch.setattr(
        adapter, "_process_message_background", fake_process_message_background
    )
    assert await adapter.connect() is True
    try:
        event = await _ask(adapter, "what time is it")
        assert Handled.is_type(event.type)
        assert Handled.from_event(event).text == "bridged reply"

        # A second utterance must also be answered — proves cleanup didn't
        # leave the adapter wedged behind a stale session guard/task.
        event2 = await _ask(adapter, "what time is it now")
        assert Handled.is_type(event2.type)
        assert Handled.from_event(event2).text == "bridged reply"
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_waiting_cap_returns_busy(rig):
    adapter, replies = rig

    async def slow(a, event):
        await asyncio.sleep(0.3)
        await a.send("home", f"answer: {event.text}")

    replies["handler"] = slow
    events = await asyncio.gather(
        *[_ask(adapter, f"q{i}", timeout=10) for i in range(9)]
    )
    texts = [Handled.from_event(e).text or "" for e in events]
    busy = [t for t in texts if "try again" in t.lower()]
    answered = [t for t in texts if t.startswith("answer:")]
    # exactly one over the cap of 8 is refused immediately; the rest queue
    assert len(busy) == 1
    assert len(answered) == 8


@pytest_asyncio.fixture
async def announce_rig(monkeypatch):
    """Adapter with fast ack + captured HA service calls."""
    def build(mode, entity=""):
        adapter = _mod.HAConversationAdapter(
            make_config(ack_after_seconds=0.05, announce_mode=mode,
                        announce_entity=entity)
        )
        calls = []

        async def fake_call_service(domain, service, entity_id, data):
            calls.append((domain, service, entity_id, data))
            return {"ok": True}

        import tools.homeassistant_tool as ha_tool
        monkeypatch.setattr(ha_tool, "async_call_service", fake_call_service)
        return adapter, calls

    return build


async def _slow_turn(a, event):
    await asyncio.sleep(0.3)
    await a.send("home", "the real answer")


@pytest.mark.asyncio
async def test_slow_turn_ack_mentions_conversation_when_announce_off(
    announce_rig, monkeypatch
):
    adapter, calls = announce_rig("off")
    monkeypatch.setattr(adapter, "handle_message",
                        lambda e: _slow_turn(adapter, e))
    assert await adapter.connect() is True
    try:
        event = await _ask(adapter, "slow question")
        text = Handled.from_event(event).text
        assert "conversation" in text.lower()   # honest: no announce coming
        await asyncio.sleep(0.5)                # let the turn finish
        assert calls == []                      # off => no REST call
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_slow_turn_late_reply_targets_originating_satellite(
    announce_rig, monkeypatch
):
    adapter, calls = announce_rig("last_active")
    monkeypatch.setattr(adapter, "handle_message",
                        lambda e: _slow_turn(adapter, e))
    assert await adapter.connect() is True
    try:
        event = await _ask(adapter, "slow question",
                           context={"satellite_id": "assist_satellite.office"})
        assert "announce" in (Handled.from_event(event).text or "").lower()
        await asyncio.sleep(0.5)
        assert calls == [(
            "assist_satellite", "announce", "assist_satellite.office",
            {"message": "the real answer"},
        )]
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_idle_announce_modes(announce_rig):
    # default_device targets the configured entity
    adapter, calls = announce_rig("default_device", "assist_satellite.kitchen")
    await adapter._announce("dinner is ready", entity=None)
    assert calls == [("assist_satellite", "announce",
                      "assist_satellite.kitchen", {"message": "dinner is ready"})]

    # broadcast targets all
    adapter, calls = announce_rig("broadcast")
    await adapter._announce("dinner is ready", entity=None)
    assert calls == [("assist_satellite", "announce", "all",
                      {"message": "dinner is ready"})]

    # last_active falls back to the most recent speaker
    adapter, calls = announce_rig("last_active")
    adapter._last_active_satellite = "assist_satellite.bedroom"
    await adapter._announce("dinner is ready", entity=None)
    assert calls[0][2] == "assist_satellite.bedroom"

    # off swallows silently and still reports success
    adapter, calls = announce_rig("off")
    result = await adapter._announce("dinner is ready", entity=None)
    assert result.success is True
    assert calls == []


@pytest.mark.asyncio
async def test_announce_failure_is_logged_not_raised(announce_rig, monkeypatch):
    adapter, calls = announce_rig("broadcast")

    async def exploding(domain, service, entity_id, data):
        raise RuntimeError("HA is down")

    import tools.homeassistant_tool as ha_tool
    monkeypatch.setattr(ha_tool, "async_call_service", exploding)
    result = await adapter._announce("hello", entity=None)
    assert result.success is False  # reported, not raised


@pytest.mark.asyncio
async def test_queued_utterance_acked_while_waiting_for_lock(
    announce_rig, monkeypatch
):
    """Important-1 regression: the ack timer must start BEFORE the turn
    lock is acquired, so a second room's utterance that is still queued
    behind a long first turn gets acked within ack_after_seconds of its
    own arrival — not only after it also acquires the lock. Before the
    fix, the ack task was created inside `async with self._turn_lock`, so
    a queued utterance's timer never even started until the first turn
    released the lock, and it could sit silent past HA's pipeline timeout.
    """
    adapter, calls = announce_rig("last_active")
    adapter._ack_after = 0.15

    async def slow(a, event):
        # "first" holds the turn lock well past the ack deadline; "second"'s
        # own processing, once it finally gets the lock, is fast — well
        # UNDER the ack deadline. This is the discriminating case: if the
        # ack timer only starts once the lock is acquired (the bug), a
        # queued "second" would finish and answer directly before its
        # (late-started) ack timer ever fires. Only starting the timer at
        # arrival makes "second" get acked while still queued.
        if event.text == "first":
            await asyncio.sleep(0.6)
        else:
            await asyncio.sleep(0.03)
        await a.send("home", f"answer: {event.text}")

    monkeypatch.setattr(adapter, "handle_message", lambda e: slow(adapter, e))
    assert await adapter.connect() is True
    try:
        first_task = asyncio.create_task(
            _ask(adapter, "first",
                 context={"satellite_id": "assist_satellite.a"}, timeout=5)
        )
        await asyncio.sleep(0.01)  # let "first" grab the turn lock first
        second_event = await _ask(
            adapter, "second",
            context={"satellite_id": "assist_satellite.b"}, timeout=5,
        )
        assert Handled.is_type(second_event.type)
        second_text = (Handled.from_event(second_event).text or "").lower()
        # Still queued (first hasn't released the lock yet) but already
        # acked: proves the timer started at arrival, not at lock-acquire.
        assert "announce" in second_text

        await first_task  # drain the first connection too

        target = (
            "assist_satellite", "announce", "assist_satellite.b",
            {"message": "answer: second"},
        )
        deadline = asyncio.get_event_loop().time() + 3.0
        while target not in calls and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.02)
        # The real answer, once "second" finally runs, must route to
        # _deliver_late/announce targeting the ORIGINATING satellite (the
        # room that actually asked "second"), never satellite "a".
        assert target in calls
    finally:
        await adapter.disconnect()

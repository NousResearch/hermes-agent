"""Telegram fire sites of ``gateway_ingress_observer`` through real PTB polling and dispatch.

One getUpdates script runs through the real Updater, TelegramApplication admission and handler
groups with no observer, then with observers that record, raise, block, run slow, are async, or are
cancelled: every core outcome must be identical, including the full getUpdates request sequence and
the ``gateway_platform_event`` envelopes. That compares against this tree with no observer
registered; the same projection was also compared against the pre-change tree out of suite.
Transport and model work are the only stand-ins. The delivery machinery itself is covered in
tests/gateway/test_ingress_observer.py.
"""

import asyncio
import contextlib
import hashlib
import json
import threading
import time
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("telegram")
from telegram import Update
from telegram.ext import TypeHandler
from telegram.request import BaseRequest

from gateway.config import PlatformConfig
from hermes_cli import plugins
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.platforms.telegram import adapter as tg_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter

BOT = 111
WAIT = 5.0
_EMPTY = b'{"ok":true,"result":[]}'


class _BotApi(BaseRequest):
    """Every Bot API call but getUpdates: getMe answers, everything else succeeds offline."""

    @property
    def read_timeout(self):
        return 1

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def do_request(self, url, method, request_data=None, **kwargs):
        if url.endswith("/getMe"):
            return 200, json.dumps({"ok": True, "result": {
                "id": BOT, "is_bot": True, "first_name": "Offline", "username": "offline_bot"}}).encode()
        return 200, b'{"ok":true,"result":true}'


class _Wire(_BotApi):
    """getUpdates: connect's readiness probe gets an empty answer; the scripted batches follow once
    ``go`` is set, after which the long poll is held open as Telegram holds an idle one. Without a
    script, polls come back empty. Every request's offset is recorded in order."""

    def __init__(self, batches):
        self.batches = [json.dumps({"ok": True, "result": batch}).encode() for batch in batches]
        self.offsets = []
        self.go = asyncio.Event()

    async def do_request(self, url, method, request_data=None, **kwargs):
        parameters = request_data.parameters if request_data is not None else {}
        timeout = parameters.get("timeout")
        self.offsets.append(parameters.get("offset"))
        if (timeout.total_seconds() if hasattr(timeout, "total_seconds") else timeout) == 0:
            return 200, _EMPTY  # the updater's read receipt while stopping
        if len(self.offsets) > 1 and self.batches:
            await self.go.wait()
            return 200, self.batches.pop(0)
        if self.go.is_set():
            await asyncio.Event().wait()
        await asyncio.sleep(0.01)
        return 200, _EMPTY


def _sender(user):
    return {"id": user, "is_bot": False, "first_name": "Human"}


def _message(update_id, *, user=88, kind="message", **fields):
    body = {"message_id": update_id, "date": 1800000000, "chat": {"id": 1000 + update_id, "type": "private"},
            "from": _sender(user), "text": "hello", **fields}
    return {"update_id": update_id, kind: body}


def _reaction(update_id):
    return {"update_id": update_id, "message_reaction": {
        "chat": {"id": 1000 + update_id, "type": "private"}, "message_id": 5, "date": 1800000000,
        "user": _sender(88), "old_reaction": [], "new_reaction": [{"type": "emoji", "emoji": "\U0001F44D"}]}}


_REPLIED = {"message_id": 1, "date": 1800000000, "chat": {"id": 1015, "type": "private"}}
SCRIPT = [
    [_message(10), _message(11, user=99)],
    [],  # an empty round trip
    [_message(12, kind="edited_message", edit_date=1800000010), _reaction(13),
     _message(14, x_unmodelled={"kept": True})],
    [_message(15, reply_to_message=_REPLIED)],  # core preparation fails before group 99
    [_message(10)],  # a redelivery: dropped by admission before any handler
]


def _digest(raw):
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


async def _until(condition):
    deadline = time.monotonic() + WAIT
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.01)


async def _fail_replies(msg, event):
    if msg.reply_to_message is not None:
        raise OSError("replied-to media unavailable")


def _delivered(event):
    source = event.source
    return [event.message_id, event.platform_update_id, event.message_type.value, event.text,
            event.reply_to_message_id, event.reply_to_text, event.media_urls, source.chat_id, source.chat_type,
            source.user_id, source.user_name, source.thread_id]


class _Gateway:
    """One offline Telegram adapter wired as the gateway wires it, plus what core produced."""

    def __init__(self, monkeypatch, home, script, observer):
        self.monkeypatch, self.home, self.wire = monkeypatch, home, _Wire(script)
        self.delivered, self.platform_events, self.later, self.errors = [], [], [], []
        manager = PluginManager()
        manager._discovered = True
        context = PluginContext(PluginManifest(name="fixture", source="user"), manager)
        context.register_hook("gateway_platform_event", lambda **event: None)
        if observer is not None:
            from gateway import ingress_observer

            monkeypatch.setattr(ingress_observer, "_dispatcher", ingress_observer._Dispatcher())
            monkeypatch.setattr(ingress_observer, "_IDLE_EXIT_SECONDS", 0.2)
            context.register_hook("gateway_ingress_observer", observer)
        monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
        monkeypatch.setenv("TELEGRAM_WEBHOOK_URL", "")
        self.adapter = TelegramAdapter(PlatformConfig(enabled=True, token=f"{BOT}:offline-test"))

    async def connect(self, *, error_handler=False):
        adapter, patch = self.adapter, self.monkeypatch.setattr
        patch(adapter, "_build_ptb_requests", AsyncMock(
            return_value=(_BotApi(), adapter._instrument_polling_request(self.wire))))
        patch(adapter, "_start_post_connect_housekeeping", lambda: None)
        patch(adapter, "_restart_task_attr", lambda name, coroutine: coroutine.close())
        patch(adapter, "_set_status_indicator", AsyncMock())
        patch(adapter, "_cache_replied_media", _fail_replies)
        patch(adapter, "_start_session_processing", lambda event, key, **_: self.delivered.append(event) or True)
        adapter._message_handler = AsyncMock()
        adapter.set_authorization_check(lambda user_id, chat_type=None, chat_id=None, **_: user_id == "88")

        async def platform_event(event, source):
            self.platform_events.append(event)

        async def later_group(update, context):
            self.later.append(update.update_id)

        async def error(update, context):
            self.errors.append(type(context.error).__name__)

        adapter.set_platform_event_handler(platform_event)
        assert await adapter.connect()
        adapter._app.add_handler(TypeHandler(Update, later_group), group=100)
        if error_handler:
            adapter._app.add_error_handler(error)

    async def play(self, script):
        adapter = self.adapter
        dispatched = adapter._updates_dispatched_total + sum(len(batch) for batch in script)
        self.wire.go.set()
        await _until(lambda: not self.wire.batches and adapter._updates_dispatched_total == dispatched
                     and not adapter._inflight_update_ids)
        while adapter._pending_text_batch_tasks:
            await asyncio.gather(*list(adapter._pending_text_batch_tasks.values()))

    def core(self):
        adapter = self.adapter
        return {
            "delivered": sorted(_delivered(event) for event in self.delivered),
            "seen": sorted(adapter._seen_update_ids), "inflight": sorted(adapter._inflight_update_ids),
            "received": adapter._updates_received_total, "dispatched": adapter._updates_dispatched_total,
            "platform_events": sorted(json.dumps(event, sort_keys=True) for event in self.platform_events),
            "later_groups": sorted(self.later), "errors": sorted(self.errors),
            "polling": [adapter._polling_progress_event.is_set(), adapter._send_path_degraded,
                        adapter._polling_network_error_count, adapter._polling_conflict_count,
                        adapter._polling_generation],
        }

    def receipts(self):
        path = self.home / f"telegram_update_receipts_{BOT}.json"
        return sorted(json.loads(path.read_text())["update_ids"])


@contextlib.contextmanager
def _home(path):
    token = set_hermes_home_override(path)
    try:
        yield path
    finally:
        reset_hermes_home_override(token)


async def _run(monkeypatch, home, *, observer=None, error_handler=False):
    with _home(home):
        gateway = _Gateway(monkeypatch, home, SCRIPT, observer)
        await gateway.connect(error_handler=error_handler)
        await gateway.play(SCRIPT)
        core = gateway.core()
        await gateway.adapter.disconnect()
        return {**core, "getupdates_offsets": gateway.wire.offsets, "receipts": gateway.receipts()}


class _Recorder:
    def __init__(self, mode="records"):
        self.mode, self.events = mode, []
        self.release = threading.Event()

    def __call__(self, **event):
        self.events.append(event)
        if self.mode == "raises":
            raise RuntimeError("observer failed")
        if self.mode == "blocks":
            self.release.wait()
        if self.mode == "slow":
            time.sleep(0.3)

    async def asynchronous(self, **event):
        self.events.append(event)
        await asyncio.sleep(0)
        if self.mode == "cancelled":
            asyncio.current_task().cancel()
            await asyncio.sleep(0)

    def callback(self):
        return self.asynchronous if self.mode in ("async", "cancelled") else self

    async def until(self, condition):
        await _until(lambda: condition(self.events))


@pytest.mark.asyncio
@pytest.mark.parametrize("error_handler", [False, True])
async def test_observers_never_change_core_ingress(monkeypatch, tmp_path, error_handler):
    baseline = await _run(monkeypatch, tmp_path / "none", error_handler=error_handler)
    if not error_handler:
        assert baseline["later_groups"] == [10, 11, 12, 13, 14]
        assert baseline["seen"] == [f"{BOT}:{update_id}" for update_id in (10, 11, 12, 13, 14)]
    # Probe, five scripted batches, the held long poll, the stop-time read receipt.
    assert len(baseline["getupdates_offsets"]) == 8
    for mode in ("records", "raises", "blocks", "slow", "async", "cancelled"):
        recorder = _Recorder(mode)
        try:
            assert await _run(monkeypatch, tmp_path / mode, observer=recorder.callback(),
                              error_handler=error_handler) == baseline, mode
        finally:
            recorder.release.set()
        assert recorder.events, mode


@pytest.mark.asyncio
async def test_fetched_and_observed_follow_the_fire_sites(monkeypatch, tmp_path):
    recorder = _Recorder()
    await _run(monkeypatch, tmp_path, observer=recorder)
    await recorder.until(lambda events: events and events[-1]["kind"] == "end")
    events = recorder.events

    assert [event["event_no"] for event in events] == list(range(1, len(events) + 1))
    assert {event["epoch"] for event in events} == {events[0]["epoch"]} and events[0]["kind"] == "start"
    assert all(event["fault"] is None and event["epoch_state"] == "live" for event in events)
    assert events[-1]["steps_clean"] is True
    assert events[-1]["prev"] == {"event_no": len(events) - 1, "outcome": "ok"}
    fetched = [event for event in events if event["kind"] == "fetched"]
    assert [len(event["updates"]) for event in fetched] == [0, 2, 0, 3, 1, 1]  # probe, then the script
    fetched_by_update = {}
    for event in fetched:
        for update in event["updates"]:
            fetched_by_update.setdefault(update["update_id"], (event["event_no"], update["raw_sha256"]))
    raw = {update["update_id"]: update for batch in SCRIPT for update in batch}
    assert {uid: digest for uid, (_, digest) in fetched_by_update.items()} == {uid: _digest(r) for uid, r in raw.items()}

    observed = {event["update_id"]: event for event in events if event["kind"] == "observed"}
    assert sorted(observed) == [10, 11, 12, 13, 14]  # not the failed 15, not the redelivered 10
    for update_id, event in observed.items():
        assert (event["fetch_event_no"], event["raw_sha256"]) == fetched_by_update[update_id]
        assert event["association"] == "ok" and event["event_no"] > event["fetch_event_no"]
    assert observed[10]["authorized"] is True and observed[10]["message"] == {
        "message_id": "10", "date": 1800000000, "edit_date": None, "text": "hello", "caption": None,
        "content_omitted": False, "reply_to_message_id": None, "is_forward": False, "has_quote": False,
        "media_kind": None}
    assert (observed[11]["authorized"], observed[11]["message"], observed[11]["user_id"]) == (False, None, "99")
    assert observed[12]["message"]["edit_date"] == 1800000010
    assert (observed[13]["authorized"], observed[13]["message"], observed[13]["chat_id"]) == (None, None, "1013")


@pytest.mark.asyncio
async def test_observations_can_arrive_out_of_fetch_order_across_chats(monkeypatch, tmp_path):
    """Updates of different chats dispatch concurrently: a later batch's update may be observed
    first, so ``fetch_event_no`` can decrease while ``event_no`` increases."""
    recorder = _Recorder()
    script = [[_message(40)], [_message(41)]]
    with _home(tmp_path):
        gateway = _Gateway(monkeypatch, tmp_path, script, recorder)
        await gateway.connect()
        released = asyncio.Event()

        async def hold_first_chat(update, context):
            if update.update_id == 40:
                await released.wait()

        gateway.adapter._app.add_handler(TypeHandler(Update, hold_first_chat), group=-1)
        playing = asyncio.ensure_future(gateway.play(script))
        await recorder.until(lambda events: any(event.get("update_id") == 41 for event in events))
        released.set()
        await playing
        await gateway.adapter.disconnect()
    await recorder.until(lambda events: events[-1]["kind"] == "end")

    observed = {event["update_id"]: event for event in recorder.events if event["kind"] == "observed"}
    assert observed[41]["event_no"] < observed[40]["event_no"]
    assert observed[41]["fetch_event_no"] > observed[40]["fetch_event_no"]
    assert observed[40]["association"] == observed[41]["association"] == "ok"


async def _observe_now(gateway, raw):
    await gateway.adapter._app.process_update(Update.de_json(raw, gateway.adapter._app.bot))


def _fetch(gateway, batch):
    """The getUpdates fire site as PTB's instrumented request reaches it."""
    adapter = gateway.adapter
    payload = json.dumps({"ok": True, "result": batch}).encode()
    adapter._observe_polling_request_result(gateway.wire, adapter._polling_generation, (200, payload))


@pytest.mark.asyncio
async def test_association_survives_a_generation_reset_and_flags_conflicts(monkeypatch, tmp_path):
    recorder = _Recorder()
    with _home(tmp_path):
        gateway = _Gateway(monkeypatch, tmp_path, [], recorder)
        await gateway.connect()
        changed = _message(21, text="edited in flight")
        _fetch(gateway, [_message(20), _message(21)])
        gateway.adapter._begin_polling_generation()
        _fetch(gateway, [_message(20), changed])
        for raw in (_message(20), changed, _message(22)):
            await _observe_now(gateway, raw)
        await gateway.adapter.disconnect()
    await recorder.until(lambda events: events[-1]["kind"] == "end")

    first_fetch = next(event for event in recorder.events if event["kind"] == "fetched" and event["updates"])
    observed = {event["update_id"]: event for event in recorder.events if event["kind"] == "observed"}
    assert (observed[20]["association"], observed[20]["fetch_event_no"], observed[20]["refetches"]) == (
        "ok", first_fetch["event_no"], 1)
    assert observed[21]["association"] == "conflict"
    assert (observed[22]["association"], observed[22]["fetch_event_no"]) == ("missing", None)


@pytest.mark.asyncio
async def test_content_is_omitted_above_the_cap_and_withheld_without_an_auth_check(monkeypatch, tmp_path):
    recorder = _Recorder()
    with _home(tmp_path):
        gateway = _Gateway(monkeypatch, tmp_path, [], recorder)
        await gateway.connect()
        gateway.adapter.set_platform_event_handler(None)  # observed fires before the group's own early return
        await _observe_now(gateway, _message(30, text="x" * (64 * 1024 + 1)))
        gateway.adapter.set_authorization_check(None)
        await _observe_now(gateway, _message(31))
        await gateway.adapter.disconnect()
    await recorder.until(lambda events: events[-1]["kind"] == "end")

    observed = {event["update_id"]: event for event in recorder.events if event["kind"] == "observed"}
    large = observed[30]["message"]
    assert (large["content_omitted"], large["text"], large["message_id"]) == (True, None, "30")
    assert (observed[31]["authorized"], observed[31]["message"]) == (None, None)


@pytest.mark.asyncio
async def test_a_long_epoch_keeps_constant_state_and_evicts_old_associations(monkeypatch, tmp_path):
    from gateway import ingress_observer

    recorder = _Recorder()
    with _home(tmp_path):
        gateway = _Gateway(monkeypatch, tmp_path, [], recorder)
        await gateway.connect()
        epoch = gateway.adapter._ingress_epoch
        for start in range(0, 6000, 100):
            _fetch(gateway, [_message(update_id) for update_id in range(start, start + 100)])
        assert len(epoch._associations) == ingress_observer._MAX_ASSOCIATIONS
        await _observe_now(gateway, _message(0))
        await _observe_now(gateway, _message(5999))
        await gateway.adapter.disconnect()
    await recorder.until(lambda events: events[-1]["kind"] == "end")

    observed = {event["update_id"]: event for event in recorder.events if event["kind"] == "observed"}
    assert (observed[0]["association"], observed[5999]["association"]) == ("missing", "ok")
    await _until(lambda: not ingress_observer._dispatcher._seals)
    assert not epoch._associations


@pytest.mark.asyncio
async def test_teardown_never_waits_for_a_blocked_observer(monkeypatch, tmp_path):
    """Shutdown stays within its own bounds; the end and the failures that happened after teardown
    began arrive once the observer returns, and a reconnect opens a new epoch on the same thread."""
    from gateway import ingress_observer

    monkeypatch.setattr(ingress_observer, "_LATE_AFTER_SECONDS", 60.0)  # held, not late: lateness is tested elsewhere
    recorder = _Recorder("blocks")

    def observe(**event):
        recorder(**event)
        if event["kind"] == "observed":
            raise RuntimeError("observer failed during teardown")

    with _home(tmp_path):
        gateway = _Gateway(monkeypatch, tmp_path, SCRIPT, observe)
        await gateway.connect()
        await gateway.play(SCRIPT)
        started = time.monotonic()
        await gateway.adapter.disconnect()
        assert time.monotonic() - started < WAIT
        first_epoch = recorder.events[0]["epoch"]
        assert [event["kind"] for event in recorder.events] == ["start"]
        thread = ingress_observer._dispatcher._thread
        gateway.wire.go.clear()
        await gateway.adapter.connect()
        await gateway.adapter.disconnect()
        assert ingress_observer._dispatcher._thread is thread
    recorder.release.set()
    await recorder.until(lambda events: sum(event["kind"] == "end" for event in events) == 2)

    first = [event for event in recorder.events if event["epoch"] == first_epoch]
    second = [event for event in recorder.events if event["epoch"] != first_epoch]
    first_observed = next(event["event_no"] for event in first if event["kind"] == "observed")
    assert first[-1]["kind"] == "end" and first[-1]["steps_clean"] is True
    assert first[-1]["fault"] == {"first_event_no": first_observed, "kinds": ("failed",), "dropped_count": 0}
    assert (second[0]["kind"], second[0]["event_no"], second[-1]["kind"]) == ("start", 1, "end")


@pytest.mark.asyncio
async def test_a_transient_rebuild_keeps_the_epoch_and_its_numbering(monkeypatch, tmp_path):
    recorder = _Recorder()
    failures = iter([OSError("connect reset")])
    original = _BotApi.do_request

    async def flaky(self, url, method, request_data=None, **kwargs):
        if url.endswith("/getMe"):
            error = next(failures, None)
            if error is not None:
                raise error
        return await original(self, url, method, request_data, **kwargs)

    monkeypatch.setattr(_BotApi, "do_request", flaky)
    with _home(tmp_path):
        gateway = _Gateway(monkeypatch, tmp_path, [], recorder)
        await gateway.connect()
        adapter = gateway.adapter
        generation = adapter._polling_generation
        assert await adapter._stop_updater_or_go_fatal(adapter._app, "test restart")
        await adapter._start_polling_once(adapter._app, drop_pending_updates=False, error_callback=None,
                                          schedule_verifier=False)
        await _until(lambda: any(event["generation"] > generation for event in recorder.events))
        await adapter.disconnect()
    await recorder.until(lambda events: events[-1]["kind"] == "end")

    events = recorder.events
    assert next(failures, None) is None  # the first connect attempt failed and the app was rebuilt
    assert len({event["epoch"] for event in events}) == 1
    assert [event["event_no"] for event in events] == list(range(1, len(events) + 1))
    assert events[0]["kind"] == "start" and events[-1]["kind"] == "end"


@pytest.mark.asyncio
@pytest.mark.parametrize("step,failure", [
    ("app.shutdown", None), ("app.shutdown", "cancelled"), ("app.shutdown", "abandoned"), ("app.shutdown", "raised"),
    ("app.stop", "cancelled"), ("updater.stop", "cancelled")])
async def test_end_follows_the_stop_steps_and_only_a_normal_finish_is_clean(monkeypatch, tmp_path, step, failure):
    from gateway.ingress_observer import IngressEpoch

    recorder = _Recorder()
    with _home(tmp_path):
        gateway = _Gateway(monkeypatch, tmp_path, [], recorder)
        adapter = gateway.adapter
        adapter._ingress_epoch = stale = IngressEpoch("telegram", str(BOT))  # never disconnected
        await gateway.connect()
        assert stale.closed and adapter._ingress_epoch is not stale
        epoch, open_during = adapter._ingress_epoch, []
        owner = type(adapter._app.updater if step == "updater.stop" else adapter._app)
        name = step.split(".")[1]
        original = getattr(owner, name)

        async def substitute(self):
            open_during.append(not epoch.closed)
            if failure == "abandoned":
                await asyncio.sleep(60)
            if failure == "raised":
                raise RuntimeError(f"{step} failed")
            await original(self)
            if failure == "cancelled":
                raise asyncio.CancelledError  # the step's own task ends cancelled, not finished

        monkeypatch.setattr(owner, name, substitute)
        monkeypatch.setattr(tg_adapter, "_DISCONNECT_STEP_TIMEOUT", 0.2)
        app = adapter._app
        await adapter.disconnect()
        if failure in ("abandoned", "raised"):
            await original(app)
    await recorder.until(lambda events: events[-1]["kind"] == "end")

    assert open_during == [True]
    assert recorder.events[-1]["steps_clean"] is (failure is None)

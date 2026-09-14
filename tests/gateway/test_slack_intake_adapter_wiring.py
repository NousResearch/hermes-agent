"""Production wiring contracts for Ring 2 Slack intake observability."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from plugins.platforms.slack.adapter import SlackAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["drop", "exception", "cancel"])
async def test_repair_missing_context_releases_failed_prefilter_claim(failure):
    adapter = SlackAdapter.__new__(SlackAdapter)
    adapter._intake_observer = SimpleNamespace(context_for_event=lambda *_args, **_kwargs: None)
    adapter._dedup = SimpleNamespace(discard=MagicMock())
    event = {"type": "message", "ts": "1", "team": "T"}
    adapter._handle_slack_message = AsyncMock(return_value="missing_mention")
    if failure == "exception":
        adapter._handle_slack_message.side_effect = ValueError("original")
    if failure == "cancel":
        adapter._handle_slack_message.side_effect = asyncio.CancelledError()
    if failure == "drop":
        assert await adapter._handle_observed_slack_message(event, {}) == "missing_mention"
    else:
        with pytest.raises(ValueError if failure == "exception" else asyncio.CancelledError):
            await adapter._handle_observed_slack_message(event, {})
    adapter._dedup.discard.assert_called_once_with(adapter._workspace_event_id("T", "1"))


@pytest.mark.asyncio
async def test_message_listener_routes_real_handler_through_observer():
    adapter = SlackAdapter.__new__(SlackAdapter)
    event = {"type": "app_mention", "ts": "1712345.200"}
    body = {"event_id": "Ev-test", "event": event}
    context = SimpleNamespace(receipt_id="a" * 64)
    async def observed(_context, callback, *args):
        await callback(*args)
        return "terminal"

    adapter._intake_observer = SimpleNamespace(
        context_for_event=lambda event_id, **_kwargs: context if event_id == "Ev-test" else None,
        run_listener=AsyncMock(side_effect=observed),
    )
    adapter._handle_slack_message = AsyncMock(return_value=None)

    result = await adapter._handle_observed_slack_message(event, body)

    assert result == "terminal"
    adapter._intake_observer.run_listener.assert_awaited_once()
    args = adapter._intake_observer.run_listener.await_args.args
    assert args[0] is context
    assert args[2:] == (event, body)
    adapter._handle_slack_message.assert_awaited_once_with(event, body)


@pytest.mark.asyncio
async def test_message_listener_fails_open_when_envelope_context_is_absent():
    adapter = SlackAdapter.__new__(SlackAdapter)
    event = {"type": "message", "ts": "1712345.300"}
    body = {"event_id": "Ev-missing", "event": event}
    adapter._intake_observer = SimpleNamespace(context_for_event=lambda _event_id, **_kwargs: None)
    adapter._handle_slack_message = AsyncMock(return_value="ignored_channel")

    result = await adapter._handle_observed_slack_message(event, body)

    assert result == "ignored_channel"
    adapter._handle_slack_message.assert_awaited_once_with(event, body)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["drop", "exception", "cancel"])
async def test_nonaccepted_first_twin_releases_prefilter_dedup_claim(failure):
    adapter = SlackAdapter.__new__(SlackAdapter)
    event = {"type": "app_mention", "ts": "1712345.400", "team": "T-test"}
    body = {"event_id": "Ev-test", "event": event}
    context = SimpleNamespace(receipt_id="a" * 64)
    dedup = SimpleNamespace(discard=MagicMock())
    adapter._dedup = dedup
    adapter._event_team_id = lambda _event, _body: "T-test"
    adapter._workspace_event_id = lambda team, ts: f"{team}:{ts}"

    async def observe_owned_attempt(_context, callback, *args):
        await callback(*args)
        return SimpleNamespace(state="dropped", reason="missing_mention")

    run_listener = AsyncMock(side_effect=observe_owned_attempt)
    adapter._intake_observer = SimpleNamespace(
        context_for_event=lambda _event_id, **_kwargs: context,
        run_listener=run_listener,
    )
    adapter._handle_slack_message = AsyncMock(return_value="missing_mention")
    if failure == "exception":
        adapter._handle_slack_message.side_effect = RuntimeError("listener failed")
    if failure == "cancel":
        adapter._handle_slack_message.side_effect = asyncio.CancelledError()

    if failure == "drop":
        await adapter._handle_observed_slack_message(event, body)
    elif failure == "exception":
        with pytest.raises(RuntimeError, match="listener failed"):
            await adapter._handle_observed_slack_message(event, body)
    else:
        with pytest.raises(asyncio.CancelledError):
            await adapter._handle_observed_slack_message(event, body)

    dedup.discard.assert_called_once_with("T-test:1712345.400")


def test_socket_handler_installs_observer_before_start(monkeypatch: pytest.MonkeyPatch):
    order: list[str] = []
    client = SimpleNamespace(logger=None)
    handler = SimpleNamespace(client=client, start_async=AsyncMock())
    observer = object()

    class _Handler:
        def __new__(cls, *_args, **_kwargs):
            order.append("construct")
            return handler

    def _install(actual_client, actual_observer):
        assert actual_client is client
        assert actual_observer is observer
        order.append("install")

    class _Task:
        def add_done_callback(self, _callback):
            order.append("callback")

    def _create_task(coro):
        assert asyncio.iscoroutine(coro)
        order.append("start")
        coro.close()
        return _Task()

    import plugins.platforms.slack.adapter as adapter_module

    monkeypatch.setattr(adapter_module, "AsyncSocketModeHandler", _Handler)
    monkeypatch.setattr(adapter_module, "install_socket_observer", _install)
    monkeypatch.setattr(asyncio, "create_task", _create_task)
    monkeypatch.setattr(adapter_module, "_apply_slack_proxy", lambda *_args: None)

    adapter = SlackAdapter.__new__(SlackAdapter)
    adapter._app = object()
    adapter._app_token = "not-a-real-token"
    adapter._proxy_url = None
    adapter._intake_observer = observer
    adapter._socket_traceback_filter = object()
    adapter._on_socket_mode_task_done = lambda _task: None

    adapter._start_socket_mode_handler()

    assert order[:3] == ["construct", "install", "start"]


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ({"subtype": "message_deleted"}, "message_deleted"),
        ({"subtype": "message_changed", "message": None}, "invalid_event"),
    ],
)
@pytest.mark.asyncio
async def test_early_handler_exits_return_fixed_drop_reasons(event, expected):
    adapter = SlackAdapter.__new__(SlackAdapter)
    adapter._dedup = SimpleNamespace(is_duplicate=lambda _key: False)
    adapter._processed_message_ts = {}

    result = await adapter._handle_slack_message(event, {})

    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("first_result", ["missing_mention", "error", "cancel", None])
async def test_concurrent_twins_wait_for_owner_disposition(first_result):
    from gateway.platforms.helpers import MessageDeduplicator

    adapter = SlackAdapter.__new__(SlackAdapter)
    adapter._intake_observer = SimpleNamespace(context_for_event=lambda *_a, **_k: None)
    adapter._dedup = MessageDeduplicator()
    entered, release, second_arrived = asyncio.Event(), asyncio.Event(), asyncio.Event()
    delivered = []
    original = ValueError("synthetic failed owner")

    async def handler(event, payload):
        key = adapter._workspace_event_id("T", "1")
        if adapter._dedup.is_duplicate(key):
            return "duplicate_event"
        if payload["event_id"] == "first":
            entered.set()
            await release.wait()
            if first_result == "error":
                raise original
            if first_result == "cancel":
                raise asyncio.CancelledError()
            if first_result:
                return first_result
        delivered.append(payload["event_id"])
        return None

    adapter._handle_slack_message = handler
    event = {"type": "message", "ts": "1", "team": "T"}
    first = asyncio.create_task(adapter._handle_observed_slack_message(event, {"event_id": "first"}))

    async def second_call():
        second_arrived.set()
        return await adapter._handle_observed_slack_message(event, {"event_id": "second"})

    second = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        second = asyncio.create_task(second_call())
        await asyncio.wait_for(second_arrived.wait(), 2)
        # The second task has reached the competing wrapper, not a timed sleep.
        release.set()
        results = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 2)
        assert delivered == (["first"] if first_result is None else ["second"])
        if first_result == "error":
            assert results[0] is original
        if first_result == "cancel":
            assert isinstance(results[0], asyncio.CancelledError)
    finally:
        release.set()
        for task in (first, second):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (first, second) if t is not None), return_exceptions=True)


@pytest.mark.asyncio
async def test_observer_failure_before_handler_does_not_discard_accepted_sibling_claim():
    from gateway.platforms.helpers import MessageDeduplicator

    adapter = SlackAdapter.__new__(SlackAdapter)
    adapter._dedup = MessageDeduplicator()
    key = adapter._workspace_event_id("T", "1")
    assert adapter._dedup.is_duplicate(key) is False
    original = asyncio.CancelledError("synthetic pre-handler cancellation")
    adapter._intake_observer = SimpleNamespace(
        context_for_event=lambda *_a, **_k: object(),
        run_listener=AsyncMock(side_effect=original),
    )
    adapter._handle_slack_message = AsyncMock()
    with pytest.raises(asyncio.CancelledError) as caught:
        await adapter._handle_observed_slack_message({"type": "message", "team": "T", "ts": "1"}, {})
    assert caught.value is original
    adapter._handle_slack_message.assert_not_awaited()
    assert adapter._dedup.is_duplicate(key) is True


@pytest.mark.asyncio
async def test_successful_handler_claim_survives_terminal_instrumentation_cancellation():
    from gateway.platforms.helpers import MessageDeduplicator

    adapter = SlackAdapter.__new__(SlackAdapter)
    adapter._dedup = MessageDeduplicator()
    key = adapter._workspace_event_id("T", "1")
    original = asyncio.CancelledError("synthetic post-handler cancellation")

    async def handler(*_args):
        assert adapter._dedup.is_duplicate(key) is False
        return None

    async def observed(_context, callback, *args):
        await callback(*args)
        raise original

    adapter._handle_slack_message = handler
    adapter._intake_observer = SimpleNamespace(
        context_for_event=lambda *_a, **_k: object(), run_listener=observed,
    )
    with pytest.raises(asyncio.CancelledError) as caught:
        await adapter._handle_observed_slack_message({"type": "message", "team": "T", "ts": "1"}, {})
    assert caught.value is original
    assert adapter._dedup.is_duplicate(key) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("twin_type", ["message", "app_mention"])
async def test_cancelled_twin_lock_waiter_terminalizes_without_releasing_owner(monkeypatch, tmp_path, twin_type):
    from gateway import slack_intake_ledger as ledger

    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    adapter = _real_offline_adapter()
    event = {"type": "message", "team": "T", "channel": "C", "channel_type": "mpim",
             "ts": "123.000001", "text": "<@UBOT> synthetic request", "user": "U"}
    twin = dict(event, type=twin_type)

    async def observe(ident, ev):
        body = {"team_id": "T", "event_id": ident, "event": ev}
        await adapter._intake_observer.observe_socket_message({
            "type": "events_api", "envelope_id": ident, "payload": body}, "{}")
        return body

    first_body, second_body = await observe("first", event), await observe("second", twin)
    entered, release, waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cancellations = []

    async def blocked(_message):
        entered.set()
        await release.wait()

    adapter.handle_message = AsyncMock(side_effect=blocked)
    first = asyncio.create_task(adapter._handle_observed_slack_message(event, first_body))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        lock = next(iter(adapter._intake_message_locks.values()))
        original_acquire = lock.acquire

        async def acquire():
            waiting.set()
            try:
                return await original_acquire()
            except asyncio.CancelledError as exc:
                cancellations.append(exc)
                raise

        monkeypatch.setattr(lock, "acquire", acquire)
        second = asyncio.create_task(adapter._handle_observed_slack_message(twin, second_body))
        await asyncio.wait_for(waiting.wait(), 2)
        assert not second.done()
        second.cancel("synthetic waiter cancellation")
        with pytest.raises(asyncio.CancelledError) as caught:
            await second
        assert caught.value is cancellations[0]
        row = next(r for r in ledger.read_receipts() if r["receipt_id"] == ledger._receipt_id("T", "second"))
        assert row["terminal_state"] == "dropped" and row["terminal_reason"] == "listener_cancelled"
        assert not first.done(), "waiter classification must not wait for owner settlement"
        release.set()
        accepted = await asyncio.wait_for(first, 2)
        third = await adapter._handle_observed_slack_message(twin, await observe("third", twin))
        assert accepted.state == "accepted"
        assert third.state == "dropped" and third.related_receipt_id == ledger._receipt_id("T", "first")
        adapter.handle_message.assert_awaited_once()
    finally:
        release.set()
        for task in (first, second):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (first, second) if t is not None), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_retry", [True, False])
async def test_inflight_same_event_retry_cannot_change_owner_receipt(monkeypatch, tmp_path, cancel_retry):
    from gateway import slack_intake_ledger as ledger

    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    adapter = _real_offline_adapter()
    event = {"type": "message", "team": "T", "channel": "C", "channel_type": "mpim",
             "ts": "123.000001", "text": "<@UBOT> synthetic request", "user": "U"}
    body = {"team_id": "T", "event_id": "same-event", "event": event}
    entered, release, retry_started = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def dispatch(_message):
        entered.set()
        await release.wait()

    adapter.handle_message = AsyncMock(side_effect=dispatch)
    await adapter._intake_observer.observe_socket_message({
        "type": "events_api", "envelope_id": "first-envelope", "payload": body}, "{}")
    owner = asyncio.create_task(adapter._handle_observed_slack_message(event, body))
    retry = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await adapter._intake_observer.observe_socket_message({
            "type": "events_api", "envelope_id": "retry-envelope", "payload": body}, "{}")

        async def retry_callback():
            retry_started.set()
            return await adapter._handle_observed_slack_message(event, body)

        retry = asyncio.create_task(retry_callback())
        await asyncio.wait_for(retry_started.wait(), 2)
        assert not retry.done()
        if cancel_retry:
            retry.cancel("synthetic retry cancellation")
            with pytest.raises(asyncio.CancelledError, match="synthetic retry cancellation"):
                await retry
            row = ledger.read_receipts()[0]
            assert row["terminal_state"] is None, "cancelled retry cannot terminalize its owner's receipt"
        assert not owner.done()
        release.set()
        assert (await asyncio.wait_for(owner, 2)).state == "accepted"
        if not cancel_retry:
            assert (await asyncio.wait_for(retry, 2)).state == "accepted"
        row = ledger.read_receipts()[0]
        assert row["terminal_state"] == "accepted"
        assert row["receive_count"] == 2
        adapter.handle_message.assert_awaited_once()
    finally:
        release.set()
        for task in (owner, retry):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (owner, retry) if t is not None), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("second_team, second_channel", [("T-two", "C"), ("T-one", "C-other")])
async def test_real_edit_completion_cache_is_scoped_to_workspace_and_channel(monkeypatch, tmp_path, second_team, second_channel):
    from gateway import slack_intake_ledger as ledger

    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    adapter = _real_offline_adapter()

    async def deliver(ident, team, ev):
        body = {"team_id": team, "event_id": ident, "event": ev}
        await adapter._intake_observer.observe_socket_message({
            "type": "events_api", "envelope_id": ident, "payload": body}, "{}")
        return await adapter._handle_observed_slack_message(ev, body)

    first_event = {"type": "message", "channel": "C", "channel_type": "mpim",
                   "ts": "123.000001", "text": "<@UBOT> synthetic request", "user": "U"}
    first = await deliver("first", "T-one", first_event)
    edited = {"type": "message", "subtype": "message_changed", "channel": second_channel,
              "channel_type": "mpim", "message": {"text": "<@UBOT> synthetic edit", "user": "U",
              "ts": "123.000001", "edited": {"ts": "124.000001", "user": "U"}}}
    second = await deliver("second", second_team, edited)
    assert adapter.handle_message.await_count == 2
    assert first.state == second.state == "accepted"
    assert first.related_receipt_id is second.related_receipt_id is None
    rows = ledger.read_receipts()
    assert len({(row["workspace_hash"], row["channel_hash"], row["message_hash"]) for row in rows}) == 2
    assert all(row["terminal_state"] == "accepted" for row in rows)
    repeated = await deliver("second-twin", second_team, edited)
    assert repeated.state == "dropped" and repeated.related_receipt_id == ledger._receipt_id(second_team, "second")
    assert adapter.handle_message.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("edited", [False, True])
async def test_channel_local_timestamps_do_not_share_inflight_lock_or_dedup(monkeypatch, tmp_path, edited):
    from gateway import slack_intake_ledger as ledger

    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    adapter = _real_offline_adapter()
    entered, release, independent = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def dispatch(message):
        if message.text == "first":
            entered.set()
            await release.wait()
        else:
            independent.set()

    adapter.handle_message = AsyncMock(side_effect=dispatch)

    async def deliver(label, channel):
        event = {"type": "message", "team": "T", "channel": channel, "channel_type": "mpim",
                 "ts": "123.000001", "text": f"<@UBOT> {label}", "user": "U"}
        if edited:
            event = {"type": "message", "subtype": "message_changed", "team": "T",
                     "channel": channel, "channel_type": "mpim", "message": {
                         "ts": "123.000001", "text": f"<@UBOT> {label}", "user": "U",
                         "edited": {"ts": "124.000001", "user": "U"}}}
        body = {"team_id": "T", "event_id": label, "event": event}
        await adapter._intake_observer.observe_socket_message({
            "type": "events_api", "envelope_id": label, "payload": body}, "{}")
        return await adapter._handle_observed_slack_message(event, body)

    first = asyncio.create_task(deliver("first", "C-one"))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        second = asyncio.create_task(deliver("second", "C-two"))
        await asyncio.wait_for(independent.wait(), 2)
        assert not first.done()
        assert (await asyncio.wait_for(second, 2)).state == "accepted"
        release.set()
        assert (await asyncio.wait_for(first, 2)).state == "accepted"
        assert adapter.handle_message.await_count == 2
    finally:
        release.set()
        for task in (first, second):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (first, second) if t is not None), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("channel, expected", [("C-one", 0), ("C-two", 1)])
async def test_file_share_fallback_uses_same_channel_scoped_dedup(monkeypatch, channel, expected):
    adapter = _real_offline_adapter()
    event = {"type": "message", "team": "T", "channel": "C-one", "channel_type": "mpim",
             "ts": "123.000001", "text": "<@UBOT> synthetic request", "user": "U"}
    await adapter._handle_observed_slack_message(event, {})
    adapter.handle_message.assert_awaited_once()
    adapter._app.client.files_info = AsyncMock(return_value={"ok": True, "file": {
        "id": "F", "user": "U", "mimetype": "video/mp4",
        "shares": {"private": {channel: [{"ts": "123.000001"}]}},
    }})
    adapter._handle_slack_message = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    await adapter._handle_slack_file_shared({"team": "T", "channel_id": channel, "file_id": "F"})
    assert adapter._handle_slack_message.await_count == expected


def _real_offline_adapter():
    from gateway.config import PlatformConfig

    adapter = SlackAdapter(PlatformConfig(enabled=True, token="synthetic-not-a-token"))
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._app.client.users_info = AsyncMock(return_value={"user": {
        "is_bot": False, "profile": {"display_name": "Synthetic User"},
        "real_name": "Synthetic User",
    }})
    adapter._app.client.conversations_info = AsyncMock(return_value={"channel": {"name": "synthetic"}})
    adapter._app.client.conversations_replies = AsyncMock(return_value={"messages": []})
    adapter._app.client.conversations_history = AsyncMock(return_value={"messages": []})
    adapter._bot_user_id = "UBOT"
    adapter._running = True
    adapter.handle_message = AsyncMock()
    return adapter


@pytest.mark.asyncio
async def test_identity_conflict_cannot_route_stale_receipt_through_adapter(monkeypatch, tmp_path):
    from gateway import slack_intake_ledger as ledger

    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    ledger._reset_initialization_for_tests()
    ledger._reset_persistence_health_for_tests()
    adapter = _real_offline_adapter()
    adapter._handle_slack_message = AsyncMock(return_value=None)
    original_event = {
        "type": "message",
        "team": "T",
        "channel": "C-original",
        "ts": "123.000001",
        "text": "synthetic original",
        "user": "U",
    }
    original_body = {"team_id": "T", "event_id": "E-conflict", "event": original_event}
    await adapter._intake_observer.observe_socket_message(
        {"type": "events_api", "envelope_id": "X-conflict", "payload": original_body},
        "{}",
    )
    original = adapter._intake_observer.context_for_event("E-conflict", workspace_id="T")
    assert original is not None
    assert adapter._intake_observer.context_for_envelope("X-conflict") is original
    # The two bounded maps can evict asymmetrically. Receipt ownership must not
    # depend on the event context still being retained when the conflict lands.
    adapter._intake_observer._event_contexts.clear()
    conflicting_event = dict(original_event, channel="C-conflicting")
    conflicting_body = {
        "team_id": "T",
        "event_id": "E-conflict",
        "event": conflicting_event,
    }
    await adapter._intake_observer.observe_socket_message(
        {"type": "events_api", "envelope_id": "X-conflict-other", "payload": conflicting_body},
        "{}",
    )
    # A later exact retry must not clear the ambiguity created by the conflict.
    await adapter._intake_observer.observe_socket_message(
        {"type": "events_api", "envelope_id": "X-fresh-retry", "payload": original_body},
        "{}",
    )

    await adapter._handle_observed_slack_message(conflicting_event, conflicting_body)
    await adapter._intake_observer.acknowledged_for_envelope("X-conflict")
    await adapter._intake_observer.acknowledged_for_envelope("X-conflict-other")
    await adapter._intake_observer.acknowledged_for_envelope("X-fresh-retry")

    assert original.receipt_id is not None
    assert adapter._intake_observer.context_for_event("E-conflict", workspace_id="T") is None
    assert adapter._intake_observer.context_for_envelope("X-conflict") is None
    assert adapter._intake_observer.context_for_envelope("X-conflict-other") is None
    assert adapter._intake_observer.context_for_envelope("X-fresh-retry") is None
    adapter._handle_slack_message.assert_awaited_once_with(conflicting_event, conflicting_body)
    receipt = ledger.read_receipts()[0]
    assert receipt["terminal_state"] is None
    assert "acknowledged" not in {event["stage"] for event in receipt["events"]}
    assert "listener_entered" not in {event["stage"] for event in receipt["events"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["message", "app_mention", "file"])
@pytest.mark.parametrize("concurrent", [False, True])
@pytest.mark.parametrize("outcome", ["success", "exception", "cancel"])
async def test_edit_owner_controls_original_and_file_dispatch(
    monkeypatch, tmp_path, kind, concurrent, outcome
):
    from gateway import slack_intake_ledger as ledger

    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    adapter = _real_offline_adapter()
    adapter.config.extra["require_mention"] = False
    assert adapter._app is not None
    adapter._app.client.files_info = AsyncMock(return_value={"ok": True, "file": {
        "id": "F", "user": "U", "mimetype": "video/mp4",
        "shares": {"private": {"C": [{"ts": "123.000001"}]}},
    }})
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    entered, release, waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = RuntimeError("synthetic edit failure") if outcome == "exception" else asyncio.CancelledError()
    calls = []

    async def dispatch(message):
        if message.text == "edit":
            entered.set()
            await release.wait()
            if outcome != "success":
                raise original
        calls.append(message.text)

    adapter.handle_message = AsyncMock(side_effect=dispatch)
    edit = {"type": "message", "subtype": "message_changed", "team": "T",
            "channel": "C", "channel_type": "mpim", "message": {
                "ts": "123.000001", "edited": {"ts": "124.000001"},
                "text": "<@UBOT> edit", "user": "U"}}

    async def sibling():
        if kind == "file":
            return await adapter._handle_slack_file_shared({
                "team": "T", "channel_id": "C", "file_id": "F"})
        event = {"type": kind, "team": "T", "channel": "C", "channel_type": "mpim",
                 "ts": "123.000001", "text": "<@UBOT> original", "user": "U"}
        return await _deliver_offline(adapter, event, "original")

    owner = asyncio.create_task(_deliver_offline(adapter, edit, "edit"))
    successor = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if concurrent:
            lock = next(iter(adapter._intake_message_locks.values()))
            acquire = lock.acquire

            async def competing_acquire():
                waiting.set()
                return await acquire()

            monkeypatch.setattr(lock, "acquire", competing_acquire)
            successor = asyncio.create_task(sibling())
            await asyncio.wait_for(waiting.wait(), 2)
            assert not successor.done()
        release.set()
        if outcome == "success":
            assert (await owner).state == "accepted"
        else:
            with pytest.raises(type(original)) as caught:
                await owner
            assert caught.value is original
        result = await asyncio.wait_for(successor, 2) if successor else await sibling()
        assert calls == (["edit"] if outcome == "success" else ["" if kind == "file" else "original"])
        if kind != "file":
            assert result is not None
            assert result.state == ("dropped" if outcome == "success" else "accepted")
            if outcome == "success":
                assert result.related_receipt_id == ledger._receipt_id("T", "edit")
    finally:
        release.set()
        for task in (owner, successor):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (owner, successor) if t is not None), return_exceptions=True)


async def _deliver_offline(adapter, event, ident, **identity):
    body = {"event_id": ident, "event": event, **identity}
    await adapter._intake_observer.observe_socket_message({
        "type": "events_api", "envelope_id": ident, "payload": body}, "{}")
    return await adapter._handle_observed_slack_message(event, body)


@pytest.mark.asyncio
@pytest.mark.parametrize("first_type", ["message", "app_mention"])
async def test_accepted_twin_suppresses_later_semantic_twin(
    monkeypatch, tmp_path, first_type
):
    from gateway import slack_intake_ledger as ledger

    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    ledger._reset_initialization_for_tests()
    ledger._reset_persistence_health_for_tests()
    adapter = _real_offline_adapter()
    adapter.config.extra["require_mention"] = False
    handler = AsyncMock()
    adapter.handle_message = handler
    first_event = {
        "type": first_type, "team": "T", "channel": "C",
        "channel_type": "mpim", "ts": "123.000001",
        "text": "<@UBOT> synthetic request", "user": "U",
    }
    second_type = "message" if first_type == "app_mention" else "app_mention"
    second_event = dict(first_event, type=second_type)

    first = await _deliver_offline(adapter, first_event, "first")
    second = await _deliver_offline(adapter, second_event, "second")
    first_receipt_id = ledger._receipt_id("T", "first")
    second_receipt_id = ledger._receipt_id("T", "second")

    assert first.state == "accepted"
    assert second.state == "dropped"
    assert second.reason == "duplicate_ts"
    assert second.related_receipt_id == first_receipt_id
    handler.assert_awaited_once()
    rows = {row["receipt_id"]: row for row in ledger.read_receipts()}
    assert rows[first_receipt_id]["terminal_state"] == "accepted"
    assert rows[second_receipt_id]["terminal_reason"] == "duplicate_ts"
    assert rows[second_receipt_id]["related_receipt_id"] == first_receipt_id
    assert all(row["terminal_state"] is not None for row in rows.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["drop", "exception", "cancel", "success"])
@pytest.mark.parametrize("concurrent", [False, True])
async def test_real_file_fallback_owns_claim_until_disposition(monkeypatch, tmp_path, outcome, concurrent):
    from gateway import slack_intake_ledger as ledger

    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    adapter = _real_offline_adapter()
    adapter.config.extra["require_mention"] = outcome == "drop"
    adapter._app.client.files_info = AsyncMock(return_value={"ok": True, "file": {
        "id": "F", "user": "U", "mimetype": "video/mp4",
        "shares": {"private": {"C": [{"ts": "123.000001"}]}},
    }})
    # No file URL: no downloads. Only the grace period and external client
    # response are controlled; policy, handler, observer and dedup stay real.
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    entered, release, competing = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = RuntimeError("synthetic fallback failure") if outcome == "exception" else asyncio.CancelledError()
    calls = []

    async def dispatch(message):
        if not message.text:
            calls.append("fallback")
            if outcome in {"exception", "cancel"}:
                raise original
        else:
            calls.append("ordinary")

    adapter.handle_message = AsyncMock(side_effect=dispatch)
    response = {"user": {"is_bot": False, "profile": {"display_name": "Synthetic"}}}

    async def user_info(**_kwargs):
        entered.set()
        await release.wait()
        return response

    adapter._app.client.users_info = AsyncMock(side_effect=user_info)
    event = {"type": "message", "team": "T", "channel": "C", "channel_type": "mpim",
             "ts": "123.000001", "text": "<@UBOT> eligible", "user": "U"}

    async def ordinary():
        # Persist before signaling arrival, then enter the real wrapper.
        body = {"team_id": "T", "event_id": "ordinary", "event": event}
        await adapter._intake_observer.observe_socket_message({
            "type": "events_api", "envelope_id": "ordinary", "payload": body}, "{}")
        competing.set()
        return await adapter._handle_observed_slack_message(event, body)

    fallback = asyncio.create_task(adapter._handle_slack_file_shared({
        "team": "T", "channel_id": "C", "file_id": "F"}))
    sibling = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if concurrent:
            sibling = asyncio.create_task(ordinary())
            await asyncio.wait_for(competing.wait(), 2)
        release.set()
        if outcome in {"exception", "cancel"}:
            with pytest.raises(type(original)) as caught:
                await fallback
            assert caught.value is original
        else:
            await fallback
        result = await asyncio.wait_for(sibling, 2) if sibling else await ordinary()
        assert calls == ({"drop": ["ordinary"], "success": ["fallback"]}.get(
            outcome, ["fallback", "ordinary"]))
        assert result.state == ("dropped" if outcome == "success" else "accepted")
        if outcome == "success":
            assert result.reason == "duplicate_event"
        else:
            assert ledger.read_receipts()[0]["terminal_state"] == "accepted"
        # A retained original exception owns traceback frames (and their lock
        # references). Release, not immediate garbage collection, is required.
        assert all(not lock.locked() for lock in adapter._intake_message_locks.values())
    finally:
        release.set()
        for task in (fallback, sibling):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (fallback, sibling) if t is not None), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("event_identity, body_identity, expected", [
    ({"team": "T"}, {"team_id": "T-envelope"}, "T"),
    ({"team_id": "T", "team": "T-other"}, {"team": {"id": "T-envelope"}}, "T"),
    ({"team": {"id": "T"}}, {"team_id": "T-envelope"}, "T"),
    ({}, {"team": "T-envelope"}, "T-envelope"),
    ({}, {"team": {"id": "T-envelope"}}, "T-envelope"),
    ({}, {"authorizations": [{"team_id": "T-authorized"}]}, "T-authorized"),
])
async def test_workspace_identity_is_consistent_from_envelope_through_dispatch(
    monkeypatch, tmp_path, event_identity, body_identity, expected
):
    from gateway import slack_intake_ledger as ledger

    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    adapter = _real_offline_adapter()
    event = {"type": "message", "channel": "C", "channel_type": "mpim",
             "ts": "123.000001", "text": "<@UBOT> synthetic", "user": "U", **event_identity}
    selected = adapter._app.client
    adapter._team_clients[expected] = selected
    adapter._app.client = _real_offline_adapter()._app.client
    result = await _deliver_offline(adapter, event, "identity", **body_identity)
    rows = ledger.read_receipts()
    assert len(rows) == 1
    assert rows[0]["receipt_id"] == ledger._receipt_id(expected, "identity")
    assert rows[0]["terminal_state"] == "accepted"
    assert "listener_entered" in [entry["stage"] for entry in rows[0]["events"]]
    assert result.state == "accepted"
    delivered = adapter.handle_message.await_args.args[0]
    assert delivered.source.scope_id == expected
    assert delivered.metadata["slack_team_id"] == expected
    assert (expected, "C", "123.000001") in adapter._processed_message_ts
    adapter._app.client.users_info.assert_not_awaited()
    selected.users_info.assert_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["drop", "exception", "cancel", "success"])
async def test_file_fallback_waits_for_ordinary_owner_before_suppression(monkeypatch, tmp_path, outcome):
    from gateway import slack_intake_ledger as ledger

    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    adapter = _real_offline_adapter()
    adapter._app.client.files_info = AsyncMock(return_value={"ok": True, "file": {
        "id": "F", "user": "U", "mimetype": "video/mp4",
        "shares": {"private": {"C": [{"ts": "123.000001"}]}},
    }})
    entered, release, fallback_ready = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def grace(_delay):
        fallback_ready.set()

    monkeypatch.setattr(asyncio, "sleep", grace)
    # Pause inside a real policy check, after dedup claimed the ordinary event.
    async def info(**_kwargs):
        entered.set()
        await release.wait()
        return {"user": {"is_bot": False, "profile": {"display_name": "Synthetic"}}}

    adapter._app.client.users_info = AsyncMock(side_effect=info)
    original = RuntimeError("synthetic ordinary failure") if outcome == "exception" else asyncio.CancelledError()
    calls = []

    async def dispatch(message):
        calls.append("ordinary" if message.text else "fallback")
        if message.text and outcome in {"exception", "cancel"}:
            raise original

    adapter.handle_message = AsyncMock(side_effect=dispatch)
    # The ordinary event is rejected as bot-authored for this control; the
    # fallback has a human sender, so no policy mutation is needed mid-race.
    event = {"type": "message", "team": "T", "channel": "C", "channel_type": "mpim",
             "ts": "123.000001", "text": "<@UBOT> ordinary", "user": "U"}
    if outcome == "drop":
        # A second lookup after the paused first lookup is unnecessary: the
        # cached bot classification is per user. Give fallback a different user.
        adapter._app.client.files_info.return_value["file"]["user"] = "U-fallback"
        async def classified_info(**kwargs):
            if kwargs["user"] == "U":
                entered.set()
                await release.wait()
            return {"user": {"is_bot": kwargs["user"] == "U", "profile": {"display_name": "Synthetic"}}}
        adapter._app.client.users_info = AsyncMock(side_effect=classified_info)
    adapter.config.extra["require_mention"] = False
    owner = asyncio.create_task(_deliver_offline(adapter, event, "owner", team_id="T"))
    fallback = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        fallback = asyncio.create_task(adapter._handle_slack_file_shared({
            "team": "T", "channel_id": "C", "file_id": "F"}))
        await asyncio.wait_for(fallback_ready.wait(), 2)
        release.set()
        if outcome in {"exception", "cancel"}:
            with pytest.raises(type(original)) as caught:
                await owner
            assert caught.value is original
        else:
            result = await owner
            assert result.state == ("dropped" if outcome == "drop" else "accepted")
        await asyncio.wait_for(fallback, 2)
        assert calls == ({"drop": ["fallback"], "success": ["ordinary"]}.get(
            outcome, ["ordinary", "fallback"]))
    finally:
        release.set()
        for task in (owner, fallback):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (owner, fallback) if t is not None), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("other_team, other_channel", [("T-other", "C"), ("T", "C-other")])
async def test_inflight_file_fallback_does_not_block_unrelated_message(
    monkeypatch, tmp_path, other_team, other_channel
):
    from gateway import slack_intake_ledger as ledger

    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    adapter = _real_offline_adapter()
    adapter.config.extra["require_mention"] = False
    adapter._app.client.files_info = AsyncMock(return_value={"ok": True, "file": {
        "id": "F", "user": "U-fallback", "mimetype": "video/mp4",
        "shares": {"private": {"C": [{"ts": "123.000001"}]}},
    }})
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    entered, release = asyncio.Event(), asyncio.Event()

    async def info(**kwargs):
        if kwargs["user"] == "U-fallback":
            entered.set()
            await release.wait()
        return {"user": {"is_bot": False, "profile": {"display_name": "Synthetic"}}}

    adapter._app.client.users_info = AsyncMock(side_effect=info)
    fallback = asyncio.create_task(adapter._handle_slack_file_shared({
        "team": "T", "channel_id": "C", "file_id": "F"}))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        event = {"type": "message", "team": other_team, "channel": other_channel,
                 "channel_type": "mpim", "ts": "123.000001", "text": "<@UBOT> ordinary", "user": "U"}
        result = await asyncio.wait_for(_deliver_offline(adapter, event, "unrelated"), 2)
        assert result.state == "accepted"
        assert not fallback.done()
        release.set()
        await asyncio.wait_for(fallback, 2)
        assert adapter.handle_message.await_count == 2
        assert ("T", "C", "123.000001") in adapter._processed_message_ts
        assert (other_team, other_channel, "123.000001") in adapter._processed_message_ts
    finally:
        release.set()
        if not fallback.done():
            fallback.cancel()
        await asyncio.gather(fallback, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_file_fallback_waiter_preserves_ordinary_owner(monkeypatch, tmp_path):
    from gateway import slack_intake_ledger as ledger

    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    adapter = _real_offline_adapter()
    adapter._app.client.files_info = AsyncMock(return_value={"ok": True, "file": {
        "id": "F", "user": "U", "mimetype": "video/mp4",
        "shares": {"private": {"C": [{"ts": "123.000001"}]}},
    }})
    entered, release, attempted = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def dispatch(_message):
        entered.set()
        await release.wait()

    async def grace(_delay):
        attempted.set()

    adapter.handle_message = AsyncMock(side_effect=dispatch)
    monkeypatch.setattr(asyncio, "sleep", grace)
    event = {"type": "message", "team": "T", "channel": "C", "channel_type": "mpim",
             "ts": "123.000001", "text": "<@UBOT> ordinary", "user": "U"}
    owner = asyncio.create_task(_deliver_offline(adapter, event, "owner"))
    fallback = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        fallback = asyncio.create_task(adapter._handle_slack_file_shared({
            "team": "T", "channel_id": "C", "file_id": "F"}))
        await asyncio.wait_for(attempted.wait(), 2)
        assert not fallback.done()
        fallback.cancel()
        with pytest.raises(asyncio.CancelledError):
            await fallback
        assert not owner.done()
        release.set()
        result = await asyncio.wait_for(owner, 2)
        assert result.state == "accepted"
        twin = await _deliver_offline(adapter, event, "twin")
        assert twin.state == "dropped" and twin.related_receipt_id == ledger._receipt_id("T", "owner")
        adapter.handle_message.assert_awaited_once()
    finally:
        release.set()
        for task in (owner, fallback):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (owner, fallback) if t is not None), return_exceptions=True)


@pytest.mark.asyncio
async def test_edited_message_keeps_envelope_identity_through_normalization(monkeypatch, tmp_path):
    from gateway import slack_intake_ledger as ledger

    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    adapter = _real_offline_adapter()
    event = {"type": "message", "subtype": "message_changed", "team": "T", "channel": "C",
             "channel_type": "mpim", "message": {"team": "T-nested", "user": "U",
             "ts": "123.000001", "edited": {"ts": "124.000001"}, "text": "<@UBOT> edit"}}
    result = await _deliver_offline(adapter, event, "edit", team_id="T-envelope")
    assert result.state == "accepted"
    message = adapter.handle_message.await_args.args[0]
    assert message.source.scope_id == "T"
    assert ("T", "C", "123.000001") in adapter._processed_message_ts
    twin = await _deliver_offline(adapter, event, "twin", team_id="T-envelope")
    assert twin.state == "dropped" and twin.related_receipt_id == ledger._receipt_id("T", "edit")
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_real_handler_edit_retry_and_twin_keep_one_dispatch_and_terminal_acceptance(monkeypatch, tmp_path):
    from gateway import slack_intake_ledger as ledger
    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    adapter = _real_offline_adapter()
    event = {"type": "message", "team": "T", "channel": "C", "channel_type": "mpim",
             "ts": "123.000001", "text": "synthetic request", "user": "U"}

    async def deliver(event_id, raw_event, envelope_id):
        body = {"team_id": "T", "event_id": event_id, "event": raw_event}
        await adapter._intake_observer.observe_socket_message({
            "type": "events_api", "envelope_id": envelope_id, "payload": body,
        }, "synthetic-raw")
        return await adapter._handle_observed_slack_message(raw_event, body)

    first = await deliver("original", event, "first-envelope")
    assert first.reason == "missing_mention"
    adapter.handle_message.assert_not_awaited()
    edited = {"type": "message", "subtype": "message_changed", "team": "T", "channel": "C",
              "channel_type": "mpim", "ts": "123.000001", "message": {
                  "text": "<@UBOT> synthetic request", "user": "U", "ts": "123.000001",
                  "edited": {"ts": "124.000001", "user": "U"},
              }}
    accepted = await deliver("edited", edited, "second-envelope")
    retried = await deliver("edited", edited, "retry-envelope")
    twin = await deliver("edited-twin", edited, "twin-envelope")
    assert accepted.state == retried.state == "accepted"
    assert twin.state == "dropped" and twin.reason == "duplicate_ts"
    assert twin.related_receipt_id == ledger._receipt_id("T", "edited")
    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].text == "synthetic request"
    assert len(adapter._intake_message_locks) == 0
    rows = ledger.read_receipts()
    assert sorted(row["receive_count"] for row in rows) == [1, 1, 2]
    assert sum(row["terminal_state"] == "accepted" for row in rows) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "cancel"])
async def test_real_failed_edit_does_not_suppress_valid_sibling(monkeypatch, tmp_path, failure):
    from gateway import slack_intake_ledger as ledger

    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    ledger._reset_initialization_for_tests()
    adapter = _real_offline_adapter()
    event = {"type": "message", "subtype": "message_changed", "team": "T", "channel": "C",
             "channel_type": "mpim", "ts": "123.000001", "message": {
                 "text": "<@UBOT> synthetic request", "user": "U", "ts": "123.000001",
                 "edited": {"ts": "124.000001", "user": "U"}}}

    async def deliver(event_id):
        body = {"team_id": "T", "event_id": event_id, "event": event}
        await adapter._intake_observer.observe_socket_message({
            "type": "events_api", "envelope_id": event_id, "payload": body}, "{}")
        return await adapter._handle_observed_slack_message(event, body)

    original = RuntimeError("synthetic failure") if failure == "exception" else asyncio.CancelledError()
    adapter.handle_message = AsyncMock(side_effect=original)
    with pytest.raises(type(original)) as caught:
        await deliver("first")
    assert caught.value is original
    adapter.handle_message = AsyncMock()
    result = await deliver("sibling")
    assert adapter.handle_message.await_count == 1, (result, ledger.read_receipts())
    assert result.state == "accepted"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "exception", "cancel"])
async def test_competing_edit_generations_wait_for_dispatch_outcome(monkeypatch, failure):
    adapter = _real_offline_adapter()
    # Context absence is a supported fail-open path. Keep the real handler;
    # only external clients and the downstream dispatcher are synthetic.
    monkeypatch.setattr(adapter, "_intake_observer", SimpleNamespace(context_for_event=lambda *_a, **_k: None))
    entered, release, competing = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def dispatch(message):
        calls.append(message.text)
        if message.text == "first":
            entered.set()
            await release.wait()
            if failure == "exception":
                raise RuntimeError("synthetic owner failure")
            if failure == "cancel":
                raise asyncio.CancelledError()

    monkeypatch.setattr(adapter, "handle_message", dispatch)

    async def deliver(label, edit_ts):
        event = {"type": "message", "subtype": "message_changed", "team": "T", "channel": "C",
                 "channel_type": "mpim", "message": {
                     "text": f"<@UBOT> {label}", "user": "U", "ts": "123.000001",
                     "edited": {"ts": edit_ts, "user": "U"}}}
        if label == "second":
            competing.set()
        return await adapter._handle_observed_slack_message(event, {"team_id": "T", "event_id": label})

    first = asyncio.create_task(deliver("first", "124.000001"))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        second = asyncio.create_task(deliver("second", "125.000001"))
        await asyncio.wait_for(competing.wait(), 2)
        assert calls == ["first"], "a competing edit entered dispatch before owner settlement"
        release.set()
        results = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 2)
        assert calls == (["first"] if failure is None else ["first", "second"])
        assert results[1] == ("duplicate_event" if failure is None else None)
        if failure:
            assert isinstance(results[0], RuntimeError if failure == "exception" else asyncio.CancelledError)
    finally:
        release.set()
        for task in (first, second):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (first, second) if t is not None), return_exceptions=True)

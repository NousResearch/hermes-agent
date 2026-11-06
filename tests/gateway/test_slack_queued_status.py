"""Three distinct Slack reactions across the real durable native FIFO."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_results import execution_result
from plugins.platforms.slack.adapter import SlackAdapter
from tests.waiting import async_wait_until


class RecordingSlack(SlackAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"))
        self.reactions = []
        self.sent = []
        self.reject_second = False
        self._running = True

    async def _react(self, channel, timestamp, emoji, team_id, *, remove):
        self.reactions.append((timestamp, "remove" if remove else "add", emoji))
        return True

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        if self.reject_second and "reply two" in content:
            return SendResult(success=False, error="invalid_auth")
        return SendResult(success=True, message_id=f"sent-{len(self.sent)}")

    async def send_typing(self, chat_id, metadata=None):
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("second_fails,delivery_fails", [(False, False), (True, False), (False, True)])
async def test_queued_slack_message_remains_read_until_its_own_execution(monkeypatch, second_fails, delivery_fails):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "fixture-user")
    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingSlack()
    adapter.reject_second = delivery_fails
    adapter.gateway_runner = runner
    runner.adapters = {Platform.SLACK: adapter}
    await initialize_session_authority(runner, profile_id="default", instance_id="reactions")
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id="D_CHAT", chat_type="dm", user_id="fixture-user")
    first_started, first_release = asyncio.Event(), asyncio.Event()
    second_started, second_release = asyncio.Event(), asyncio.Event()

    async def turn(event, source, quick_key, run_generation):
        started, release = (first_started, first_release) if event.text == "one" else (second_started, second_release)
        started.set()
        await release.wait()
        execution_result.get()["result"] = {
            "final_response": f"reply {event.text}", "messages": [],
            "failed": second_fails and event.text == "two",
        }
        return f"reply {event.text}"

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    first = MessageEvent(text="one", source=source, message_id="1234567890.000001")
    second = MessageEvent(text="two", source=source, message_id="1234567890.000002")
    try:
        for event in (first, second):
            adapter._track_reacting_message("", event.message_id)
            await adapter._react(source.chat_id, event.message_id, "eyes", "", remove=False)
        await adapter.handle_message(first)
        await asyncio.wait_for(first_started.wait(), 5)
        await asyncio.wait_for(adapter.handle_message(second), 5)
        assert not second_started.is_set()
        assert (second.message_id, "add", "hourglass_flowing_sand") not in adapter.reactions
        first_release.set()
        await asyncio.wait_for(second_started.wait(), 5)
        assert (second.message_id, "add", "hourglass_flowing_sand") in adapter.reactions
        assert (second.message_id, "add", "white_check_mark") not in adapter.reactions
        second_release.set()
        await async_wait_until(lambda: "reply two" in adapter.sent and not adapter._background_tasks, timeout=10)
        for event in (first, second):
            operations = [(kind, emoji) for timestamp, kind, emoji in adapter.reactions if timestamp == event.message_id]
            assert operations[0] == ("add", "eyes")
            assert ("add", "hourglass_flowing_sand") in operations
            final = "x" if (second_fails or delivery_fails) and event is second else "white_check_mark"
            assert operations[-1] == ("add", final)
    finally:
        first_release.set()
        second_release.set()
        await asyncio.gather(*adapter._background_tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "shared-busy"])
@pytest.mark.parametrize("strict", [True, False], ids=["slack", "legacy-policy"])
async def test_mixed_text_attachment_delivery_never_checks_slack(monkeypatch, busy, strict):
    from types import SimpleNamespace
    from gateway.session_ingress import dispatch_shared_busy

    adapter = RecordingSlack()
    # A non-opted-in adapter must retain the existing any-success policy.
    if not strict:
        adapter.processing_requires_all_deliveries = False
    source = adapter.build_source(chat_id="D_CHAT", chat_type="dm", user_id="fixture-user")
    event = MessageEvent(text="attachment please", source=source, message_id="1234567890.000020")
    adapter._track_reacting_message("", event.message_id)
    await adapter._react(source.chat_id, event.message_id, "eyes", "", remove=False)
    adapter._message_handler = AsyncMock(return_value="reply with attachment")
    adapter._extract_response_content = AsyncMock(return_value=SimpleNamespace(
        text_content="reply with attachment", media_files=[], images=[], local_files=[]))
    adapter._wants_auto_tts = lambda *args: False

    async def fail_attachment(_event, _extracted, _metadata, *, anything_sent, record_delivery):
        assert anything_sent
        record_delivery(SendResult(success=False, error="attachment rejected"))

    adapter._deliver_attachments = fail_attachment
    key = adapter._event_session_key(event)
    if busy:
        await adapter.on_processing_start(event)
        await dispatch_shared_busy(adapter, event, key)
        await asyncio.gather(*adapter._background_tasks)
    else:
        await adapter._process_message_background(event, key)
    assert adapter.sent == ["reply with attachment"]
    final = "x" if strict else "white_check_mark"
    assert adapter.reactions[-1] == (event.message_id, "add", final)
    if strict:
        assert (event.message_id, "add", "white_check_mark") not in adapter.reactions


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True], ids=["idle-pause", "busy-pause"])
@pytest.mark.parametrize("failed,rejected,reply_mode", [
    (False, False, "text"), (True, False, "text"), (False, True, "text"),
    (False, False, "empty"), (False, False, "none"), (True, False, "raised"),
])
async def test_paused_slack_receipt_survives_resumed_no_waiter_delivery(monkeypatch, busy, failed, rejected, reply_mode):
    from gateway import session_envelope, session_settlement_recovery
    from gateway.session_contract import SessionRef
    from gateway.session_ingress import dispatch_shared_busy
    from hermes_state_runtime import RuntimeStoreError, list_session_admissions

    monkeypatch.setenv("SLACK_ALLOWED_USERS", "fixture-user")
    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingSlack()
    adapter.gateway_runner = runner
    runner.adapters = {Platform.SLACK: adapter}
    await initialize_session_authority(runner, profile_id="default", instance_id="pause-reactions")
    runner._wire_adapter_handlers(adapter)
    authority = runner.session_authority
    source = adapter.build_source(chat_id="D_CHAT", chat_type="dm", user_id="fixture-user")
    event = MessageEvent(text="paused", source=source, message_id="1234567890.000030")
    adapter._track_reacting_message("", event.message_id)
    await adapter._react(source.chat_id, event.message_id, "eyes", "", remove=False)
    paused = True
    started, finish, sending, delivered = (asyncio.Event() for _ in range(4))
    settling, settle_release = asyncio.Event(), asyncio.Event()
    original_check, original_send = session_envelope.check_native_route, adapter.send
    original_settle = session_settlement_recovery.settle_with_retry

    async def settle(*args, **kwargs):
        settling.set()
        await settle_release.wait()
        return await original_settle(*args, **kwargs)

    async def preflight(*args, **kwargs):
        await original_check(*args, **kwargs)
        if paused:
            await async_wait_until(lambda: bool(authority.native_waiters), timeout=5)
            raise RuntimeStoreError("permission_denied")

    async def turn(event, source, quick_key, run_generation):
        started.set()
        await finish.wait()
        if reply_mode == "raised":
            raise RuntimeError("owned empty-response lifecycle control")
        response = {"text": "reply paused", "empty": "", "none": None}[reply_mode]
        execution_result.get()["result"] = {
            "final_response": response or "", "messages": [], "failed": failed,
        }
        return response

    async def send(chat_id, content, reply_to=None, metadata=None):
        if content == "reply paused":
            sending.set()
            await delivered.wait()
            if rejected:
                adapter.sent.append(content)
                return SendResult(success=False, error="invalid_auth")
        return await original_send(chat_id, content, reply_to, metadata)

    monkeypatch.setattr(session_envelope, "check_native_route", preflight)
    monkeypatch.setattr(session_settlement_recovery, "settle_with_retry", settle)
    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    # _send_final_text remains real; this is the offline transport/retry boundary.
    monkeypatch.setattr(adapter, "_send_with_retry", send)
    try:
        if busy:
            await dispatch_shared_busy(adapter, event, adapter._event_session_key(event))
        else:
            await adapter.handle_message(event)
        await async_wait_until(lambda: bool(adapter.sent) and not adapter._background_tasks, timeout=10)
        sid = next(iter(authority.sessions))
        rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
        assert len(rows) == 1 and rows[0]["status"] == "queued"
        assert not authority.native_waiters and not authority.waiters
        assert event.message_id in adapter._reacting_message_ids, "paused queued receipt was discarded"
        assert adapter.reactions == [(event.message_id, "add", "eyes")]
        assert not started.is_set()
        paused = False
        authority._schedule(SessionRef("default", sid))
        await asyncio.wait_for(started.wait(), 5)
        assert (event.message_id, "add", "hourglass_flowing_sand") in adapter.reactions
        assert (event.message_id, "add", "white_check_mark") not in adapter.reactions
        finish.set()
        await asyncio.wait_for(settling.wait(), 5)
        row = list_session_admissions(authority.db, session_id=sid, pending_only=False)[0]
        assert row["status"] == "started"
        assert event.message_id in adapter._reacting_message_ids
        assert (event.message_id, "remove", "hourglass_flowing_sand") not in adapter.reactions
        assert (event.message_id, "add", "white_check_mark") not in adapter.reactions
        settle_release.set()
        if reply_mode == "text":
            await asyncio.wait_for(sending.wait(), 5)
            assert (event.message_id, "add", "white_check_mark") not in adapter.reactions
            delivered.set()
        await asyncio.wait_for(authority.sessions[sid].task, 5)
        row = list_session_admissions(authority.db, session_id=sid, pending_only=False)[0]
        assert row["status"] == "terminal"
        assert row["outcome"] == ("failed" if failed else "completed")
        if reply_mode == "text":
            assert adapter.sent.count("reply paused") == 1
            assert adapter.reactions[-1] == (event.message_id, "add", "x" if failed or rejected else "white_check_mark")
        else:
            assert (event.message_id, "remove", "hourglass_flowing_sand") in adapter.reactions
            assert (event.message_id, "add", "white_check_mark") not in adapter.reactions
            assert ((event.message_id, "add", "x") in adapter.reactions) is failed
        assert event.message_id not in adapter._reacting_message_ids
        assert not authority.pending_deliveries and not authority.held_turn_markers
    finally:
        finish.set()
        delivered.set()
        settle_release.set()
        await asyncio.gather(*adapter._background_tasks, return_exceptions=True)
        await asyncio.gather(*(live.task for live in authority.sessions.values() if live.task), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "busy"])
@pytest.mark.parametrize("reply", ["late reply", "", None], ids=["text", "empty", "none"])
@pytest.mark.parametrize("strict", [True, False], ids=["slack", "legacy"])
async def test_late_live_slack_waiter_owns_its_only_completion(monkeypatch, busy, reply, strict):
    from gateway.session_ingress import dispatch_shared_busy
    from hermes_state_runtime import list_session_admissions
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "fixture-user")
    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingSlack()
    adapter.gateway_runner = runner
    runner.adapters = {Platform.SLACK: adapter}
    adapter.processing_requires_all_deliveries = strict
    await initialize_session_authority(runner, profile_id="default", instance_id="late-waiter")
    runner._wire_adapter_handlers(adapter)
    authority = runner.session_authority
    source = adapter.build_source(chat_id="D_CHAT", chat_type="dm", user_id="fixture-user")
    event = MessageEvent(text="late waiter", source=source, message_id="1234567890.000031")
    adapter._track_reacting_message("", event.message_id)
    started, finish = asyncio.Event(), asyncio.Event()
    original_admit, original_complete = authority.admit_native, adapter.on_processing_complete
    completions = []

    async def admit(received):
        receipt = await original_admit(received)
        await started.wait()  # the claim snapshots no waiter before ingress can register one
        return receipt

    async def turn(event, source, quick_key, run_generation):
        started.set()
        await finish.wait()
        execution_result.get()["result"] = {"final_response": reply or "", "messages": []}
        return reply

    async def complete(received, outcome):
        completions.append(received)
        await original_complete(received, outcome)

    monkeypatch.setattr(authority, "admit_native", admit)
    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    monkeypatch.setattr(adapter, "on_processing_complete", complete)
    try:
        if busy:
            await dispatch_shared_busy(adapter, event, adapter._event_session_key(event))
        else:
            await adapter.handle_message(event)
        await asyncio.wait_for(started.wait(), 5)
        await async_wait_until(lambda: bool(authority.native_waiters), timeout=5)
        sid = next(iter(authority.sessions))
        assert (event.message_id, "add", "hourglass_flowing_sand") in adapter.reactions
        assert not completions and not adapter.sent
        finish.set()
        await async_wait_until(lambda: not adapter._background_tasks, timeout=10)
        await asyncio.wait_for(authority.sessions[sid].task, 5)
        row = list_session_admissions(authority.db, session_id=sid, pending_only=False)[0]
        assert (row["status"], row["outcome"]) == ("terminal", "completed")
        assert adapter.sent == ([reply] if reply else [])
        assert completions == [event], "drain completed a live waiter's lifecycle"
        assert (event.message_id, "remove", "hourglass_flowing_sand") in adapter.reactions
        assert event.message_id not in adapter._reacting_message_ids
        if reply or not strict:
            assert adapter.reactions[-1] == (event.message_id, "add", "white_check_mark")
        else:
            assert (event.message_id, "add", "white_check_mark") not in adapter.reactions
        assert not authority.pending_deliveries and not authority.native_waiters
    finally:
        finish.set()
        await asyncio.gather(*adapter._background_tasks, return_exceptions=True)
        await asyncio.gather(*(live.task for live in authority.sessions.values() if live.task), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("captured", [True, False], ids=["captured", "discarded"])
async def test_empty_no_waiter_slack_completion_waits_for_unknown_resolution(monkeypatch, captured):
    from gateway import session_results, session_settlement_recovery
    from gateway.session_contract import Principal, SessionRef
    from hermes_state_runtime import get_session_admission

    monkeypatch.setenv("SLACK_ALLOWED_USERS", "fixture-user")
    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingSlack()
    adapter.gateway_runner = runner
    runner.adapters = {Platform.SLACK: adapter}
    await initialize_session_authority(runner, profile_id="default", instance_id="unknown-empty")
    runner._wire_adapter_handlers(adapter)
    authority = runner.session_authority
    source = adapter.build_source(chat_id="D_CHAT", chat_type="dm", user_id="fixture-user")
    event = MessageEvent(text="unknown empty", source=source, message_id="1234567890.000032")
    adapter._track_reacting_message("", event.message_id)
    original_finish = session_results.finish_result

    async def turn(event, source, quick_key, run_generation):
        execution_result.get()["result"] = {"final_response": "", "messages": []}

    def unavailable(*args, **kwargs):
        raise OSError("owned settlement failure")

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    monkeypatch.setattr(session_results, "finish_result", unavailable)
    monkeypatch.setattr(session_settlement_recovery, "_SETTLE_RETRY_DELAYS_S", ())
    try:
        receipt = await authority.admit_native(event)  # no platform delivery waiter
        sid = receipt.ref.session_id
        await asyncio.wait_for(authority.sessions[sid].task, 5)
        row = get_session_admission(authority.db, admission_id=receipt.admission_id)
        assert row["status"] == "unknown"
        assert event.message_id in adapter._reacting_message_ids
        assert (event.message_id, "add", "hourglass_flowing_sand") in adapter.reactions
        assert (event.message_id, "remove", "hourglass_flowing_sand") not in adapter.reactions
        assert (event.message_id, "add", "white_check_mark") not in adapter.reactions
        assert receipt.admission_id in authority.unsettled_deliveries
        assert not authority.pending_deliveries and not adapter.sent
        monkeypatch.setattr(session_results, "finish_result", original_finish)
        if not captured:
            # Explicitly absent captured result selects the actual owed=False caller.
            authority.pending_results.pop(receipt.admission_id)
        removing, cleanup_release = asyncio.Event(), asyncio.Event()
        original_react = adapter._react

        async def react(channel, timestamp, emoji, team_id, *, remove):
            if remove and emoji == "hourglass_flowing_sand":
                removing.set()
                await cleanup_release.wait()
            return await original_react(channel, timestamp, emoji, team_id, remove=remove)

        monkeypatch.setattr(adapter, "_react", react)
        actor = Principal("operator", "default", frozenset({"session:submit", "session:control"}), "fixture")
        ref = SessionRef("default", sid)
        resolution = asyncio.create_task(authority.resolve_unknown(actor, ref, receipt.admission_id, row["generation"]))
        entered = asyncio.create_task(removing.wait())
        try:
            await asyncio.wait_for(asyncio.wait({resolution, entered}, return_when=asyncio.FIRST_COMPLETED), 5)
            resolved = get_session_admission(authority.db, admission_id=receipt.admission_id)
            assert resolved["status"] == "terminal"
            assert resolved["outcome"] == ("completed" if captured else "interrupted")
            assert removing.is_set(), "Discard dropped the retained Slack completion owner"
        finally:
            cleanup_release.set()
            entered.cancel()
            await asyncio.gather(resolution, entered, return_exceptions=True)
        await resolution
        assert event.message_id not in adapter._reacting_message_ids
        assert (event.message_id, "remove", "hourglass_flowing_sand") in adapter.reactions
        assert (event.message_id, "add", "white_check_mark") not in adapter.reactions
        assert not authority.pending_deliveries and not authority.unsettled_deliveries
        assert not authority.held_turn_markers and not adapter.sent
        reactions = list(adapter.reactions)
        await authority.resolve_unknown(actor, ref, receipt.admission_id, row["generation"])
        assert adapter.reactions == reactions
    finally:
        await asyncio.gather(*(live.task for live in authority.sessions.values() if live.task), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "busy"])
async def test_fast_unknown_slack_outer_completion_retains_resolution(monkeypatch, busy):
    from gateway.session_ingress import dispatch_shared_busy
    from gateway import session_results, session_settlement_recovery
    from gateway.session_contract import Principal, SessionRef
    from hermes_state_runtime import get_session_admission

    monkeypatch.setenv("SLACK_ALLOWED_USERS", "fixture-user")
    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingSlack()
    adapter.gateway_runner = runner
    runner.adapters = {Platform.SLACK: adapter}
    await initialize_session_authority(runner, profile_id="default", instance_id="unknown-empty")
    runner._wire_adapter_handlers(adapter)
    authority = runner.session_authority
    source = adapter.build_source(chat_id="D_CHAT", chat_type="dm", user_id="fixture-user")
    event = MessageEvent(text="unknown empty", source=source, message_id="1234567890.000032")
    adapter._track_reacting_message("", event.message_id)
    original_finish = session_results.finish_result
    original_admit = authority.admit_native
    receipts = []

    async def admit(received):
        receipt = await original_admit(received)
        receipts.append(receipt)
        # Settle unknown before the actual ingress caller registers its waiter.
        await authority.sessions[receipt.ref.session_id].task
        return receipt

    monkeypatch.setattr(authority, "admit_native", admit)

    async def turn(event, source, quick_key, run_generation):
        execution_result.get()["result"] = {"final_response": "", "messages": []}

    def unavailable(*args, **kwargs):
        raise OSError("owned settlement failure")

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    monkeypatch.setattr(session_results, "finish_result", unavailable)
    monkeypatch.setattr(session_settlement_recovery, "_SETTLE_RETRY_DELAYS_S", ())
    try:
        if busy:
            await dispatch_shared_busy(adapter, event, adapter._event_session_key(event))
        else:
            await adapter.handle_message(event)
        await async_wait_until(lambda: bool(adapter.sent) and not adapter._background_tasks, timeout=10)
        receipt = receipts[0]
        sid = receipt.ref.session_id
        notices = list(adapter.sent)
        assert len(notices) == 1 and "paused" in notices[0].lower()
        row = get_session_admission(authority.db, admission_id=receipt.admission_id)
        assert row["status"] == "unknown"
        assert event.message_id in adapter._reacting_message_ids
        assert (event.message_id, "add", "hourglass_flowing_sand") in adapter.reactions
        assert (event.message_id, "remove", "hourglass_flowing_sand") not in adapter.reactions
        assert (event.message_id, "add", "white_check_mark") not in adapter.reactions
        assert receipt.admission_id in authority.unsettled_deliveries
        assert not authority.pending_deliveries and adapter.sent == notices
        monkeypatch.setattr(session_results, "finish_result", original_finish)
        actor = Principal("operator", "default", frozenset({"session:submit", "session:control"}), "fixture")
        ref = SessionRef("default", sid)
        await authority.resolve_unknown(actor, ref, receipt.admission_id, row["generation"])
        resolved = get_session_admission(authority.db, admission_id=receipt.admission_id)
        assert (resolved["status"], resolved["outcome"]) == ("terminal", "completed")
        assert event.message_id not in adapter._reacting_message_ids
        assert (event.message_id, "remove", "hourglass_flowing_sand") in adapter.reactions
        assert (event.message_id, "add", "white_check_mark") not in adapter.reactions
        assert not authority.pending_deliveries and not authority.unsettled_deliveries
        assert not authority.held_turn_markers and adapter.sent == notices
        reactions = list(adapter.reactions)
        await authority.resolve_unknown(actor, ref, receipt.admission_id, row["generation"])
        assert adapter.reactions == reactions
    finally:
        await asyncio.gather(*adapter._background_tasks, return_exceptions=True)
        await asyncio.gather(*(live.task for live in authority.sessions.values() if live.task), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "busy"])
async def test_fast_terminal_slack_drain_keeps_completion_until_send(monkeypatch, busy):
    from gateway.platforms.event import ProcessingOutcome
    from gateway.session_ingress import dispatch_shared_busy
    from hermes_state_runtime import get_session_admission

    monkeypatch.setenv("SLACK_ALLOWED_USERS", "fixture-user")
    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingSlack()
    adapter.gateway_runner = runner
    runner.adapters = {Platform.SLACK: adapter}
    await initialize_session_authority(runner, profile_id="default", instance_id="fast-terminal")
    runner._wire_adapter_handlers(adapter)
    authority = runner.session_authority
    source = adapter.build_source(chat_id="D_CHAT", chat_type="dm", user_id="fixture-user")
    event = MessageEvent(text="fast terminal", source=source, message_id="1234567890.000035")
    adapter._track_reacting_message("", event.message_id)
    await adapter._react(source.chat_id, event.message_id, "eyes", "", remove=False)
    sending, release_send, confirmed = (asyncio.Event() for _ in range(3))
    original_admit, original_send = authority.admit_native, adapter.send
    original_complete = adapter.on_processing_complete
    receipts, completions = [], []

    async def admit(received):
        receipt = await original_admit(received)
        receipts.append(receipt)
        # Keep the real queued receipt until the no-waiter drain commits and starts its send.
        await sending.wait()
        return receipt

    async def turn(event, source, quick_key, run_generation):
        execution_result.get()["result"] = {"final_response": "fast reply", "messages": []}
        return "fast reply"

    async def send(chat_id, content, reply_to=None, metadata=None):
        sending.set()
        await release_send.wait()
        result = await original_send(chat_id, content, reply_to, metadata)
        confirmed.set()
        return result

    async def complete(received, outcome):
        tracked = event.message_id in adapter._reacting_message_ids
        await original_complete(received, outcome)
        if tracked and event.message_id not in adapter._reacting_message_ids:
            completions.append((received, outcome, confirmed.is_set()))

    monkeypatch.setattr(authority, "admit_native", admit)
    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    monkeypatch.setattr(adapter, "_send_with_retry", send)
    monkeypatch.setattr(adapter, "on_processing_complete", complete)
    try:
        if busy:
            await asyncio.wait_for(dispatch_shared_busy(adapter, event, adapter._event_session_key(event)), 5)
        else:
            await adapter.handle_message(event)
        await asyncio.wait_for(sending.wait(), 5)
        await async_wait_until(lambda: not adapter._background_tasks, timeout=10)
        receipt = receipts[0]
        row = get_session_admission(authority.db, admission_id=receipt.admission_id)
        assert receipt.status == "queued" and (row["status"], row["outcome"]) == ("terminal", "completed")
        assert receipt.admission_id in authority.pending_deliveries
        assert not authority.sessions[receipt.ref.session_id].task.done()
        assert not adapter.sent and not confirmed.is_set()
        assert not completions, "outer lifecycle completed while the drain still owned an unconfirmed send"
        assert event.message_id in adapter._reacting_message_ids
        assert (event.message_id, "add", "hourglass_flowing_sand") in adapter.reactions
        assert (event.message_id, "remove", "hourglass_flowing_sand") not in adapter.reactions
        assert (event.message_id, "add", "white_check_mark") not in adapter.reactions
        assert not authority.native_waiters and not authority.waiters and not authority.marker_adopters
        release_send.set()
        await asyncio.wait_for(authority.sessions[receipt.ref.session_id].task, 5)
        assert adapter.sent == ["fast reply"] and confirmed.is_set()
        assert len(completions) == 1
        assert completions[0][0] is not event and completions[0][0].message_id == event.message_id
        assert completions[0][1:] == (ProcessingOutcome.SUCCESS, True)
        assert adapter.reactions.count((event.message_id, "remove", "hourglass_flowing_sand")) == 1
        assert adapter.reactions.count((event.message_id, "add", "white_check_mark")) == 1
        assert event.message_id not in adapter._reacting_message_ids
        assert not authority.pending_deliveries and not authority.held_turn_markers
        assert not authority.native_completion
    finally:
        release_send.set()
        await asyncio.gather(*adapter._background_tasks, return_exceptions=True)
        await asyncio.gather(*(live.task for live in authority.sessions.values() if live.task), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "busy"])
async def test_terminal_receipt_slack_redelivery_keeps_drain_completion_until_send(monkeypatch, busy):
    from dataclasses import replace
    from gateway.platforms.event import ProcessingOutcome
    from gateway.session_ingress import dispatch_shared_busy
    from hermes_state_runtime import get_session_admission, list_session_admissions

    monkeypatch.setenv("SLACK_ALLOWED_USERS", "fixture-user")
    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingSlack()
    adapter.gateway_runner = runner
    runner.adapters = {Platform.SLACK: adapter}
    await initialize_session_authority(runner, profile_id="default", instance_id="terminal-receipt")
    runner._wire_adapter_handlers(adapter)
    authority = runner.session_authority
    source = adapter.build_source(chat_id="D_CHAT", chat_type="dm", user_id="fixture-user")
    event = MessageEvent(text="terminal receipt", source=source, message_id="1234567890.000037")
    adapter._track_reacting_message("", event.message_id)
    await adapter._react(source.chat_id, event.message_id, "eyes", "", remove=False)
    sending, release_send, confirmed = (asyncio.Event() for _ in range(3))
    original_admit, original_send = authority.admit_native, adapter.send
    original_complete = adapter.on_processing_complete
    receipts, completions, turns = [], [], []

    async def admit(received):
        receipt = await original_admit(received)
        receipts.append(receipt)
        # Let the real no-waiter drain commit terminal and block at confirmed-send first.
        await sending.wait()
        return receipt

    async def turn(received, source, quick_key, run_generation):
        turns.append(received.message_id)
        execution_result.get()["result"] = {"final_response": "receipt reply", "messages": []}
        return "receipt reply"

    async def send(chat_id, content, reply_to=None, metadata=None):
        sending.set()
        await release_send.wait()
        result = await original_send(chat_id, content, reply_to, metadata)
        confirmed.set()
        return result

    async def complete(received, outcome):
        tracked = event.message_id in adapter._reacting_message_ids
        await original_complete(received, outcome)
        if tracked and event.message_id not in adapter._reacting_message_ids:
            completions.append((received, outcome, confirmed.is_set()))

    async def receive(received):
        if busy:
            await asyncio.wait_for(dispatch_shared_busy(adapter, received, adapter._event_session_key(received)), 5)
        else:
            await adapter.handle_message(received)
        await async_wait_until(lambda: not adapter._background_tasks, timeout=10)

    monkeypatch.setattr(authority, "admit_native", admit)
    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    monkeypatch.setattr(adapter, "_send_with_retry", send)
    monkeypatch.setattr(adapter, "on_processing_complete", complete)
    try:
        await receive(event)
        receipt = receipts[0]
        row = get_session_admission(authority.db, admission_id=receipt.admission_id)
        assert receipt.status == "queued" and (row["status"], row["outcome"]) == ("terminal", "completed")
        assert receipt.admission_id in authority.pending_deliveries
        assert event.message_id in adapter._reacting_message_ids and not completions
        redelivery = replace(event, source=replace(source))
        await receive(redelivery)
        assert [r.status for r in receipts] == ["queued", "terminal"]
        assert receipts[1].admission_id == receipt.admission_id
        assert not authority.sessions[receipt.ref.session_id].task.done()
        assert receipt.admission_id in authority.pending_deliveries
        assert not adapter.sent and not confirmed.is_set()
        assert not completions, "terminal redelivery completed while the drain still owned an unconfirmed send"
        assert event.message_id in adapter._reacting_message_ids
        assert getattr(redelivery, "_gateway_processing_deferred", False)
        assert (event.message_id, "add", "hourglass_flowing_sand") in adapter.reactions
        assert (event.message_id, "remove", "hourglass_flowing_sand") not in adapter.reactions
        assert (event.message_id, "add", "white_check_mark") not in adapter.reactions
        assert not authority.native_waiters and not authority.waiters and not authority.marker_adopters
        release_send.set()
        await asyncio.wait_for(authority.sessions[receipt.ref.session_id].task, 5)
        assert adapter.sent == ["receipt reply"] and confirmed.is_set()
        assert len(completions) == 1
        assert completions[0][0] is not event and completions[0][0] is not redelivery
        assert completions[0][0].message_id == event.message_id
        assert completions[0][1:] == (ProcessingOutcome.SUCCESS, True)
        finished_redelivery = replace(event, source=replace(source))
        await receive(finished_redelivery)
        assert [r.status for r in receipts] == ["queued", "terminal", "terminal"]
        assert all(r.admission_id == receipt.admission_id for r in receipts)
        assert not getattr(finished_redelivery, "_gateway_processing_deferred", False)
        assert len(list_session_admissions(authority.db, session_id=receipt.ref.session_id, pending_only=False)) == 1
        assert turns == [event.message_id] and adapter.sent == ["receipt reply"]
        assert len(completions) == 1
        assert adapter.reactions.count((event.message_id, "remove", "hourglass_flowing_sand")) == 1
        assert adapter.reactions.count((event.message_id, "add", "white_check_mark")) == 1
        assert event.message_id not in adapter._reacting_message_ids
        assert not authority.pending_deliveries and not authority.held_turn_markers and not authority.native_completion
        assert not authority.native_waiters and not authority.waiters and not authority.marker_adopters
    finally:
        release_send.set()
        await asyncio.gather(*adapter._background_tasks, return_exceptions=True)
        await asyncio.gather(*(live.task for live in authority.sessions.values() if live.task), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "busy"])
async def test_terminal_redelivery_during_live_waiter_send(monkeypatch, busy):
    from dataclasses import replace
    from gateway.session_ingress import dispatch_shared_busy
    from hermes_state_runtime import list_session_admissions

    monkeypatch.setenv("SLACK_ALLOWED_USERS", "fixture-user")
    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingSlack()
    adapter.gateway_runner = runner
    runner.adapters = {Platform.SLACK: adapter}
    await initialize_session_authority(runner, profile_id="default", instance_id="live-send")
    runner._wire_adapter_handlers(adapter)
    authority = runner.session_authority
    source = adapter.build_source(chat_id="D_CHAT", chat_type="dm", user_id="fixture-user")
    event = MessageEvent(text="live send", source=source, message_id="1234567890.000038")
    adapter._track_reacting_message("", event.message_id)
    registered, sending, release_send = (asyncio.Event() for _ in range(3))
    original_claim, original_send = authority._claim_with_retry, adapter.send
    original_admit = authority.admit_native
    receipts, turns = [], []

    async def claim(ref, live):
        await registered.wait()
        return await original_claim(ref, live)

    async def admit(received):
        receipt = await original_admit(received)
        receipts.append(receipt)
        return receipt

    async def turn(received, source, quick_key, run_generation):
        turns.append(received.message_id)
        execution_result.get()["result"] = {"final_response": "live reply", "messages": []}
        return "live reply"

    async def send(chat_id, content, reply_to=None, metadata=None):
        sending.set()
        await release_send.wait()
        return await original_send(chat_id, content, reply_to, metadata)

    async def duplicate():
        received = replace(event, source=replace(source))
        existing = set(adapter._background_tasks)
        await dispatch_shared_busy(adapter, received, adapter._event_session_key(received))
        await asyncio.gather(*(set(adapter._background_tasks) - existing))
        return received

    monkeypatch.setattr(authority, "_claim_with_retry", claim)
    monkeypatch.setattr(authority, "admit_native", admit)
    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    monkeypatch.setattr(adapter, "_send_with_retry", send)
    try:
        if busy:
            await dispatch_shared_busy(adapter, event, adapter._event_session_key(event))
        else:
            await adapter.handle_message(event)
        await async_wait_until(lambda: bool(authority.native_waiters), timeout=5)
        registered.set()
        await asyncio.wait_for(sending.wait(), 5)
        sid = receipts[0].ref.session_id
        row = list_session_admissions(authority.db, session_id=sid, pending_only=False)[0]
        assert (row["status"], row["outcome"]) == ("terminal", "completed")
        assert not authority.pending_deliveries, "this must be the live waiter, not a drain send"
        assert not adapter.sent
        redelivery = await duplicate()
        assert [r.status for r in receipts] == ["queued", "terminal"]
        assert event.message_id in adapter._reacting_message_ids, "duplicate erased live send completion ownership"
        assert getattr(redelivery, "_gateway_processing_deferred", False)
        assert (event.message_id, "remove", "hourglass_flowing_sand") not in adapter.reactions
        assert (event.message_id, "add", "white_check_mark") not in adapter.reactions
        release_send.set()
        await async_wait_until(lambda: not adapter._background_tasks, timeout=10)
        await asyncio.wait_for(authority.sessions[sid].task, 5)
        assert adapter.sent == ["live reply"]
        assert adapter.reactions.count((event.message_id, "add", "white_check_mark")) == 1
        finished = await duplicate()
        assert not getattr(finished, "_gateway_processing_deferred", False)
        assert all(r.admission_id == row["admission_id"] for r in receipts)
        assert turns == [event.message_id] and adapter.sent == ["live reply"]
        assert adapter.reactions.count((event.message_id, "remove", "hourglass_flowing_sand")) == 1
        assert adapter.reactions.count((event.message_id, "add", "white_check_mark")) == 1
        assert not authority.pending_deliveries and not authority.native_completion
        assert not authority.waiter_replies and not authority.native_waiters and not authority.waiters
        assert not getattr(authority, "live_delivery_owners", {})
    finally:
        registered.set()
        release_send.set()
        await asyncio.gather(*adapter._background_tasks, return_exceptions=True)
        await asyncio.gather(*(live.task for live in authority.sessions.values() if live.task), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "busy"])
@pytest.mark.parametrize("reply", ["finished reply", "", None], ids=["text", "empty", "none"])
async def test_fast_terminal_slack_finished_drain_does_not_replay(monkeypatch, busy, reply):
    from dataclasses import replace
    from gateway.session_ingress import dispatch_shared_busy
    from hermes_state_runtime import get_session_admission

    monkeypatch.setenv("SLACK_ALLOWED_USERS", "fixture-user")
    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingSlack()
    adapter.gateway_runner = runner
    runner.adapters = {Platform.SLACK: adapter}
    await initialize_session_authority(runner, profile_id="default", instance_id="finished-terminal")
    runner._wire_adapter_handlers(adapter)
    authority = runner.session_authority
    source = adapter.build_source(chat_id="D_CHAT", chat_type="dm", user_id="fixture-user")
    event = MessageEvent(text="finished terminal", source=source, message_id="1234567890.000036")
    adapter._track_reacting_message("", event.message_id)
    original_admit = authority.admit_native
    receipts, turns = [], []

    async def admit(received):
        receipt = await original_admit(received)
        receipts.append(receipt)
        # Even no-response cleanup finishes before the original queued receipt returns.
        await authority.sessions[receipt.ref.session_id].task
        return receipt

    async def turn(event, source, quick_key, run_generation):
        turns.append(event.message_id)
        execution_result.get()["result"] = {"final_response": reply or "", "messages": []}
        return reply

    monkeypatch.setattr(authority, "admit_native", admit)
    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    try:
        for received in (event, replace(event, source=replace(source))):
            if busy:
                await asyncio.wait_for(dispatch_shared_busy(adapter, received, adapter._event_session_key(received)), 5)
            else:
                await adapter.handle_message(received)
            await async_wait_until(lambda: not adapter._background_tasks, timeout=10)
            assert not getattr(received, "_gateway_processing_deferred", False)
            assert not authority.native_waiters and not authority.waiters and not authority.marker_adopters
            assert not authority.pending_deliveries and not authority.held_turn_markers
            assert not authority.native_completion
        assert [receipt.status for receipt in receipts] == ["queued", "terminal"]
        assert receipts[0].admission_id == receipts[1].admission_id
        row = get_session_admission(authority.db, admission_id=receipts[0].admission_id)
        assert (row["status"], row["outcome"]) == ("terminal", "completed")
        assert turns == [event.message_id]
        assert adapter.sent == ([reply] if reply else [])
        assert event.message_id not in adapter._reacting_message_ids
        assert adapter.reactions.count((event.message_id, "remove", "hourglass_flowing_sand")) == 1
        assert adapter.reactions.count((event.message_id, "add", "white_check_mark")) == int(bool(reply))
    finally:
        await asyncio.gather(*adapter._background_tasks, return_exceptions=True)
        await asyncio.gather(*(live.task for live in authority.sessions.values() if live.task), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "busy"])
@pytest.mark.parametrize("reply,failed,raised", [
    ("", False, False), (None, False, False),
    ("", True, False), (None, True, False), (None, True, True),
], ids=["empty", "none", "failed-empty", "failed-none", "raised"])
async def test_registered_live_waiter_terminal_releases_completion(monkeypatch, busy, reply, failed, raised):
    from gateway import session_settlement_recovery
    from gateway.session_ingress import dispatch_shared_busy
    from hermes_state_runtime import list_session_admissions

    monkeypatch.setenv("SLACK_ALLOWED_USERS", "fixture-user")
    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingSlack()
    adapter.gateway_runner = runner
    runner.adapters = {Platform.SLACK: adapter}
    await initialize_session_authority(runner, profile_id="default", instance_id="terminal-live")
    runner._wire_adapter_handlers(adapter)
    authority = runner.session_authority
    source = adapter.build_source(chat_id="D_CHAT", chat_type="dm", user_id="fixture-user")
    event = MessageEvent(text="terminal live", source=source, message_id="1234567890.000034")
    adapter._track_reacting_message("", event.message_id)
    claimed, registered, settling, release = (asyncio.Event() for _ in range(4))
    original_claim = authority._claim_with_retry
    original_settle = session_settlement_recovery.settle_with_retry
    original_complete = adapter.on_processing_complete
    completions = []

    async def claim(ref, live):
        claimed.set()
        await registered.wait()
        return await original_claim(ref, live)

    async def turn(event, source, quick_key, run_generation):
        if raised:
            raise RuntimeError("owned terminal completion control")
        execution_result.get()["result"] = {"final_response": reply or "", "messages": [], "failed": failed}
        return reply

    async def settle(*args, **kwargs):
        settling.set()
        await release.wait()
        return await original_settle(*args, **kwargs)

    async def complete(received, outcome):
        completions.append(received)
        await original_complete(received, outcome)

    monkeypatch.setattr(authority, "_claim_with_retry", claim)
    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    monkeypatch.setattr(session_settlement_recovery, "settle_with_retry", settle)
    monkeypatch.setattr(adapter, "on_processing_complete", complete)
    try:
        if busy:
            await dispatch_shared_busy(adapter, event, adapter._event_session_key(event))
        else:
            await adapter.handle_message(event)
        await asyncio.wait_for(claimed.wait(), 5)
        await async_wait_until(lambda: bool(authority.native_waiters), timeout=5)
        registered.set()
        await asyncio.wait_for(settling.wait(), 5)
        sid = next(iter(authority.sessions))
        row = list_session_admissions(authority.db, session_id=sid, pending_only=False)[0]
        assert row["status"] == "started"
        completion = authority.native_completion[row["admission_id"]]
        assert completion[0] is adapter and completion[1].message_id == event.message_id
        assert event.message_id in adapter._reacting_message_ids
        assert (event.message_id, "add", "hourglass_flowing_sand") in adapter.reactions
        assert (event.message_id, "remove", "hourglass_flowing_sand") not in adapter.reactions
        assert not adapter.sent and not completions
        release.set()
        await async_wait_until(lambda: not adapter._background_tasks, timeout=10)
        await asyncio.wait_for(authority.sessions[sid].task, 5)
        row = list_session_admissions(authority.db, session_id=sid, pending_only=False)[0]
        assert (row["status"], row["outcome"]) == ("terminal", "failed" if failed else "completed")
        assert completions == [event], "drain completed a live waiter's lifecycle"
        assert adapter.sent == (["The admitted turn failed."] if raised else [])
        assert event.message_id not in adapter._reacting_message_ids
        assert (event.message_id, "remove", "hourglass_flowing_sand") in adapter.reactions
        assert (event.message_id, "add", "white_check_mark") not in adapter.reactions
        assert ((event.message_id, "add", "x") in adapter.reactions) is failed
        assert not authority.pending_deliveries and not authority.native_waiters and not authority.waiters
        assert not authority.native_completion, "terminal admission retained adapter/event completion ownership"
    finally:
        registered.set()
        release.set()
        await asyncio.gather(*adapter._background_tasks, return_exceptions=True)
        await asyncio.gather(*(live.task for live in authority.sessions.values() if live.task), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "busy"])
@pytest.mark.parametrize("reply", ["", None], ids=["empty", "none"])
async def test_registered_live_waiter_unknown_retains_completion(monkeypatch, busy, reply):
    from gateway import session_results, session_settlement_recovery
    from gateway.session_contract import Principal, SessionRef
    from gateway.session_ingress import dispatch_shared_busy
    from hermes_state_runtime import get_session_admission

    monkeypatch.setenv("SLACK_ALLOWED_USERS", "fixture-user")
    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingSlack()
    adapter.gateway_runner = runner
    runner.adapters = {Platform.SLACK: adapter}
    await initialize_session_authority(runner, profile_id="default", instance_id="unknown-live")
    runner._wire_adapter_handlers(adapter)
    authority = runner.session_authority
    source = adapter.build_source(chat_id="D_CHAT", chat_type="dm", user_id="fixture-user")
    event = MessageEvent(text="unknown live", source=source, message_id="1234567890.000033")
    adapter._track_reacting_message("", event.message_id)
    started, claimed, registered, release = (asyncio.Event() for _ in range(4))
    original_finish = session_results.finish_result
    original_claim = authority._claim_with_retry

    async def claim(ref, live):
        claimed.set()
        await registered.wait()
        return await original_claim(ref, live)

    async def turn(event, source, quick_key, run_generation):
        started.set()
        await release.wait()
        execution_result.get()["result"] = {"final_response": reply or "", "messages": []}
        return reply

    def unavailable(*args, **kwargs):
        raise OSError("owned settlement failure")

    monkeypatch.setattr(authority, "_claim_with_retry", claim)
    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    monkeypatch.setattr(session_results, "finish_result", unavailable)
    monkeypatch.setattr(session_settlement_recovery, "_SETTLE_RETRY_DELAYS_S", ())
    try:
        if busy:
            await dispatch_shared_busy(adapter, event, adapter._event_session_key(event))
        else:
            await adapter.handle_message(event)
        await asyncio.wait_for(claimed.wait(), 5)
        await async_wait_until(lambda: bool(authority.native_waiters), timeout=5)
        registered.set()
        await asyncio.wait_for(started.wait(), 5)
        sid = next(iter(authority.sessions))
        from hermes_state_runtime import list_session_admissions
        row = list_session_admissions(authority.db, session_id=sid, pending_only=False)[0]
        assert row["status"] == "started"
        assert authority.native_completion[row["admission_id"]][0] is adapter
        assert (event.message_id, "add", "hourglass_flowing_sand") in adapter.reactions
        release.set()
        await async_wait_until(lambda: bool(adapter.sent) and not adapter._background_tasks, timeout=10)
        row = list_session_admissions(authority.db, session_id=sid, pending_only=False)[0]
        assert row["status"] == "unknown"
        assert event.message_id in adapter._reacting_message_ids
        assert (event.message_id, "remove", "hourglass_flowing_sand") not in adapter.reactions
        assert (event.message_id, "add", "white_check_mark") not in adapter.reactions
        assert row["admission_id"] in getattr(authority, "unsettled_deliveries", {})
        assert authority.unsettled_deliveries[row["admission_id"]][0] is adapter
        assert not authority.native_completion and not authority.pending_deliveries
        notices = list(adapter.sent)
        assert len(notices) == 1 and "paused" in notices[0].lower()
        monkeypatch.setattr(session_results, "finish_result", original_finish)
        actor = Principal("operator", "default", frozenset({"session:submit", "session:control"}), "fixture")
        ref = SessionRef("default", sid)
        await authority.resolve_unknown(actor, ref, row["admission_id"], row["generation"])
        resolved = get_session_admission(authority.db, admission_id=row["admission_id"])
        assert (resolved["status"], resolved["outcome"]) == ("terminal", "completed")
        assert event.message_id not in adapter._reacting_message_ids
        assert (event.message_id, "remove", "hourglass_flowing_sand") in adapter.reactions
        assert (event.message_id, "add", "white_check_mark") not in adapter.reactions
        assert not authority.pending_deliveries and not authority.unsettled_deliveries
        assert not authority.held_turn_markers and adapter.sent == notices
        reactions = list(adapter.reactions)
        await authority.resolve_unknown(actor, ref, row["admission_id"], row["generation"])
        assert adapter.reactions == reactions
    finally:
        release.set()
        await asyncio.gather(*adapter._background_tasks, return_exceptions=True)
        await asyncio.gather(*(live.task for live in authority.sessions.values() if live.task), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cmd", ["status", "approve", "deny", "stop", "new", "reset"])
@pytest.mark.parametrize("send_ok", [True, False], ids=["delivered", "rejected"])
async def test_busy_slack_inline_commands_have_reactions_without_queueing(cmd, send_ok):
    from hermes_cli.commands import is_interrupt_then_dispatch

    adapter = RecordingSlack()
    source = adapter.build_source(chat_id="D_CHAT", chat_type="dm", user_id="fixture-user")
    event = MessageEvent(text=f"/{cmd}", source=source, message_id="1234567890.000040")
    key = adapter._event_session_key(event)
    adapter._track_reacting_message("", event.message_id)
    await adapter._react(source.chat_id, event.message_id, "eyes", "", remove=False)
    guard = asyncio.Event()
    owner_release, entered, reply_release = (asyncio.Event() for _ in range(3))
    owner = asyncio.create_task(owner_release.wait())
    adapter._active_sessions[key] = guard
    adapter._session_tasks[key] = owner
    dispatch = None
    delivered_before_cancel = []

    async def handler(received):
        entered.set()
        await reply_release.wait()
        return "command reply"

    async def send(chat_id, content, reply_to=None, metadata=None):
        delivered_before_cancel.append(not owner.done())
        adapter.sent.append(content)
        return SendResult(success=send_ok, message_id="command-reply" if send_ok else None,
                          error=None if send_ok else "invalid_auth")

    adapter._message_handler = AsyncMock(side_effect=handler)
    adapter._send_with_retry = send
    try:
        dispatch = asyncio.create_task(adapter._handle_message_while_active(event, key))
        await asyncio.wait_for(entered.wait(), 5)
        assert not owner.done()
        assert not adapter._pending_messages
        assert (event.message_id, "add", "hourglass_flowing_sand") in adapter.reactions
        assert (event.message_id, "add", "white_check_mark") not in adapter.reactions
        reply_release.set()
        await asyncio.wait_for(dispatch, 5)
        adapter._message_handler.assert_awaited_once_with(event)
        assert delivered_before_cancel and all(delivered_before_cancel)
        assert not adapter._pending_messages
        if is_interrupt_then_dispatch(cmd):
            assert owner.cancelled()
            assert key not in adapter._active_sessions
        else:
            assert not owner.done()
            assert adapter._active_sessions[key] is guard
        assert adapter.reactions[-1] == (event.message_id, "add", "white_check_mark" if send_ok else "x")
    finally:
        reply_release.set()
        owner_release.set()
        if dispatch is not None:
            await asyncio.gather(dispatch, return_exceptions=True)
        await asyncio.gather(owner, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "busy"])
@pytest.mark.parametrize("cancel_at", ["hourglass_flowing_sand", "eyes"])
async def test_completion_cancellation_retries_transient_cleanup(monkeypatch, busy, cancel_at):
    from dataclasses import replace
    from gateway.session_ingress import dispatch_shared_busy
    from hermes_state_runtime import list_session_admissions

    monkeypatch.setenv("SLACK_ALLOWED_USERS", "fixture-user")
    runner = GatewayRunner(GatewayConfig())
    adapter = RecordingSlack()
    adapter.gateway_runner = runner
    runner.adapters = {Platform.SLACK: adapter}
    await initialize_session_authority(runner, profile_id="default", instance_id="cancel-cleanup")
    runner._wire_adapter_handlers(adapter)
    authority = runner.session_authority
    source = adapter.build_source(chat_id="D_CHAT", chat_type="dm", user_id="fixture-user")
    event = MessageEvent(text="cancel cleanup", source=source, message_id="1234567890.000041")
    adapter._track_reacting_message("", event.message_id)
    registered, removing, retrying, release = (asyncio.Event() for _ in range(4))
    original_claim, original_react = authority._claim_with_retry, adapter._react
    original_admit = authority.admit_native
    receipts, turns, owners = [], [], []
    attempts = 0

    async def claim(ref, live):
        await registered.wait()
        return await original_claim(ref, live)

    async def admit(received):
        receipt = await original_admit(received)
        receipts.append(receipt)
        return receipt

    async def turn(received, source, quick_key, run_generation):
        turns.append(received.message_id)
        execution_result.get()["result"] = {"final_response": "cancel reply", "messages": []}
        return "cancel reply"

    async def react(channel, timestamp, emoji, team_id, *, remove):
        nonlocal attempts
        if remove and emoji == cancel_at and adapter.sent:
            attempts += 1
            if attempts == 1:
                owners.append(asyncio.current_task())
                removing.set()
            else:
                retrying.set()
            await release.wait()
        return await original_react(channel, timestamp, emoji, team_id, remove=remove)

    async def duplicate():
        received = replace(event, source=replace(source))
        existing = set(adapter._background_tasks)
        await dispatch_shared_busy(adapter, received, adapter._event_session_key(received))
        await asyncio.gather(*(set(adapter._background_tasks) - existing))
        return received

    monkeypatch.setattr(authority, "_claim_with_retry", claim)
    monkeypatch.setattr(authority, "admit_native", admit)
    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    monkeypatch.setattr(adapter, "_react", react)
    entered = None
    try:
        if busy:
            await dispatch_shared_busy(adapter, event, adapter._event_session_key(event))
        else:
            await adapter.handle_message(event)
        await async_wait_until(lambda: bool(authority.native_waiters), timeout=5)
        registered.set()
        await asyncio.wait_for(removing.wait(), 5)
        row = list_session_admissions(authority.db, session_id=receipts[0].ref.session_id, pending_only=False)[0]
        assert (row["status"], row["outcome"]) == ("terminal", "completed")
        assert adapter.sent == ["cancel reply"]
        assert (event.message_id, "add", "white_check_mark") not in adapter.reactions
        owner = owners[0]
        adapter._expected_cancelled_tasks.add(owner)
        owner.cancel()  # interrupt the real completion hook's held first/second removal
        entered = asyncio.create_task(retrying.wait())
        await asyncio.wait_for(asyncio.wait({owner, entered}, return_when=asyncio.FIRST_COMPLETED), 5)
        assert retrying.is_set(), "Cancellation lost the tracked transient-reaction cleanup target"
        assert event.message_id in adapter._reacting_message_ids
        assert authority.live_delivery_owners.get(row["admission_id"]) is event, "Cancellation released the live completion owner before its retry"
        redelivery = await duplicate()
        assert getattr(redelivery, "_gateway_processing_deferred", False)
        assert attempts == 2, "a terminal duplicate competed with the cancelled owner's retry"
        assert (event.message_id, "add", "white_check_mark") not in adapter.reactions
        release.set()
        await asyncio.gather(owner, return_exceptions=True)
        assert owner.cancelled()
        await asyncio.wait_for(authority.sessions[receipts[0].ref.session_id].task, 5)
        assert event.message_id not in adapter._reacting_message_ids
        assert (event.message_id, "remove", "hourglass_flowing_sand") in adapter.reactions
        assert (event.message_id, "remove", "eyes") in adapter.reactions
        assert (event.message_id, "add", "white_check_mark") not in adapter.reactions
        reactions = list(adapter.reactions)
        finished = await duplicate()
        assert not getattr(finished, "_gateway_processing_deferred", False)
        assert adapter.reactions == reactions
        assert [r.status for r in receipts] == ["queued", "terminal", "terminal"]
        assert all(r.admission_id == row["admission_id"] for r in receipts)
        assert turns == [event.message_id] and adapter.sent == ["cancel reply"]
        assert not authority.live_delivery_owners and not authority.pending_deliveries
        assert not authority.native_completion and not authority.waiter_replies
        assert not authority.waiters and not authority.native_waiters and not authority.held_turn_markers
    finally:
        registered.set()
        release.set()
        if entered is not None:
            entered.cancel()
            await asyncio.gather(entered, return_exceptions=True)
        await asyncio.gather(*adapter._background_tasks, *owners, return_exceptions=True)
        await asyncio.gather(*(live.task for live in authority.sessions.values() if live.task), return_exceptions=True)

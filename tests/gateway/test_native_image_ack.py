"""Busy image acknowledgements follow durable native acceptance or FIFO fallback."""
import asyncio
import threading

import pytest

from agent.content_steering import ContentSteering
from gateway.platforms.base import SendResult
from hermes_state_runtime import list_session_admissions
from tests.gateway.test_native_image_transfer import owner
from tests.waiting import async_wait_until

pytest_plugins = ["tests.gateway.test_native_image_steering"]


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [True, False])
@pytest.mark.parametrize("ack_mode", ["on", "off", "steer-off", "debounced"])
async def test_busy_image_ack_waits_for_native_transfer(setup, monkeypatch, accepted, ack_mode):
    import gateway.run_image_steering as native
    from agent.i18n import t

    runner, agent, images, adapter, key, event, authority, running = await owner(setup, monkeypatch)
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false" if ack_mode == "off" else "true")
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED", "false" if ack_mode == "steer-off" else "true")
    agent._content_steering = ContentSteering()
    agent._content_steering.open()
    monkeypatch.setattr(runner, "_agent_has_active_subagents", lambda _agent: True)
    if ack_mode == "debounced":
        import time
        runner._session_state(key).turn.busy_ack_ts = time.time()
    adapter._active_sessions[key] = asyncio.Event()
    adapter._session_tasks[key] = asyncio.current_task()
    sent = []

    def rows():
        return list_session_admissions(authority.db, session_id=agent.session_id, pending_only=False)

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append((content, chat_id, reply_to, metadata, rows()[-1], bool(agent._content_steering._pending)))
        return SendResult(success=True, message_id="busy-ack")

    monkeypatch.setattr(adapter, "send", send)
    entered, release = threading.Event(), threading.Event()
    prepare = native._prepare

    def held(*args):
        entered.set()
        assert release.wait(10)
        return prepare(*args)

    monkeypatch.setattr(native, "_prepare", held)
    task = asyncio.create_task(adapter.handle_message(event))
    try:
        await async_wait_until(entered.is_set, timeout=10, what="native preparation")
        assert len(rows()) == 2, "pixels must be captured and durably admitted first"
        assert rows()[-1]["status"] == "queued"
        assert sent == [], "preparation has not decided steering versus FIFO yet"
        if not accepted:
            agent._content_steering.seal()  # the real agent refuses the prepared correction
        release.set()
        await asyncio.wait_for(task, 10)
        await async_wait_until(lambda: not authority._mutation_tasks, timeout=10, what="transfer completion")
        row = rows()[-1]
        assert row["outcome"] == "steering" if accepted else row["status"] == "queued"
        assert row["generation"] == (running["generation"] if accepted else None)
        assert len(row["payload"]["native_text_v1"]["media"]) == len(images)
        expected_ack = ack_mode not in ("off", "debounced") and not (accepted and ack_mode == "steer-off")
        assert len(sent) == int(expected_ack)
        if expected_ack:
            content, chat_id, reply_to, metadata, at_send, native_pending = sent[0]
            assert native_pending is accepted, "the ack must follow real agent acceptance or rejection"
            expected, forbidden = ("steered", "queued") if accepted else ("queued", "steered")
            assert t(f"gateway.busy.{expected}_head") in content
            assert t(f"gateway.busy.{forbidden}_head") not in content
            assert t("gateway.busy.steered_subagents_head") not in content
            assert chat_id == event.source.chat_id
            assert reply_to == event.message_id
            assert metadata == runner._thread_metadata_for_source(event.source, runner._reply_anchor_for_event(event))
            assert at_send["outcome"] == "steering" if accepted else at_send["status"] == "queued"
        await asyncio.wait_for(adapter.handle_message(event), 10)
        await async_wait_until(lambda: not authority._mutation_tasks, timeout=10, what="redelivery completion")
        assert len(rows()) == 2
        assert len(sent) == int(expected_ack), "redelivery must not send a second acknowledgement"
        assert bool(agent.withdraw_steer(row["admission_id"])) is accepted
        assert not agent.withdraw_steer(row["admission_id"])
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        for waiter in authority.waiters.values():
            if not waiter.done():
                waiter.set_result(None)
        await async_wait_until(lambda: not authority._mutation_tasks, timeout=10, what="admission cleanup")
        await asyncio.wait_for(asyncio.gather(*adapter._background_tasks, return_exceptions=True), 10)
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True])
async def test_uncertain_native_transfer_does_not_ack_a_guessed_outcome(setup, monkeypatch, committed):
    import gateway.run_image_steering as native
    from unittest.mock import AsyncMock

    _runner, agent, _images, adapter, key, event, authority, _running = await owner(setup, monkeypatch)
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED", "true")
    agent._content_steering = ContentSteering()
    agent._content_steering.open()
    adapter._active_sessions[key] = asyncio.Event()
    adapter._session_tasks[key] = asyncio.current_task()
    send = AsyncMock(return_value=SendResult(success=True, message_id="busy-ack"))
    monkeypatch.setattr(adapter, "send", send)
    write = native.steer_session_input
    attempted = []

    def uncertain(*args, **kwargs):
        attempted.append(kwargs["admission_id"])
        if committed:
            write(*args, **kwargs)
        raise OSError("synthetic transfer write uncertainty")

    monkeypatch.setattr(native, "steer_session_input", uncertain)
    try:
        await asyncio.wait_for(adapter.handle_message(event), 10)
        await async_wait_until(lambda: not authority._mutation_tasks, timeout=10, what="uncertain transfer")
        rows = list_session_admissions(authority.db, session_id=agent.session_id, pending_only=False)
        assert len(rows) == 2
        assert attempted == [rows[-1]["admission_id"]]
        assert rows[-1]["outcome"] == "steering" if committed else rows[-1]["status"] == "queued"
        assert not agent.withdraw_steer(rows[-1]["admission_id"])
        send.assert_not_awaited()
    finally:
        for waiter in authority.waiters.values():
            if not waiter.done():
                waiter.set_result(None)
        await asyncio.wait_for(asyncio.gather(*adapter._background_tasks, return_exceptions=True), 10)
        authority.db.close()

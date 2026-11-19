"""Synthetic completion turns must not adopt other platforms' inbound-message UI."""

import asyncio
import json

import pytest

from gateway.session_ingress import deliver_settled, execute_admission
from hermes_state_runtime import claim_session_input, settle_session_input
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_restored_automation_never_reacts_to_its_producer_identity(tmp_path, monkeypatch):
    from gateway.run_turn_followup_ack import _followup_processing_hooks_apply
    from gateway.session_contract import SessionRef
    from tests.gateway.relay.test_relay_native_turn_lifecycle import (
        EYES, OK, _actions, _gateway, _reacts, _settled, _stub_model,
    )
    from tests.gateway.test_completion_admission import pending

    async with _gateway(tmp_path, "automation-reactions") as (runner, authority, send, outgoing, adapter):
        _stub_model(runner, monkeypatch)
        await send("1700000000.000100", "D1", "hello")
        await async_wait_until(_settled(outgoing, "1700000000.000100", (OK, False)), timeout=20,
                               what="genuine relay input finished")
        assert _reacts(outgoing, "1700000000.000100") == [(EYES, False), (EYES, True), (OK, False)]
        sid, live = next(iter(authority.sessions.items()))
        ref = SessionRef(authority.profile_id, sid)
        monkeypatch.setattr(authority, "_schedule", lambda ref: None)
        completion = dict(pending(live.route, "synthetic"), parent_session_id=sid)
        assert await runner._deliver_async_delegation_group([completion]) is True
        row = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
        before = len(_actions(outgoing))
        seen = []

        async def handler(event):
            seen.append(event)
            assert event.internal and json.loads(event.message_id)[0] == "async_delegation"
            assert not _followup_processing_hooks_apply(adapter, event)
            return "completed"

        monkeypatch.setattr(runner, "_handle_message", handler)
        await execute_admission(authority, ref, row)
        settle_session_input(authority.db, epoch=authority.epoch, admission_id=row["admission_id"],
                             generation=row["generation"], outcome="completed")
        await deliver_settled(authority, row["admission_id"])
        assert len(seen) == 1
        assert not [a for a in _actions(outgoing)[before:] if a.get("op") in {"react", "typing"}]
        assert not authority.producer_turns


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["failed", "cancelled", "no_reply", "reply"])
async def test_google_chat_automation_does_not_create_inbound_typing_cards(tmp_path, monkeypatch, ending):
    from unittest.mock import AsyncMock
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from plugins.platforms.google_chat.adapter import GoogleChatAdapter
    from tests.gateway.test_automation_compression_media import _admit_human
    from tests.gateway.test_completion_admission import pending

    monkeypatch.setenv("GOOGLE_CHAT_ALLOWED_USERS", "42")
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = GoogleChatAdapter(PlatformConfig(enabled=True))
    runner.adapters = {adapter.platform: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="google-automation")
    runner._wire_adapter_handlers(adapter)
    monkeypatch.setattr(authority, "_schedule", lambda ref: None)
    source = adapter.build_source(chat_id="spaces/S", chat_type="dm", user_id="42")
    typing_finished, entered = asyncio.Event(), asyncio.Event()
    refreshes = []
    start_typing, send_typing = adapter._start_typing_refresh, adapter.send_typing

    def start(*args):
        task = start_typing(*args)
        if task is not None:
            refreshes.append(task)
        return task

    async def typing(*args, **kwargs):
        await send_typing(*args, **kwargs)
        typing_finished.set()

    monkeypatch.setattr(adapter, "_start_typing_refresh", start)
    monkeypatch.setattr(adapter, "send_typing", typing)
    create = AsyncMock(return_value=SendResult(success=True, message_id="spaces/S/messages/new"))
    patch = AsyncMock(return_value=SendResult(success=True, message_id="spaces/S/messages/new"))
    monkeypatch.setattr(adapter, "_create_message", create)
    monkeypatch.setattr(adapter, "_patch_message", patch)
    task = None
    try:
        receipt = await _admit_human(runner, authority, MessageEvent(text="human", source=source, message_id="human"))
        ref = receipt.ref
        row = claim_session_input(authority.db, epoch=authority.epoch, session_id=ref.session_id)
        settle_session_input(authority.db, epoch=authority.epoch, admission_id=row["admission_id"],
                             generation=row["generation"], outcome="completed")
        key = adapter._source_session_key(source)
        completion = dict(pending(key, "google-completion"), parent_session_id=ref.session_id)
        assert await runner._deliver_async_delegation_group([completion]) is True
        row = claim_session_input(authority.db, epoch=authority.epoch, session_id=ref.session_id)

        async def handler(event):
            assert event.internal
            # If the lifecycle started a refresh, let the real card create complete before
            # ending the model turn. No timing race can hide a newly stranded thinking card.
            if refreshes:
                await asyncio.wait_for(typing_finished.wait(), 5)
            entered.set()
            if ending == "failed":
                raise RuntimeError("model failed")
            if ending == "cancelled":
                await asyncio.Event().wait()
            return "completed" if ending == "reply" else None

        monkeypatch.setattr(runner, "_handle_message", handler)
        task = asyncio.create_task(execute_admission(authority, ref, row))
        await asyncio.wait_for(entered.wait(), 5)
        if ending == "cancelled":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif ending == "failed":
            with pytest.raises(RuntimeError, match="model failed"):
                await task
        else:
            await task
        settle_session_input(authority.db, epoch=authority.epoch, admission_id=row["admission_id"],
                             generation=row["generation"], outcome="completed")
        await deliver_settled(authority, row["admission_id"])
        assert not adapter._typing_messages, "synthetic completion stranded a thinking card"
        assert not refreshes, "automation must preserve Google Chat's preexisting no-typing behavior"
        assert create.await_count == (1 if ending == "reply" else 0)
        patch.assert_not_awaited()
        assert not authority.producer_turns
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await adapter.cancel_background_tasks()
        authority.db.close()

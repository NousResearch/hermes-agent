"""A committed async completion uses voice mode at delivery, not dispatch."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import session_ingress, session_settlement_recovery
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.session_results import execution_result
from tests.gateway.test_auto_voice_reply_format import _make_adapter, _make_runner
from tests.gateway.test_session_authority_cancel_settlement import REF, _authority, _submit


# Adapted from Kevin Rajan's regression test (commit 3167946b0ad456b85d3e0f5db376d036a6adc45d).
@pytest.mark.asyncio
async def test_streamed_async_completion_reads_voice_mode_after_commit(
    tmp_path, monkeypatch,
):
    db, authority = _authority(tmp_path, monkeypatch, Platform.TELEGRAM)
    monkeypatch.setattr(authority, "_schedule", lambda ref: None)
    runner = _make_runner()
    adapter = _make_adapter(Platform.TELEGRAM)
    adapter._streaming_tts_turn_completed = MagicMock(return_value=False)
    adapter._should_auto_tts_for_chat = MagicMock(return_value=False)
    runner.adapters[Platform.TELEGRAM] = adapter
    runner._voice_mode["telegram:123"] = "off"
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._adapter_for_source = lambda source: adapter
    runner._delivery_adapter_for = lambda source: adapter
    runner._adapter_profile_for_source = lambda source: None
    authority.runner = runner
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="human")
    authority.sessions["s"].source = source
    monkeypatch.setattr(
        session_ingress, "MessageEvent",
        lambda **kw: MessageEvent(
            text=kw["text"], source=source, internal=True,
            metadata={"presentation_origin": "voice"},
        ),
    )
    spoken = []

    async def send_audio(event, answer):
        admission = authority.persistence.admissions.get(admission_id=receipt.admission_id)
        assert (admission.status, admission.outcome) == ("terminal", "completed")
        spoken.append(answer)

    runner._send_voice_reply = AsyncMock(side_effect=send_audio)

    async def handle(event):
        answer = "async final answer"
        result = {
            "final_response": answer, "messages": [], "completed": True,
            "already_sent": True, "media_already_delivered": True,
        }
        execution_result.get()["result"] = result
        return await runner._hmwa_deliver_turn_response(
            event, source, SimpleNamespace(session_id="s"), "s", 1,
            result, [], answer, "", False,
        )

    runner._handle_message = handle
    settle = session_settlement_recovery.settle_with_retry

    async def switch_mode_before_commit(*args, **kwargs):
        runner._voice_mode["telegram:123"] = "voice_only"
        return await settle(*args, **kwargs)

    monkeypatch.setattr(
        session_settlement_recovery, "settle_with_retry", switch_mode_before_commit,
    )
    with db:
        receipt = await _submit(authority, "completion")
        await asyncio.wait_for(authority._drain(REF), 5)
        assert spoken == ["async final answer"]

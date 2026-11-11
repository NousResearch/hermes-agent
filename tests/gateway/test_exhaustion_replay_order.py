"""A request whose turn exhausted the context is not re-run behind follow-ups the reset re-homed onto
the fresh session: it would answer out of order (#137565 review). Real GatewayRunner, DiscordAdapter
and SessionAuthority over SQLite; the turn is a stand-in."""
import pytest

from gateway.session_results import execution_result
from tests.gateway import test_native_fifo_context_exhaustion as fifo

_admit = fifo._admit
native_fifo = fifo.native  # the shared real-authority fixture, registered here under its own name


@pytest.mark.asyncio
async def test_exhausted_request_is_not_rerun_behind_rehomed_follow_ups(native_fifo, monkeypatch):
    native = native_fifo
    runner, authority, adapter = native.runner, native.authority, native.adapter
    admitted = [await _admit(native, text) for text in ("A", "B")]
    executed, queued_after_reset = [], []

    async def turn(event, source, session_key, run_generation):
        entry = runner.session_store.lookup_by_session_key(session_key)
        executed.append(event.text)
        captured = execution_result.get()
        if event.text == "A" and executed.count("A") == 1:
            result = {"failed": True, "compression_exhausted": True, "messages": []}
            captured["result"] = result
            await runner._hmwa_compression_exhaustion_reset(result, "copy", entry, session_key, source, event=event)
            queued_after_reset.append(adapter._pending_messages.get(session_key))
        else:
            captured["result"] = {"final_response": "reply", "messages": []}
        return "reply"

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    await authority._drain(admitted[0][1].ref)
    # B was re-homed onto the fresh session; A going back in the head slot would run behind it.
    assert queued_after_reset == [None], "the exhausted request was queued behind a re-homed follow-up"
    assert executed == ["A"], executed

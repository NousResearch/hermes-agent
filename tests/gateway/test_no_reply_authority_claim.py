"""Reply-clock ownership follows the real inbound claim and native authority waiter."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.run_turn_persistence import GatewayTurnPersistenceMixin
from tests.gateway import test_platform_no_reply as platform_tests
from tests.gateway.test_platform_no_reply import Adapter, event, no_replies
from tests.gateway.test_post_gateway_admission import _register, _runner

rows = platform_tests.rows


def claim_runner(adapter):
    runner = _runner()
    runner._delivery_adapter_for = lambda _: adapter
    runner._intake_adapter_for = lambda _: adapter
    runner._hm_admit_event = AsyncMock(side_effect=lambda ev: (ev, ev.source, False))
    runner._hm_dispatch_idle_commands = AsyncMock(return_value=(False, None))
    runner._is_telegram_topic_root_lobby = lambda _: False
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [None, "plugin answer"])
async def test_real_claim_plugin_consume_is_silent_or_replied(rows, reply):
    adapter = Adapter()
    ev = event(adapter)
    runner = claim_runner(adapter)
    _register(lambda **_: {"action": "handled", "reply": reply})
    adapter.set_message_handler(runner._handle_message)
    await adapter._process_message_background(ev, "session")
    assert ev._reply_clock_token is not None  # the production claim must attach it
    assert no_replies(rows) == ([] if reply else [{"platform": "telegram", "reason": "silent"}])
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("delivered", [False, True], ids=["empty", "streamed-media"])
async def test_authority_waiter_adopts_executed_claim_before_adapter_finish(rows, tmp_path, monkeypatch, delivered):
    from gateway.platforms import base
    from gateway.session_contract import Principal, SessionRef, Submission
    from gateway.session_ingress import admit_message
    from tests.gateway.test_prompt_attachments import _authority

    adapter, executed, finished = Adapter(), [], []
    entered, release = asyncio.Event(), asyncio.Event()
    runner = claim_runner(adapter)

    async def answer(ev, source, key, generation):
        executed.append(ev)
        entered.set()
        await asyncio.wait_for(release.wait(), 5)
        if delivered:
            return await GatewayTurnPersistenceMixin._hmwa_deliver_turn_response(
                runner, ev, source, SimpleNamespace(session_id="s"), key, generation,
                {"already_sent": True, "media_already_delivered": True}, [], "MEDIA:image.png", "", False)
        return None

    runner._handle_message_with_agent = AsyncMock(side_effect=answer)
    runner._should_send_voice_reply = lambda *_a, **_k: False
    authority = await _authority(tmp_path, monkeypatch, runner._handle_message)
    actor = Principal("human", "p", frozenset({"session:submit"}), "t")

    async def admit_native(ev):
        return await authority.submit(actor, Submission("m1", SessionRef("p", "s"), {"text": ev.text}, "queue"))

    monkeypatch.setattr(authority, "admit_native", admit_native)
    incoming = event(adapter)
    incoming.source = authority.sessions["s"].source
    adapter.gateway_runner = SimpleNamespace(_clear_durable_active_turn=AsyncMock())
    adapter.set_message_handler(lambda ev: admit_message(authority, ev))
    finish = base.finish_reply_clock

    def capture_finish(*args, **kwargs):
        finished.append((getattr(incoming, "_reply_clock_token", None),
                         getattr(incoming, "_turn_reply_delivered", False)))
        finish(*args, **kwargs)

    monkeypatch.setattr(base, "finish_reply_clock", capture_finish)
    lifecycle = asyncio.create_task(adapter._process_message_background(incoming, "session"))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert not hasattr(incoming, "_reply_clock_token")
        assert no_replies(rows) == []  # an unfinished FIFO turn is never classified
        release.set()
        await asyncio.wait_for(lifecycle, 10)
        await asyncio.wait_for(authority.sessions["s"].task, 10)
        assert len(executed) == 1 and executed[0] is not incoming
        assert finished and finished[0][0] is not None
        assert finished[0][1] == delivered
        assert not hasattr(executed[0], "_reply_clock_token")
        assert not hasattr(executed[0], "_turn_reply_delivered")
        assert no_replies(rows) == ([] if delivered else [{"platform": "telegram", "reason": "empty"}])
        assert not authority.marker_adopters
    finally:
        release.set()
        await lifecycle
        authority.db.close()

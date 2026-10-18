"""A photo sent while a gateway turn is running is one next turn, not two.

The authority path interrupts the running turn with the caption and also admits the
photo to the durable FIFO. The drain used to promote that same caption into a
follow-up, so the send ran twice. Outside an admission the interrupt text is still
the only copy, and it must still become the next turn.
"""
import threading

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session_ingress import executing_admission


class _Slot:
    def __init__(self):
        self._pending_messages = {}

    def get_pending_message(self, session_key):
        return self._pending_messages.pop(session_key, None)


def _source():
    from gateway.session import SessionSource
    return SessionSource(platform=Platform.DISCORD, chat_id="42", chat_type="dm", user_id="42")


@pytest.mark.asyncio
async def test_interrupt_text_still_becomes_the_next_turn_outside_an_admission():
    runner = GatewayRunner(GatewayConfig())
    pending_event, pending = await runner._run_agent_drain_pending(
        {"interrupted": True, "interrupt_message": "keep me"}, _Slot(), _source(), "sk")
    assert pending_event is None
    assert pending == "keep me"


@pytest.mark.asyncio
async def test_interrupt_text_is_not_a_second_turn_during_an_admission():
    runner = GatewayRunner(GatewayConfig())
    token = executing_admission.set(True)
    try:
        pending_event, pending = await runner._run_agent_drain_pending(
            {"interrupted": True, "interrupt_message": "caption"}, _Slot(), _source(), "sk")
    finally:
        executing_admission.reset(token)
    assert pending_event is None
    assert pending is None


async def _followup_while_running(monkeypatch, event, marker):
    """Run one turn, deliver ``event`` while it is in progress, and return what ran next."""
    from gateway.session_authority import initialize_session_authority
    from plugins.platforms.discord.adapter import DiscordAdapter
    from tests.waiting import async_wait_until

    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="s1")

    monkeypatch.setattr(adapter, "send", send)

    async def transcribed(self, event, adapter, source, text, *, log_context, metadata=None):
        return "heard this", []

    monkeypatch.setattr(GatewayRunner, "_transcribe_and_echo_pending_voice", transcribed)
    adapter.gateway_runner = runner
    runner.adapters = {Platform.DISCORD: adapter}
    await initialize_session_authority(runner, profile_id="default", instance_id="owner")
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    event.source = source
    seen = []
    started = threading.Event()
    agent_ready = threading.Event()
    first = {"open": True}

    class Agent:
        def __init__(self):
            self._interrupt_message = None

        def interrupt(self, message=None, *args, **kwargs):
            self._interrupt_message = message
            started.set()

    agent = Agent()

    def run_sync(self):
        message = self._ctx.message
        seen.append(message)
        if first["open"]:
            first["open"] = False
            self._runner._session_state(self._ctx.session_key).turn.agent = agent
            agent_ready.set()
            got_it = started.wait(10)
            result = {
                "final_response": "", "messages": [], "completed": False, "interrupted": True,
                "interrupt_message": agent._interrupt_message if got_it else None,
            }
        else:
            result = {"final_response": "ok", "messages": [], "completed": True}
        self._ctx.result_holder[0] = result
        return result

    monkeypatch.setattr(TurnRunner, "run_sync", run_sync)
    import asyncio
    opening = asyncio.create_task(adapter.handle_message(MessageEvent(text="A", source=source, message_id="A")))
    key = runner.session_store._generate_session_key(source)
    await async_wait_until(agent_ready.is_set, what="the running agent")
    await adapter.handle_message(event)
    await opening
    await async_wait_until(
        lambda: len(seen) >= 2 and not runner._is_session_running(key), what="the follow-up to finish")
    followups = seen[1:]
    assert len(followups) == 1, seen
    assert marker in followups[0], seen
    return agent._interrupt_message


@pytest.mark.asyncio
async def test_photo_during_a_running_turn_is_admitted_once(monkeypatch, tmp_path):
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(bytes.fromhex("ffd8ffe000104a46494600010100000100010000ffd9"))
    interrupt = await _followup_while_running(monkeypatch, MessageEvent(
        text="caption", message_type=MessageType.PHOTO, message_id="P",
        media_urls=[str(photo)], media_types=["image/jpeg"],
    ), "photo.jpg")
    assert interrupt == "caption"


@pytest.mark.asyncio
async def test_voice_during_a_running_turn_is_admitted_once(monkeypatch, tmp_path):
    voice = tmp_path / "note.ogg"
    voice.write_bytes(b"OggS" + b"\x00" * 32)
    interrupt = await _followup_while_running(monkeypatch, MessageEvent(
        text="", message_type=MessageType.VOICE, message_id="V",
        media_urls=[str(voice)], media_types=["audio/ogg"],
    ), "note.ogg")
    assert interrupt == "heard this"

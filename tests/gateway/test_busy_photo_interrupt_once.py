"""A photo sent while a gateway turn is running is one next turn, not two.

The authority path interrupts the running turn with the caption and also admits the
photo to the durable FIFO. The drain used to promote that same caption into a
follow-up, so the send ran twice. A refused admit is the authority's answer: the send
never runs as an unadmitted legacy turn behind its back.
"""
import threading

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner


async def _followup_while_running(monkeypatch, event, marker, *, fail_admit=False):
    """Run one turn, deliver ``event`` while it is in progress; return the follow-up turns, the
    interrupt text and what was sent to the chat.

    The running turn stays inside its tool until the inbound handler returns, so a failed
    admit is visible to the drain. ``fail_admit`` makes that admit refuse (``admission_conflict``).
    """
    from gateway.session_authority import initialize_session_authority
    from plugins.platforms.discord.adapter import DiscordAdapter
    from tests.waiting import async_wait_until

    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))

    sent = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append(content)
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
    release = threading.Event()
    agent_ready = threading.Event()
    first = {"open": True}

    class Agent:
        def __init__(self):
            self._interrupt_message = None

        def interrupt(self, message=None, *args, **kwargs):
            self._interrupt_message = message

    agent = Agent()

    def run_sync(self):
        message = self._ctx.message
        seen.append(message)
        if first["open"]:
            first["open"] = False
            self._runner._session_state(self._ctx.session_key).turn.agent = agent
            agent_ready.set()
            held = release.wait(10)
            result = {
                "final_response": "", "messages": [], "completed": False, "interrupted": True,
                "interrupt_message": agent._interrupt_message if held else None,
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
    if fail_admit:
        async def deny(authority, event):
            from hermes_state_runtime import RuntimeStoreError
            raise RuntimeStoreError("admission_conflict")

        monkeypatch.setattr("gateway.session_ingress.admit_message", deny)
    from hermes_state_runtime import RuntimeStoreError
    try:
        await adapter.handle_message(event)
    except RuntimeStoreError as exc:  # the refusal surfaces to the caller, as on main
        if not fail_admit:
            raise
        sent.append(exc.reason)
    finally:
        release.set()
    await opening
    await async_wait_until(lambda: len(seen) >= 1 + (not fail_admit) and not runner._is_session_running(key)
                           and not adapter._background_tasks, what="the follow-up to finish")
    return seen[1:], agent._interrupt_message, sent


def _photo(tmp_path):
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(bytes.fromhex("ffd8ffe000104a46494600010100000100010000ffd9"))
    return MessageEvent(text="caption", message_type=MessageType.PHOTO, message_id="P",
                        media_urls=[str(photo)], media_types=["image/jpeg"]), "photo.jpg", "caption"


def _voice(tmp_path):
    voice = tmp_path / "note.ogg"
    voice.write_bytes(b"OggS" + b"\x00" * 32)
    return MessageEvent(text="", message_type=MessageType.VOICE, message_id="V",
                        media_urls=[str(voice)], media_types=["audio/ogg"]), "note.ogg", "heard this"


def _text(tmp_path):
    return MessageEvent(text="later", message_id="T"), "later", "later"


@pytest.mark.asyncio
@pytest.mark.parametrize("make", [_photo, _voice, _text], ids=["photo", "voice", "text"])
async def test_send_during_a_running_turn_is_admitted_once(monkeypatch, tmp_path, make):
    event, marker, interrupt_text = make(tmp_path)
    followups, interrupt, _ = await _followup_while_running(monkeypatch, event, marker)
    assert len(followups) == 1 and marker in followups[0], followups
    assert interrupt == interrupt_text


@pytest.mark.asyncio
async def test_refused_admit_never_runs_as_an_unadmitted_turn(monkeypatch):
    """A refused replay (``admission_conflict``) is not saved: it must not run anyway through the
    legacy pending slot, outside durable admission and FIFO order."""
    followups, _, sent = await _followup_while_running(
        monkeypatch, MessageEvent(text="changed replay", message_id="T"), "changed replay", fail_admit=True)
    assert followups == []
    assert "admission_conflict" in sent, sent  # refused, not swallowed

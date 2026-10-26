"""``post_transcription_admission``: let a plugin consume a voice message by its SPOKEN text (#145125).

``post_gateway_admission`` fires before speech-to-text runs, so for a captionless voice note the
plugin sees an empty ``text`` and cannot act on what the user actually said. This hook fires once
the transcript exists, with the same fail-open ``{"action": "handled", "reply": ...}`` contract,
and a consumed message skips the agent turn.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource

HOOK = "post_transcription_admission"


def _register(callback) -> None:
    """Register *callback* on the ACTIVE home's plugin manager (what a loaded plugin does)."""
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    manager._discovered = True  # no on-disk plugins in the test home; skip discovery
    manager._hooks.setdefault(HOOK, []).append(callback)


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)},
    )
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.pairing_store = MagicMock()
    runner.session_store = MagicMock()
    runner._is_user_authorized_for_source = lambda _s: True
    runner._admit_bot_message_for_source = lambda _s: True
    runner._delivery_adapter_for = lambda _s: adapter
    runner._intake_adapter_for = lambda _s: adapter
    runner._rescue_orphaned_overflow = lambda _k, _a: None
    runner._enqueue_fifo = MagicMock()
    runner._handle_message_with_agent = AsyncMock(return_value="agent-ok")
    runner._run_post_turn_hooks = AsyncMock()
    runner._session_key_for_source = lambda _s: "wa:123"
    return runner


def _voice_event(text: str = "", path: str = "/tmp/voice.ogg"):
    source = SessionSource(
        platform=Platform.WHATSAPP, user_id="15551234567@s.whatsapp.net",
        chat_id="15551234567@s.whatsapp.net", user_name="tester", chat_type="dm",
    )
    return MessageEvent(
        text=text, message_id="m1", source=source, message_type=MessageType.VOICE,
        media_urls=[path], media_types=["audio/ogg"],
    )


def _patch_stt(transcript: str = "lights out"):
    """Patch the STT entry points the transcription mixin calls."""
    return patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": transcript, "provider": "test"},
    )


@pytest.fixture(autouse=True)
def _allow_all(monkeypatch):
    for key in ("WHATSAPP_ALLOWED_USERS", "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")


@pytest.mark.asyncio
async def test_the_hook_receives_the_transcript_not_the_empty_caption():
    """The whole point of the hook: the plugin sees what the user SAID."""
    seen = []

    def consume(**kwargs):
        seen.append(kwargs)

    _register(consume)
    runner = _runner()
    event = _voice_event(text="")

    with _patch_stt("lights out"):
        await runner._enrich_inbound_voice(event, event.source, "", ["/tmp/voice.ogg"], "wa:123")

    assert seen and seen[0]["text"] == "lights out"
    assert seen[0]["source_kind"] == "voice"
    assert seen[0]["message_id"] == "m1"
    # Snapshot-only payload: no live runner or session-store handles.
    assert "gateway" not in seen[0] and "session_store" not in seen[0]


@pytest.mark.asyncio
async def test_a_handled_transcript_skips_the_agent_turn():
    """A consumed voice message is answered by the plugin, not by the model."""
    _register(lambda **_k: {"action": "handled", "reply": "Lights are off."})
    runner = _runner()
    event = _voice_event(text="")

    with _patch_stt("lights out"):
        await runner._enrich_inbound_voice(event, event.source, "", ["/tmp/voice.ogg"], "wa:123")

    assert event._gateway_transcription_consumed is True
    assert event._gateway_transcription_reply == "Lights are off."


@pytest.mark.asyncio
async def test_the_prepared_text_step_returns_the_plugin_reply_instead_of_a_prompt():
    """The turn must not reach the model: the prepare step short-circuits with the reply."""
    _register(lambda **_k: {"action": "handled", "reply": "Lights are off."})
    runner = _runner()
    event = _voice_event(text="")

    with _patch_stt("lights out"):
        prepared = await runner._prepare_inbound_message_text(
            event=event, source=event.source, history=[], session_key="wa:123",
        )

    # None means "no prompt for the model"; the refusal channel carries the plugin's reply.
    assert prepared is None
    assert event._gateway_turn_refusal == "Lights are off."


@pytest.mark.asyncio
async def test_an_unhandled_transcript_leaves_the_turn_alone():
    """A plugin that does not consume the message must not change the turn's behaviour."""
    _register(lambda **_k: None)
    runner = _runner()
    event = _voice_event(text="")

    with _patch_stt("lights out"):
        prepared = await runner._prepare_inbound_message_text(
            event=event, source=event.source, history=[], session_key="wa:123",
        )

    assert prepared is not None
    assert "lights out" in prepared
    assert not getattr(event, "_gateway_transcription_consumed", False)


@pytest.mark.asyncio
async def test_a_raising_transcript_consumer_fails_open_and_the_turn_continues():
    """Fail-open: a buggy plugin must never blackhole the profile's traffic."""
    def broken(**_kwargs):
        raise RuntimeError("plugin bug")

    _register(broken)
    runner = _runner()
    event = _voice_event(text="")

    with _patch_stt("lights out"):
        prepared = await runner._prepare_inbound_message_text(
            event=event, source=event.source, history=[], session_key="wa:123",
        )

    assert prepared is not None
    assert "lights out" in prepared


@pytest.mark.asyncio
async def test_no_transcript_fires_no_hook():
    """A failed transcription has no spoken text, so there is nothing to offer."""
    seen = []
    _register(lambda **kwargs: seen.append(kwargs))
    runner = _runner()
    event = _voice_event(text="")

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": False, "error": "no provider"},
    ), patch(
        "tools.transcription_tools.transcribe_audio_local_fallback",
        return_value={"success": False, "error": "no provider"},
    ):
        await runner._enrich_inbound_voice(event, event.source, "", ["/tmp/voice.ogg"], "wa:123")

    assert seen == []


@pytest.mark.asyncio
async def test_the_existing_admission_hook_payload_is_unchanged():
    """Adding the transcription hook must not alter what ``post_gateway_admission`` receives."""
    from hermes_cli.plugins import get_plugin_manager

    seen = []
    manager = get_plugin_manager()
    manager._discovered = True
    manager._hooks.setdefault("post_gateway_admission", []).append(
        lambda **kwargs: seen.append(kwargs)
    )
    runner = _runner()
    source = SessionSource(
        platform=Platform.WHATSAPP, user_id="15551234567@s.whatsapp.net",
        chat_id="15551234567@s.whatsapp.net", user_name="tester", chat_type="dm",
    )
    event = MessageEvent(text="queue my report", message_id="m1", source=source)

    from gateway.run_inbound_consumer import run_post_admission_hook

    await run_post_admission_hook(runner, event, source, "wa:123")

    assert seen and "source_kind" not in seen[0]
    assert seen[0]["text"] == "queue my report"


@pytest.mark.asyncio
async def test_the_plugin_reply_reaches_the_user_and_the_model_is_not_invoked():
    """The whole point of consuming a message: the plugin answers it, the model does not.

    A double reply — the plugin's reply AND a model reply for the same message — is the worst
    defect this hook could have, so the test asserts BOTH halves: the reply is delivered and
    the model turn never runs.
    """
    _register(lambda **_k: {"action": "handled", "reply": "Lights are off."})
    runner = _runner()
    event = _voice_event(text="")

    # The real turn path up to the model call: transcription happens inside the turn
    # preparation, so a stubbed agent method would skip the hook under test with it.
    model_ran = False

    async def _fake_agent(event_, source, _quick_key, _run_generation):
        nonlocal model_ran
        prepared = await runner._prepare_profile_scoped_inbound_message_text(
            event=event_, source=source, history=[], session_key="wa:123",
        )
        if prepared is None:
            return getattr(event_, "_gateway_turn_refusal", None)
        model_ran = True
        return "model-reply"

    runner._handle_message_with_agent = AsyncMock(side_effect=_fake_agent)

    with _patch_stt("lights out"):
        result = await runner._handle_message(event)

    # The plugin's reply is the answer; the model never ran.
    assert result == "Lights are off."
    assert model_ran is False


@pytest.mark.asyncio
async def test_the_hook_receives_the_turn_session_key_not_a_re_derived_one():
    """A plugin must see the session the turn actually runs in — the same key
    ``post_gateway_admission`` received, not one re-derived from the source."""
    seen = []
    _register(lambda **kwargs: seen.append(kwargs))
    runner = _runner()
    event = _voice_event(text="")

    with _patch_stt("lights out"):
        await runner._prepare_inbound_message_text(
            event=event, source=event.source, history=[], session_key="wa:turn-authority",
        )

    assert seen and seen[0]["session_key"] == "wa:turn-authority"

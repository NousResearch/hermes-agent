"""Byte-stable gateway system prompts (the ephemeral session-context pin).

The composed system prompt used to change bytes nearly every gateway turn:
the "## Current Session Context" block was re-rendered from live platform
state per message (thread renames, voice-channel member/speaking state,
one-shot onboarding and auto-reset notes).  Every byte change re-keys the
provider prompt cache AND changes the gateway agent-cache signature, forcing
a full AIAgent rebuild per message.

The fix pins the rendered session-context bytes per session keyed by a hash
of the exact renderer inputs (``_ephemeral_change_key``) and relocates
must-deliver per-turn facts onto the current user message (the api_content
sidecar), so a key hit reuses the pinned bytes verbatim.

The maintained invariant — every rendered input appears in the change key —
is guarded by the parity test below.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.session import (
    SessionContext,
    SessionSource,
    build_session_context_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner(**attrs):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._session_ephemeral_pin = {}
    runner._session_vc_last = {}
    runner._pending_turn_sidecar_notes = {}
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner.adapters = {}
    runner.session_store = MagicMock()
    for key, value in attrs.items():
        setattr(runner, key, value)
    return runner


def _make_context(
    *,
    platform: Platform = Platform.DISCORD,
    chat_id: str = "111222333",
    chat_name: str = "general",
    chat_type: str = "channel",
    thread_id: str | None = "444555666",
    parent_chat_id: str | None = "111222333",
    chat_topic: str | None = "ops chatter",
    user_name: str | None = "pix",
    user_id: str | None = "9001",
    guild_id: str | None = "777888999",
    message_id: str | None = "1357",
    shared_multi_user: bool = False,
    connected: list[Platform] | None = None,
    home_channels: dict | None = None,
) -> SessionContext:
    source = SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_name=chat_name,
        chat_type=chat_type,
        user_id=user_id,
        user_name=user_name,
        thread_id=thread_id,
        chat_topic=chat_topic,
        parent_chat_id=parent_chat_id,
        scope_id=guild_id,
        message_id=message_id,
    )
    connected = connected if connected is not None else [Platform.DISCORD, Platform.TELEGRAM]
    if home_channels is None:
        home_channels = {
            Platform.DISCORD: HomeChannel(
                platform=Platform.DISCORD, chat_id="111222333", name="general"
            ),
        }
    return SessionContext(
        source=source,
        connected_platforms=connected,
        home_channels=home_channels,
        shared_multi_user_session=shared_multi_user,
    )


@pytest.fixture(autouse=True)
def _stable_discord_tools(monkeypatch):
    """Pin the config/env-dependent renderer gates so key<->render parity is
    evaluated on the same footing in every environment."""
    monkeypatch.setattr("gateway.session._discord_tools_loaded", lambda: True)
    monkeypatch.setattr("gateway.session._slack_tools_loaded", lambda: False)


def _key(runner, context, redact_pii=False):
    return runner._ephemeral_change_key(context, redact_pii)


def _render(context, redact_pii=False):
    return build_session_context_prompt(context, redact_pii=redact_pii)


# ---------------------------------------------------------------------------
# 1. Parity: key <-> render (the maintained invariant)
# ---------------------------------------------------------------------------

class TestEphemeralChangeKeyParity:
    # Single-field mutations spanning every rendered input.  For each:
    # if the rendered bytes change, the key MUST change (staleness guard).
    _MUTATIONS = [
        ("chat_name", dict(chat_name="renamed-thread")),
        ("chat_topic", dict(chat_topic="new topic")),
        ("chat_topic_cleared", dict(chat_topic=None)),
        ("thread_id", dict(thread_id="000111222")),
        ("thread_cleared", dict(thread_id=None, parent_chat_id=None)),
        ("chat_type", dict(chat_type="group")),
        ("user_name", dict(user_name="somebody-else")),
        ("user_name_cleared", dict(user_name=None)),
        ("user_id", dict(user_name=None, user_id="1234")),
        ("shared_multi_user", dict(shared_multi_user=True)),
        ("guild_id", dict(guild_id="123123123")),
        ("parent_chat_id", dict(parent_chat_id="999000111")),
        ("chat_id", dict(chat_id="999999999", parent_chat_id="999999999")),
        ("platform", dict(platform=Platform.TELEGRAM)),
        ("connected_platforms", dict(connected=[Platform.DISCORD])),
        (
            "home_channel_renamed",
            dict(
                home_channels={
                    Platform.DISCORD: HomeChannel(
                        platform=Platform.DISCORD, chat_id="111222333", name="ops-home"
                    )
                }
            ),
        ),
        (
            "home_channel_added",
            dict(
                home_channels={
                    Platform.DISCORD: HomeChannel(
                        platform=Platform.DISCORD, chat_id="111222333", name="general"
                    ),
                    Platform.TELEGRAM: HomeChannel(
                        platform=Platform.TELEGRAM, chat_id="tg1", name="tg-home"
                    ),
                }
            ),
        ),
        ("message_id_cleared", dict(message_id=None)),
    ]


    def test_redact_pii_flip_changes_key(self):
        # PII redaction only rewrites bytes on pii-safe platforms; the key
        # must react wherever the render does.
        runner = _make_runner()
        ctx = _make_context(platform=Platform.TELEGRAM, thread_id=None, parent_chat_id=None)
        assert _render(ctx, False) != _render(ctx, True)
        assert _key(runner, ctx, False) != _key(runner, ctx, True)




# ---------------------------------------------------------------------------
# 2. The pin: reuse verbatim on hit, exactly one legit bust on change
# ---------------------------------------------------------------------------

class TestSessionContextPin:
    def test_pin_hit_returns_identical_object(self):
        runner = _make_runner()
        ctx = _make_context()
        first = runner._pinned_session_context_prompt(ctx, False, "sk")
        second = runner._pinned_session_context_prompt(_make_context(), False, "sk")
        # Identity, not just equality: the pinned bytes are reused verbatim,
        # immunizing against renderer nondeterminism.
        assert second is first



# ---------------------------------------------------------------------------
# 4. Voice-channel sidecar note: only-when-changed
# ---------------------------------------------------------------------------

def _source():
    return SessionSource(
        platform=Platform.DISCORD, chat_id="c1", chat_type="channel", user_id="u1"
    )


class _VcAdapter:
    def __init__(self, value):
        self.value = value

    def get_voice_channel_context(self, guild_id):
        return self.value


def _vc_runner(vc_value):
    adapter = _VcAdapter(vc_value)
    runner = _make_runner(adapters={Platform.DISCORD: adapter})
    return runner, adapter


def _vc_event():
    return SimpleNamespace(raw_message=SimpleNamespace(guild_id="777"))


class TestVoiceChannelSidecarNote:
    def test_first_sighting_injects(self):
        runner, _ = _vc_runner("**Voice:** dev-vc (2 members)")
        note = runner._voice_channel_sidecar_note(_vc_event(), _source(), "sk")
        assert note and "dev-vc (2 members)" in note

    def test_unchanged_state_injects_nothing(self):
        runner, _ = _vc_runner("**Voice:** dev-vc (2 members)")
        assert runner._voice_channel_sidecar_note(_vc_event(), _source(), "sk")
        assert runner._voice_channel_sidecar_note(_vc_event(), _source(), "sk") is None

    def test_member_change_injects_again(self):
        runner, adapter = _vc_runner("**Voice:** dev-vc (2 members)")
        runner._voice_channel_sidecar_note(_vc_event(), _source(), "sk")
        adapter.value = "**Voice:** dev-vc (3 members)"
        note = runner._voice_channel_sidecar_note(_vc_event(), _source(), "sk")
        assert note and "dev-vc (3 members)" in note

    def test_leaving_channel_injects_disconnect_note(self):
        runner, adapter = _vc_runner("**Voice:** dev-vc (2 members)")
        runner._voice_channel_sidecar_note(_vc_event(), _source(), "sk")
        adapter.value = ""
        note = runner._voice_channel_sidecar_note(_vc_event(), _source(), "sk")
        assert note

    def test_never_in_channel_injects_nothing(self):
        runner, _ = _vc_runner("")
        assert runner._voice_channel_sidecar_note(_vc_event(), _source(), "sk") is None

    def test_secondary_profile_reads_its_own_bot_voice_context(self):
        # Multiplexed gateway: the voice binding lives on the adapter whose bot is
        # in the call, so the note must be read from the adapter that answers the
        # source -- not from the default profile's adapter map.
        primary = _VcAdapter("")
        secondary = _VcAdapter("**Voice:** dev-vc (2 members)")
        runner = _make_runner(
            adapters={Platform.DISCORD: primary},
            _profile_adapters={"otto": {Platform.DISCORD: secondary}},
        )
        source = _source()
        source._transport_adapter_ref = lambda: secondary
        note = runner._voice_channel_sidecar_note(_vc_event(), source, "sk")
        assert note and "dev-vc (2 members)" in note


def _multiplex_vc_runner(primary_vc, secondary_vc):
    primary = _VcAdapter(primary_vc)
    secondary = _VcAdapter(secondary_vc)
    runner = _make_runner(adapters={Platform.DISCORD: primary})
    runner._profile_adapters = {"otto": {Platform.DISCORD: secondary}}
    runner.config = SimpleNamespace(multiplex_profiles=True)
    return runner, primary, secondary


def _secondary_source():
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="c1",
        chat_type="channel",
        user_id="u1",
        profile="otto",
    )


class TestVoiceChannelSidecarNoteMultiplexed:
    """#139345: the note must never be read from another bot's call state.

    The two negative halves of the report, ported from the competing draft
    #139359 (closed in favour of this one, whose author offered them there).
    """

    def test_secondary_never_borrows_the_primary_call(self):
        runner, _primary, _secondary = _multiplex_vc_runner(
            "**Voice:** ops-vc (5 members)", ""
        )
        note = runner._voice_channel_sidecar_note(_vc_event(), _secondary_source(), "sk")
        assert note is None

    def test_disconnected_secondary_fails_closed(self):
        runner, _primary, _secondary = _multiplex_vc_runner(
            "**Voice:** ops-vc (5 members)", ""
        )
        runner._profile_adapters = {}
        note = runner._voice_channel_sidecar_note(_vc_event(), _secondary_source(), "sk")
        assert note is None


class _AckAdapter:
    """A Discord bot bound to text channel c1's voice call in guild 77."""

    def __init__(self, bound):
        self._voice_text_channels = {77: "c1"} if bound else {}
        self.acks = []

    def voice_mixer_active(self, guild_id):
        return guild_id in self._voice_text_channels

    async def play_ack_in_voice(self, guild_id):
        self.acks.append(guild_id)


class TestVoiceAckFollowsTheAnsweringBot:
    """#139345, same class: the first-tool-call verbal ack resolves its guild on, and plays through,
    the bot whose call is bound to the turn's text channel, not the default profile's bot."""

    def _runner(self, primary_bound, secondary_bound):
        primary, secondary = _AckAdapter(primary_bound), _AckAdapter(secondary_bound)
        runner = _make_runner(adapters={Platform.DISCORD: primary})
        runner._profile_adapters = {"otto": {Platform.DISCORD: secondary}}
        runner.config = SimpleNamespace(multiplex_profiles=True)
        return runner, primary, secondary

    def test_secondary_in_call_gets_its_guild(self):
        runner, _primary, secondary = self._runner(False, True)
        source = _secondary_source()
        source._transport_adapter_ref = lambda: secondary
        assert runner._voice_ack_guild_for(source) == 77

    def test_secondary_never_borrows_the_primary_call(self):
        runner, _primary, secondary = self._runner(True, False)
        source = _secondary_source()
        source._transport_adapter_ref = lambda: secondary
        assert runner._voice_ack_guild_for(source) is None

    def test_ack_plays_through_the_answering_bot(self):
        from gateway.run_turn_progress import GatewayTurnProgressMixin
        runner, _primary, secondary = self._runner(True, True)
        source = _secondary_source()
        source._transport_adapter_ref = lambda: secondary
        scheduled = []
        progress = SimpleNamespace(
            _runner=runner,
            _ctx=SimpleNamespace(_voice_ack_fired=[False], _voice_ack_guild=[77], _run_still_current=lambda: True,
                                 _voice_ack_loop=None, source=source),
            _schedule=lambda coro, msg, loop=None: scheduled.append(coro),
        )
        GatewayTurnProgressMixin.voice_ack_callback(progress, "call1", "web_search", {})
        assert len(scheduled) == 1
        asyncio.run(scheduled[0])
        assert secondary.acks == [77]
        assert _primary.acks == []


# ---------------------------------------------------------------------------
# 5. Sidecar note staging: one-shot per turn
# ---------------------------------------------------------------------------

class TestSidecarNoteStaging:
    def test_set_then_consume_once(self):
        runner = _make_runner()
        runner._set_pending_turn_sidecar_notes("sk", ["[System note: reset]"])
        assert runner._consume_pending_turn_sidecar_notes("sk") == ["[System note: reset]"]
        assert runner._consume_pending_turn_sidecar_notes("sk") == []


# ---------------------------------------------------------------------------
# 6. Connected platforms: stable order
# ---------------------------------------------------------------------------

class TestConnectedPlatformsOrder:
    def test_sorted_regardless_of_insertion_order(self):
        cfg_a = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="t"),
                Platform.DISCORD: PlatformConfig(enabled=True, token="d"),
            }
        )
        cfg_b = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(enabled=True, token="d"),
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="t"),
            }
        )
        assert cfg_a.get_connected_platforms() == cfg_b.get_connected_platforms()
        values = [p.value for p in cfg_a.get_connected_platforms()]
        assert values == sorted(values)

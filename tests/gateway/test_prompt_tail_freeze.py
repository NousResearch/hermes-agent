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

import hashlib
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
    """Pin the config/env-dependent renderer gate so key<->render parity is
    evaluated on the same footing in every environment."""
    monkeypatch.setattr("gateway.session._discord_tools_loaded", lambda: True)


def _key(runner, context, redact_pii=False):
    return runner._ephemeral_change_key(context, redact_pii)  # noqa: SLF001


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

    @pytest.mark.parametrize("name,mutation", _MUTATIONS)
    def test_render_change_implies_key_change(self, name, mutation):
        runner = _make_runner()
        base = _make_context()
        mutated = _make_context(**mutation)

        render_changed = _render(base) != _render(mutated)
        key_changed = _key(runner, base) != _key(runner, mutated)

        if render_changed:
            assert key_changed, (
                f"mutation {name!r} changed the rendered bytes but not the "
                "change key — the pin would serve STALE context"
            )

    def test_redact_pii_flip_changes_key(self):
        # PII redaction only rewrites bytes on pii-safe platforms; the key
        # must react wherever the render does.
        runner = _make_runner()
        ctx = _make_context(platform=Platform.TELEGRAM, thread_id=None, parent_chat_id=None)
        assert _render(ctx, False) != _render(ctx, True)
        assert _key(runner, ctx, False) != _key(runner, ctx, True)

    def test_discord_tools_gate_flip_changes_key(self, monkeypatch):
        runner = _make_runner()
        ctx = _make_context()
        render_on, key_on = _render(ctx), _key(runner, ctx)
        monkeypatch.setattr("gateway.session._discord_tools_loaded", lambda: False)
        assert _render(ctx) != render_on
        assert _key(runner, ctx) != key_on

    def test_message_id_value_change_is_not_a_bust(self):
        """Only message-id PRESENCE renders (the id itself rides the user
        message) — a new id every turn must not re-render."""
        runner = _make_runner()
        a = _make_context(message_id="1357")
        b = _make_context(message_id="2468")
        assert _render(a) == _render(b)
        assert _key(runner, a) == _key(runner, b)

    def test_key_is_deterministic(self):
        runner = _make_runner()
        ctx = _make_context()
        assert _key(runner, ctx) == _key(runner, ctx)


# ---------------------------------------------------------------------------
# 2. The pin: reuse verbatim on hit, exactly one legit bust on change
# ---------------------------------------------------------------------------

class TestSessionContextPin:
    def test_pin_hit_returns_identical_object(self):
        runner = _make_runner()
        ctx = _make_context()
        first = runner._pinned_session_context_prompt(ctx, False, "sk")  # noqa: SLF001
        second = runner._pinned_session_context_prompt(_make_context(), False, "sk")  # noqa: SLF001
        # Identity, not just equality: the pinned bytes are reused verbatim,
        # immunizing against renderer nondeterminism.
        assert second is first

    def test_auto_thread_rename_busts_exactly_once(self):
        """Turn 1: placeholder title.  Turn 2: gateway auto-rename lands (one
        legit bust — Source line AND origin delivery line move together).
        Turn 3+: byte-stable."""
        runner = _make_runner()
        t1 = runner._pinned_session_context_prompt(  # noqa: SLF001
            _make_context(chat_name="new-chat-1357"), False, "sk"
        )
        t2 = runner._pinned_session_context_prompt(  # noqa: SLF001
            _make_context(chat_name="Fixing the flaky deploy"), False, "sk"
        )
        t3 = runner._pinned_session_context_prompt(  # noqa: SLF001
            _make_context(chat_name="Fixing the flaky deploy"), False, "sk"
        )
        assert t1 != t2
        assert t3 is t2
        assert "Fixing the flaky deploy" in t2

    def test_eviction_drops_pin_and_vc_state(self):
        runner = _make_runner(
            _agent_cache={}, _running_agents={},
        )
        runner._session_ephemeral_pin["sk"] = ("k", "text")
        runner._session_vc_last["sk"] = "vc"
        runner._evict_cached_agent("sk")  # noqa: SLF001
        assert "sk" not in runner._session_ephemeral_pin
        assert "sk" not in runner._session_vc_last

    def test_no_session_key_never_pins(self):
        runner = _make_runner()
        ctx = _make_context()
        out = runner._pinned_session_context_prompt(ctx, False, None)  # noqa: SLF001
        assert out == _render(ctx)
        assert runner._session_ephemeral_pin == {}


# ---------------------------------------------------------------------------
# 3. Two-turn byte test: composed system prompt sha256 + codex cache key
# ---------------------------------------------------------------------------

def _compose(context_prompt: str) -> str:
    """Compose base + ephemeral exactly like conversation_loop does."""
    base = "BASE IDENTITY PROMPT\n" + "x" * 8000
    return (base + "\n\n" + context_prompt).strip()


class TestComposedPromptByteStability:
    def test_turn2_equals_turn3_sha256(self):
        runner = _make_runner()
        name = "Fixing the flaky deploy"
        t2 = _compose(
            runner._pinned_session_context_prompt(  # noqa: SLF001
                _make_context(chat_name=name), False, "sk"
            )
        )
        t3 = _compose(
            runner._pinned_session_context_prompt(  # noqa: SLF001
                _make_context(chat_name=name), False, "sk"
            )
        )
        assert hashlib.sha256(t2.encode()).hexdigest() == hashlib.sha256(t3.encode()).hexdigest()

    def test_codex_cache_key_constant_across_turns(self):
        """The codex transport content-addresses its prompt cache key from
        (instructions + tools); pinned ephemeral bytes keep it warm."""
        from agent.transports.codex import _content_cache_key

        runner = _make_runner()
        tools = [{"type": "function", "name": "read_file"}]
        keys = [
            _content_cache_key(
                _compose(
                    runner._pinned_session_context_prompt(  # noqa: SLF001
                        _make_context(), False, "sk"
                    )
                ),
                tools,
            )
            for _ in range(3)
        ]
        assert keys[0] is not None
        assert len(set(keys)) == 1


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
        note = runner._voice_channel_sidecar_note(_vc_event(), _source(), "sk")  # noqa: SLF001
        assert note == "[Voice channel now: **Voice:** dev-vc (2 members)]"

    def test_unchanged_state_injects_nothing(self):
        runner, _ = _vc_runner("**Voice:** dev-vc (2 members)")
        assert runner._voice_channel_sidecar_note(_vc_event(), _source(), "sk")  # noqa: SLF001
        assert runner._voice_channel_sidecar_note(_vc_event(), _source(), "sk") is None  # noqa: SLF001

    def test_member_change_injects_again(self):
        runner, adapter = _vc_runner("**Voice:** dev-vc (2 members)")
        runner._voice_channel_sidecar_note(_vc_event(), _source(), "sk")  # noqa: SLF001
        adapter.value = "**Voice:** dev-vc (3 members)"
        note = runner._voice_channel_sidecar_note(_vc_event(), _source(), "sk")  # noqa: SLF001
        assert note == "[Voice channel now: **Voice:** dev-vc (3 members)]"

    def test_leaving_channel_injects_disconnect_note(self):
        runner, adapter = _vc_runner("**Voice:** dev-vc (2 members)")
        runner._voice_channel_sidecar_note(_vc_event(), _source(), "sk")  # noqa: SLF001
        adapter.value = ""
        note = runner._voice_channel_sidecar_note(_vc_event(), _source(), "sk")  # noqa: SLF001
        assert note == "[Voice channel now: not connected to a voice channel]"

    def test_never_in_channel_injects_nothing(self):
        runner, _ = _vc_runner("")
        assert runner._voice_channel_sidecar_note(_vc_event(), _source(), "sk") is None  # noqa: SLF001

    def test_non_discord_platform_is_noop(self):
        runner, _ = _vc_runner("**Voice:** dev-vc")
        src = SessionSource(platform=Platform.TELEGRAM, chat_id="c", user_id="u")
        assert runner._voice_channel_sidecar_note(_vc_event(), src, "sk") is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# 5. Sidecar note staging: one-shot per turn
# ---------------------------------------------------------------------------

class TestSidecarNoteStaging:
    def test_set_then_consume_once(self):
        runner = _make_runner()
        runner._set_pending_turn_sidecar_notes("sk", ["[System note: reset]"])  # noqa: SLF001
        assert runner._consume_pending_turn_sidecar_notes("sk") == ["[System note: reset]"]  # noqa: SLF001
        assert runner._consume_pending_turn_sidecar_notes("sk") == []  # noqa: SLF001

    def test_empty_inputs_are_noops(self):
        runner = _make_runner()
        runner._set_pending_turn_sidecar_notes("", ["x"])  # noqa: SLF001
        runner._set_pending_turn_sidecar_notes("sk", [])  # noqa: SLF001
        assert runner._consume_pending_turn_sidecar_notes("sk") == []  # noqa: SLF001
        assert runner._consume_pending_turn_sidecar_notes("") == []  # noqa: SLF001


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

"""A voice-channel turn must render the same session-context system prompt as a text message
from the same person in the same channel. It used to differ in two places: ``**User:**`` (and the
``**Source:**`` description) carried the bare Discord user ID instead of the display name, and the
Discord note's "Triggering message" line was only rendered when the turn had a message id. Either
difference changed the top of the system prompt, so every switch between typing and talking
re-prefilled the whole conversation (a 90K-token cache miss: 24 s for a one-word reply).
"""

import pytest

from gateway.config import HomeChannel, Platform
from gateway.run_voice import GatewayVoiceMixin
from gateway.session import SessionContext, SessionSource, build_session_context_prompt

GUILD, TEXT, USER, FRIEND = 10, 30, 42, 43


@pytest.fixture(autouse=True)
def _discord_tools(monkeypatch):
    monkeypatch.setattr("gateway.session._discord_tools_loaded", lambda: True)


class FakeAdapter:
    """What ``_voice_input_source`` reads from the Discord adapter (weakref-able, unlike SimpleNamespace)."""

    _owner_profile = None

    def __init__(self, bound: SessionSource, names: dict):
        self._voice_sources = {GUILD: bound.to_dict()}
        self._names = names

    def voice_member_display_name(self, guild_id, user_id):
        return self._names.get(int(user_id))


def _channel_source(**overrides) -> SessionSource:
    """#iris as the Discord adapter builds it for a text message or a /voice join there."""
    fields = dict(platform=Platform.DISCORD, chat_id=str(TEXT), chat_name="HomeLab / #iris",
                  chat_type="group", user_id=str(USER), user_name="Matt", chat_topic="Iris",
                  guild_id=str(GUILD))
    fields.update(overrides)
    return SessionSource(**fields)


def _render(source: SessionSource) -> str:
    return build_session_context_prompt(SessionContext(
        source=source, connected_platforms=[Platform.LOCAL, Platform.DISCORD],
        home_channels={Platform.DISCORD: HomeChannel(platform=Platform.DISCORD, chat_id=str(TEXT), name="iris")},
    ))


def _voice_source(adapter, user_id=USER) -> SessionSource:
    return GatewayVoiceMixin._voice_input_source(adapter, GUILD, user_id, TEXT)


def test_voice_turn_renders_the_same_prompt_as_a_text_turn():
    text = _channel_source(message_id="1555000000000000000")
    # A typed `/voice join` binds that message's source, id included: the voice turn must not
    # inherit it as its own trigger.
    voice = _voice_source(FakeAdapter(_channel_source(message_id="1554000000000000000"), {USER: "Matt"}))
    assert voice.user_name == "Matt" and voice.message_id is None
    assert _render(voice) == _render(text)


def test_ephemeral_change_key_matches_too():
    """Same key: the pinned session prompt is reused, not even re-rendered, across the switch."""
    from gateway.run import GatewayRunner
    text = _channel_source(message_id="1555000000000000000")
    voice = _voice_source(FakeAdapter(_channel_source(), {USER: "Matt"}))
    ctx = lambda src: SessionContext(source=src, connected_platforms=[Platform.DISCORD], home_channels={})  # noqa: E731
    assert GatewayRunner._ephemeral_change_key(ctx(voice), False) == GatewayRunner._ephemeral_change_key(ctx(text), False)


def test_triggering_message_note_does_not_depend_on_a_message_id():
    with_id, without = _render(_channel_source(message_id="1")), _render(_channel_source())
    assert with_id == without and "Triggering message: when the turn comes from a Discord message" in with_id


def test_another_speaker_gets_their_own_display_name():
    voice = _voice_source(FakeAdapter(_channel_source(), {USER: "Matt", FRIEND: "Sam"}), user_id=FRIEND)
    assert (voice.user_id, voice.user_name) == (str(FRIEND), "Sam")


@pytest.mark.parametrize("speaker,expected", [(USER, "Matt"), (FRIEND, str(FRIEND))])
def test_without_a_lookup_falls_back_to_the_bound_name_or_the_id(speaker, expected):
    """Member not cached: the /voice join invoker keeps the name bound at join; anyone else gets the ID."""
    voice = _voice_source(FakeAdapter(_channel_source(), {}), user_id=speaker)
    assert voice.user_name == expected


def test_discord_adapter_looks_up_the_member_display_name():
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from plugins.platforms.discord.adapter import DiscordAdapter
    adapter = object.__new__(DiscordAdapter)
    adapter._client = MagicMock()
    members = {USER: SimpleNamespace(display_name="Matt")}
    users = {USER: SimpleNamespace(display_name="matt_global"), FRIEND: SimpleNamespace(display_name="Sam")}
    adapter._client.get_guild.return_value = SimpleNamespace(get_member=members.get)
    adapter._client.get_user.side_effect = users.get
    assert adapter.voice_member_display_name(GUILD, USER) == "Matt"  # member cached: guild nick wins
    assert adapter.voice_member_display_name(GUILD, FRIEND) == "Sam"  # member miss, global user cached
    assert adapter.voice_member_display_name(GUILD, 44) is None  # cached nowhere
    adapter._client.get_guild.return_value = None
    assert adapter.voice_member_display_name(GUILD, USER) == "matt_global"

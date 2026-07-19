"""Focused hardening coverage for the three Discord ingress gates.

Pins three policy-safe blockers across the live-admission
(``_discord_message_admission``), recovered pre-gate
(``_dispatch_recovered_message``) and final ``_handle_message`` gates:

* P0-1  The final ``_handle_message`` mention gate honors an in-scope
        configured group-role mention (humans only). A sibling bot must
        still type a real inline self-mention -- a role mention alone
        never hands off to a bot.
* P0-2  A voice-linked text channel keeps its free-response exemption on
        every gate, not only inside ``_handle_message``. The exemption is
        exact-channel-only: a thread under a voice-linked parent still
        requires a mention.
* P1-3  ``allow_bots`` is a strict fail-closed enum ``{mentions, all}``.
        Any other value -- ``none``, ``false``, empty, or a typo -- denies.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    """Install a mock discord module when discord.py isn't available."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(
        View=object, button=lambda *a, **k: lambda fn: fn, Button=object
    )
    discord_mod.ButtonStyle = SimpleNamespace(
        success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3
    )
    discord_mod.Color = SimpleNamespace(
        orange=lambda: 1,
        green=lambda: 2,
        blue=lambda: 3,
        red=lambda: 4,
        purple=lambda: 5,
    )
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.Object = lambda *, id: SimpleNamespace(id=id)
    discord_mod.Message = type("Message", (), {})
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: lambda fn: fn,
        choices=lambda **kwargs: lambda fn: fn,
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


BOT_ID = 999
AUTHOR_ID = 42
OPS_ID = 100  # non-free parent (mention required)
VOICE_CH_ID = 555  # a text channel bound to an active voice session
ROLE_ID = 777
GUILD_ID = 1


class FakeDMChannel:
    def __init__(self, channel_id: int = 1, name: str = "dm"):
        self.id = channel_id
        self.name = name


class FakeTextChannel:
    def __init__(self, channel_id: int, name: str = "general"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(id=GUILD_ID, name="Hermes Server")
        self.topic = None


class FakeThread:
    def __init__(self, channel_id: int, name: str = "thread", parent=None):
        self.id = channel_id
        self.name = name
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.guild = getattr(parent, "guild", None) or SimpleNamespace(
            id=GUILD_ID, name="Hermes Server"
        )
        self.topic = None


def _install_channel_types(monkeypatch):
    monkeypatch.setattr(
        discord_platform.discord, "DMChannel", FakeDMChannel, raising=False
    )
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)


def _clear_discord_env(monkeypatch):
    for _var in (
        "DISCORD_REQUIRE_MENTION",
        "DISCORD_THREAD_REQUIRE_MENTION",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_ALLOW_BOTS",
        "DISCORD_BOTS_REQUIRE_INLINE_MENTION",
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_IGNORED_CHANNELS",
        "DISCORD_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "DISCORD_IGNORE_NO_MENTION",
        "DISCORD_AUTO_THREAD",
        "DISCORD_NO_THREAD_CHANNELS",
    ):
        monkeypatch.delenv(_var, raising=False)


@pytest.fixture
def admission_adapter(monkeypatch):
    """Adapter for the two pre-``_handle_message`` gates.

    ``_handle_message`` is stubbed so an admit is observable as a bool /
    awaited call without dragging in the downstream handler.
    """
    _install_channel_types(monkeypatch)
    _clear_discord_env(monkeypatch)

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=BOT_ID, bot=True))
    adapter._allowed_user_ids = {str(AUTHOR_ID)}
    adapter._text_batch_delay_seconds = 0
    adapter._handle_message = AsyncMock(return_value=True)
    return adapter


@pytest.fixture
def handle_adapter(monkeypatch):
    """Adapter that runs the REAL ``_handle_message`` gate end-to-end.

    ``handle_message`` (the downstream dispatch callback) is stubbed, so an
    admit is observable as ``handle_message`` being awaited.
    """
    _install_channel_types(monkeypatch)
    _clear_discord_env(monkeypatch)
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "true")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "false")

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=BOT_ID, bot=True))
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    # Keep these gate tests focused on admission, not on the history-backfill
    # I/O that a mention-gap admit would otherwise trigger.
    adapter._fetch_channel_context = AsyncMock(return_value=None)
    return adapter


def make_message(
    *,
    channel,
    content: str = "hello",
    mentions=None,
    bot: bool = False,
    msg_id: int = 123,
):
    author = SimpleNamespace(
        id=(BOT_ID + 1 if bot else AUTHOR_ID),
        display_name="Author",
        name="Author",
        bot=bot,
    )
    is_dm = isinstance(channel, FakeDMChannel)
    return SimpleNamespace(
        id=msg_id,
        content=content,
        mentions=list(mentions or []),
        attachments=[],
        reference=None,
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=author,
        guild=None if is_dm else channel.guild,
        type=discord_platform.discord.MessageType.default,
    )


def _voice_channel() -> FakeTextChannel:
    return FakeTextChannel(VOICE_CH_ID, "voice-text")


def _ops_channel() -> FakeTextChannel:
    return FakeTextChannel(OPS_ID, "ops")


def _admitted(adapter, message) -> bool:
    admitted, _role = adapter._discord_message_admission(message, claim=True)
    return admitted


# ---------------------------------------------------------------------------
# P0-2 voice-linked exemption -- live admission gate
# ---------------------------------------------------------------------------


def test_admission_admits_voice_linked_channel_no_mention(admission_adapter):
    """A no-mention message in an active voice-linked channel is admitted."""
    admission_adapter._voice_text_channels = {GUILD_ID: VOICE_CH_ID}
    message = make_message(
        channel=_voice_channel(), content="follow-up from voice chat, no mention"
    )

    assert _admitted(admission_adapter, message) is True


def test_admission_denies_voice_linked_parent_thread_no_mention(admission_adapter):
    """The exemption is exact-channel-only: a thread under a voice-linked
    parent still requires a mention on the live gate."""
    admission_adapter._voice_text_channels = {GUILD_ID: VOICE_CH_ID}
    thread = FakeThread(790, name="topic", parent=_voice_channel())
    message = make_message(channel=thread, content="thread reply, no mention")

    assert _admitted(admission_adapter, message) is False


def test_admission_denies_nonvoice_channel_no_mention(admission_adapter):
    """Control: with no voice link, a no-mention channel message is denied."""
    admission_adapter._voice_text_channels = {}
    message = make_message(channel=_voice_channel(), content="no mention, no voice")

    assert _admitted(admission_adapter, message) is False


# ---------------------------------------------------------------------------
# P0-2 voice-linked exemption -- recovered pre-gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recovered_admits_voice_linked_channel_no_mention(admission_adapter):
    """A recovered no-mention message in a voice-linked channel is admitted."""
    admission_adapter._voice_text_channels = {GUILD_ID: VOICE_CH_ID}
    message = make_message(
        channel=_voice_channel(), content="recovered voice-chat follow-up"
    )

    admitted = await admission_adapter._dispatch_recovered_message(message)

    assert admitted is True
    admission_adapter._handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_recovered_denies_voice_linked_parent_thread_no_mention(
    admission_adapter,
):
    """Recovered exemption is also exact-channel-only."""
    admission_adapter._voice_text_channels = {GUILD_ID: VOICE_CH_ID}
    thread = FakeThread(790, name="topic", parent=_voice_channel())
    message = make_message(channel=thread, content="recovered thread reply, no mention")

    admitted = await admission_adapter._dispatch_recovered_message(message)

    assert admitted is False
    admission_adapter._handle_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# P0-2 voice-linked exemption -- final _handle_message gate (still honored)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_admits_voice_linked_channel_no_mention(handle_adapter):
    """The final gate keeps the voice-linked free-response exemption."""
    handle_adapter._voice_text_channels = {GUILD_ID: VOICE_CH_ID}
    message = make_message(channel=_voice_channel(), content="voice text follow-up")

    await handle_adapter._handle_message(message)

    handle_adapter.handle_message.assert_awaited_once()


# ---------------------------------------------------------------------------
# P0-1 group-role mention -- final _handle_message gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_admits_configured_group_role_mention_in_scope(handle_adapter):
    """A human in-scope group-role mention passes the final gate.

    The earlier admission gate already honors this; the final gate must
    agree, otherwise a valid handoff is admitted then silently dropped.
    """
    handle_adapter.config.extra["group_mention_role_ids"] = [str(ROLE_ID)]
    handle_adapter.config.extra["group_mention_channel_ids"] = [str(OPS_ID)]
    message = make_message(
        channel=_ops_channel(),
        content=f"<@&{ROLE_ID}> hermes team, handoff",
    )

    await handle_adapter._handle_message(message)

    handle_adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_denies_group_role_mention_out_of_scope(handle_adapter):
    """An out-of-scope group-role mention is denied by the final gate."""
    handle_adapter.config.extra["group_mention_role_ids"] = [str(ROLE_ID)]
    handle_adapter.config.extra["group_mention_channel_ids"] = ["999999"]
    message = make_message(
        channel=_ops_channel(),
        content=f"<@&{ROLE_ID}> hermes team, handoff",
    )

    await handle_adapter._handle_message(message)

    handle_adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_denies_bot_role_only_handoff(handle_adapter, monkeypatch):
    """A sibling bot must NOT be handed off by a role mention alone.

    Even with ``allow_bots=all`` and the role configured in-scope, a bot that
    did not type a real inline self-mention is denied at the final gate.
    """
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    handle_adapter.config.extra["group_mention_role_ids"] = [str(ROLE_ID)]
    handle_adapter.config.extra["group_mention_channel_ids"] = [str(OPS_ID)]
    message = make_message(
        channel=_ops_channel(),
        content=f"<@&{ROLE_ID}> sibling bot, no self-mention",
        bot=True,
    )

    await handle_adapter._handle_message(message)

    handle_adapter.handle_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# P1-3 allow_bots strict fail-closed enum {mentions, all}
# ---------------------------------------------------------------------------


def test_admission_admits_bot_allow_bots_all(admission_adapter):
    """Positive control: allow_bots=all admits a bot."""
    admission_adapter.config.extra["allow_bots"] = "all"
    message = make_message(channel=_ops_channel(), content="bot chatter", bot=True)

    assert _admitted(admission_adapter, message) is True


def test_admission_admits_bot_allow_bots_mentions_with_self_mention(admission_adapter):
    """Positive control: allow_bots=mentions admits a bot that self-mentions."""
    admission_adapter.config.extra["allow_bots"] = "mentions"
    message = make_message(
        channel=_ops_channel(),
        content=f"<@{BOT_ID}> deliberate handoff",
        mentions=[admission_adapter._client.user],
        bot=True,
    )

    assert _admitted(admission_adapter, message) is True


def test_admission_denies_bot_allow_bots_mentions_without_mention(admission_adapter):
    """allow_bots=mentions denies a bot with no self-mention."""
    admission_adapter.config.extra["allow_bots"] = "mentions"
    message = make_message(channel=_ops_channel(), content="bot chatter", bot=True)

    assert _admitted(admission_adapter, message) is False


def test_admission_denies_bot_allow_bots_mentions_group_role_only(admission_adapter):
    """allow_bots=mentions denies a bot that only pings the in-scope group role.

    A configured group-role ping is treated as an explicit self-mention for a
    human author (so co-located bots wake), but a sibling bot must still type a
    real inline self-mention. A role-only ping must never hand off to a bot.
    """
    admission_adapter.config.extra["allow_bots"] = "mentions"
    admission_adapter.config.extra["group_mention_role_ids"] = [str(ROLE_ID)]
    admission_adapter.config.extra["group_mention_channel_ids"] = [str(OPS_ID)]
    message = make_message(
        channel=_ops_channel(),
        content=f"<@&{ROLE_ID}> sibling bot, no self-mention",
        bot=True,
    )

    assert _admitted(admission_adapter, message) is False


def test_admission_denies_bot_allow_bots_false_even_with_self_mention(
    admission_adapter,
):
    """allow_bots="false" is not a valid enum value: fail closed.

    A YAML ``allow_bots: false`` serializes to "false" and must deny, even
    when the bot types a real self-mention -- not fall through to "all".
    """
    admission_adapter.config.extra["allow_bots"] = "false"
    message = make_message(
        channel=_ops_channel(),
        content=f"<@{BOT_ID}> self-mention but disallowed",
        mentions=[admission_adapter._client.user],
        bot=True,
    )

    assert _admitted(admission_adapter, message) is False


def test_admission_denies_bot_allow_bots_typo_even_with_self_mention(admission_adapter):
    """A typo (e.g. "mention") is not a valid enum value: fail closed."""
    admission_adapter.config.extra["allow_bots"] = "mention"
    message = make_message(
        channel=_ops_channel(),
        content=f"<@{BOT_ID}> self-mention but typo enum",
        mentions=[admission_adapter._client.user],
        bot=True,
    )

    assert _admitted(admission_adapter, message) is False


def test_admission_denies_bot_allow_bots_empty_even_with_self_mention(
    admission_adapter,
):
    """An empty value is not a valid enum value: fail closed."""
    admission_adapter.config.extra["allow_bots"] = ""
    message = make_message(
        channel=_ops_channel(),
        content=f"<@{BOT_ID}> self-mention but empty enum",
        mentions=[admission_adapter._client.user],
        bot=True,
    )

    assert _admitted(admission_adapter, message) is False


@pytest.mark.asyncio
async def test_recovered_denies_bot_allow_bots_false(admission_adapter):
    """The fail-closed enum also holds on the recovered path."""
    admission_adapter.config.extra["allow_bots"] = "false"
    message = make_message(
        channel=_ops_channel(),
        content=f"<@{BOT_ID}> recovered self-mention but disallowed",
        mentions=[admission_adapter._client.user],
        bot=True,
    )

    admitted = await admission_adapter._dispatch_recovered_message(message)

    assert admitted is False
    admission_adapter._handle_message.assert_not_awaited()

"""Free-response channels must not be silenced by a quoted other-bot mention.

``_discord_message_admission`` suppresses any non-DM message that mentions
another bot but not Hermes. That guard is correct for normal channels, but a
free-response channel is *explicitly configured* to answer without being
mentioned at all — and such channels routinely carry quoted bot mentions from
migration notes and prior context. Applying the suppression there dropped
messages the channel was configured to answer.

The direct-address case (the message *begins* with another bot's mention) stays
suppressed, so this does not turn Hermes into a cross-bot interloper.
"""

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
        View=object, button=lambda *a, **k: (lambda fn: fn), Button=object
    )
    discord_mod.ButtonStyle = SimpleNamespace(
        success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3
    )
    discord_mod.Color = SimpleNamespace(
        orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5
    )
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.Object = lambda *, id: SimpleNamespace(id=id)
    discord_mod.Message = type("Message", (), {})
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
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

SELF_ID = 999
OTHER_BOT_ID = 555


class FakeDMChannel:
    def __init__(self, channel_id: int = 1):
        self.id = channel_id
        self.name = "dm"


class FakeTextChannel:
    def __init__(self, channel_id: int = 789, name: str = "agents"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name="Hermes Server")
        self.topic = None


class FakeThread:
    def __init__(self, channel_id: int = 4242, parent=None):
        self.id = channel_id
        self.name = "thread"
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.guild = getattr(parent, "guild", None) or SimpleNamespace(name="Hermes Server")
        self.topic = None


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDMChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)

    for var in (
        "DISCORD_REQUIRE_MENTION",
        "DISCORD_THREAD_REQUIRE_MENTION",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_IGNORE_NO_MENTION",
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_IGNORED_CHANNELS",
        "DISCORD_ALLOW_BOTS",
    ):
        monkeypatch.delenv(var, raising=False)

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=SELF_ID, bot=True))
    adapter.handle_message = AsyncMock()
    # Admission is what we're testing; keep the user/role gate out of the way.
    monkeypatch.setattr(adapter, "_is_allowed_user", lambda *a, **k: True)
    return adapter


def make_message(*, channel, content, mentions=None):
    return SimpleNamespace(
        id=123,
        content=content,
        mentions=list(mentions or []),
        attachments=[],
        reference=None,
        channel=channel,
        type=discord_platform.discord.MessageType.default,
        author=SimpleNamespace(id=42, display_name="Jezza", name="Jezza", bot=False),
    )


def other_bot():
    return SimpleNamespace(id=OTHER_BOT_ID, bot=True, name="OtherBot")


def _free(adapter, *channel_ids):
    adapter.config.extra["free_response_channels"] = [str(c) for c in channel_ids]


def test_free_response_channel_allows_an_inline_other_bot_mention(adapter):
    """The bug: a quoted bot mention silenced a channel configured to answer."""
    _free(adapter, 789)
    channel = FakeTextChannel(channel_id=789)
    message = make_message(
        channel=channel,
        content=f"earlier we used to ping <@{OTHER_BOT_ID}> for this — how do we do it now?",
        mentions=[other_bot()],
    )

    admitted, _ = adapter._discord_message_admission(message, claim=False)

    assert admitted is True


def test_free_response_channel_still_yields_on_direct_address(adapter):
    """A message that BEGINS with another bot's mention is addressed to it."""
    _free(adapter, 789)
    channel = FakeTextChannel(channel_id=789)
    message = make_message(
        channel=channel,
        content=f"<@{OTHER_BOT_ID}> please summarize the thread",
        mentions=[other_bot()],
    )

    admitted, _ = adapter._discord_message_admission(message, claim=False)

    assert admitted is False


def test_free_response_channel_yields_on_the_nickname_mention_form(adapter):
    """Discord's legacy ``<@!ID>`` nickname form is the same direct address."""
    _free(adapter, 789)
    channel = FakeTextChannel(channel_id=789)
    message = make_message(
        channel=channel,
        content=f"<@!{OTHER_BOT_ID}> please summarize the thread",
        mentions=[other_bot()],
    )

    admitted, _ = adapter._discord_message_admission(message, claim=False)

    assert admitted is False


def test_normal_channel_still_ignores_other_bot_mentions(adapter):
    """The suppression is unchanged outside free-response channels."""
    adapter.config.extra["free_response_channels"] = []
    channel = FakeTextChannel(channel_id=789)
    message = make_message(
        channel=channel,
        content=f"we used to ping <@{OTHER_BOT_ID}> for this",
        mentions=[other_bot()],
    )

    admitted, _ = adapter._discord_message_admission(message, claim=False)

    assert admitted is False


def test_free_response_thread_inherits_its_parent_channel(adapter):
    """A thread under a free-response parent gets the same treatment."""
    _free(adapter, 789)
    parent = FakeTextChannel(channel_id=789)
    thread = FakeThread(channel_id=4242, parent=parent)
    message = make_message(
        channel=thread,
        content=f"context: the old bot was <@{OTHER_BOT_ID}>",
        mentions=[other_bot()],
    )

    admitted, _ = adapter._discord_message_admission(message, claim=False)

    assert admitted is True


def test_wildcard_free_response_also_allows_inline_mentions(adapter):
    """``*`` means every channel is free-response, including for this guard."""
    _free(adapter, "*")
    channel = FakeTextChannel(channel_id=31337)
    message = make_message(
        channel=channel,
        content=f"the old bot was <@{OTHER_BOT_ID}>",
        mentions=[other_bot()],
    )

    admitted, _ = adapter._discord_message_admission(message, claim=False)

    assert admitted is True


def test_self_mention_is_admitted_everywhere_as_before(adapter):
    """An explicit @Hermes was always admitted; that must not regress."""
    adapter.config.extra["free_response_channels"] = []
    channel = FakeTextChannel(channel_id=789)
    message = make_message(
        channel=channel,
        content=f"<@{SELF_ID}> hi (and <@{OTHER_BOT_ID}> was the old one)",
        mentions=[other_bot(), SimpleNamespace(id=SELF_ID, bot=True)],
    )

    admitted, _ = adapter._discord_message_admission(message, claim=False)

    assert admitted is True


def test_no_mention_in_a_normal_channel_is_still_ignored(adapter):
    """The unrelated ignore_no_mention guard keeps its behavior."""
    adapter.config.extra["free_response_channels"] = []
    channel = FakeTextChannel(channel_id=789)
    message = make_message(
        channel=channel,
        content="just talking to myself",
        mentions=[SimpleNamespace(id=77, bot=False, name="Human")],
    )

    admitted, _ = adapter._discord_message_admission(message, claim=False)

    assert admitted is False


def test_a_disconnected_client_does_not_classify_every_bot_as_another_bot(adapter):
    """With no client, we cannot recognize ourselves — so admit nothing.

    ``self_user`` used to be computed as
    ``self._client.user if self._client is not None else None``. When the
    client is absent that yields ``None``, and ``mentioned != self_user`` is
    then true for EVERY bot in ``message.mentions`` — including Hermes itself.
    A message that mentions only Hermes would be reclassified as an
    "other bot" mention and silently suppressed, which inverts the admission
    decision this module exists to get right.

    The guard now short-circuits instead of guessing. Main reaches
    ``self._client.user`` directly and would raise ``AttributeError`` here;
    returning ``(False, False)`` states the invariant without depending on an
    exception escaping the ingress path.
    """
    _free(adapter, 789)
    adapter._client = None
    channel = FakeTextChannel(channel_id=789)
    # Mentions ONLY Hermes. With the old fallback this became an "other bot"
    # mention because SELF_ID != None.
    message = make_message(
        channel=channel,
        content=f"<@{SELF_ID}> what is the status?",
        mentions=[SimpleNamespace(id=SELF_ID, bot=True, name="Hermes")],
    )

    admitted, role_authorized = adapter._discord_message_admission(message, claim=False)

    assert admitted is False
    assert role_authorized is False


def test_a_connected_client_still_recognizes_its_own_mention(adapter):
    """Companion to the guard above: the normal path must be unchanged.

    Guards that short-circuit are easy to over-apply, so pin the positive
    case — a live client seeing only its own mention admits the message and
    must NOT treat itself as another bot.
    """
    _free(adapter, 789)
    channel = FakeTextChannel(channel_id=789)
    message = make_message(
        channel=channel,
        content=f"<@{SELF_ID}> what is the status?",
        mentions=[SimpleNamespace(id=SELF_ID, bot=True, name="Hermes")],
    )

    admitted, _ = adapter._discord_message_admission(message, claim=False)

    assert admitted is True

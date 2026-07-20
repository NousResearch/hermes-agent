"""Regression coverage for Discord thread mention-admission policy.

These tests pin the two authoritative ingress gates that the live and
recovered dispatch paths run *before* ``_handle_message``:

* ``_discord_message_admission`` (live path, ``_dispatch_discord_message``)
* ``_dispatch_recovered_message`` (missed-message backfill / recovered path)

Shared policy under test (intended config: ``require_mention=true``,
``thread_require_mention=false``; a non-free parent like ``#ops`` and a
free-response parent like ``#lounge``):

    A human no-mention message in a thread inherits ONLY its parent's
    free-response / default-bot policy. Bot *participation* in a thread must
    NOT bypass a non-free parent's mention requirement.

The historical ``in_bot_thread`` bypass -- which let any thread the bot had
previously spoken in skip the mention gate -- was removed from these two gates
on purpose. This file proves the removal holds and that the surrounding
admission behaviour is preserved, so the bypass is not silently reintroduced.
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
        View=object, button=lambda *a, **k: (lambda fn: fn), Button=object
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


BOT_ID = 999
AUTHOR_ID = 42
OPS_ID = 100  # non-free parent (e.g. #ops)
LOUNGE_ID = 200  # free-response parent (e.g. #lounge)
ROLE_ID = 777


class FakeDMChannel:
    def __init__(self, channel_id: int = 1, name: str = "dm"):
        self.id = channel_id
        self.name = name


class FakeTextChannel:
    def __init__(self, channel_id: int, name: str = "general"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(id=1, name="Hermes Server")
        self.topic = None


class FakeThread:
    def __init__(self, channel_id: int, name: str = "thread", parent=None):
        self.id = channel_id
        self.name = name
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.guild = getattr(parent, "guild", None) or SimpleNamespace(
            id=1, name="Hermes Server"
        )
        self.topic = None


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(
        discord_platform.discord, "DMChannel", FakeDMChannel, raising=False
    )
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)

    # Isolate from the contributor's shell: every DISCORD_* knob this file
    # exercises must come from config.extra / per-test env, not leaked state.
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
    ):
        monkeypatch.delenv(_var, raising=False)

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=BOT_ID, bot=True))
    adapter._allowed_user_ids = {str(AUTHOR_ID)}
    adapter._text_batch_delay_seconds = 0
    # The two gates under test run before _handle_message. Stub it so a
    # recovered admit is observable as a bool / awaited call without dragging
    # in the full downstream handler (which has its own, separate gate).
    adapter._handle_message = AsyncMock(return_value=True)
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


def _ops_thread(thread_id: int = 456) -> FakeThread:
    """A thread whose parent is a non-free channel (mention required)."""
    return FakeThread(thread_id, name="topic", parent=FakeTextChannel(OPS_ID, "ops"))


def _lounge_thread(thread_id: int = 456) -> FakeThread:
    """A thread whose parent is a free-response channel (no mention required)."""
    return FakeThread(
        thread_id, name="topic", parent=FakeTextChannel(LOUNGE_ID, "lounge")
    )


def _free_config(adapter):
    adapter.config.extra["free_response_channels"] = [str(LOUNGE_ID)]


def _admitted(adapter, message) -> bool:
    admitted, _role_authorized = adapter._discord_message_admission(message, claim=True)
    return admitted


# ---------------------------------------------------------------------------
# Admission path (_discord_message_admission)
# ---------------------------------------------------------------------------


def test_admission_denies_participated_nonfree_parent_thread_no_mention(adapter):
    """(1) Participation does NOT bypass a non-free parent on the live gate."""
    _free_config(adapter)
    adapter._threads.mark("456")  # bot has previously spoken here
    message = make_message(
        channel=_ops_thread(), content="ambient follow-up, no mention"
    )

    assert _admitted(adapter, message) is False


def test_admission_admits_participated_free_parent_thread_no_mention(adapter):
    """(2) A thread under a free-response parent inherits its free policy."""
    _free_config(adapter)
    adapter._threads.mark("456")
    message = make_message(channel=_lounge_thread(), content="casual chat, no mention")

    assert _admitted(adapter, message) is True


def test_admission_denies_unparticipated_nonfree_thread_no_mention(adapter):
    """(3) An unparticipated thread under a non-free parent still requires @mention."""
    _free_config(adapter)
    # thread 789 is NOT marked as participated
    message = make_message(
        channel=_ops_thread(789), content="hello from unknown thread"
    )

    assert _admitted(adapter, message) is False


def test_admission_thread_require_mention_denies_participated_nonfree_no_mention(
    adapter,
):
    """(4) thread_require_mention=true keeps a participated non-free thread gated.

    Enabling the stricter knob must not loosen anything: a no-mention
    follow-up in a participated thread under a non-free parent stays denied.
    """
    _free_config(adapter)
    adapter.config.extra["thread_require_mention"] = True
    adapter._threads.mark("456")
    message = make_message(channel=_ops_thread(), content="ambient chatter, not for me")

    assert _admitted(adapter, message) is False


def test_admission_free_parent_inheritance_survives_thread_require_mention(adapter):
    """(4, companion) Free-response inheritance is by-parent and is NOT

    overridden by thread_require_mention on this gate. This pins the current,
    policy-consistent behaviour ("a thread inherits ONLY its parent
    free-response policy"): a thread under a free parent is still admitted
    without a mention even when thread_require_mention=true. If the shared
    policy ever changes to make thread_require_mention override a free parent,
    THIS is the test that must be updated deliberately.
    """
    _free_config(adapter)
    adapter.config.extra["thread_require_mention"] = True
    adapter._threads.mark("456")
    message = make_message(channel=_lounge_thread(), content="casual chat, no mention")

    assert _admitted(adapter, message) is True


def test_admission_admits_direct_self_mention_in_nonfree_thread(adapter):
    """(5) An explicit @bot mention is admitted even under a non-free parent."""
    _free_config(adapter)
    message = make_message(
        channel=_ops_thread(),
        content=f"<@{BOT_ID}> please look at this",
        mentions=[adapter._client.user],
    )

    assert _admitted(adapter, message) is True


def test_admission_admits_configured_group_role_mention_in_scope(adapter):
    """(5) A configured group-role mention in a configured channel is admitted."""
    _free_config(adapter)
    adapter.config.extra["group_mention_role_ids"] = [str(ROLE_ID)]
    adapter.config.extra["group_mention_channel_ids"] = [str(OPS_ID)]
    message = make_message(
        channel=_ops_thread(),
        content=f"<@&{ROLE_ID}> hermes team, handoff",
    )

    assert _admitted(adapter, message) is True


def test_admission_denies_group_role_mention_out_of_configured_scope(adapter):
    """(5, negative) A group-role mention outside its configured channels is denied."""
    _free_config(adapter)
    adapter.config.extra["group_mention_role_ids"] = [str(ROLE_ID)]
    adapter.config.extra["group_mention_channel_ids"] = ["999999"]  # not the ops parent
    message = make_message(
        channel=_ops_thread(),
        content=f"<@&{ROLE_ID}> hermes team, handoff",
    )

    assert _admitted(adapter, message) is False


def test_admission_admits_group_role_mention_wildcard_channel_scope(adapter):
    """(5) A wildcard channel scope honors a human group-role mention anywhere.

    ``group_mention_channel_ids: ["*"]`` means the configured role addresses
    the bot in any channel, including a thread under a non-free parent.
    """
    _free_config(adapter)
    adapter.config.extra["group_mention_role_ids"] = [str(ROLE_ID)]
    adapter.config.extra["group_mention_channel_ids"] = ["*"]
    message = make_message(
        channel=_ops_thread(),
        content=f"<@&{ROLE_ID}> hermes team, handoff",
    )

    assert _admitted(adapter, message) is True


def test_admission_denies_bot_no_mention_when_inline_required(adapter):
    """(6) A bot's ambient (no-mention) message is denied when inline is required."""
    _free_config(adapter)
    adapter.config.extra["allow_bots"] = "all"
    adapter.config.extra["bots_require_inline_mention"] = True
    message = make_message(channel=_ops_thread(), content="just chatter", bot=True)

    assert _admitted(adapter, message) is False


def test_admission_denies_bot_reply_ping_when_inline_required(adapter):
    """(6) A reply-ping (bot in message.mentions, no literal token) is denied.

    Discord's reply-ping silently adds us to ``message.mentions`` without the
    author typing our handle; the inline-required gate must reject it so two
    bots don't ping-pong replies forever.
    """
    _free_config(adapter)
    adapter.config.extra["allow_bots"] = "all"
    adapter.config.extra["bots_require_inline_mention"] = True
    message = make_message(
        channel=_ops_thread(),
        content="reply body, no literal token",
        mentions=[adapter._client.user],  # reply-ping populates mentions only
        bot=True,
    )

    assert _admitted(adapter, message) is False


def test_admission_admits_bot_real_inline_mention(adapter):
    """(6, positive control) A bot that types a real <@bot> token is admitted."""
    _free_config(adapter)
    adapter.config.extra["allow_bots"] = "all"
    adapter.config.extra["bots_require_inline_mention"] = True
    message = make_message(
        channel=_ops_thread(),
        content=f"<@{BOT_ID}> deliberate handoff",
        mentions=[adapter._client.user],
        bot=True,
    )

    assert _admitted(adapter, message) is True



def test_admission_admits_bot_ambient_followup_allow_bots_all(adapter):
    """(7) Admission admits ambient allow_bots=all bot traffic in a thread.

    The in-thread shortcut / thread_require_mention decision for this traffic
    happens later, in _handle_message (pinned end-to-end in
    test_discord_free_response.py); admission's job here is only the
    allow_bots policy, so the message passes this gate.
    """
    _free_config(adapter)
    adapter.config.extra["allow_bots"] = "all"
    adapter._threads.mark("456")
    message = make_message(channel=_ops_thread(), content="ambient bot chatter", bot=True)

    assert _admitted(adapter, message) is True


# ---------------------------------------------------------------------------
# Bot-admission config-vs-env precedence. config.extra["allow_bots"] wins over
# the DISCORD_ALLOW_BOTS env var when present; the env var is the fallback.
# Pins the documented precedence at the admission resolution site so it is not
# silently flipped. ``adapter`` and the test share one monkeypatch instance,
# and the fixture has already cleared DISCORD_ALLOW_BOTS from the environment.
# ---------------------------------------------------------------------------


def test_admission_allow_bots_config_wins_over_env_permissive(adapter, monkeypatch):
    """config allow_bots="all" admits even when env says "none"."""
    _free_config(adapter)
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "none")
    adapter.config.extra["allow_bots"] = "all"
    message = make_message(channel=_ops_thread(), content="bot handoff", bot=True)

    assert _admitted(adapter, message) is True


def test_admission_allow_bots_config_wins_over_env_restrictive(adapter, monkeypatch):
    """config allow_bots="none" denies even when env says "all"."""
    _free_config(adapter)
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    adapter.config.extra["allow_bots"] = "none"
    message = make_message(channel=_ops_thread(), content="bot chatter", bot=True)

    assert _admitted(adapter, message) is False


def test_admission_allow_bots_env_fallback_when_config_absent(adapter, monkeypatch):
    """With no config allow_bots, the DISCORD_ALLOW_BOTS env var governs."""
    _free_config(adapter)
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    # config.extra has no "allow_bots" key -> env is the fallback source.
    message = make_message(channel=_ops_thread(), content="bot handoff", bot=True)

    assert _admitted(adapter, message) is True


def test_admission_allow_bots_default_denies_when_neither_set(adapter):
    """Absent both config and env, bots default to denied ("none")."""
    _free_config(adapter)
    # Fixture already cleared DISCORD_ALLOW_BOTS; no config key set.
    message = make_message(channel=_ops_thread(), content="bot handoff", bot=True)

    assert _admitted(adapter, message) is False


# ---------------------------------------------------------------------------
# Recovered path (_dispatch_recovered_message) -- where the in_bot_thread
# bypass was removed. These mirror the key admission cases end-to-end.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recovered_denies_participated_nonfree_parent_thread_no_mention(adapter):
    """(1, recovered) The removed bypass stays removed on the recovered path."""
    _free_config(adapter)
    adapter._threads.mark("456")
    message = make_message(
        channel=_ops_thread(), content="recovered ambient, no mention"
    )

    admitted = await adapter._dispatch_recovered_message(message)

    assert admitted is False
    adapter._handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovered_admits_participated_free_parent_thread_no_mention(adapter):
    """(2, recovered) Free-response parent inheritance still admits on recovery."""
    _free_config(adapter)
    adapter._threads.mark("456")
    message = make_message(channel=_lounge_thread(), content="recovered casual chat")

    admitted = await adapter._dispatch_recovered_message(message)

    assert admitted is True
    adapter._handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_recovered_denies_unparticipated_nonfree_thread_no_mention(adapter):
    """(3, recovered) Unparticipated non-free thread still requires @mention."""
    _free_config(adapter)
    message = make_message(channel=_ops_thread(789), content="recovered unknown thread")

    admitted = await adapter._dispatch_recovered_message(message)

    assert admitted is False
    adapter._handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovered_admits_direct_self_mention(adapter):
    """(5, recovered) An explicit @bot mention is recovered under a non-free parent."""
    _free_config(adapter)
    message = make_message(
        channel=_ops_thread(),
        content=f"<@{BOT_ID}> recovered mention",
        mentions=[adapter._client.user],
    )

    admitted = await adapter._dispatch_recovered_message(message)

    assert admitted is True
    adapter._handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_recovered_denies_bot_reply_ping_when_inline_required(adapter):
    """(6, recovered) Bot reply-ping denial also holds on the recovered path."""
    _free_config(adapter)
    adapter.config.extra["allow_bots"] = "all"
    adapter.config.extra["bots_require_inline_mention"] = True
    message = make_message(
        channel=_ops_thread(),
        content="recovered reply body, no token",
        mentions=[adapter._client.user],
        bot=True,
    )

    admitted = await adapter._dispatch_recovered_message(message)

    assert admitted is False
    adapter._handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovered_admits_configured_group_role_mention_in_scope(adapter):
    """(5, recovered) A configured in-scope group-role mention is recovered.

    Mirrors ``test_admission_admits_configured_group_role_mention_in_scope`` on
    the recovered path: the recovered pre-gate must honor the same group-role
    exception the live gate does, so a valid handoff missed while offline is
    not silently dropped on backfill.
    """
    _free_config(adapter)
    adapter.config.extra["group_mention_role_ids"] = [str(ROLE_ID)]
    adapter.config.extra["group_mention_channel_ids"] = [str(OPS_ID)]
    message = make_message(
        channel=_ops_thread(),
        content=f"<@&{ROLE_ID}> hermes team, recovered handoff",
    )

    admitted = await adapter._dispatch_recovered_message(message)

    assert admitted is True
    adapter._handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_recovered_denies_bot_role_only_handoff_in_participated_thread(adapter):
    """Backfill cannot hand off a configured role ping from a sibling bot.

    This pins the formerly divergent recovered pre-gate. Even with permissive
    bot policy, inline mentions disabled, an in-scope role, and prior thread
    participation, only a human author may use a group-role handoff.
    """
    _free_config(adapter)
    adapter.config.extra["allow_bots"] = "all"
    adapter.config.extra["bots_require_inline_mention"] = False
    adapter.config.extra["group_mention_role_ids"] = [str(ROLE_ID)]
    adapter.config.extra["group_mention_channel_ids"] = [str(OPS_ID)]
    adapter._threads.mark("456")
    message = make_message(
        channel=_ops_thread(),
        content=f"<@&{ROLE_ID}> sibling bot, recovered role-only handoff",
        bot=True,
    )

    admitted = await adapter._dispatch_recovered_message(message)

    assert admitted is False
    adapter._handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovered_denies_group_role_mention_out_of_configured_scope(adapter):
    """(5, recovered, negative) An out-of-scope group-role mention stays denied.

    The group-role exception must not become a blanket bypass on recovery: a
    role mention whose channel is not in ``group_mention_channel_ids`` is
    denied exactly as on the live gate.
    """
    _free_config(adapter)
    adapter.config.extra["group_mention_role_ids"] = [str(ROLE_ID)]
    adapter.config.extra["group_mention_channel_ids"] = ["999999"]  # not the ops parent
    message = make_message(
        channel=_ops_thread(),
        content=f"<@&{ROLE_ID}> hermes team, recovered handoff",
    )

    admitted = await adapter._dispatch_recovered_message(message)

    assert admitted is False
    adapter._handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovered_denies_bot_ambient_followup_even_with_allow_bots_all(adapter):
    """(7, recovered) Backfill never rides the in-thread shortcut.

    The recovered pre-gate is stricter than live dispatch: a mention-free bot
    follow-up in a participated thread under a non-free parent is not
    recovered even though allow_bots=all admits it live. As a consequence,
    thread_require_mention has no effect at all on the recovered path.
    """
    _free_config(adapter)
    adapter.config.extra["allow_bots"] = "all"
    adapter._threads.mark("456")
    message = make_message(
        channel=_ops_thread(), content="recovered bot chatter", bot=True
    )

    admitted = await adapter._dispatch_recovered_message(message)

    assert admitted is False
    adapter._handle_message.assert_not_awaited()

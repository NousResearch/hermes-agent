"""``thread_ignore_no_mention``: the opt-in mention drop inside a thread the bot already joined.

The bot-thread exemption stays the default — a thread the bot participates in is its own
conversation, not someone else's (#116568 / PR #116581). An operator running multi-bot or
human+bot threads can opt into the channel-level "mentions someone else, not me" policy inside
those threads too (``DISCORD_THREAD_IGNORE_NO_MENTION`` / YAML ``thread_ignore_no_mention``)
without ``thread_require_mention``, which demotes everything that is not a typed mention and so
also drops the plain follow-ups a thread exists for.

The matrix below is the report's own (#142739): only a mentions-others message inside an
otherwise-exempt bot thread changes behaviour; every other shape keeps the path it had.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import discord

from gateway.platforms.helpers import MessageDeduplicator
from plugins.platforms.discord.adapter import DiscordAdapter

BOT_ID = 99
THREAD_ID = 7
OTHER_USER_ID = 820
ENV_KEY = "DISCORD_THREAD_IGNORE_NO_MENTION"
LOG_NAME = "plugins.platforms.discord.adapter"

OTHER_USER = SimpleNamespace(id=OTHER_USER_ID, bot=False)


def _adapter(*, thread_require_mention: bool = False, free_response=(),
             extra: dict | None = None) -> DiscordAdapter:
    adapter = object.__new__(DiscordAdapter)
    # self.name (used by the drop-path debug log) reads platform.value.title().
    adapter.platform = SimpleNamespace(value="discord")
    adapter.config = SimpleNamespace(extra=extra if extra is not None else {})
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=BOT_ID, bot=True))
    adapter._dedup = MessageDeduplicator()
    adapter._is_allowed_user = Mock(return_value=True)
    adapter._get_parent_channel_id = Mock(return_value=None)
    adapter._discord_free_response_channels = Mock(return_value=set(free_response))
    adapter._discord_channel_keys = Mock(return_value={str(THREAD_ID)})
    adapter._discord_thread_require_mention = Mock(return_value=thread_require_mention)
    adapter._threads = {str(THREAD_ID)}
    return adapter


def _thread_channel():
    # spec= keeps ``isinstance(channel, discord.Thread)`` true inside _in_bot_thread().
    channel = Mock(spec=discord.Thread)
    channel.id = THREAD_ID
    channel.parent_id = None
    return channel


def _plain_channel():
    return SimpleNamespace(id=THREAD_ID, parent_id=None)


def _message(channel, mentions, content: str = ""):
    return SimpleNamespace(
        id=123,
        author=SimpleNamespace(id=42, bot=False),
        channel=channel,
        content=content,
        mentions=list(mentions),
        type=discord.MessageType.default,
        guild=SimpleNamespace(id=1),
    )


def _admitted(adapter: DiscordAdapter, message) -> bool:
    admitted, _ = adapter._discord_message_admission(message, claim=False)
    return admitted


def test_gate_is_off_by_default_and_the_thread_exemption_still_admits(monkeypatch):
    monkeypatch.delenv(ENV_KEY, raising=False)

    assert _admitted(_adapter(), _message(_thread_channel(), [OTHER_USER])) is True


def test_mentions_others_in_a_bot_thread_is_dropped_and_logged(monkeypatch, caplog):
    monkeypatch.setenv(ENV_KEY, "true")
    caplog.set_level("DEBUG", logger=LOG_NAME)

    assert _admitted(_adapter(), _message(_thread_channel(), [OTHER_USER])) is False
    assert any("thread_ignore_no_mention" in record.getMessage() for record in caplog.records)


def test_a_no_mention_follow_up_in_a_bot_thread_is_still_admitted(monkeypatch):
    # The thread's normal conversational flow is exactly what this gate must not take away.
    monkeypatch.setenv(ENV_KEY, "true")

    assert _admitted(_adapter(), _message(_thread_channel(), [])) is True


def test_a_message_that_mentions_the_bot_is_not_dropped(monkeypatch):
    monkeypatch.setenv(ENV_KEY, "true")
    adapter = _adapter()

    message = _message(_thread_channel(), [adapter._client.user])
    assert _admitted(adapter, message) is True


def test_the_gate_is_a_no_op_under_thread_require_mention(monkeypatch):
    # Strict multi-bot mode already demotes every non-typed mention, so the thread exemption
    # never applies and this gate must not change that verdict either way.
    monkeypatch.setenv(ENV_KEY, "true")
    strict = _adapter(thread_require_mention=True)
    with_gate = _admitted(strict, _message(_thread_channel(), [OTHER_USER]))

    monkeypatch.delenv(ENV_KEY, raising=False)
    without_gate = _admitted(strict, _message(_thread_channel(), [OTHER_USER]))

    assert with_gate is without_gate is False


def test_a_free_response_channel_keeps_admitting_its_bot_threads(monkeypatch):
    # Free response is an explicit "answer everything here" opt-in; the channel gate already
    # honours it, and a thread inside such a channel must not become stricter than its channel.
    monkeypatch.setenv(ENV_KEY, "true")

    assert _admitted(_adapter(free_response={"*"}), _message(_thread_channel(), [OTHER_USER])) is True


def test_a_plain_channel_drop_is_unchanged_by_the_gate(monkeypatch):
    monkeypatch.delenv(ENV_KEY, raising=False)
    without_gate = _admitted(_adapter(), _message(_plain_channel(), [OTHER_USER]))

    monkeypatch.setenv(ENV_KEY, "true")
    with_gate = _admitted(_adapter(), _message(_plain_channel(), [OTHER_USER]))

    assert with_gate is without_gate is False


def test_the_yaml_key_enables_the_gate_without_the_env_var(monkeypatch):
    monkeypatch.delenv(ENV_KEY, raising=False)

    adapter = _adapter(extra={"thread_ignore_no_mention": True})
    assert _admitted(adapter, _message(_thread_channel(), [OTHER_USER])) is False

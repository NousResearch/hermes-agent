"""Topic-bound bots: a Telegram topic titled ``$Name`` routes to that bot.

Coverage for ``agent.bot_chain.bind_topic_bot`` (pure chain-request shaping)
and ``gateway.run.GatewayRunner._telegram_topic_bound_bot`` (topic title ->
bot profile resolution).

Routing contract: a topic whose title does NOT start with ``$`` is ordinary
and returns None. Once the title DOES start with ``$`` the binding is an
explicit route identity: an unresolved, disabled, unconfigured, or
unreadable bot fails CLOSED with ``BotTopicBindingError`` and the gateway
turns it into a user-visible refusal — the message never falls through to
the default agent.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent.bot_chain import (
    BotChainRequest,
    BotTopicBindingError,
    bind_topic_bot,
    parse_bot_chain_message,
)
from gateway.config import Platform
from gateway.run import GatewayRunner


# ---------------------------------------------------------------- bind_topic_bot


def test_bind_topic_bot_plain_message_becomes_single_bot_chain():
    request = bind_topic_bot(None, "writer", "Draft a concise release note.")
    assert request is not None
    assert request.names == ("writer",)
    assert request.prompt == "Draft a concise release note."


def test_bind_topic_bot_prepends_bound_bot_to_explicit_chain():
    explicit = parse_bot_chain_message("$reviewer check the proposal")
    assert explicit is not None
    request = bind_topic_bot(explicit, "writer", "$reviewer check the proposal")
    assert request.names == ("writer", "reviewer")
    assert request.prompt == "check the proposal"


def test_bind_topic_bot_dedupes_bound_bot_case_insensitively():
    explicit = parse_bot_chain_message("$WRITER $reviewer improve this draft")
    assert explicit is not None
    request = bind_topic_bot(
        explicit, "writer", "$WRITER $reviewer improve this draft"
    )
    assert request.names == ("WRITER", "reviewer")
    assert request.prompt == "improve this draft"


def test_bind_topic_bot_no_bound_bot_returns_request_unchanged():
    assert bind_topic_bot(None, None, "ordinary message") is None
    explicit = parse_bot_chain_message("$writer draft a summary")
    assert bind_topic_bot(explicit, None, "$writer draft a summary") is explicit


# ---------------------------------------------- _telegram_topic_bound_bot


@pytest.fixture()
def bot_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "model:\n  provider: nous\n  default: default/model\n",
        encoding="utf-8",
    )
    (home / "SOUL.md").write_text("Default system prompt\n", encoding="utf-8")

    def _make_profile(name, *, model="kimi-k3", provider="kimi-coding", enabled=True):
        profile_dir = home / "profiles" / name
        profile_dir.mkdir(parents=True)
        (profile_dir / "config.yaml").write_text(
            yaml.safe_dump(
                {"model": {"provider": provider, "default": model}},
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        if not enabled:
            (profile_dir / "profile.yaml").write_text(
                yaml.safe_dump({"bot": {"enabled": False}}),
                encoding="utf-8",
            )

    _make_profile("writer")
    _make_profile("reviewer", provider="deepseek", model="deepseek-v4-flash")
    _make_profile("offline", enabled=False)
    return home


def _runner():
    return object.__new__(GatewayRunner)


def test_topic_bound_bot_resolves_dollar_topic_to_bot(bot_home):
    source = SimpleNamespace(
        platform=Platform.TELEGRAM, chat_type="dm", chat_topic="$writer"
    )
    assert _runner()._telegram_topic_bound_bot(source) == "writer"


def test_topic_bound_bot_ignores_plain_topic_titles(bot_home):
    source = SimpleNamespace(
        platform=Platform.TELEGRAM, chat_type="dm", chat_topic="General"
    )
    assert _runner()._telegram_topic_bound_bot(source) is None


def test_topic_bound_bot_ignores_non_telegram_platforms(bot_home):
    source = SimpleNamespace(
        platform=Platform.SLACK, chat_type="dm", chat_topic="$writer"
    )
    assert _runner()._telegram_topic_bound_bot(source) is None


def test_topic_bound_bot_rejects_unknown_bot(bot_home):
    source = SimpleNamespace(
        platform=Platform.TELEGRAM, chat_type="dm", chat_topic="$nobody"
    )
    with pytest.raises(BotTopicBindingError, match="no profile"):
        _runner()._telegram_topic_bound_bot(source)


def test_topic_bound_bot_rejects_disabled_bot(bot_home):
    source = SimpleNamespace(
        platform=Platform.TELEGRAM, chat_type="dm", chat_topic="$offline"
    )
    with pytest.raises(BotTopicBindingError, match="disabled"):
        _runner()._telegram_topic_bound_bot(source)


def test_topic_bound_bot_rejects_bare_dollar_topic(bot_home):
    source = SimpleNamespace(
        platform=Platform.TELEGRAM, chat_type="dm", chat_topic="$"
    )
    with pytest.raises(BotTopicBindingError, match="names no bot"):
        _runner()._telegram_topic_bound_bot(source)


def test_topic_bound_bot_rejects_unconfigured_bot(bot_home):
    profile_dir = bot_home / "profiles" / "blank"
    profile_dir.mkdir(parents=True)
    (profile_dir / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    source = SimpleNamespace(
        platform=Platform.TELEGRAM, chat_type="dm", chat_topic="$blank"
    )
    with pytest.raises(BotTopicBindingError, match="no model/provider"):
        _runner()._telegram_topic_bound_bot(source)


def test_topic_bound_bot_rejects_corrupt_profile_metadata(bot_home):
    """A bound topic whose profile.yaml is corrupt fails closed."""
    profile_dir = bot_home / "profiles" / "writer"
    (profile_dir / "profile.yaml").write_text("bot: [unclosed\n", encoding="utf-8")
    source = SimpleNamespace(
        platform=Platform.TELEGRAM, chat_type="dm", chat_topic="$writer"
    )
    with pytest.raises(BotTopicBindingError, match="disabled|unreadable"):
        _runner()._telegram_topic_bound_bot(source)


# ------------------------------------- gateway-level fail-closed refusals


def _make_refusal_runner(bot_home, monkeypatch):
    """A GatewayRunner stub that proves the ordinary agent path is unreachable."""

    async def _inline_to_thread(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("gateway.run.asyncio.to_thread", _inline_to_thread)
    runner = object.__new__(GatewayRunner)

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("ordinary AIAgent path must not be reached")

    # Everything past the bot-chain routing block must never run for a
    # refused $-topic message.
    runner._recover_telegram_topic_thread_id = _forbidden
    return runner


@pytest.mark.parametrize("topic", ["$nobody", "$offline"])
def test_gateway_refuses_plain_text_in_unbound_dollar_topics(
    bot_home, monkeypatch, topic
):
    """Plain text in a ``$missing``/``$disabled`` topic: user-visible refusal,
    and the default AIAgent session path is never invoked."""
    import asyncio

    from gateway.platforms.base import MessageType

    runner = _make_refusal_runner(bot_home, monkeypatch)
    source = SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_type="dm",
        chat_id="chat-1",
        user_id="user-1",
        user_name="user",
        chat_topic=topic,
        thread_id="99",
    )
    event = SimpleNamespace(
        text="hello, are you there?",
        message_type=MessageType.TEXT,
        message_id="tg-900",
        internal=False,
        reply_to_message_id=None,
        reply_to_text=None,
    )

    response = asyncio.run(
        runner._handle_message_with_agent(event, source, "quick", 1)
    )

    assert isinstance(response, str)
    assert topic in response
    assert "default agent" in response or "does not answer" in response


def test_topic_bound_bot_rejects_invalid_profile_name(bot_home):
    source = SimpleNamespace(
        platform=Platform.TELEGRAM, chat_type="dm", chat_topic="$hello world"
    )
    with pytest.raises(BotTopicBindingError, match="not a valid profile name"):
        _runner()._telegram_topic_bound_bot(source)

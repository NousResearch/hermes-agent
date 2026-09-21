"""Tests for the bot-to-bot loop-guard.

Two bots that @mention each other would otherwise reply forever. The guard
counts consecutive bot messages from a per-channel in-process author log
(populated by ``_note_channel_author``) and stays silent once the chain runs
past ``max_bot_hops`` with no human turn. A human message resets it.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402

_CHANNEL_ID = 222


def _adapter(*, max_hops=None, env=None):
    extra = {} if max_hops is None else {"max_bot_hops": max_hops}
    config = PlatformConfig(enabled=True, token="fake-token", extra=extra)
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    return adapter


def _seed(adapter, sequence):
    """Record a chronological sequence of authors. True = bot, False = human."""
    for is_bot in sequence:
        msg = SimpleNamespace(
            author=SimpleNamespace(bot=is_bot),
            channel=SimpleNamespace(id=_CHANNEL_ID),
        )
        adapter._note_channel_author(msg)


def _trigger():
    return SimpleNamespace(
        author=SimpleNamespace(bot=True), channel=SimpleNamespace(id=_CHANNEL_ID)
    )


def test_no_prior_messages_allows(monkeypatch):
    monkeypatch.delenv("DISCORD_MAX_BOT_HOPS", raising=False)
    adapter = _adapter()
    assert adapter._bot_loop_would_exceed(_trigger()) is False


def test_chain_past_cap_blocks(monkeypatch):
    monkeypatch.delenv("DISCORD_MAX_BOT_HOPS", raising=False)
    adapter = _adapter()  # default cap 6
    # 6 prior bot messages + the trigger bot message = 7 trailing bots > 6.
    _seed(adapter, [True] * 6 + [True])
    assert adapter._bot_loop_would_exceed(_trigger()) is True


def test_human_resets_chain(monkeypatch):
    monkeypatch.delenv("DISCORD_MAX_BOT_HOPS", raising=False)
    adapter = _adapter()  # default cap 6
    # human, then 5 bots, then the trigger = 6 trailing bots, not > 6.
    _seed(adapter, [False] + [True] * 5 + [True])
    assert adapter._bot_loop_would_exceed(_trigger()) is False


def test_cap_disabled_when_zero(monkeypatch):
    monkeypatch.delenv("DISCORD_MAX_BOT_HOPS", raising=False)
    adapter = _adapter(max_hops=0)
    _seed(adapter, [True] * 20 + [True])
    assert adapter._bot_loop_would_exceed(_trigger()) is False


def test_config_driven_cap(monkeypatch):
    monkeypatch.delenv("DISCORD_MAX_BOT_HOPS", raising=False)
    adapter = _adapter(max_hops=2)
    # 2 prior bots + trigger = 3 trailing bots > 2.
    _seed(adapter, [True, True, True])
    assert adapter._bot_loop_would_exceed(_trigger()) is True


def test_unknown_channel_does_not_block(monkeypatch):
    monkeypatch.delenv("DISCORD_MAX_BOT_HOPS", raising=False)
    adapter = _adapter()
    no_channel = SimpleNamespace(author=SimpleNamespace(bot=True), channel=SimpleNamespace(id=None))
    assert adapter._bot_loop_would_exceed(no_channel) is False


def test_invalid_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("DISCORD_MAX_BOT_HOPS", "not-a-number")
    adapter = _adapter()
    assert adapter._discord_max_bot_hops() == 6

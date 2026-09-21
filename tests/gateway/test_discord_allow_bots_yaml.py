"""Tests for wiring ``discord.allow_bots`` / ``max_bot_hops`` config into env.

The adapter reads ``DISCORD_ALLOW_BOTS`` and ``DISCORD_MAX_BOT_HOPS`` via
``os.getenv``. Before this wiring, setting ``discord.extra.allow_bots`` in a
profile config.yaml was a silent no-op — bot-to-bot co-working never fired.
``_apply_yaml_config`` is the YAML→env translation hook.
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

import os  # noqa: E402

import pytest  # noqa: E402

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402

_apply = discord_platform._apply_yaml_config


@pytest.fixture(autouse=True)
def _restore_env():
    """``_apply_yaml_config`` writes os.environ directly; snapshot/restore so
    these tests don't leak DISCORD_ALLOW_BOTS into sibling test modules."""
    keys = ("DISCORD_ALLOW_BOTS", "DISCORD_MAX_BOT_HOPS")
    saved = {k: os.environ.get(k) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _clear(monkeypatch):
    monkeypatch.delenv("DISCORD_ALLOW_BOTS", raising=False)
    monkeypatch.delenv("DISCORD_MAX_BOT_HOPS", raising=False)


def test_allow_bots_from_extra(monkeypatch):
    _clear(monkeypatch)
    _apply({}, {"extra": {"allow_bots": "mentions"}})
    assert discord_platform.os.getenv("DISCORD_ALLOW_BOTS") == "mentions"


def test_allow_bots_top_level(monkeypatch):
    _clear(monkeypatch)
    _apply({}, {"allow_bots": "All"})
    assert discord_platform.os.getenv("DISCORD_ALLOW_BOTS") == "all"


def test_allow_bots_env_precedence(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "none")
    monkeypatch.delenv("DISCORD_MAX_BOT_HOPS", raising=False)
    _apply({}, {"extra": {"allow_bots": "mentions"}})
    assert discord_platform.os.getenv("DISCORD_ALLOW_BOTS") == "none"


def test_allow_bots_absent_leaves_env_unset(monkeypatch):
    _clear(monkeypatch)
    _apply({}, {"require_mention": True})
    assert discord_platform.os.getenv("DISCORD_ALLOW_BOTS") is None


def test_max_bot_hops_from_extra(monkeypatch):
    _clear(monkeypatch)
    _apply({}, {"extra": {"max_bot_hops": 3}})
    assert discord_platform.os.getenv("DISCORD_MAX_BOT_HOPS") == "3"


def test_max_bot_hops_env_precedence(monkeypatch):
    monkeypatch.setenv("DISCORD_MAX_BOT_HOPS", "9")
    monkeypatch.delenv("DISCORD_ALLOW_BOTS", raising=False)
    _apply({}, {"extra": {"max_bot_hops": 3}})
    assert discord_platform.os.getenv("DISCORD_MAX_BOT_HOPS") == "9"

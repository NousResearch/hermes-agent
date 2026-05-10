"""Tests for the Discord ``allowed_mentions`` safe-default helper.

Ensures the bot defaults to blocking ``@everyone`` / ``@here`` / role pings
so an LLM response (or echoed user content) can't spam a whole server —
and that the four ``DISCORD_ALLOW_MENTION_*`` env vars correctly opt back
in when an operator explicitly wants a different policy.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _FakeAllowedMentions:
    """Stand-in for ``discord.AllowedMentions`` that exposes the same four
    boolean flags as real attributes so the test can assert on them.
    """

    def __init__(self, *, everyone=True, roles=True, users=True, replied_user=True):
        self.everyone = everyone
        self.roles = roles
        self.users = users
        self.replied_user = replied_user

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"AllowedMentions(everyone={self.everyone}, roles={self.roles}, "
            f"users={self.users}, replied_user={self.replied_user})"
        )


def _ensure_discord_mock():
    """Install (or augment) a mock ``discord`` module.

    Other test modules in this directory stub ``discord`` via
    ``sys.modules.setdefault`` — whichever test file imports first wins and
    our full module is then silently dropped. We therefore ALWAYS force
    ``AllowedMentions`` onto whatever is currently in ``sys.modules["discord"]``;
    that's the only attribute this test file actually needs real behavior from.
    """
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        sys.modules["discord"].AllowedMentions = _FakeAllowedMentions
        return

    if sys.modules.get("discord") is None:
        discord_mod = MagicMock()
        discord_mod.Intents.default.return_value = MagicMock()
        discord_mod.Client = MagicMock
        discord_mod.File = MagicMock
        discord_mod.DMChannel = type("DMChannel", (), {})
        discord_mod.Thread = type("Thread", (), {})
        discord_mod.ForumChannel = type("ForumChannel", (), {})
        discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
        discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, danger=3, green=1, blurple=2, red=3, grey=4, secondary=5)
        discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4)
        discord_mod.Interaction = object
        discord_mod.Embed = MagicMock
        discord_mod.app_commands = SimpleNamespace(
            describe=lambda **kwargs: (lambda fn: fn),
            choices=lambda **kwargs: (lambda fn: fn),
            Choice=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        discord_mod.opus = SimpleNamespace(is_loaded=lambda: True)

        ext_mod = MagicMock()
        commands_mod = MagicMock()
        commands_mod.Bot = MagicMock
        ext_mod.commands = commands_mod

        sys.modules["discord"] = discord_mod
        sys.modules.setdefault("discord.ext", ext_mod)
        sys.modules.setdefault("discord.ext.commands", commands_mod)

    # Whether we just installed the mock OR the mock was already installed
    # by another test's _ensure_discord_mock, force the AllowedMentions
    # stand-in onto it — _build_allowed_mentions() reads this attribute.
    sys.modules["discord"].AllowedMentions = _FakeAllowedMentions


_ensure_discord_mock()

from gateway.platforms.discord import _build_allowed_mentions  # noqa: E402


# The four DISCORD_ALLOW_MENTION_* env vars that _build_allowed_mentions reads.
# Cleared before each test so env leakage from other tests never masks a regression.
_ENV_VARS = (
    "DISCORD_ALLOW_MENTION_EVERYONE",
    "DISCORD_ALLOW_MENTION_ROLES",
    "DISCORD_ALLOW_MENTION_USERS",
    "DISCORD_ALLOW_MENTION_REPLIED_USER",
)


@pytest.fixture(autouse=True)
def _clear_allowed_mention_env(monkeypatch):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_safe_defaults_block_everyone_and_roles():
    am = _build_allowed_mentions()
    assert am.everyone is False, "default must NOT allow @everyone/@here pings"
    assert am.roles is False, "default must NOT allow role pings"
    assert am.users is True, "default must allow user pings so replies work"
    assert am.replied_user is True, "default must allow reply-reference pings"


def test_env_var_opts_back_into_everyone(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_MENTION_EVERYONE", "true")
    am = _build_allowed_mentions()
    assert am.everyone is True
    # other defaults unaffected
    assert am.roles is False
    assert am.users is True
    assert am.replied_user is True


def test_env_var_can_disable_users(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_MENTION_USERS", "false")
    am = _build_allowed_mentions()
    assert am.users is False
    # safe defaults elsewhere remain
    assert am.everyone is False
    assert am.roles is False
    assert am.replied_user is True


@pytest.mark.parametrize("raw, expected", [
    ("true", True), ("True", True), ("TRUE", True),
    ("1", True), ("yes", True), ("YES", True), ("on", True),
    ("false", False), ("False", False), ("0", False),
    ("no", False), ("off", False),
    ("", False),                 # empty falls back to default (False for everyone)
    ("garbage", False),          # unknown falls back to default
    (" true ", True),            # whitespace tolerated
])
def test_everyone_boolean_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("DISCORD_ALLOW_MENTION_EVERYONE", raw)
    am = _build_allowed_mentions()
    assert am.everyone is expected


def test_all_four_knobs_together(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_MENTION_EVERYONE", "true")
    monkeypatch.setenv("DISCORD_ALLOW_MENTION_ROLES", "true")
    monkeypatch.setenv("DISCORD_ALLOW_MENTION_USERS", "false")
    monkeypatch.setenv("DISCORD_ALLOW_MENTION_REPLIED_USER", "false")
    am = _build_allowed_mentions()
    assert am.everyone is True
    assert am.roles is True
    assert am.users is False
    assert am.replied_user is False

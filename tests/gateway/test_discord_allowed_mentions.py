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

from plugins.platforms.discord.adapter import _build_allowed_mentions  # noqa: E402


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


# ── handoff-thread seed message (cron channel-summary seed_text) ────────────

import asyncio  # noqa: E402

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class _FakeSeedMessage:
    async def create_thread(self, **kwargs):
        return SimpleNamespace(id=777)


class _FakeParentChannel:
    """Text-channel double WITHOUT create_thread, so create_handoff_thread
    takes the seed-message fallback path; records every send() call."""

    def __init__(self):
        self.send_calls = []

    async def send(self, content, **kwargs):
        self.send_calls.append((content, kwargs))
        return _FakeSeedMessage()


def _make_handoff_adapter(parent):
    adapter = object.__new__(DiscordAdapter)
    client = MagicMock()
    client.get_channel = lambda pid: parent
    adapter._client = client
    return adapter


class TestHandoffSeedMentionSuppression:
    """REGRESSION: seed_text is model output posted raw on the seed-message
    fallback — a per-send AllowedMentions override must pin @everyone/@here
    and role pings off regardless of the env-overridable client default.
    The fixed-label send keeps the legacy no-kwarg call untouched."""

    def test_seed_text_send_pins_safe_allowed_mentions(self):
        parent = _FakeParentChannel()
        adapter = _make_handoff_adapter(parent)
        thread_id = asyncio.run(
            adapter.create_handoff_thread(
                "123", "Daily Brief", seed_text="@everyone 3 alerts fired."
            )
        )
        assert thread_id == "777"
        (content, kwargs), = parent.send_calls
        assert content == "@everyone 3 alerts fired."
        am = kwargs["allowed_mentions"]
        assert am.everyone is False
        assert am.roles is False
        assert am.users is True
        assert am.replied_user is False

    def test_label_send_keeps_legacy_call_without_allowed_mentions(self):
        parent = _FakeParentChannel()
        adapter = _make_handoff_adapter(parent)
        thread_id = asyncio.run(adapter.create_handoff_thread("123", "Daily Brief"))
        assert thread_id == "777"
        (content, kwargs), = parent.send_calls
        assert content == "\U0001f9f5 Hermes handoff: **Daily Brief**"
        assert kwargs == {}



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
    """Delegate to central comprehensive mock in gateway/conftest.py."""
    from tests.gateway.conftest import _ensure_discord_mock as _central
    _central()

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

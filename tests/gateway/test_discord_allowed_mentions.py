"""Tests for the Discord ``allowed_mentions`` safe-default helper.

Ensures the bot defaults to blocking ``@everyone`` / ``@here`` / role pings
so an LLM response (or echoed user content) can't spam a whole server —
and that the four ``DISCORD_ALLOW_MENTION_*`` env vars correctly opt back
in when an operator explicitly wants a different policy.
"""


import pytest

from tests.discord_mock import ensure_discord_module as _ensure_discord_mock

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


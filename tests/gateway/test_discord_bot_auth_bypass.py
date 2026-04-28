"""Regression guard for #4466: DISCORD_ALLOW_BOTS works without DISCORD_ALLOWED_USERS.

The bug had two sequential gates both rejecting bot messages:

  Gate 1 — `on_message` in gateway/platforms/discord.py ran the user-allowlist
  check BEFORE the bot filter, so bot senders were dropped with a warning
  before the DISCORD_ALLOW_BOTS policy was ever evaluated.

  Gate 2 — `_is_user_authorized` in gateway/run.py rejected bots at the
  gateway level even if they somehow reached that layer.

These tests assert both gates now pass a bot message through when
DISCORD_ALLOW_BOTS permits it AND no user allowlist entry exists.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.session import Platform, SessionSource


@pytest.fixture(autouse=True)
def _isolate_discord_env(monkeypatch):
    """Make every test start with a clean Discord env so prior tests in the
    session (or CI setups) can't leak DISCORD_ALLOWED_ROLES / DISCORD_ALLOWED_USERS
    / DISCORD_ALLOW_BOTS and silently flip the auth result.
    """
    for var in (
        "DISCORD_ALLOW_BOTS",
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOWED_ROLES",
        "DISCORD_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


# -----------------------------------------------------------------------------
# Gate 2: _is_user_authorized bypasses allowlist for permitted bots
# -----------------------------------------------------------------------------


def _make_bare_runner():
    """Build a GatewayRunner skeleton with just enough wiring for the auth test.

    Uses ``object.__new__`` to skip the heavy __init__ — many gateway tests
    use this pattern (see AGENTS.md pitfall #17).
    """
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    # _is_user_authorized reads self.pairing_store.is_approved(...) before
    # any allowlist check succeeds; stub it to never approve so we exercise
    # the real allowlist path.
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def _make_discord_bot_source(bot_id: str = "999888777"):
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_type="channel",
        user_id=bot_id,
        user_name="SomeBot",
        is_bot=True,
    )


def _make_discord_human_source(user_id: str = "100200300"):
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_type="channel",
        user_id=user_id,
        user_name="SomeHuman",
        is_bot=False,
    )


def test_discord_bot_authorized_when_allow_bots_mentions(monkeypatch):
    """DISCORD_ALLOW_BOTS=mentions must authorize a bot sender even when
    DISCORD_ALLOWED_USERS is set and the bot's ID is NOT in it.

    This is the exact scenario from #4466 — a Cloudflare Worker webhook
    posts Notion events to Discord, the Hermes bot gets @mentioned, and
    the webhook's bot ID is not (and shouldn't be) on the human
    allowlist.
    """
    runner = _make_bare_runner()

    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "100200300")  # human-only allowlist

    source = _make_discord_bot_source(bot_id="999888777")
    assert runner._is_user_authorized(source) is True


def test_discord_bot_authorized_when_allow_bots_all(monkeypatch):
    """DISCORD_ALLOW_BOTS=all is a superset of =mentions — should also bypass."""
    runner = _make_bare_runner()

    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "100200300")

    source = _make_discord_bot_source()
    assert runner._is_user_authorized(source) is True


def test_discord_bot_NOT_authorized_when_allow_bots_none(monkeypatch):
    """DISCORD_ALLOW_BOTS=none (default) must still reject bots that aren't
    in DISCORD_ALLOWED_USERS — preserves the original security behavior.
    """
    runner = _make_bare_runner()

    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "none")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "100200300")

    source = _make_discord_bot_source(bot_id="999888777")
    assert runner._is_user_authorized(source) is False


def test_discord_bot_NOT_authorized_when_allow_bots_unset(monkeypatch):
    """Unset DISCORD_ALLOW_BOTS must behave like 'none'."""
    runner = _make_bare_runner()

    monkeypatch.delenv("DISCORD_ALLOW_BOTS", raising=False)
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "100200300")

    source = _make_discord_bot_source(bot_id="999888777")
    assert runner._is_user_authorized(source) is False


def test_discord_human_still_checked_against_allowlist_when_bot_policy_set(monkeypatch):
    """DISCORD_ALLOW_BOTS=all must NOT open the gate for humans — they
    still need to be in DISCORD_ALLOWED_USERS (or a pairing approval).
    """
    runner = _make_bare_runner()

    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "100200300")

    # Human NOT on the allowlist → must be rejected.
    source = _make_discord_human_source(user_id="999999999")
    assert runner._is_user_authorized(source) is False

    # Human ON the allowlist → accepted.
    source_allowed = _make_discord_human_source(user_id="100200300")
    assert runner._is_user_authorized(source_allowed) is True


def test_bot_bypass_does_not_leak_to_other_platforms(monkeypatch):
    """The is_bot bypass is Discord-specific — a Telegram bot source with
    is_bot=True must NOT be authorized just because DISCORD_ALLOW_BOTS=all.
    """
    runner = _make_bare_runner()

    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "100200300")

    telegram_bot = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="channel",
        user_id="999888777",
        is_bot=True,
    )
    assert runner._is_user_authorized(telegram_bot) is False


# -----------------------------------------------------------------------------
# DISCORD_ALLOWED_ROLES gateway-layer bypass (#7871)
# -----------------------------------------------------------------------------


def test_discord_role_config_bypasses_gateway_allowlist(monkeypatch):
    """When DISCORD_ALLOWED_ROLES is set, _is_user_authorized must trust
    the adapter's pre-filter and authorize. Without this, role-only setups
    (DISCORD_ALLOWED_ROLES populated, DISCORD_ALLOWED_USERS empty) would
    hit the 'no allowlists configured' branch and get rejected.
    """
    runner = _make_bare_runner()

    monkeypatch.setenv("DISCORD_ALLOWED_ROLES", "1493705176387948674")
    # Note: DISCORD_ALLOWED_USERS is NOT set — the entire point.

    source = _make_discord_human_source(user_id="999888777")
    assert runner._is_user_authorized(source) is True


def test_discord_role_config_still_authorizes_alongside_users(monkeypatch):
    """Sanity: setting both DISCORD_ALLOWED_ROLES and DISCORD_ALLOWED_USERS
    doesn't break the user-id path. Users in the allowlist should still be
    authorized even if they don't have a role. (OR semantics.)
    """
    runner = _make_bare_runner()

    monkeypatch.setenv("DISCORD_ALLOWED_ROLES", "1493705176387948674")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "100200300")

    # User on the user allowlist, no role → still authorized at gateway
    # level via the role bypass (adapter already approved them).
    source = _make_discord_human_source(user_id="100200300")
    assert runner._is_user_authorized(source) is True


def test_discord_role_bypass_does_not_leak_to_other_platforms(monkeypatch):
    """DISCORD_ALLOWED_ROLES must only affect Discord. Setting it should
    not suddenly start authorizing Telegram users whose platform has its
    own empty allowlist.
    """
    runner = _make_bare_runner()

    monkeypatch.setenv("DISCORD_ALLOWED_ROLES", "1493705176387948674")
    # Telegram has its own empty allowlist and no allow-all flag.

    telegram_user = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="channel",
        user_id="999888777",
    )
    assert runner._is_user_authorized(telegram_user) is False

"""Discord bot-authored events are no longer an authorization surface."""

from types import SimpleNamespace

import pytest

from gateway.session import Platform, SessionSource


@pytest.fixture(autouse=True)
def _isolate_discord_env(monkeypatch):
    for var in (
        "DISCORD_ALLOW_BOTS",
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOWED_ROLES",
        "DISCORD_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "FEISHU_ALLOW_BOTS",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_bare_runner():
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
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


def test_discord_bot_not_authorized_when_allow_bots_mentions(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "100200300")

    assert runner._is_user_authorized(_make_discord_bot_source()) is False


def test_discord_bot_not_authorized_when_allow_bots_all(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "100200300")

    assert runner._is_user_authorized(_make_discord_bot_source()) is False


def test_discord_human_still_checked_against_allowlist_when_old_bot_policy_set(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "100200300")

    assert runner._is_user_authorized(_make_discord_human_source(user_id="999999999")) is False
    assert runner._is_user_authorized(_make_discord_human_source(user_id="100200300")) is True


def test_feishu_bot_bypass_is_preserved(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_ALLOW_BOTS", "all")
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "human-only")

    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="chat-1",
        chat_type="channel",
        user_id="bot-1",
        user_name="FeishuBot",
        is_bot=True,
    )
    assert runner._is_user_authorized(source) is True


def test_discord_role_config_does_not_bypass_gateway_allowlist(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("DISCORD_ALLOWED_ROLES", "1493705176387948674")

    assert runner._is_user_authorized(_make_discord_human_source(user_id="999888777")) is False


def test_discord_user_allowlist_still_authorizes_when_role_is_also_configured(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("DISCORD_ALLOWED_ROLES", "1493705176387948674")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "100200300")

    assert runner._is_user_authorized(_make_discord_human_source(user_id="100200300")) is True

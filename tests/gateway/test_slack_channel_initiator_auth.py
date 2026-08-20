"""Security-boundary tests for Slack channel-scoped initiator grants."""

from types import SimpleNamespace

import pytest

from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource


class _Runner(GatewayAuthorizationMixin):
    pass


def _adapter(open_user_channels):
    return SimpleNamespace(
        config=PlatformConfig(
            enabled=True,
            extra={"open_user_channels": open_user_channels},
        )
    )


@pytest.fixture
def runner(monkeypatch):
    for key in (
        "SLACK_ALLOWED_USERS",
        "SLACK_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "SLACK_ALLOW_BOTS",
    ):
        monkeypatch.delenv(key, raising=False)

    instance = _Runner()
    instance.adapters = {Platform.SLACK: _adapter(["C_TEAM"])}
    instance._profile_adapters = {}
    instance.pairing_store = None
    return instance


def _source(*, chat_id="C_TEAM", chat_type="group", user_id="U_MEMBER", profile=None, is_bot=False):
    return SessionSource(
        platform=Platform.SLACK,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        profile=profile,
        is_bot=is_bot,
    )


def test_exact_channel_grant_unions_with_owner_allowlist(runner, monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_OWNER")

    assert runner._is_user_authorized(_source()) is True
    assert runner._is_user_authorized(_source(chat_id="C_OTHER")) is False
    assert runner._is_user_authorized(
        _source(chat_id="C_OTHER", user_id="U_OWNER")
    ) is True


@pytest.mark.parametrize("chat_type", ["dm", "interaction"])
def test_channel_grant_does_not_widen_dm_or_interactive_access(runner, chat_type):
    assert runner._is_user_authorized(_source(chat_type=chat_type)) is False


def test_channel_chat_type_is_an_ordinary_initiator_surface(runner):
    assert runner._is_user_authorized(_source(chat_type="channel")) is True


def test_wildcard_is_not_interpreted_as_all_channels(runner):
    runner.adapters[Platform.SLACK] = _adapter("*")

    assert runner._is_user_authorized(_source(chat_id="C_TEAM")) is False


def test_channel_grant_does_not_authorize_bot_senders(runner):
    assert runner._is_user_authorized(_source(is_bot=True)) is False


def test_multiplex_profile_uses_selected_adapter_policy(runner):
    runner._profile_adapters = {
        "coder": {Platform.SLACK: _adapter("C_CODER,C_SECOND")},
    }

    assert runner._is_user_authorized(
        _source(chat_id="C_CODER", profile="coder")
    ) is True
    assert runner._is_user_authorized(
        _source(chat_id="C_TEAM", profile="coder")
    ) is False
    assert runner._is_user_authorized(
        _source(chat_id="C_SECOND", profile="coder")
    ) is True


def test_missing_multiplex_adapter_fails_closed(runner):
    assert runner._is_user_authorized(
        _source(chat_id="C_TEAM", profile="missing")
    ) is False

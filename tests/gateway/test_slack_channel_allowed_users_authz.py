"""Channel-scoped Slack sender authorization."""

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session import SessionSource


EXECUTIVE = "C_EXECUTIVE"
OTHER = "C_OTHER"
SOPHIE = "U_SOPHIE"
VIRGINIA = "U_VIRGINIA"
TROY = "U_TROY"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for name in (
        "SLACK_ALLOWED_USERS",
        "SLACK_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(name, raising=False)


def _runner(channel_users=None):
    from gateway.run import GatewayRunner

    extra = {}
    if channel_users is not None:
        extra["channel_allowed_users"] = channel_users
    adapter = SimpleNamespace(config=SimpleNamespace(extra=extra))

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.SLACK: adapter}  # type: ignore[assignment]
    runner.pairing_store = SimpleNamespace(  # type: ignore[assignment]
        is_approved=lambda *_a, **_kw: False
    )
    return runner


def _source(*, channel=EXECUTIVE, user=SOPHIE, chat_type="group"):
    return SessionSource(
        platform=Platform.SLACK,
        chat_id=channel,
        chat_type=chat_type,
        user_id=user,
        user_name="Tester",
        is_bot=False,
    )


def test_explicit_users_are_authorized_only_in_mapped_channel(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", TROY)
    runner = _runner({EXECUTIVE: [TROY, SOPHIE, VIRGINIA]})

    assert runner._is_user_authorized(_source(user=SOPHIE)) is True
    assert runner._is_user_authorized(_source(user=VIRGINIA)) is True
    assert runner._is_user_authorized(_source(channel=OTHER, user=SOPHIE)) is False


def test_channel_map_does_not_open_dms(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", TROY)
    runner = _runner({EXECUTIVE: [SOPHIE]})

    assert runner._is_user_authorized(_source(user=SOPHIE, chat_type="dm")) is False


def test_unlisted_channel_member_remains_denied(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", TROY)
    runner = _runner({EXECUTIVE: [VIRGINIA]})

    assert runner._is_user_authorized(_source(user=SOPHIE)) is False


def test_comma_separated_channel_users_are_supported(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", TROY)
    runner = _runner({EXECUTIVE: f"{SOPHIE},{VIRGINIA}"})

    assert runner._is_user_authorized(_source(user=SOPHIE)) is True
    assert runner._is_user_authorized(_source(user=VIRGINIA)) is True


def test_channel_map_does_not_honor_wildcard(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", TROY)
    runner = _runner({EXECUTIVE: ["*"]})

    assert runner._is_user_authorized(_source(user=SOPHIE)) is False

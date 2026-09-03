"""Authorization for Slack trigger-only reporting-channel members."""
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session import SessionSource


LOCKED = "C_LOCKED"
OTHER = "C_OTHER"
UPDATE_PATTERN = r"(?i)^\s*update\s*[.!]?\s*$"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for name in (
        "SLACK_ALLOWED_USERS",
        "SLACK_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(name, raising=False)


def _runner(*, trigger_only=True, member_channels=LOCKED):
    from gateway.run import GatewayRunner

    extra = {"trigger_only_member_channels": member_channels}
    if trigger_only:
        extra["trigger_only_channels"] = {LOCKED: UPDATE_PATTERN}
    adapter = SimpleNamespace(config=SimpleNamespace(extra=extra))

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.SLACK: adapter}
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def _source(channel=LOCKED, *, chat_type="group", user="U_NEW_MEMBER"):
    return SessionSource(
        platform=Platform.SLACK,
        chat_id=channel,
        chat_type=chat_type,
        user_id=user,
        user_name="Invited Member",
        is_bot=False,
    )


def test_invited_member_authorized_only_in_locked_trigger_channel(monkeypatch):
    runner = _runner()
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_OWNER")

    assert runner._is_user_authorized(_source()) is True
    assert runner._is_user_authorized(_source(OTHER)) is False


def test_member_channel_setting_requires_hard_trigger_gate(monkeypatch):
    runner = _runner(trigger_only=False)
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_OWNER")

    assert runner._is_user_authorized(_source()) is False


def test_member_authorization_never_opens_dms(monkeypatch):
    runner = _runner()
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_OWNER")

    assert runner._is_user_authorized(_source(chat_type="dm")) is False


def test_string_and_list_channel_configuration_are_supported(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_OWNER")

    assert _runner(member_channels=LOCKED)._is_user_authorized(_source()) is True
    assert _runner(member_channels=[LOCKED])._is_user_authorized(_source()) is True

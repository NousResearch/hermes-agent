from __future__ import annotations

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _signal_group_source(*, chat_id: str = "group:raw-signal-group-id", chat_id_alt: str | None = "raw-signal-group-id") -> SessionSource:
    return SessionSource(
        platform=Platform.SIGNAL,
        user_id="+15551234567",
        user_name="Sickan",
        chat_id=chat_id,
        chat_id_alt=chat_id_alt,
        chat_name="DroneProject",
        chat_type="group",
    )


def test_signal_group_allowed_users_authorizes_raw_group_id(monkeypatch):
    """SIGNAL_GROUP_ALLOWED_USERS is a group-id allowlist, not a sender allowlist.

    The Signal adapter already drops messages from unlisted groups before they
    reach GatewayRunner. Once such a group message reaches run.py, the gateway
    must not deny a participant just because their phone number is absent from
    SIGNAL_ALLOWED_USERS.
    """
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "raw-signal-group-id")
    monkeypatch.delenv("SIGNAL_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

    runner = object.__new__(GatewayRunner)

    assert runner._is_user_authorized(_signal_group_source()) is True


def test_signal_group_allowed_users_authorizes_gateway_prefixed_group_id(monkeypatch):
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "group:raw-signal-group-id")
    monkeypatch.delenv("SIGNAL_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

    runner = object.__new__(GatewayRunner)

    assert runner._is_user_authorized(_signal_group_source()) is True


def test_signal_group_allowed_users_wildcard_authorizes_group(monkeypatch):
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "*")
    monkeypatch.delenv("SIGNAL_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

    runner = object.__new__(GatewayRunner)

    assert runner._is_user_authorized(_signal_group_source(chat_id="group:any-group", chat_id_alt="any-group")) is True

"""Signal group allowlist authorization tests."""

from typing import Optional
from unittest.mock import Mock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.signal import SignalAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_signal_adapter(monkeypatch, group_allowed="abc123", dm_allowed="+155****2222"):
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", group_allowed)
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", dm_allowed)
    config = PlatformConfig()
    config.enabled = True
    config.extra = {
        "http_url": "http://localhost:8080",
        "account": "+155****9999",
    }
    return SignalAdapter(config)


def _event(chat_type="group", chat_id="group:abc123", chat_id_alt: Optional[str] = "abc123", user_id="+155****1111"):
    return MessageEvent(
        source=SessionSource(
            platform=Platform.SIGNAL,
            chat_id=chat_id,
            chat_type=chat_type,
            user_id=user_id,
            user_name="Alice",
            chat_id_alt=chat_id_alt,
        ),
        text="ping",
        message_type=MessageType.TEXT,
    )


def _runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {}
    runner.pairing_store = Mock()
    runner.pairing_store.is_approved.return_value = False
    return runner


def test_signal_group_allowed_users_authorizes_group_chat_by_raw_group_id(monkeypatch):
    monkeypatch.delenv("SIGNAL_ALLOWED_USERS", raising=False)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "abc123")

    source = SessionSource(
        platform=Platform.SIGNAL,
        chat_id="group:abc123",
        chat_type="group",
        user_id="+15550001111",
        user_name="Alice",
        chat_id_alt="abc123",
    )

    assert _runner()._is_user_authorized(source) is True


def test_signal_group_allowed_users_does_not_authorize_dm(monkeypatch):
    monkeypatch.delenv("SIGNAL_ALLOWED_USERS", raising=False)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "abc123")

    source = SessionSource(
        platform=Platform.SIGNAL,
        chat_id="+15550001111",
        chat_type="dm",
        user_id="+15550001111",
        user_name="Alice",
    )

    assert _runner()._is_user_authorized(source) is False


def test_signal_group_allowed_users_authorizes_group_chat_shape(monkeypatch):
    monkeypatch.delenv("SIGNAL_ALLOWED_USERS", raising=False)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "group:abc123")

    source = SessionSource(
        platform=Platform.SIGNAL,
        chat_id="group:abc123",
        chat_type="group",
        user_id="+155****1111",
        user_name="Alice",
        chat_id_alt="abc123",
    )

    assert _runner()._is_user_authorized(source) is True


def test_signal_group_only_dm_block_overrides_stale_pairing_approval(monkeypatch):
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "")
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "abc123")

    runner = _runner()
    pairing_store = Mock()
    pairing_store.is_approved.return_value = True
    runner.pairing_store = pairing_store
    source = SessionSource(
        platform=Platform.SIGNAL,
        chat_id="+155****8189",
        chat_type="dm",
        user_id="+155****8189",
        user_name="Signal Bot",
    )

    assert runner._is_user_authorized(source) is False
    pairing_store.is_approved.assert_not_called()


def test_signal_group_only_unauthorized_dm_behavior_forces_ignore(monkeypatch):
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "")
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "abc123")

    runner = _runner()

    assert runner._get_unauthorized_dm_behavior(Platform.SIGNAL) == "ignore"


def test_signal_reactions_allowed_for_allowed_group_even_when_sender_not_dm_allowed(monkeypatch):
    adapter = _make_signal_adapter(monkeypatch, group_allowed="abc123", dm_allowed="+155****2222")

    assert adapter._reactions_enabled(_event()) is True


def test_signal_reactions_reject_unlisted_group(monkeypatch):
    adapter = _make_signal_adapter(monkeypatch, group_allowed="other", dm_allowed="*")

    assert adapter._reactions_enabled(_event()) is False


def test_signal_reactions_still_apply_dm_allowlist(monkeypatch):
    adapter = _make_signal_adapter(monkeypatch, group_allowed="abc123", dm_allowed="+155****2222")

    assert adapter._reactions_enabled(_event(chat_type="dm", chat_id="+155****1111", chat_id_alt=None)) is False

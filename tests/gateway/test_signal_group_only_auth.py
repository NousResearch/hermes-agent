"""Signal group chat grants and explicit group-only DMs remain profile-scoped."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.secret_scope import reset_secret_scope, set_secret_scope
from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource


def runner():
    host = GatewayAuthorizationMixin()
    host.adapters = {}
    host.config = GatewayConfig()
    host.pairing_store = Mock()
    host.pairing_store.is_approved.return_value = False
    return host


def source(kind="group", chat="group:abc"):
    return SessionSource(platform=Platform.SIGNAL, chat_id=chat, chat_type=kind, user_id="alice")


@pytest.mark.parametrize("allowed", ["abc", "group:abc", "*"])
def test_group_allowlist_authorizes_chat_not_sender(monkeypatch, allowed):
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "")
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", allowed)
    host = runner()
    assert host._is_user_authorized(source()) is True
    assert host._is_user_authorized(source("dm", "alice")) is False
    if allowed != "*":
        assert host._is_user_authorized(source(chat="group:other")) is False


def test_group_only_blocks_old_pairing_and_pairing_code(monkeypatch):
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "")
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "abc")
    host = runner()
    host.pairing_store.is_approved.return_value = True
    assert host._is_user_authorized(source("dm", "alice")) is False
    host.pairing_store.is_approved.assert_not_called()
    # Even an explicit pairing reply must not undermine disabled DMs.
    host.config.platforms[Platform.SIGNAL] = SimpleNamespace(extra={"unauthorized_dm_behavior": "pair"})
    assert host._get_unauthorized_dm_behavior(Platform.SIGNAL) == "ignore"


@pytest.mark.parametrize("dm_allowlist", [None, "alice"])
def test_unset_or_nonempty_dm_allowlist_preserves_pairing(monkeypatch, dm_allowlist):
    if dm_allowlist is None:
        monkeypatch.delenv("SIGNAL_ALLOWED_USERS", raising=False)
    else:
        monkeypatch.setenv("SIGNAL_ALLOWED_USERS", dm_allowlist)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "abc")
    host = runner()
    host.pairing_store.is_approved.return_value = True
    assert host._is_user_authorized(source("dm", "alice")) is True


def test_group_only_scope_a_b_a_does_not_leak(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "*")
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "*")
    host = runner()
    host.pairing_store.is_approved.return_value = True
    for name, secrets, expected_dm in [
        ("a", {"SIGNAL_ALLOWED_USERS": "", "SIGNAL_GROUP_ALLOWED_USERS": "abc"}, False),
        ("b", {"SIGNAL_ALLOWED_USERS": "alice"}, True),
        ("a", {"SIGNAL_ALLOWED_USERS": "", "SIGNAL_GROUP_ALLOWED_USERS": "abc"}, False),
    ]:
        home = tmp_path / name
        home.mkdir(exist_ok=True)
        token = set_secret_scope(secrets, profile_home=str(home))
        try:
            assert host._is_user_authorized(source("dm", "alice")) is expected_dm
            if not expected_dm:
                assert host._get_unauthorized_dm_behavior(Platform.SIGNAL) == "ignore"
                assert host._is_user_authorized(source()) is True
        finally:
            reset_secret_scope(token)


@pytest.mark.asyncio
async def test_real_signal_intake_and_pairing_store_keep_groups_only(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.pairing import PairingStore
    from gateway.platforms.signal import SignalAdapter

    host = runner()
    host.pairing_store = PairingStore()
    host.pairing_store._approve_user("signal", "alice")
    assert host.pairing_store.is_approved("signal", "alice")
    # Simulate disabling DMs after an earlier pairing approval.
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "")
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "abc")
    adapter = SignalAdapter(PlatformConfig(enabled=True, extra={
        "http_url": "http://localhost:8080", "account": "+15550000000",
    }))
    host.adapters[Platform.SIGNAL] = adapter
    verdicts = []

    async def admit(event):
        verdicts.append(host._is_user_authorized(event.source))

    adapter.handle_message = admit
    for group in (None, "abc", "other"):
        message = {"message": "hello"}
        if group:
            message["groupInfo"] = {"groupId": group}
        await adapter._handle_envelope({"sourceNumber": "alice", "dataMessage": message})
    assert verdicts == [False, True]  # unlisted group is rejected at intake
    assert host._get_unauthorized_dm_behavior(Platform.SIGNAL) == "ignore"

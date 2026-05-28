import re
from dataclasses import FrozenInstanceError

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.permissions import (
    AuthDecision,
    PermissionManager,
    PermissionSnapshot,
    PlatformPermissionSnapshot,
)
from gateway.session import SessionSource


@pytest.fixture(autouse=True)
def _clear_telegram_permission_env(monkeypatch):
    for name in (
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_ALLOWED_CHATS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
        "TELEGRAM_ALLOWED_TOPICS",
        "TELEGRAM_FREE_RESPONSE_CHATS",
        "TELEGRAM_MENTION_PATTERNS",
        "TELEGRAM_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(name, raising=False)


class _PairingStore:
    def __init__(self, approved=None, fail=False):
        self.approved = approved or {}
        self.fail = fail

    def list_approved(self, platform=None):
        if self.fail:
            raise ValueError("invalid approved users json")
        if platform:
            return [
                {"platform": platform, "user_id": user_id, **info}
                for user_id, info in self.approved.get(platform, {}).items()
            ]
        return []


def _config(extra=None):
    return GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="test-token",
                extra=extra or {},
            )
        }
    )


def test_permission_snapshot_normalizes_ids_to_strings():
    snapshot = PlatformPermissionSnapshot(
        platform=Platform.TELEGRAM,
        approved_users={123},
        allowed_users=[456, "789"],
        group_allowed_users=(),
        allowed_chats=[-100, "-200"],
        group_allowed_chats={-300},
        allowed_topics=[1],
        free_response_chats=[],
        mention_patterns=["rei"],
        allow_all=False,
        allow_bots=False,
        extra={"require_mention": True},
    )

    assert snapshot.approved_users == frozenset({"123"})
    assert snapshot.allowed_users == frozenset({"456", "789"})
    assert snapshot.allowed_chats == frozenset({"-100", "-200"})
    assert snapshot.group_allowed_chats == frozenset({"-300"})
    assert snapshot.allowed_topics == frozenset({"1"})
    assert snapshot.compiled_mention_patterns[0].search("oi rei")


def test_permission_snapshot_is_immutable():
    snapshot = PlatformPermissionSnapshot(platform=Platform.TELEGRAM, allowed_users=["1"])

    try:
        snapshot.allowed_users = frozenset({"2"})
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("snapshot should be frozen")


def test_permission_manager_loads_approved_pairing_users():
    manager = PermissionManager(
        config_loader=lambda: _config(),
        pairing_store=_PairingStore({"telegram": {"42": {"user_name": "Alice"}}}),
    )

    result = manager.reload()

    assert result.ok is True
    telegram = manager.snapshot.platforms[Platform.TELEGRAM]
    assert telegram.approved_users == frozenset({"42"})


def test_permission_manager_rejects_invalid_pairing_without_swapping():
    pairing = _PairingStore({"telegram": {"42": {}}})
    manager = PermissionManager(config_loader=lambda: _config(), pairing_store=pairing)
    assert manager.reload().ok is True
    before = manager.snapshot

    pairing.fail = True
    result = manager.reload()

    assert result.ok is False
    assert "approved users" in result.reason
    assert manager.snapshot is before


def test_permission_manager_loads_telegram_allowed_chats_from_config():
    manager = PermissionManager(
        config_loader=lambda: _config(
            {
                "allow_from": ["10"],
                "group_allow_from": ["11"],
                "allowed_chats": [-100],
                "group_allowed_chats": ["-200"],
                "allowed_topics": [1],
                "free_response_chats": ["-300"],
                "require_mention": True,
            }
        ),
        pairing_store=_PairingStore(),
    )

    assert manager.reload().ok is True
    telegram = manager.snapshot.platforms[Platform.TELEGRAM]
    assert telegram.allowed_users == frozenset({"10"})
    assert telegram.group_allowed_users == frozenset({"11"})
    assert telegram.allowed_chats == frozenset({"-100"})
    assert telegram.group_allowed_chats == frozenset({"-200"})
    assert telegram.allowed_topics == frozenset({"1"})
    assert telegram.extra["require_mention"] is True


def test_permission_manager_uses_explicit_empty_config_over_stale_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    manager = PermissionManager(
        config_loader=lambda: _config({"allow_from": []}),
        pairing_store=_PairingStore(),
    )

    assert manager.reload().ok is True
    telegram = manager.snapshot.platforms[Platform.TELEGRAM]
    assert "42" not in telegram.allowed_users


def test_permission_manager_rejects_invalid_mention_regex_and_keeps_old_snapshot():
    good = _config({"mention_patterns": ["rei"]})
    bad = _config({"mention_patterns": ["["]})
    configs = iter([good, bad])
    manager = PermissionManager(config_loader=lambda: next(configs), pairing_store=_PairingStore())

    assert manager.reload().ok is True
    before = manager.snapshot
    result = manager.reload()

    assert result.ok is False
    assert "mention pattern" in result.reason
    assert manager.snapshot is before


def test_auth_decision_allows_approved_user():
    manager = PermissionManager(
        config_loader=lambda: _config(),
        pairing_store=_PairingStore({"telegram": {"42": {}}}),
    )
    manager.reload()

    decision = manager.authorize(
        SessionSource(platform=Platform.TELEGRAM, user_id="42", chat_id="42", chat_type="dm")
    )

    assert decision == AuthDecision(True, "approved_user", Platform.TELEGRAM, "42", "42")


def test_auth_decision_allows_group_allowed_chat_without_user_id():
    manager = PermissionManager(
        config_loader=lambda: _config({"group_allowed_chats": ["-100"]}),
        pairing_store=_PairingStore(),
    )
    manager.reload()

    decision = manager.authorize(
        SessionSource(platform=Platform.TELEGRAM, user_id=None, chat_id="-100", chat_type="group")
    )

    assert decision.allowed is True
    assert decision.reason == "group_allowed_chat"


def test_auth_decision_denies_unknown_user_and_chat():
    manager = PermissionManager(
        config_loader=lambda: _config({"allow_from": ["1"]}),
        pairing_store=_PairingStore(),
    )
    manager.reload()

    decision = manager.authorize(
        SessionSource(platform=Platform.TELEGRAM, user_id="2", chat_id="2", chat_type="dm")
    )

    assert decision.allowed is False
    assert decision.reason == "not_allowed"

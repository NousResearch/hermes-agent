"""WHATSAPP_GROUP_ALLOWED_SENDERS: a group-scoped sender allowlist for WhatsApp (#41371).

Telegram parity (``TELEGRAM_GROUP_ALLOWED_USERS``): a sender listed here may use the bot in a
group the bridge admitted, without DM access. ``WHATSAPP_GROUP_ALLOWED_USERS`` already means
group JIDs (``group_allow_from``), so the sender list needs its own name.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource

OWNER = "15550000001"
MEMBER = "15550000002"
GROUP_JID = "120363000000000001@g.us"


@pytest.fixture(autouse=True)
def _clean_whatsapp_auth_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in (
        "WHATSAPP_ALLOWED_USERS", "WHATSAPP_ALLOW_ALL_USERS", "WHATSAPP_GROUP_ALLOWED_USERS",
        "WHATSAPP_GROUP_ALLOWED_SENDERS", "WHATSAPP_GROUP_ALLOW_FROM", "WHATSAPP_DM_POLICY",
        "WHATSAPP_GROUP_POLICY", "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.setenv(key, "")  # registers the undo, so values a test bridges in do not leak
        monkeypatch.delenv(key)


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)})
    runner.adapters = {Platform.WHATSAPP: SimpleNamespace()}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    return runner


def _group(user_id: str) -> SessionSource:
    return SessionSource(platform=Platform.WHATSAPP, user_id=user_id, chat_id=GROUP_JID,
                         user_name="tester", chat_type="group")


def _dm(user_id: str) -> SessionSource:
    return SessionSource(platform=Platform.WHATSAPP, user_id=user_id, chat_id=f"{user_id}@s.whatsapp.net",
                         user_name="tester", chat_type="dm")


def test_group_sender_is_authorized_in_group_but_not_in_dm(monkeypatch):
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", OWNER)
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_SENDERS", MEMBER)
    runner = _runner()

    assert runner._is_user_authorized(_group(f"{MEMBER}@s.whatsapp.net")) is True
    assert runner._is_user_authorized(_dm(f"{MEMBER}@s.whatsapp.net")) is False
    # The platform-wide allowlist keeps working in both scopes.
    assert runner._is_user_authorized(_group(f"{OWNER}@s.whatsapp.net")) is True
    assert runner._is_user_authorized(_dm(f"{OWNER}@s.whatsapp.net")) is True


def test_unlisted_group_sender_stays_rejected_without_the_new_allowlist(monkeypatch):
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", OWNER)
    runner = _runner()

    assert runner._is_user_authorized(_group(f"{MEMBER}@s.whatsapp.net")) is False
    assert runner._is_user_authorized(_group(f"{OWNER}@s.whatsapp.net")) is True


def test_group_sender_listed_by_phone_matches_lid_sender(monkeypatch, tmp_path):
    """Group participants often arrive as ``<id>@lid``; the phone-number entry must still match."""
    lid = "900000000000002"
    session_dir = tmp_path / "platforms" / "whatsapp" / "session"
    session_dir.mkdir(parents=True)
    (session_dir / f"lid-mapping-{MEMBER}.json").write_text(f'"{lid}"', encoding="utf-8")
    (session_dir / f"lid-mapping-{lid}_reverse.json").write_text(f'"{MEMBER}"', encoding="utf-8")
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", OWNER)
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_SENDERS", MEMBER)
    runner = _runner()

    assert runner._is_user_authorized(_group(f"{lid}@lid")) is True
    assert runner._is_user_authorized(_dm(f"{lid}@lid")) is False


def test_config_yaml_group_allowed_senders_bridges_to_env():
    from plugins.platforms.whatsapp.adapter import _apply_yaml_config

    extra = _apply_yaml_config({}, {"group_allowed_senders": [MEMBER], "group_allow_from": [GROUP_JID]})

    assert extra["group_allowed_senders"] == [MEMBER]
    assert MEMBER in (os.environ.get("WHATSAPP_GROUP_ALLOWED_SENDERS") or "")
    # The group-JID list keeps its own carrier; the two never mix.
    assert MEMBER not in (os.environ.get("WHATSAPP_GROUP_ALLOWED_USERS") or "")

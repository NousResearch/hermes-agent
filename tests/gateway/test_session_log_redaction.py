"""Session diagnostics mask WhatsApp identities while live session keys remain exact."""

import logging
from unittest.mock import Mock

import pytest

from gateway.config import Platform
from gateway.log_redaction import log_safe_gateway_identity, session_key_for_log
from gateway.session import SessionSource, SessionStore, build_session_key


@pytest.mark.parametrize("platform", [Platform.WHATSAPP, Platform.WHATSAPP_CLOUD, Platform.TELEGRAM])
def test_active_process_failure_keeps_session_and_raw_lookup_key(caplog, platform):
    number = "15551234567"
    source = SessionSource(platform=platform, chat_id=number, user_id=number)
    key = build_session_key(source)
    store = SessionStore.__new__(SessionStore)
    store._has_active_processes_fn = Mock(side_effect=RuntimeError(f"lookup failed for {number}"))

    with caplog.at_level(logging.WARNING, logger="gateway.session"):
        assert store._has_active_processes_safe(key, context="expiry") is True

    store._has_active_processes_fn.assert_called_once_with(key)
    assert number in key
    if platform == Platform.TELEGRAM:
        assert f"lookup failed for {number}" in caplog.text
    else:
        assert number not in caplog.text
        assert "RuntimeError" in caplog.text


@pytest.mark.parametrize("platform", [Platform.WHATSAPP, Platform.WHATSAPP_CLOUD])
def test_scoped_identity_projection_preserves_runtime_identity(platform):
    value = "15551234567"
    assert log_safe_gateway_identity(platform, value) == "15****67"
    assert value == "15551234567"
    assert log_safe_gateway_identity(platform, "opaque-private-identity") == "present"
    assert log_safe_gateway_identity(platform, None) == "absent"
    assert log_safe_gateway_identity(Platform.TELEGRAM, value) == value


def test_platform_must_occupy_the_session_key_platform_slot():
    key = "agent:main:telegram:dm:whatsapp_cloud:15551234567"
    assert session_key_for_log(key) == key


@pytest.mark.parametrize("identity", ["Private Name 15551234567", "123456789012345678@lid", "private-name:15551234567"])
@pytest.mark.parametrize("platform", [Platform.WHATSAPP, Platform.WHATSAPP_CLOUD])
def test_non_phone_identifiers_and_mixed_names_are_presence_only(identity, platform):
    # Historical or malformed keys still reach diagnostic helpers directly.
    key = f"agent:main:{platform.value}:dm:{identity}"
    assert log_safe_gateway_identity(platform, identity) == "present"
    assert session_key_for_log(key) == f"agent:main:{platform.value}:dm:present"
    assert identity in key
    assert log_safe_gateway_identity(Platform.TELEGRAM, identity) == identity


@pytest.mark.parametrize("identity", ["15551234567", "15551234567@s.whatsapp.net", "+15551234567"])
def test_diagnostic_identity_projection_is_idempotent(identity):
    safe = log_safe_gateway_identity(Platform.WHATSAPP, identity)
    assert log_safe_gateway_identity(Platform.WHATSAPP, safe) == safe


def test_live_whatsapp_lid_key_is_opaque_only_in_logs():
    source = SessionSource(platform=Platform.WHATSAPP, chat_id="123456789012345678@lid", user_id="123456789012345678@lid")
    key = build_session_key(source)
    assert session_key_for_log(key) == "agent:main:whatsapp:dm:present"
    assert key == "agent:main:whatsapp:dm:123456789012345678"
    assert source.chat_id == source.user_id == "123456789012345678@lid"

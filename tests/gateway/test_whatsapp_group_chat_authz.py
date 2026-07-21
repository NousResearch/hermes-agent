"""Gateway authz must treat WHATSAPP_GROUP_ALLOWED_USERS as chat-scoped.

WHATSAPP_GROUP_ALLOWED_USERS holds group JIDs (same shape as
TELEGRAM_GROUP_ALLOWED_CHATS), not sender user IDs. Without wiring it into
GatewayAuthorizationMixin's chat-allowlist maps, a message that already
passed the adapter's group_policy still hits Unauthorized user for any
sender who is not also on WHATSAPP_ALLOWED_USERS — so customer-support
groups only work for the owner DM allowlist.

This is the multi-member / support-group path: any participant in an
allowlisted @g.us chat is authorized; DMs stay on the DM allowlist.
"""

from __future__ import annotations

import os

import pytest

from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.config import Platform
from gateway.session import SessionSource


class _Authz(GatewayAuthorizationMixin):
    def __init__(self):
        self.adapters = {}
        self.config = None
        self.pairing_store = None
        self.pairing_stores = {}


def _clear_allow_env(monkeypatch):
    for key in list(os.environ):
        if any(
            tok in key
            for tok in (
                "ALLOW",
                "WHATSAPP",
                "GATEWAY",
                "TELEGRAM",
                "DISCORD",
                "SIGNAL",
                "SLACK",
            )
        ):
            monkeypatch.delenv(key, raising=False)


def _src(**kwargs) -> SessionSource:
    return SessionSource(platform=Platform.WHATSAPP, **kwargs)


def test_allowlisted_group_authorizes_any_member(monkeypatch):
    _clear_allow_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "15550000001")
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", "120363001234567890@g.us")
    authz = _Authz()

    assert authz._is_user_authorized(
        _src(
            chat_id="120363001234567890@g.us",
            chat_type="group",
            user_id="999888777@lid",
            user_name="Customer",
        )
    )
    assert authz._is_user_authorized(
        _src(
            chat_id="120363001234567890@g.us",
            chat_type="group",
            user_id="15550000001",
            user_name="Owner",
        )
    )


def test_non_allowlisted_group_still_denies_strangers(monkeypatch):
    _clear_allow_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "15550000001")
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", "120363001234567890@g.us")
    authz = _Authz()

    assert not authz._is_user_authorized(
        _src(
            chat_id="120363009999999999@g.us",
            chat_type="group",
            user_id="999888777@lid",
            user_name="Customer",
        )
    )


def test_dm_stranger_still_denied(monkeypatch):
    _clear_allow_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "15550000001")
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", "120363001234567890@g.us")
    authz = _Authz()

    assert not authz._is_user_authorized(
        _src(
            chat_id="999888777@s.whatsapp.net",
            chat_type="dm",
            user_id="999888777",
            user_name="Stranger",
        )
    )
    assert authz._is_user_authorized(
        _src(
            chat_id="15550000001@s.whatsapp.net",
            chat_type="dm",
            user_id="15550000001",
            user_name="Owner",
        )
    )


def test_bare_group_id_in_env_matches_full_jid(monkeypatch):
    _clear_allow_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", "120363001234567890")
    authz = _Authz()

    assert authz._is_user_authorized(
        _src(
            chat_id="120363001234567890@g.us",
            chat_type="group",
            user_id="anyone@lid",
            user_name="Customer",
        )
    )


def test_wildcard_group_allowlist(monkeypatch):
    _clear_allow_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", "*")
    authz = _Authz()

    assert authz._is_user_authorized(
        _src(
            chat_id="120363001234567890@g.us",
            chat_type="group",
            user_id="anyone@lid",
            user_name="Customer",
        )
    )

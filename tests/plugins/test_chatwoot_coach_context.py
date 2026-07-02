"""Unit tests for the CRWD Coach context injector.

Covers the member-id resolver (contact-attr hit, Mongo fallback, cache, platform
gate) and the pre_llm_call hook, with the Chatwoot HTTP GET and Mongo lookup
mocked. No live Chatwoot/Mongo is touched.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from plugins.platforms.chatwoot import coach_context as cc


@pytest.fixture(autouse=True)
def _clear_cache():
    cc._reset_cache()
    yield
    cc._reset_cache()


@pytest.fixture
def chatwoot_env(monkeypatch):
    monkeypatch.setenv("CHATWOOT_BASE_URL", "https://chat.example.com")
    monkeypatch.setenv("CHATWOOT_TOKEN", "bot-tok")
    monkeypatch.setenv("CRWD_MONGO_URI", "mongodb://localhost:27017/")
    # Session context: chatwoot conversation "7:42".
    monkeypatch.setattr(
        "gateway.session_context.get_session_env",
        lambda name, default="": {
            "HERMES_SESSION_PLATFORM": "chatwoot",
            "HERMES_SESSION_CHAT_ID": "7:42",
        }.get(name, default),
    )


class TestResolve:
    def test_returns_id_from_contact_attribute(self, chatwoot_env):
        contact = {"custom_attributes": {"joincrwd_user_id": "abc123"}}
        with patch.object(cc, "_get_contact", return_value=contact) as g:
            assert cc.resolve_member_crwd_id("55") == "abc123"
            g.assert_called_once_with("7", "55")

    def test_falls_back_to_mongo_when_attr_missing(self, chatwoot_env):
        contact = {"custom_attributes": {}, "email": "m@x.com", "phone_number": "+1555"}
        with patch.object(cc, "_get_contact", return_value=contact), patch(
            "plugins.platforms.chatwoot.enrichment.fetch_user",
            return_value={"_id": "deadbeef"},
        ) as fu:
            assert cc.resolve_member_crwd_id("55") == "deadbeef"
            fu.assert_called_once_with("m@x.com", "+1555")

    def test_none_when_no_contact_and_no_match(self, chatwoot_env):
        with patch.object(cc, "_get_contact", return_value=None):
            assert cc.resolve_member_crwd_id("55") is None

    def test_caches_result_no_second_http(self, chatwoot_env):
        contact = {"custom_attributes": {"joincrwd_user_id": "abc123"}}
        with patch.object(cc, "_get_contact", return_value=contact) as g:
            assert cc.resolve_member_crwd_id("55") == "abc123"
            assert cc.resolve_member_crwd_id("55") == "abc123"
            g.assert_called_once()

    def test_blank_contact_id_returns_none(self, chatwoot_env):
        assert cc.resolve_member_crwd_id("") is None


class TestHook:
    def test_injects_context_when_resolved(self, chatwoot_env):
        with patch.object(cc, "resolve_member_crwd_id", return_value="abc123"):
            out = cc.member_context_hook(platform="chatwoot", sender_id="55")
        assert out is not None
        assert "abc123" in out["context"]
        assert "get_user_gigs" in out["context"]

    def test_none_off_chatwoot(self, chatwoot_env):
        with patch.object(cc, "_is_chatwoot", return_value=False):
            assert cc.member_context_hook(platform="telegram", sender_id="55") is None

    def test_none_without_mongo_uri(self, chatwoot_env, monkeypatch):
        monkeypatch.delenv("CRWD_MONGO_URI", raising=False)
        assert cc.member_context_hook(platform="chatwoot", sender_id="55") is None

    def test_none_without_sender_id(self, chatwoot_env):
        assert cc.member_context_hook(platform="chatwoot", sender_id="") is None

    def test_none_when_unresolved(self, chatwoot_env):
        with patch.object(cc, "resolve_member_crwd_id", return_value=None):
            assert cc.member_context_hook(platform="chatwoot", sender_id="55") is None

    def test_hook_never_raises(self, chatwoot_env):
        with patch.object(cc, "resolve_member_crwd_id", side_effect=RuntimeError("boom")):
            assert cc.member_context_hook(platform="chatwoot", sender_id="55") is None

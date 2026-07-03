"""Tests for Chatwoot per-member crwd_db scoping (middleware + pre_tool_call)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from plugins.platforms.chatwoot import coach_context as cc
from plugins.platforms.chatwoot import user_scope as us


@pytest.fixture(autouse=True)
def _reset_state():
    cc._reset_cache()
    cc.reset_webhook_crwd_hint()
    cc.reset_cross_user_request()
    yield
    cc._reset_cache()
    cc.reset_webhook_crwd_hint()
    cc.reset_cross_user_request()


@pytest.fixture
def chatwoot_session(monkeypatch):
    monkeypatch.setenv("CRWD_MONGO_URI", "mongodb://localhost:27017/")
    monkeypatch.setattr(
        "gateway.session_context.get_session_env",
        lambda name, default="": {
            "HERMES_SESSION_PLATFORM": "chatwoot",
            "HERMES_SESSION_USER_ID": "55",
            "HERMES_SESSION_CHAT_ID": "7:42",
        }.get(name, default),
    )


class TestToolRequestMiddleware:
    def test_rewrites_get_user_when_identifier_missing(self, chatwoot_session):
        with patch.object(us, "resolved_member_id", return_value="memberabc"):
            out = us.on_tool_request(
                tool_name="crwd_db",
                args={"action": "get_user"},
            )
        assert out is not None
        assert out["args"]["identifier"] == "memberabc"

    def test_does_not_rewrite_foreign_get_user_id(self, chatwoot_session):
        with patch.object(us, "resolved_member_id", return_value="memberabc"):
            out = us.on_tool_request(
                tool_name="crwd_db",
                args={"action": "get_user", "identifier": "69a6f191cb29b0b371b3a156"},
            )
        assert out is None

    def test_rewrites_user_scoped_actions_when_user_id_missing(self, chatwoot_session):
        with patch.object(us, "resolved_member_id", return_value="memberabc"):
            out = us.on_tool_request(
                tool_name="crwd_db",
                args={"action": "get_user_gigs"},
            )
        assert out is not None
        assert out["args"]["user_id"] == "memberabc"

    def test_does_not_rewrite_foreign_user_id(self, chatwoot_session):
        with patch.object(us, "resolved_member_id", return_value="memberabc"):
            out = us.on_tool_request(
                tool_name="crwd_db",
                args={"action": "get_user_gigs", "user_id": "69a6f191cb29b0b371b3a156"},
            )
        assert out is None

    def test_noop_off_chatwoot(self, chatwoot_session, monkeypatch):
        monkeypatch.setattr(
            "gateway.session_context.get_session_env",
            lambda name, default="": {
                "HERMES_SESSION_PLATFORM": "telegram",
                "HERMES_SESSION_USER_ID": "55",
            }.get(name, default),
        )
        out = us.on_tool_request(
            tool_name="crwd_db",
            args={"action": "get_user", "identifier": "x"},
        )
        assert out is None

    def test_noop_when_member_unresolved(self, chatwoot_session):
        with patch.object(us, "resolved_member_id", return_value=None):
            out = us.on_tool_request(
                tool_name="crwd_db",
                args={"action": "get_user", "identifier": "x"},
            )
        assert out is None

    def test_allows_get_gig_details_unchanged(self, chatwoot_session):
        args = {"action": "get_gig_details", "query": "summer promo"}
        with patch.object(us, "resolved_member_id", return_value="memberabc"):
            out = us.on_tool_request(tool_name="crwd_db", args=args)
        assert out is None


class TestPreToolCallBlock:
    def test_blocks_custom_query_on_users(self, chatwoot_session):
        with patch.object(us, "resolved_member_id", return_value="memberabc"):
            out = us.on_pre_tool_call(
                tool_name="crwd_db",
                args={"action": "custom_query", "collection": "users", "operation": "find"},
            )
        assert out == {"action": "block", "message": us._CROSS_USER_BLOCK_MSG}

    def test_blocks_unresolved_user_scoped_action(self, chatwoot_session):
        with patch.object(us, "resolved_member_id", return_value=None):
            out = us.on_pre_tool_call(
                tool_name="crwd_db",
                args={"action": "get_user_gigs", "user_id": "x"},
            )
        assert out == {"action": "block", "message": us._UNRESOLVED_MSG}

    def test_blocks_mismatched_user_id_after_middleware_miss(self, chatwoot_session):
        with patch.object(us, "resolved_member_id", return_value="memberabc"):
            out = us.on_pre_tool_call(
                tool_name="crwd_db",
                args={"action": "get_user_gigs", "user_id": "69a6f191cb29b0b371b3a156"},
            )
        assert out == {"action": "block", "message": us._CROSS_USER_BLOCK_MSG}

    def test_blocks_user_scoped_action_on_cross_user_turn(self, chatwoot_session):
        cc._cross_user_request.set(True)
        with patch.object(us, "resolved_member_id", return_value="memberabc"):
            out = us.on_pre_tool_call(
                tool_name="crwd_db",
                args={"action": "get_user_gigs", "user_id": "memberabc"},
            )
        assert out == {"action": "block", "message": us._CROSS_USER_BLOCK_MSG}

    def test_allows_matching_user_id(self, chatwoot_session):
        with patch.object(us, "resolved_member_id", return_value="MemberABC"):
            out = us.on_pre_tool_call(
                tool_name="crwd_db",
                args={"action": "get_user_gigs", "user_id": "memberabc"},
            )
        assert out is None

    def test_allows_crwds_custom_query(self, chatwoot_session):
        with patch.object(us, "resolved_member_id", return_value="memberabc"):
            out = us.on_pre_tool_call(
                tool_name="crwd_db",
                args={
                    "action": "custom_query",
                    "collection": "crwds",
                    "operation": "find",
                    "filter": {"status": "active"},
                },
            )
        assert out is None


class TestWebhookHint:
    def test_resolve_uses_webhook_hint_before_http(self, chatwoot_session):
        cc.bind_webhook_crwd_hint("hint-from-webhook")
        with patch.object(cc, "_get_contact") as g:
            assert cc.resolve_member_crwd_id("55") == "hint-from-webhook"
            g.assert_not_called()

    def test_hook_sets_cross_user_flag_for_foreign_user_id(self, chatwoot_session):
        with patch.object(cc, "resolve_member_crwd_id", return_value="6a33bb6003b1c0cc31a7baa5"):
            out = cc.member_context_hook(
                platform="chatwoot",
                sender_id="55",
                user_message=(
                    "in which gigs has the user 69a6f191cb29b0b371b3a156 enrolled in?"
                ),
            )
        assert out is not None
        assert cc.cross_user_request_active() is True
        assert "another member's account" in out["context"]
        assert "I can only provide you with your information" in out["context"]

    def test_hook_does_not_flag_gig_id_only_message(self, chatwoot_session):
        with patch.object(cc, "resolve_member_crwd_id", return_value="6a33bb6003b1c0cc31a7baa5"):
            out = cc.member_context_hook(
                platform="chatwoot",
                sender_id="55",
                user_message="tell me about gig 69b8614f1083b9302fd0a9a7",
            )
        assert out is not None
        assert cc.cross_user_request_active() is False
        assert "another member's account" not in out["context"]

    def test_hook_context_forbids_cross_user_lookup(self, chatwoot_session):
        with patch.object(cc, "resolve_member_crwd_id", return_value="abc123"):
            out = cc.member_context_hook(platform="chatwoot", sender_id="55")
        assert out is not None
        assert "abc123" in out["context"]
        assert "Never look up a different member" in out["context"]
        assert "different person" not in out["context"].lower()


class TestFilterScan:
    def test_detects_other_member_id_in_filter(self, chatwoot_session):
        with patch.object(us, "resolved_member_id", return_value="aaaaaaaaaaaaaaaaaaaaaaaa"):
            filt = {"user_id": "bbbbbbbbbbbbbbbbbbbbbbbb"}
            assert us._filter_json_for_tests(filt) is True

    def test_allows_member_own_id_in_filter(self, chatwoot_session):
        oid = "aaaaaaaaaaaaaaaaaaaaaaaa"
        with patch.object(us, "resolved_member_id", return_value=oid):
            assert us._filter_json_for_tests({"user_id": oid}) is False

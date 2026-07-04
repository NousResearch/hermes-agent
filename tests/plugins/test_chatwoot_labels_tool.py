"""Tests for the Chatwoot labels tool (no live Chatwoot required)."""

import json
from unittest.mock import patch

import pytest

from plugins.platforms.chatwoot import labels_tool as t


@pytest.fixture
def chatwoot_env(monkeypatch):
    monkeypatch.setenv("CHATWOOT_BASE_URL", "https://chat.example.com")
    monkeypatch.setenv("CHATWOOT_AGENT_TOKEN", "agent-tok")
    monkeypatch.setenv("CHATWOOT_ACCOUNT_ID", "1")
    monkeypatch.delenv("CHATWOOT_TOKEN", raising=False)


class TestAvailability:
    def test_unavailable_without_base_url(self, monkeypatch):
        monkeypatch.delenv("CHATWOOT_BASE_URL", raising=False)
        monkeypatch.setenv("CHATWOOT_AGENT_TOKEN", "x")
        assert t.check_chatwoot_labels_requirements() is False

    def test_unavailable_without_token(self, monkeypatch):
        monkeypatch.setenv("CHATWOOT_BASE_URL", "https://chat.example.com")
        monkeypatch.delenv("CHATWOOT_AGENT_TOKEN", raising=False)
        monkeypatch.delenv("CHATWOOT_TOKEN", raising=False)
        assert t.check_chatwoot_labels_requirements() is False

    def test_available_with_creds(self, chatwoot_env):
        assert t.check_chatwoot_labels_requirements() is True

    def test_falls_back_to_plain_token(self, monkeypatch):
        monkeypatch.setenv("CHATWOOT_BASE_URL", "https://chat.example.com")
        monkeypatch.delenv("CHATWOOT_AGENT_TOKEN", raising=False)
        monkeypatch.setenv("CHATWOOT_TOKEN", "plain-tok")
        assert t.check_chatwoot_labels_requirements() is True
        assert t._agent_token() == "plain-tok"


class TestMergeLabels:
    def test_merge_preserves_existing_order(self):
        assert t._merge_labels(["a"], ["b", "c"]) == ["a", "b", "c"]

    def test_merge_dedupes(self):
        assert t._merge_labels(["payment-payout"], ["payment-payout", "troubleshooting"]) == [
            "payment-payout",
            "troubleshooting",
        ]

    def test_merge_normalizes_case(self):
        assert t._merge_labels(["Gig-Discovery"], ["gig-discovery"]) == ["gig-discovery"]


class TestResolveConversation:
    def _with_session(self, platform, chat_id):
        return patch(
            "gateway.session_context.get_session_env",
            side_effect=lambda name, default="": {
                "HERMES_SESSION_PLATFORM": platform,
                "HERMES_SESSION_CHAT_ID": chat_id,
            }.get(name, default),
        )

    def test_parses_account_and_conversation(self, chatwoot_env):
        with self._with_session("chatwoot", "7:42"):
            assert t._resolve_conversation() == ("7", "42")

    def test_bare_id_uses_account_env(self, chatwoot_env):
        with self._with_session("chatwoot", "42"):
            assert t._resolve_conversation() == ("1", "42")

    def test_override_params(self, chatwoot_env):
        assert t._resolve_conversation("9", "99") == ("9", "99")

    def test_wrong_platform_returns_none(self, chatwoot_env):
        with self._with_session("telegram", "7:42"):
            assert t._resolve_conversation() == (None, None)


class TestCreateLabelsIfNotExists:
    def test_creates_only_missing(self, chatwoot_env):
        existing_payload = {
            "payload": [{"title": "gig-discovery", "id": 1}],
        }

        def fake_api(method, path, body=None):
            if method == "GET" and path.endswith("/labels") and "conversations" not in path:
                return True, existing_payload, ""
            if method == "POST" and path.endswith("/labels"):
                return True, {"payload": body}, ""
            return False, None, "unexpected"

        with patch.object(t, "_api_request", side_effect=fake_api):
            out = t._create_labels_if_not_exists("1")

        assert "gig-discovery" not in out["created"]
        assert len(out["created"]) == 7  # 8 predefined minus gig-discovery

    def test_get_failure(self, chatwoot_env):
        with patch.object(t, "_api_request", return_value=(False, None, "HTTP 401")):
            out = t._create_labels_if_not_exists("1")
        assert out["success"] is False
        assert out["error"] == "HTTP 401"


class TestAssignLabels:
    def test_merge_mode(self, chatwoot_env):
        calls = []

        def fake_api(method, path, body=None):
            calls.append((method, path, body))
            if method == "GET" and "conversations" in path:
                return True, {"payload": ["gig-discovery"]}, ""
            if method == "POST" and "conversations" in path:
                return True, {"payload": body["labels"]}, ""
            if method == "GET":
                return True, {"payload": [{"title": x} for x in t.PREDEFINED_LABEL_TITLES]}, ""
            return False, None, "unexpected"

        with patch.object(t, "_api_request", side_effect=fake_api):
            out = t._assign_labels("1", "42", ["payment-payout", "troubleshooting"], replace=False)

        assert out["success"] is True
        assert out["labels"] == ["gig-discovery", "payment-payout", "troubleshooting"]
        post_calls = [c for c in calls if c[0] == "POST" and "conversations" in c[1]]
        assert post_calls[-1][2] == {
            "labels": ["gig-discovery", "payment-payout", "troubleshooting"],
        }

    def test_replace_mode(self, chatwoot_env):
        def fake_api(method, path, body=None):
            if method == "GET" and "conversations" in path:
                return True, {"payload": ["old-label"]}, ""
            if method == "POST" and "conversations" in path:
                return True, {"payload": body["labels"]}, ""
            if method == "GET":
                return True, {"payload": [{"title": "payment-payout"}]}, ""
            return False, None, "unexpected"

        with patch.object(t, "_api_request", side_effect=fake_api):
            out = t._assign_labels("1", "42", ["payment-payout"], replace=True)

        assert out["success"] is True
        assert out["labels"] == ["payment-payout"]
        assert out["replaced"] is True

    def test_empty_labels_error(self, chatwoot_env):
        out = t._assign_labels("1", "42", [], replace=False)
        assert out["success"] is False

    def test_post_422_returns_error(self, chatwoot_env):
        def fake_api(method, path, body=None):
            if method == "GET" and "conversations" in path:
                return True, {"payload": []}, ""
            if method == "POST" and "conversations" in path:
                return False, {"message": "Invalid labels: fake-label"}, "HTTP 422"
            if method == "GET":
                return True, {"payload": [{"title": "payment-payout"}]}, ""
            return False, None, "unexpected"

        with patch.object(t, "_api_request", side_effect=fake_api):
            out = t._assign_labels("1", "42", ["fake-label"], replace=True)

        assert out["success"] is False
        assert "422" in out["error"]


class TestHandler:
    def test_noop_when_not_configured(self, monkeypatch):
        monkeypatch.delenv("CHATWOOT_BASE_URL", raising=False)
        out = json.loads(t.chatwoot_labels_tool({"action": "assign_labels", "labels": ["a"]}))
        assert out["success"] is False
        assert out["error"] is None

    def test_invalid_action(self, chatwoot_env):
        out = json.loads(t.chatwoot_labels_tool({"action": "bogus"}))
        assert out["success"] is False
        assert "action must be" in out["error"]

    def test_get_all_labels(self, chatwoot_env):
        with patch.object(
            t,
            "_get_all_labels",
            return_value={"success": True, "labels": ["a"], "error": None},
        ):
            out = json.loads(
                t.chatwoot_labels_tool({"action": "get_all_labels", "account_id": "1"})
            )
        assert out["success"] is True
        assert out["labels"] == ["a"]

    def test_assign_no_conversation(self, chatwoot_env):
        with patch.object(t, "_resolve_conversation", return_value=(None, None)):
            out = json.loads(
                t.chatwoot_labels_tool(
                    {"action": "assign_labels", "labels": ["payment-payout"]}
                )
            )
        assert out["success"] is False
        assert "No current Chatwoot conversation" in out["reason"]

    def test_assign_success(self, chatwoot_env):
        with patch.object(t, "_resolve_conversation", return_value=("1", "42")), patch.object(
            t,
            "_assign_labels",
            return_value={"success": True, "labels": ["payment-payout"], "error": None},
        ) as assign:
            out = json.loads(
                t.chatwoot_labels_tool(
                    {
                        "action": "assign_labels",
                        "labels": ["payment-payout", "troubleshooting"],
                        "replace": True,
                    }
                )
            )
        assert out["success"] is True
        assign.assert_called_once_with("1", "42", ["payment-payout", "troubleshooting"], True)

    def test_assign_requires_labels_array(self, chatwoot_env):
        with patch.object(t, "_resolve_conversation", return_value=("1", "42")):
            out = json.loads(t.chatwoot_labels_tool({"action": "assign_labels"}))
        assert out["success"] is False
        assert "labels must be an array" in out["error"]

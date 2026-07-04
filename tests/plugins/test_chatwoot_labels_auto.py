"""Tests for automatic Chatwoot conversation labeling."""

import json
from unittest.mock import patch

import pytest

from plugins.platforms.chatwoot import labels_auto as auto


class TestClassifyConversationLabels:
    def test_amazon_gig_details(self):
        labels = auto.classify_conversation_labels(
            "give me details about the amazon gig ?"
        )
        assert "gig-execution" in labels

    def test_find_gigs(self):
        labels = auto.classify_conversation_labels("what gigs are near me?")
        assert "gig-discovery" in labels

    def test_payment(self):
        labels = auto.classify_conversation_labels("did I get paid yet?")
        assert "payment-payout" in labels

    def test_multi_label_payment_and_troubleshooting(self):
        labels = auto.classify_conversation_labels(
            "my payout page won't load, when will I get paid?"
        )
        assert "payment-payout" in labels
        assert "troubleshooting" in labels

    def test_general_inquiry_fallback(self):
        labels = auto.classify_conversation_labels("hello there")
        assert labels == ["general-inquiry"]

    def test_handoff(self):
        labels = auto.classify_conversation_labels("I'm so frustrated, get me a human")
        assert "handoff-escalation" in labels


class TestAutoLabelConversation:
    @pytest.fixture
    def chatwoot_env(self, monkeypatch):
        monkeypatch.setenv("CHATWOOT_BASE_URL", "https://chat.example.com")
        monkeypatch.setenv("CHATWOOT_AGENT_TOKEN", "agent-tok")
        monkeypatch.setenv("CHATWOOT_ACCOUNT_ID", "1")

    def test_skips_without_creds(self, monkeypatch):
        monkeypatch.delenv("CHATWOOT_BASE_URL", raising=False)
        out = auto.auto_label_conversation("hello")
        assert out["skipped"] is True

    def test_applies_labels(self, chatwoot_env):
        with patch.object(auto, "_resolve_conversation", return_value=("1", "42")), patch.object(
            auto, "_create_labels_if_not_exists",
            return_value={"success": True, "existing": ["gig-execution"]},
        ), patch.object(
            auto, "_assign_labels",
            return_value={"success": True, "labels": ["gig-execution"], "error": None},
        ) as assign:
            out = auto.auto_label_conversation("give me details about the amazon gig")
        assert out["success"] is True
        assert out["classified"] == ["gig-execution"]
        assign.assert_called_once_with("1", "42", ["gig-execution"], replace=True)


class TestAutoLabelHook:
    def test_ignores_non_chatwoot(self):
        with patch.object(auto, "auto_label_conversation") as fn:
            auto.auto_label_hook(platform="telegram", user_message="hi")
        fn.assert_not_called()

    def test_runs_on_chatwoot(self):
        with patch.object(auto, "auto_label_conversation") as fn:
            auto.auto_label_hook(
                platform="chatwoot",
                user_message="give me details about the amazon gig",
            )
        fn.assert_called_once()

    def test_reminder_hook_chatwoot_only(self, monkeypatch):
        monkeypatch.setenv("CHATWOOT_BASE_URL", "https://chat.example.com")
        monkeypatch.setenv("CHATWOOT_AGENT_TOKEN", "tok")
        assert auto.labeling_reminder_hook(platform="chatwoot") is not None
        assert auto.labeling_reminder_hook(platform="cli") is None

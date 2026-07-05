"""Tests for Chatwoot gig context prefetch hook."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from plugins.platforms.chatwoot import coach_context as cc
from plugins.platforms.chatwoot import gig_context as gc


@pytest.fixture(autouse=True)
def _reset():
    cc._reset_cache()
    cc.reset_cross_user_request()
    yield
    cc._reset_cache()
    cc.reset_cross_user_request()


class TestIntentDetection:
    @pytest.mark.parametrize("msg", [
        "what are my next steps?",
        "what's my status?",
        "my gigs",
        "how is Pul Tool going?",
    ])
    def test_gig_intent_matches(self, msg):
        assert gc.should_prefetch_gig_context(msg) is True

    def test_generic_identity_does_not_match(self):
        assert gc.should_prefetch_gig_context("who are you?") is False


class TestGigContextHook:
    def test_skips_non_chatwoot(self, monkeypatch):
        monkeypatch.setenv("CRWD_MONGO_URI", "mongodb://x/")
        assert gc.gig_context_hook(
            platform="telegram",
            sender_id="55",
            user_message="what are my next steps?",
        ) is None

    def test_skips_without_mongo_uri(self, monkeypatch):
        monkeypatch.delenv("CRWD_MONGO_URI", raising=False)
        assert gc.gig_context_hook(
            platform="chatwoot",
            sender_id="55",
            user_message="what are my next steps?",
        ) is None

    def test_skips_non_gig_message(self, monkeypatch):
        monkeypatch.setenv("CRWD_MONGO_URI", "mongodb://x/")
        assert gc.gig_context_hook(
            platform="chatwoot",
            sender_id="55",
            user_message="who are you?",
        ) is None

    def test_injects_context_on_intent(self, monkeypatch):
        monkeypatch.setenv("CRWD_MONGO_URI", "mongodb://x/")
        payload = {
            "_type": "user_gig_status",
            "items": [{
                "gig_id": "g1",
                "gig_name": "Pul Tool",
                "gig_type": "web",
                "stage": "receipt_review",
                "next_step": "Receipt under review.",
                "buy_link": None,
                "handoff_recommended": False,
            }],
        }
        with patch.object(gc, "resolve_member_crwd_id", return_value="user1"), patch(
            "tools.crwd_db_tool.build_user_gig_status",
            return_value=payload,
        ):
            out = gc.gig_context_hook(
                platform="chatwoot",
                sender_id="55",
                user_message="what are my next steps?",
            )
        assert out is not None
        assert "[CRWD gig context]" in out["context"]
        data = json.loads(out["context"].split("\n", 2)[2])
        assert data["active_gigs"][0]["gig_name"] == "Pul Tool"
        assert data["active_gigs"][0]["stage"] == "receipt_review"

    def test_skips_cross_user_turn(self, monkeypatch):
        monkeypatch.setenv("CRWD_MONGO_URI", "mongodb://x/")
        cc._cross_user_request.set(True)
        assert gc.gig_context_hook(
            platform="chatwoot",
            sender_id="55",
            user_message="what are my next steps?",
        ) is None

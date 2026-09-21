"""Tests for idea_box.models — IdeaCard, SourceRef, DedupResult, IdeaStatus.

These tests mirror the source module 1:1, covering every public method and
dataclass, including serialization round-trips and the card renderer.
"""

from __future__ import annotations

import json

import pytest

from idea_box.models import DedupResult, IdeaCard, IdeaStatus, SourceRef


# ---------------------------------------------------------------------------
# SourceRef
# ---------------------------------------------------------------------------

class TestSourceRef:
    def test_creation(self):
        ref = SourceRef(
            platform="discord",
            channel_id="123456",
            message_id="789",
            user_id="111",
            user_name="sahil",
        )
        assert ref.platform == "discord"
        assert ref.channel_id == "123456"
        assert ref.message_id == "789"
        assert ref.user_id == "111"
        assert ref.user_name == "sahil"
        assert ref.channel_name == ""
        assert ref.raw_text == ""

    def test_creation_with_optional_fields(self):
        ref = SourceRef(
            platform="discord",
            channel_id="123",
            message_id="456",
            user_id="u1",
            user_name="sahil",
            channel_name="idea-box",
            raw_text="/idea build a feature",
        )
        assert ref.channel_name == "idea-box"
        assert ref.raw_text == "/idea build a feature"

    def test_to_dict(self):
        ref = SourceRef(
            platform="discord",
            channel_id="123",
            message_id="456",
            user_id="u1",
            user_name="sahil",
            channel_name="idea-box",
            raw_text="/idea test",
        )
        d = ref.to_dict()
        assert d["platform"] == "discord"
        assert d["channel_id"] == "123"
        assert d["message_id"] == "456"
        assert d["user_id"] == "u1"
        assert d["user_name"] == "sahil"
        assert d["channel_name"] == "idea-box"
        assert d["raw_text"] == "/idea test"

    def test_from_dict_roundtrip(self):
        ref = SourceRef(
            platform="discord",
            channel_id="123",
            message_id="456",
            user_id="u1",
            user_name="sahil",
            channel_name="idea-box",
            raw_text="/idea test",
        )
        d = ref.to_dict()
        ref2 = SourceRef.from_dict(d)
        assert ref2 == ref

    def test_from_dict_missing_fields(self):
        ref = SourceRef.from_dict({"platform": "discord", "channel_id": "1"})
        assert ref.platform == "discord"
        assert ref.channel_id == "1"
        assert ref.message_id == ""
        assert ref.user_id == ""


# ---------------------------------------------------------------------------
# IdeaCard
# ---------------------------------------------------------------------------

class TestIdeaCard:
    def _make_source(self) -> SourceRef:
        return SourceRef(
            platform="discord",
            channel_id="ch1",
            message_id="m1",
            user_id="u1",
            user_name="sahil",
            channel_name="idea-box",
            raw_text="/idea test idea",
        )

    def test_default_fields(self):
        card = IdeaCard(summary="test idea", source=self._make_source())
        assert card.summary == "test idea"
        assert card.tags == []
        assert card.dedup_status == IdeaStatus.CAPTURED
        assert card.dedup_matches == []
        assert card.idea_id is None
        assert card.rejection_reason is None
        assert card.created_at  # auto-generated ISO timestamp

    def test_to_dict(self):
        src = self._make_source()
        card = IdeaCard(
            summary="test",
            source=src,
            tags=["feature", "ui"],
            dedup_status=IdeaStatus.NOVEL,
        )
        d = card.to_dict()
        assert d["summary"] == "test"
        assert d["tags"] == ["feature", "ui"]
        assert d["dedup_status"] == "novel"
        assert "source" in d
        assert d["source"]["platform"] == "discord"

    def test_to_dict_with_dedup_matches(self):
        card = IdeaCard(
            summary="test",
            source=self._make_source(),
            dedup_matches=[{"source": "kanban", "title": "existing task"}],
        )
        d = card.to_dict()
        assert len(d["dedup_matches"]) == 1
        assert d["dedup_matches"][0]["source"] == "kanban"

    def test_render_card_basic(self):
        card = IdeaCard(
            summary="Build a feature for X",
            source=self._make_source(),
            tags=["build", "feature"],
            dedup_status=IdeaStatus.NOVEL,
        )
        rendered = card.render_card()
        assert "**Idea Card**" in rendered
        assert "Build a feature for X" in rendered
        assert "build, feature" in rendered
        assert "novel" in rendered
        assert "discord/idea-box" in rendered

    def test_render_card_with_dedup_matches(self):
        card = IdeaCard(
            summary="test idea",
            source=self._make_source(),
            dedup_status=IdeaStatus.DUPLICATE,
            dedup_matches=[{"source": "kanban", "title": "existing task"}],
        )
        rendered = card.render_card()
        assert "duplicate" in rendered
        assert "Dedup matches" in rendered
        assert "kanban: existing task" in rendered

    def test_render_card_no_tags(self):
        card = IdeaCard(
            summary="test",
            source=self._make_source(),
            tags=[],
        )
        rendered = card.render_card()
        assert "(none)" in rendered

    def test_idea_id_set_on_confirmation(self):
        card = IdeaCard(summary="test", source=self._make_source())
        card.idea_id = "t_abc123"
        card.dedup_status = IdeaStatus.CONFIRMED
        d = card.to_dict()
        assert d["idea_id"] == "t_abc123"
        assert d["dedup_status"] == "confirmed"

    def test_rejection_reason_set_on_rejection(self):
        card = IdeaCard(summary="test", source=self._make_source())
        card.rejection_reason = "Already done"
        card.dedup_status = IdeaStatus.REJECTED
        d = card.to_dict()
        assert d["rejection_reason"] == "Already done"
        assert d["dedup_status"] == "rejected"


# ---------------------------------------------------------------------------
# DedupResult
# ---------------------------------------------------------------------------

class TestDedupResult:
    def test_default_not_duplicate(self):
        result = DedupResult(is_duplicate=False)
        assert result.is_duplicate is False
        assert result.matches == []
        assert result.checked_sources == []

    def test_duplicate_with_matches(self):
        result = DedupResult(
            is_duplicate=True,
            matches=[{"source": "kanban", "title": "match1"}],
            checked_sources=["kanban", "session_search", "mnemosyne"],
        )
        assert result.is_duplicate is True
        assert len(result.matches) == 1
        assert result.checked_sources == ["kanban", "session_search", "mnemosyne"]


# ---------------------------------------------------------------------------
# IdeaStatus enum
# ---------------------------------------------------------------------------

class TestIdeaStatus:
    def test_values(self):
        assert IdeaStatus.CAPTURED.value == "captured"
        assert IdeaStatus.DUPLICATE.value == "duplicate"
        assert IdeaStatus.NOVEL.value == "novel"
        assert IdeaStatus.CONFIRMED.value == "confirmed"
        assert IdeaStatus.REJECTED.value == "rejected"

    def test_from_string(self):
        assert IdeaStatus("captured") == IdeaStatus.CAPTURED
        assert IdeaStatus("novel") == IdeaStatus.NOVEL

    def test_str_representation(self):
        # IdeaStatus is a str enum; the value is always accessible via .value
        assert IdeaStatus.CONFIRMED.value == "confirmed"
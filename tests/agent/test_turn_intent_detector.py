"""Unit tests for agent.turn_intent_detector (S-0518-01 direction B)."""

from __future__ import annotations

import pytest

from agent import turn_intent_detector as tid


# =========================================================================
# detect_turn_intent — skip paths + happy paths
# =========================================================================

class TestDetectTurnIntent:
    def test_short_message_is_skipped(self, monkeypatch):
        called = {"aux": False}

        def _fake_call_llm(**kwargs):
            called["aux"] = True
            raise AssertionError("should not be called")

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm", _fake_call_llm, raising=False
        )
        result = tid.detect_turn_intent("ok")
        assert result["checked"] is False
        assert result["skipped"] == "msg_too_short"
        assert result["route_to_subagent"] is False
        assert called["aux"] is False

    def test_route_true_returns_full_slots(self, monkeypatch):
        class _FakeMsg:
            content = (
                '{"route_to_subagent": true, "sub_agent": "analyst", '
                '"id_slug": "draft-interview-prep-flashcards", '
                '"suggested_action": "Draft interview-prep flashcards", '
                '"suggested_announcement": "Analyst will put the '
                'flashcards together.", "confidence": "high", '
                '"reasoning": "User explicitly asks for a saveable '
                'reference artifact."}'
            )

        class _FakeChoice:
            message = _FakeMsg()

        class _FakeResponse:
            choices = [_FakeChoice()]

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _FakeResponse(),
            raising=False,
        )

        result = tid.detect_turn_intent(
            "interview's on friday — can you put together flashcards "
            "covering the common product-sense questions?"
        )

        assert result["checked"] is True
        assert result["route_to_subagent"] is True
        assert result["sub_agent"] == "analyst"
        assert result["id_slug"] == "draft-interview-prep-flashcards"
        assert "flashcards" in result["suggested_action"]
        assert "Analyst will" in result["suggested_announcement"]
        assert result["confidence"] == "high"

    def test_route_false_for_emotional_turn(self, monkeypatch):
        class _FakeMsg:
            content = (
                '{"route_to_subagent": false, "sub_agent": null, '
                '"suggested_action": null, "suggested_announcement": '
                'null, "confidence": "high", '
                '"reasoning": "User is sharing how they feel, not asking '
                'for a deliverable."}'
            )

        class _FakeChoice:
            message = _FakeMsg()

        class _FakeResponse:
            choices = [_FakeChoice()]

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _FakeResponse(),
            raising=False,
        )

        result = tid.detect_turn_intent(
            "i'm just feeling really stuck about all this honestly"
        )

        assert result["checked"] is True
        assert result["route_to_subagent"] is False
        assert result["sub_agent"] is None
        assert result["suggested_action"] is None

    def test_aux_failure_is_silent(self, monkeypatch):
        def _fake_call_llm(**kwargs):
            raise RuntimeError("network timeout")

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm", _fake_call_llm, raising=False
        )

        result = tid.detect_turn_intent("can you draft me a cover letter?")
        assert result["checked"] is False
        assert "aux_call_failed" in result["skipped"]
        assert result["route_to_subagent"] is False

    def test_aux_returns_garbage(self, monkeypatch):
        class _FakeMsg:
            content = "not json"

        class _FakeChoice:
            message = _FakeMsg()

        class _FakeResponse:
            choices = [_FakeChoice()]

        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: _FakeResponse(),
            raising=False,
        )

        result = tid.detect_turn_intent("can you draft me a cover letter?")
        assert result["checked"] is False
        assert result["skipped"] == "aux_parse_failed"

    def test_empty_user_message(self, monkeypatch):
        called = {"aux": False}
        monkeypatch.setattr(
            "agent.auxiliary_client.call_llm",
            lambda **kw: (called.__setitem__("aux", True), None)[1],
            raising=False,
        )
        result = tid.detect_turn_intent("")
        assert result["checked"] is False
        assert result["skipped"] == "msg_too_short"
        assert called["aux"] is False


# =========================================================================
# render_injection_block
# =========================================================================

class TestRenderInjectionBlock:
    def test_renders_when_route_true(self):
        detection = {
            "checked": True,
            "route_to_subagent": True,
            "sub_agent": "analyst",
            "id_slug": "draft-flashcards",
            "suggested_action": "Draft interview-prep flashcards",
            "suggested_announcement": "Analyst will draft them.",
        }
        block = tid.render_injection_block(detection)
        assert block is not None
        assert "Detected user intent" in block
        assert "analyst" in block
        assert 'id="coach-commit-draft-flashcards"' in block
        assert "Draft interview-prep flashcards" in block
        assert "Analyst will draft them." in block
        assert "enqueue_action" in block
        assert "announce_subagent" in block

    def test_returns_none_when_route_false(self):
        detection = {
            "checked": True,
            "route_to_subagent": False,
        }
        assert tid.render_injection_block(detection) is None

    def test_returns_none_when_not_checked(self):
        detection = {"checked": False, "route_to_subagent": True}
        assert tid.render_injection_block(detection) is None

    def test_returns_none_when_required_slots_missing(self):
        """Defensive — if the LLM forgot a field, skip injection
        rather than inject malformed guidance."""
        detection = {
            "checked": True,
            "route_to_subagent": True,
            "sub_agent": "analyst",
            "id_slug": "x",
            "suggested_action": None,  # missing
            "suggested_announcement": "x",
        }
        assert tid.render_injection_block(detection) is None

    def test_returns_none_when_id_slug_missing(self):
        detection = {
            "checked": True,
            "route_to_subagent": True,
            "sub_agent": "analyst",
            "id_slug": None,
            "suggested_action": "x",
            "suggested_announcement": "x",
        }
        assert tid.render_injection_block(detection) is None


# =========================================================================
# log_result — single structured line
# =========================================================================

class TestLogResult:
    def test_emits_single_turn_intent_line(self, caplog):
        detection = {
            "checked": True,
            "skipped": None,
            "route_to_subagent": True,
            "sub_agent": "scout",
            "suggested_action": "Find new agency roles",
            "confidence": "high",
            "reasoning": "User asked about agency openings.",
        }
        with caplog.at_level("INFO", logger="agent.turn_intent_detector"):
            tid.log_result("DTEST", detection)
        lines = [
            r.message for r in caplog.records
            if r.message.startswith("turn-intent:")
        ]
        assert len(lines) == 1
        assert "chat=DTEST" in lines[0]
        assert "route=True" in lines[0]
        assert "sub_agent=scout" in lines[0]


# =========================================================================
# _parse_response — direct
# =========================================================================

class TestSanitizeSlug:
    def test_clean_slug_passes_through(self):
        assert tid._sanitize_slug("draft-cover-letter") == "draft-cover-letter"

    def test_uppercase_lowercased(self):
        assert tid._sanitize_slug("Draft-Cover-Letter") == "draft-cover-letter"

    def test_spaces_become_dashes(self):
        assert tid._sanitize_slug("draft cover letter") == "draft-cover-letter"

    def test_underscores_become_dashes(self):
        assert tid._sanitize_slug("draft_cover_letter") == "draft-cover-letter"

    def test_double_dashes_collapsed(self):
        assert tid._sanitize_slug("draft--cover") == "draft-cover"

    def test_leading_coach_commit_stripped(self):
        assert tid._sanitize_slug("coach-commit-draft-x") == "draft-x"

    def test_special_chars_dropped(self):
        assert tid._sanitize_slug("draft!cover@letter#") == "draftcoverletter"

    def test_trims_edges(self):
        assert tid._sanitize_slug("---draft-x---") == "draft-x"

    def test_empty_returns_none(self):
        assert tid._sanitize_slug("") is None
        assert tid._sanitize_slug(None) is None
        assert tid._sanitize_slug("---") is None

    def test_too_long_returns_none(self):
        long = "a-" * 40
        assert tid._sanitize_slug(long) is None

    def test_non_string_returns_none(self):
        assert tid._sanitize_slug(42) is None
        assert tid._sanitize_slug(["a", "b"]) is None


class TestParseResponse:
    def test_clean_json(self):
        raw = '{"route_to_subagent": true}'
        assert tid._parse_response(raw) == {"route_to_subagent": True}

    def test_with_markdown_fence(self):
        raw = '```json\n{"route_to_subagent": true}\n```'
        assert tid._parse_response(raw) == {"route_to_subagent": True}

    def test_empty_returns_none(self):
        assert tid._parse_response("") is None

    def test_garbage_returns_none(self):
        assert tid._parse_response("nope") is None

    def test_non_dict_returns_none(self):
        assert tid._parse_response("[1,2]") is None

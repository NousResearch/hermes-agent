"""Tests for idea_box.flow — IdeaBoxFlow full lifecycle.

Tests the full capture → dedup → confirm/reject flow with mocked backends.
Uses fixture Discord messages to simulate the #idea-box channel.

Verifies:
  - /idea command parsing
  - Novel idea produces a presentable card
  - Duplicate idea surfaces matches
  - Confirmation creates a Kanban triage task
  - Rejection logs to LLM Wiki
  - No auto-promotion ever occurs
  - Wiki logging is optional per-idea
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from idea_box.flow import (
    FlowResult,
    IdeaBoxFlow,
    IDEA_COMMAND_RE,
    IDEA_CHANNEL_NAME,
    parse_idea_command,
)
from idea_box.models import IdeaCard, IdeaStatus, SourceRef


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_source(text: str = "/idea Build a new analytics dashboard") -> SourceRef:
    return SourceRef(
        platform="discord",
        channel_id="111222333",
        message_id="444555666",
        user_id="999888777",
        user_name="sahil",
        channel_name=IDEA_CHANNEL_NAME,
        raw_text=text,
    )


def _make_checker(is_dup: bool = False) -> MagicMock:
    """Return a mock DedupChecker with configurable dup result."""
    from idea_box.models import DedupResult
    checker = MagicMock()
    checker.check.return_value = DedupResult(
        is_duplicate=is_dup,
        matches=[{"source": "kanban", "title": "existing task", "ref_id": "t_001", "score": 0.8}] if is_dup else [],
        checked_sources=["kanban", "session_search", "mnemosyne"],
    )
    return checker


# ---------------------------------------------------------------------------
# parse_idea_command
# ---------------------------------------------------------------------------

class TestParseIdeaCommand:
    def test_basic(self):
        assert parse_idea_command("/idea build a feature") == "build a feature"

    def test_with_extra_spaces(self):
        assert parse_idea_command("/idea    build a feature") == "build a feature"

    def test_multiline(self):
        text = "/idea Build a feature\nwith multiple lines\nof detail"
        result = parse_idea_command(text)
        assert result is not None
        assert "Build a feature" in result
        assert "multiple lines" in result

    def test_no_text(self):
        assert parse_idea_command("/idea") is None

    def test_not_idea_command(self):
        assert parse_idea_command("build a feature") is None
        assert parse_idea_command("/other command") is None

    def test_empty_string(self):
        assert parse_idea_command("") is None

    def test_idea_regex(self):
        assert IDEA_COMMAND_RE.match("/idea test")
        assert not IDEA_COMMAND_RE.match("/ideal test")  # /ideal is not /idea


# ---------------------------------------------------------------------------
# IdeaBoxFlow.capture — novel idea
# ---------------------------------------------------------------------------

class TestCaptureNovel:
    def test_novel_idea_produces_presentable_card(self):
        flow = IdeaBoxFlow(dedup_checker=_make_checker(is_dup=False))
        result = flow.capture("Build a new analytics dashboard", _make_source())

        assert result.action == "present"
        assert result.card.dedup_status == IdeaStatus.NOVEL
        assert result.card.summary == "Build a new analytics dashboard"
        assert result.card.source.platform == "discord"
        assert result.card.source.channel_name == "idea-box"
        assert "Reply" in result.message
        assert "confirm" in result.message

    def test_novel_idea_has_tags(self):
        flow = IdeaBoxFlow(dedup_checker=_make_checker(is_dup=False))
        result = flow.capture("Build dashboard feature for analytics", _make_source())
        # Tags should be extracted (words > 4 chars, not stop words)
        assert len(result.card.tags) > 0
        # "dashboard", "analytics" should be in tags
        assert "dashboard" in result.card.tags
        assert "analytics" in result.card.tags

    def test_empty_idea_text_returns_error(self):
        flow = IdeaBoxFlow(dedup_checker=_make_checker(is_dup=False))
        result = flow.capture("", _make_source())
        assert result.action == "error"
        assert "No idea text" in result.message

    def test_whitespace_only_idea_text_returns_error(self):
        flow = IdeaBoxFlow(dedup_checker=_make_checker(is_dup=False))
        result = flow.capture("   ", _make_source())
        assert result.action == "error"


# ---------------------------------------------------------------------------
# IdeaBoxFlow.capture — duplicate idea
# ---------------------------------------------------------------------------

class TestCaptureDuplicate:
    def test_duplicate_idea_surfaces_matches(self):
        flow = IdeaBoxFlow(dedup_checker=_make_checker(is_dup=True))
        result = flow.capture("Build a dashboard", _make_source())

        assert result.action == "duplicate"
        assert result.card.dedup_status == IdeaStatus.DUPLICATE
        assert len(result.card.dedup_matches) > 0
        assert "Potential duplicate" in result.message
        assert "**kanban**: existing task" in result.message

    def test_duplicate_card_can_still_be_confirmed(self):
        """Even duplicates can be confirmed — the card is presented for Sahil's choice."""
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=True),
            kanban_create_fn=MagicMock(return_value="t_new_001"),
        )
        result = flow.capture("Build a dashboard", _make_source())
        assert result.action == "duplicate"

        # Sahil can still confirm a duplicate
        confirm_result = flow.confirm(result.card)
        assert confirm_result.action == "confirmed"
        assert confirm_result.kanban_task_id == "t_new_001"
        assert result.card.dedup_status == IdeaStatus.CONFIRMED


# ---------------------------------------------------------------------------
# IdeaBoxFlow.confirm
# ---------------------------------------------------------------------------

class TestConfirm:
    def test_confirm_creates_kanban_triage_task(self):
        mock_create = MagicMock(return_value="t_triage_001")
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=False),
            kanban_create_fn=mock_create,
        )
        result = flow.capture("Build a new feature", _make_source())
        confirm_result = flow.confirm(result.card)

        assert confirm_result.action == "confirmed"
        assert confirm_result.kanban_task_id == "t_triage_001"
        assert result.card.idea_id == "t_triage_001"
        assert result.card.dedup_status == IdeaStatus.CONFIRMED

    def test_confirm_calls_kanban_create_with_triage_true(self):
        mock_create = MagicMock(return_value="t_triage_002")
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=False),
            kanban_create_fn=mock_create,
        )
        result = flow.capture("Build a new feature", _make_source())
        flow.confirm(result.card)

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        assert call_kwargs.kwargs.get("triage") is True

    def test_confirm_task_title_truncated(self):
        """Long idea text should be truncated in the task title."""
        long_idea = "Build a very long feature description that exceeds the one hundred twenty character limit for task titles in the kanban system"
        mock_create = MagicMock(return_value="t_001")
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=False),
            kanban_create_fn=mock_create,
        )
        result = flow.capture(long_idea, _make_source())
        flow.confirm(result.card)

        call_kwargs = mock_create.call_args
        title = call_kwargs.kwargs.get("title", "")
        assert len(title) <= 120

    def test_confirm_task_body_contains_source_ref(self):
        mock_create = MagicMock(return_value="t_001")
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=False),
            kanban_create_fn=mock_create,
        )
        src = _make_source()
        result = flow.capture("Build a feature", src)
        flow.confirm(result.card)

        call_kwargs = mock_create.call_args
        body = call_kwargs.kwargs.get("body", "")
        assert "discord/idea-box" in body
        assert src.message_id in body
        assert "sahil" in body

    def test_confirm_already_confirmed_rejects(self):
        """Confirming an already-confirmed idea should return error."""
        mock_create = MagicMock(return_value="t_001")
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=False),
            kanban_create_fn=mock_create,
        )
        result = flow.capture("Build a feature", _make_source())
        flow.confirm(result.card)
        # Second confirmation
        second = flow.confirm(result.card)
        assert second.action == "error"
        assert "already confirmed" in second.message.lower()

    def test_confirm_with_wiki(self):
        """Confirmation with create_wiki=True also logs to wiki."""
        mock_create = MagicMock(return_value="t_001")
        mock_wiki = MagicMock(return_value="/wiki/raw/ideas/confirmed-t_001.md")
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=False),
            kanban_create_fn=mock_create,
            wiki_log_fn=mock_wiki,
        )
        result = flow.capture("Build a feature", _make_source())
        confirm_result = flow.confirm(result.card, create_wiki=True)

        assert confirm_result.wiki_path == "/wiki/raw/ideas/confirmed-t_001.md"
        assert "LLM Wiki" in confirm_result.message
        mock_wiki.assert_called_once()

    def test_confirm_without_wiki(self):
        """Confirmation without wiki flag should not log to wiki."""
        mock_create = MagicMock(return_value="t_001")
        mock_wiki = MagicMock(return_value=None)
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=False),
            kanban_create_fn=mock_create,
            wiki_log_fn=mock_wiki,
        )
        result = flow.capture("Build a feature", _make_source())
        confirm_result = flow.confirm(result.card, create_wiki=False)

        assert confirm_result.wiki_path is None
        mock_wiki.assert_not_called()

    def test_custom_assignee(self):
        """The assignee should be configurable."""
        mock_create = MagicMock(return_value="t_001")
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=False),
            kanban_create_fn=mock_create,
        )
        result = flow.capture("Build a feature", _make_source())
        flow.confirm(result.card, assignee="kensei-intake")

        call_kwargs = mock_create.call_args
        assert call_kwargs.kwargs.get("assignee") == "kensei-intake"


# ---------------------------------------------------------------------------
# IdeaBoxFlow.reject
# ---------------------------------------------------------------------------

class TestReject:
    def test_reject_logs_to_wiki(self):
        mock_wiki = MagicMock(return_value="/wiki/raw/ideas/rejected-rejected.md")
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=False),
            wiki_log_fn=mock_wiki,
        )
        result = flow.capture("Build a feature", _make_source())
        reject_result = flow.reject(result.card, reason="Already exists")

        assert reject_result.action == "rejected"
        assert reject_result.wiki_path is not None
        assert result.card.dedup_status == IdeaStatus.REJECTED
        assert result.card.rejection_reason == "Already exists"

    def test_reject_default_reason(self):
        mock_wiki = MagicMock(return_value="/wiki/raw/ideas/rejected.md")
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=False),
            wiki_log_fn=mock_wiki,
        )
        result = flow.capture("Build a feature", _make_source())
        reject_result = flow.reject(result.card)

        assert result.card.rejection_reason == "No reason given"

    def test_reject_wiki_unavailable(self):
        """Rejecting without a wiki backend should not crash."""
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=False),
        )
        result = flow.capture("Build a feature", _make_source())
        # Use temp dir for wiki fallback
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"WIKI_PATH": tmpdir}):
                reject_result = flow.reject(result.card, reason="Not needed")
        assert reject_result.action == "rejected"
        assert reject_result.wiki_path is not None

    def test_reject_default_wiki_root_is_canonical(self, tmp_path, monkeypatch):
        """The no-backend fallback writes only under the canonical HOME-relative wiki."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("WIKI_PATH", raising=False)
        flow = IdeaBoxFlow(dedup_checker=_make_checker(is_dup=False))
        result = flow.capture("Build a feature", _make_source())

        rejected = flow.reject(result.card, reason="Not needed")

        assert rejected.wiki_path is not None
        assert rejected.wiki_path.startswith(
            str(tmp_path / "docs" / "wiki" / "raw" / "ideas")
        )

    def test_reject_wiki_filename_unique_per_idea(self):
        """Each rejected idea should get a unique wiki filename to preserve provenance."""
        import tempfile, os
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=False),
        )

        # Reject two different ideas from different messages
        paths = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"WIKI_PATH": tmpdir}):
                src1 = SourceRef(
                    platform="discord", channel_id="c1", message_id="msg_001",
                    user_id="u1", user_name="sahil", channel_name="idea-box",
                )
                result1 = flow.capture("Build feature alpha", src1)
                reject1 = flow.reject(result1.card, reason="Duplicate")
                paths.append(reject1.wiki_path)

                src2 = SourceRef(
                    platform="discord", channel_id="c1", message_id="msg_002",
                    user_id="u1", user_name="sahil", channel_name="idea-box",
                )
                result2 = flow.capture("Build feature beta", src2)
                reject2 = flow.reject(result2.card, reason="Out of scope")
                paths.append(reject2.wiki_path)

        # Both paths should exist and be different
        assert paths[0] != paths[1], "Wiki filenames must be unique per rejection"
        assert "msg_001" in paths[0] or "msg_002" not in paths[0]  # First uses message_id
        assert "msg_002" in paths[1]


# ---------------------------------------------------------------------------
# IdeaBoxFlow — no auto-promotion (acceptance criteria #6)
# ---------------------------------------------------------------------------

class TestNoAutoPromotion:
    def test_capture_does_not_create_kanban_task(self):
        """Capturing an idea must NOT create a Kanban task."""
        mock_create = MagicMock()
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=False),
            kanban_create_fn=mock_create,
        )
        flow.capture("Build a feature", _make_source())
        mock_create.assert_not_called()

    def test_capture_does_not_dispatch_worker(self):
        """Capturing an idea must NOT dispatch a worker or trigger runtime action."""
        mock_create = MagicMock()
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=False),
            kanban_create_fn=mock_create,
        )
        result = flow.capture("Build a feature", _make_source())
        assert result.kanban_task_id is None
        mock_create.assert_not_called()

    def test_duplicate_does_not_create_kanban_task(self):
        """A duplicate idea must NOT create a Kanban task."""
        mock_create = MagicMock()
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=True),
            kanban_create_fn=mock_create,
        )
        flow.capture("Build a dashboard", _make_source())
        mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# IdeaBoxFlow — full lifecycle
# ---------------------------------------------------------------------------

class TestFullLifecycle:
    def test_novel_to_confirmed(self):
        mock_create = MagicMock(return_value="t_final_001")
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=False),
            kanban_create_fn=mock_create,
        )
        # Step 1: Capture
        capture_result = flow.capture("Build analytics dashboard", _make_source())
        assert capture_result.action == "present"

        # Step 2: Confirm
        confirm_result = flow.confirm(capture_result.card)
        assert confirm_result.action == "confirmed"
        assert confirm_result.kanban_task_id == "t_final_001"
        assert capture_result.card.dedup_status == IdeaStatus.CONFIRMED

    def test_novel_to_confirmed_with_wiki(self):
        mock_create = MagicMock(return_value="t_final_002")
        mock_wiki = MagicMock(return_value="/wiki/raw/ideas/confirmed-t_final_002.md")
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=False),
            kanban_create_fn=mock_create,
            wiki_log_fn=mock_wiki,
        )
        # Capture
        capture_result = flow.capture("Build analytics dashboard", _make_source())
        # Confirm with wiki
        confirm_result = flow.confirm(capture_result.card, create_wiki=True)
        assert confirm_result.kanban_task_id == "t_final_002"
        assert confirm_result.wiki_path is not None
        mock_wiki.assert_called_once()

    def test_duplicate_to_confirmed(self):
        """Duplicate can still be confirmed if Sahil chooses."""
        mock_create = MagicMock(return_value="t_dup_001")
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=True),
            kanban_create_fn=mock_create,
        )
        capture_result = flow.capture("Build analytics dashboard", _make_source())
        assert capture_result.action == "duplicate"
        confirm_result = flow.confirm(capture_result.card)
        assert confirm_result.action == "confirmed"

    def test_novel_to_rejected(self):
        mock_wiki = MagicMock(return_value="/wiki/raw/ideas/rejected-rejected.md")
        flow = IdeaBoxFlow(
            dedup_checker=_make_checker(is_dup=False),
            wiki_log_fn=mock_wiki,
        )
        capture_result = flow.capture("Build analytics dashboard", _make_source())
        reject_result = flow.reject(capture_result.card, reason="Out of scope")
        assert reject_result.action == "rejected"
        assert capture_result.card.dedup_status == IdeaStatus.REJECTED
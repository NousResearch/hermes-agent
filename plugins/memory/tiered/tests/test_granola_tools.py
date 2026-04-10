"""Tests for granola_tools.py — Granola tool functions (MCP proxy)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.tiered.granola_tools import (
    GRANOLA_CONTENT_LIMIT,
    get_granola_meeting,
    list_granola_meetings,
    query_granola_meetings,
    truncate_granola_content,
)


# ===========================================================================
# TestTruncateGranolaContent
# ===========================================================================


class TestTruncateGranolaContent:
    """Tests for truncate_granola_content() recursive truncation."""

    def test_truncates_long_string_fields_in_object(self) -> None:
        """Long string fields in a dict get truncated with marker."""
        data = {
            "transcript": "T" * 3000,
            "title": "Short title",
            "count": 42,
        }
        result = truncate_granola_content(data)

        assert len(result["transcript"]) == GRANOLA_CONTENT_LIMIT + len(" [truncated]")
        assert result["transcript"].endswith(" [truncated]")
        assert result["title"] == "Short title"
        assert result["count"] == 42

    def test_handles_primitives_as_is(self) -> None:
        """Non-object/array types returned unchanged."""
        assert truncate_granola_content("just a string") == "just a string"
        assert truncate_granola_content(42) == 42
        assert truncate_granola_content(None) is None
        assert truncate_granola_content(True) is True

    def test_handles_nested_arrays(self) -> None:
        """Arrays of objects with long fields get truncated recursively."""
        data = {
            "meetings": [
                {"id": "m1", "notes": "N" * 3000},
                {"id": "m2", "notes": "Short"},
            ],
        }
        result = truncate_granola_content(data)

        assert len(result["meetings"][0]["notes"]) == GRANOLA_CONTENT_LIMIT + len(" [truncated]")
        assert result["meetings"][1]["notes"] == "Short"

    def test_preserves_exactly_limit_content(self) -> None:
        """Content exactly at the limit is NOT truncated."""
        data = {"notes": "E" * GRANOLA_CONTENT_LIMIT}
        result = truncate_granola_content(data)

        assert len(result["notes"]) == GRANOLA_CONTENT_LIMIT
        assert "[truncated]" not in result["notes"]

    def test_truncates_top_level_long_strings(self) -> None:
        """Top-level strings exceeding limit get truncated."""
        long_string = "x" * 3000
        result = truncate_granola_content(long_string)

        assert isinstance(result, str)
        assert len(result) == GRANOLA_CONTENT_LIMIT + len(" [truncated]")
        assert result.endswith(" [truncated]")

    def test_handles_empty_dict(self) -> None:
        """Empty dict returns empty dict."""
        assert truncate_granola_content({}) == {}

    def test_handles_empty_list(self) -> None:
        """Empty list returns empty list."""
        assert truncate_granola_content([]) == []

    def test_respects_max_recursion_depth(self) -> None:
        """Deeply nested structures stop recursing at MAX_RECURSION_DEPTH."""
        # Build deeply nested structure
        data: dict = {"value": "x" * 3000}
        for _ in range(25):
            data = {"nested": data}

        result = truncate_granola_content(data)

        # Navigate to the deepest level — past depth 20, truncation should stop
        current = result
        for _ in range(25):
            current = current["nested"]
        # The innermost value should NOT be truncated (depth > 20)
        assert current["value"] == "x" * 3000


# ===========================================================================
# TestListGranolaMeetings
# ===========================================================================


class TestListGranolaMeetings:
    """Tests for list_granola_meetings()."""

    @patch("plugins.memory.tiered.granola_tools._create_client")
    def test_default_time_range(self, mock_create: MagicMock) -> None:
        """Default time_range is last_30_days."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {"meetings": []}
        mock_create.return_value = mock_client

        result = list_granola_meetings()

        mock_client.call_tool.assert_called_once_with(
            "list_meetings", {"time_range": "last_30_days"}
        )
        assert json.loads(result["content"]) == {"meetings": []}

    @patch("plugins.memory.tiered.granola_tools._create_client")
    def test_custom_time_range(self, mock_create: MagicMock) -> None:
        """Passes non-default time_range to MCP."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {"meetings": []}
        mock_create.return_value = mock_client

        list_granola_meetings(time_range="last_week")

        mock_client.call_tool.assert_called_once_with(
            "list_meetings", {"time_range": "last_week"}
        )

    @patch("plugins.memory.tiered.granola_tools._create_client")
    def test_custom_date_range(self, mock_create: MagicMock) -> None:
        """Passes custom start/end dates when time_range is custom."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {"meetings": []}
        mock_create.return_value = mock_client

        list_granola_meetings(
            time_range="custom",
            custom_start="2026-03-16T00:00:00Z",
            custom_end="2026-03-22T23:59:59Z",
        )

        mock_client.call_tool.assert_called_once_with(
            "list_meetings",
            {
                "time_range": "custom",
                "custom_start": "2026-03-16T00:00:00Z",
                "custom_end": "2026-03-22T23:59:59Z",
            },
        )

    def test_rejects_invalid_time_range(self) -> None:
        """Invalid time_range raises ValueError."""
        with pytest.raises(ValueError, match="time_range"):
            list_granola_meetings(time_range="yesterday")


# ===========================================================================
# TestGetGranolaMeeting
# ===========================================================================


class TestGetGranolaMeeting:
    """Tests for get_granola_meeting()."""

    @patch("plugins.memory.tiered.granola_tools._create_client")
    def test_get_meeting_without_transcript(self, mock_create: MagicMock) -> None:
        """Without transcript, calls get_meetings with meeting_ids array."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {"id": "m1", "title": "Standup"}
        mock_create.return_value = mock_client

        result = get_granola_meeting(meeting_id="m1")

        mock_client.call_tool.assert_called_once_with(
            "get_meetings", {"meeting_ids": ["m1"]}
        )
        assert json.loads(result["content"])["title"] == "Standup"

    @patch("plugins.memory.tiered.granola_tools._create_client")
    def test_get_meeting_with_transcript(self, mock_create: MagicMock) -> None:
        """With include_transcript=True, calls get_meeting_transcript."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {"transcript": "Hello..."}
        mock_create.return_value = mock_client

        get_granola_meeting(meeting_id="m1", include_transcript=True)

        mock_client.call_tool.assert_called_once_with(
            "get_meeting_transcript", {"meeting_id": "m1"}
        )

    def test_rejects_empty_meeting_id(self) -> None:
        """Empty meeting ID raises ValueError."""
        with pytest.raises(ValueError, match="Meeting ID"):
            get_granola_meeting(meeting_id="")


# ===========================================================================
# TestQueryGranolaMeetings
# ===========================================================================


class TestQueryGranolaMeetings:
    """Tests for query_granola_meetings()."""

    @patch("plugins.memory.tiered.granola_tools._create_client")
    def test_normal_query(self, mock_create: MagicMock) -> None:
        """Passes query to query_granola_meetings MCP tool."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {"answer": "Meeting summary..."}
        mock_create.return_value = mock_client

        result = query_granola_meetings(query="sprint planning")

        mock_client.call_tool.assert_called_once_with(
            "query_granola_meetings", {"query": "sprint planning"}
        )
        assert json.loads(result["content"])["answer"] == "Meeting summary..."

    def test_rejects_query_too_short(self) -> None:
        """Query shorter than 2 characters raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            query_granola_meetings(query="a")


# ===========================================================================
# TestContentTruncation
# ===========================================================================


class TestContentTruncation:
    """Tests for content truncation in tool outputs."""

    @patch("plugins.memory.tiered.granola_tools._create_client")
    def test_truncates_transcript_content(self, mock_create: MagicMock) -> None:
        """Long transcript content is truncated to GRANOLA_CONTENT_LIMIT."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {"transcript": "W" * 5000}
        mock_create.return_value = mock_client

        result = get_granola_meeting(meeting_id="m1", include_transcript=True)
        parsed = json.loads(result["content"])

        assert len(parsed["transcript"]) == GRANOLA_CONTENT_LIMIT + len(" [truncated]")
        assert parsed["transcript"].endswith(" [truncated]")

    @patch("plugins.memory.tiered.granola_tools._create_client")
    def test_does_not_truncate_short_content(self, mock_create: MagicMock) -> None:
        """Short content is left unchanged."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {"transcript": "Hello world"}
        mock_create.return_value = mock_client

        result = get_granola_meeting(meeting_id="m1", include_transcript=True)
        parsed = json.loads(result["content"])

        assert parsed["transcript"] == "Hello world"

    @patch("plugins.memory.tiered.granola_tools._create_client")
    def test_truncates_query_answer(self, mock_create: MagicMock) -> None:
        """Long query answer is truncated."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {"answer": "A" * 5000}
        mock_create.return_value = mock_client

        result = query_granola_meetings(query="sprint planning")
        parsed = json.loads(result["content"])

        assert len(parsed["answer"]) == GRANOLA_CONTENT_LIMIT + len(" [truncated]")


# ===========================================================================
# TestNoTokenLeakage
# ===========================================================================


class TestNoTokenLeakage:
    """Tests that OAuth tokens don't leak into tool output."""

    @patch("plugins.memory.tiered.granola_tools._create_client")
    def test_no_token_in_output(self, mock_create: MagicMock, monkeypatch) -> None:
        """OAuth token must not appear in tool output."""
        monkeypatch.setenv("GRANOLA_OAUTH_TOKEN", "super-secret-token")
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {"meetings": []}
        mock_create.return_value = mock_client

        result = list_granola_meetings()

        assert "super-secret-token" not in result["content"]

"""Tests for dashboard session list pagination clamping."""

import asyncio
from unittest.mock import MagicMock, patch

from hermes_cli.web_server import (
    _clamp_sessions_list_limit,
    _clamp_sessions_list_offset,
    get_sessions,
)


class TestClampSessionsListLimit:
    def test_invalid_value_uses_default(self):
        assert _clamp_sessions_list_limit("bad") == 20

    def test_zero_clamped_to_one(self):
        assert _clamp_sessions_list_limit(0) == 1

    def test_excessive_limit_capped(self):
        assert _clamp_sessions_list_limit(9999) == 100


class TestClampSessionsListOffset:
    def test_invalid_value_uses_zero(self):
        assert _clamp_sessions_list_offset("bad") == 0

    def test_negative_offset_clamped_to_zero(self):
        assert _clamp_sessions_list_offset(-5) == 0


class TestGetSessionsEndpoint:
    def test_passes_clamped_pagination_to_session_db(self):
        mock_db = MagicMock()
        mock_db.list_sessions_rich.return_value = []
        mock_db.session_count.return_value = 0

        with patch("hermes_state.SessionDB", return_value=mock_db):
            result = asyncio.run(get_sessions(limit=9999, offset=-10))

        mock_db.list_sessions_rich.assert_called_once()
        assert mock_db.list_sessions_rich.call_args.kwargs["limit"] == 100
        assert mock_db.list_sessions_rich.call_args.kwargs["offset"] == 0
        assert result["limit"] == 100
        assert result["offset"] == 0
        mock_db.close.assert_called_once()

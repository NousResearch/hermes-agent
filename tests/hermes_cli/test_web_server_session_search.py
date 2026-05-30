"""Tests for dashboard session search limit handling."""

import asyncio
from unittest.mock import MagicMock, patch

from hermes_cli.web_server import (
    _clamp_session_search_limit,
    search_sessions,
)


class TestClampSessionSearchLimit:
    def test_invalid_value_uses_default(self):
        assert _clamp_session_search_limit("bad") == 20

    def test_zero_clamped_to_one(self):
        assert _clamp_session_search_limit(0) == 1

    def test_excessive_limit_capped(self):
        assert _clamp_session_search_limit(9999) == 100


class TestSearchSessionsEndpoint:
    def test_passes_clamped_limit_to_session_db(self):
        mock_db = MagicMock()
        mock_db.search_messages.return_value = []

        with patch("hermes_state.SessionDB", return_value=mock_db):
            asyncio.run(search_sessions(q="hello", limit=9999))

        mock_db.search_messages.assert_called_once()
        assert mock_db.search_messages.call_args.kwargs["limit"] == 100
        mock_db.close.assert_called_once()

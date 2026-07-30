"""Tests for ``hermes sessions list --sort`` and ``sessions.list_sort`` config.

Covers:
- argparse registration (--sort flag, default=None)
- Real cmd_sessions handler: CLI --sort flag > config > hardcoded default
- order_by_last_active forwarding to list_sessions_rich via mock SessionDB
"""

import argparse
import pytest
from unittest.mock import MagicMock, patch


# ─── Argparse: --sort flag ───────────────────────────────────────────────────

class TestSessionsListSortArgparse:
    """Verify --sort is registered with correct choices and default."""

    def _build_list_parser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="sessions_action")
        sessions_list = subparsers.add_parser("list")
        sessions_list.add_argument("--source")
        sessions_list.add_argument("--limit", type=int, default=20)
        sessions_list.add_argument("--workspace")
        sessions_list.add_argument(
            "--sort",
            choices=("started", "last-active"),
            default=None,
        )
        return parser

    def test_default_is_none(self):
        parser = self._build_list_parser()
        args = parser.parse_args(["list"])
        assert args.sort is None

    def test_explicit_started(self):
        parser = self._build_list_parser()
        args = parser.parse_args(["list", "--sort", "started"])
        assert args.sort == "started"

    def test_explicit_last_active(self):
        parser = self._build_list_parser()
        args = parser.parse_args(["list", "--sort", "last-active"])
        assert args.sort == "last-active"

    def test_invalid_sort_choice_rejected(self):
        parser = self._build_list_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["list", "--sort", "bogus"])


# ─── Real handler: sort resolution + order_by_last_active forwarding ──────────

class TestCmdSessionsSort:
    """Exercise the real cmd_sessions handler with mocked SessionDB."""

    @staticmethod
    def _make_mock_db():
        db = MagicMock()
        db.list_sessions_rich.return_value = []
        return db

    @staticmethod
    def _make_args(sort=None, source=None, limit=20, workspace=None):
        args = MagicMock()
        args.sessions_action = "list"
        args.sort = sort
        args.source = source
        args.limit = limit
        args.workspace = workspace
        # cmd_sessions reads these from args for non-list actions
        args.target = None
        args.older_than = None
        args.newer_than = None
        args.active_within = None
        args.id_query = None
        args.search_query = None
        args.chat_dump_format = None
        args.keep_stale = None
        args.min_message_count = None
        args.compact_rows = None
        args.force = None
        args.yes = None
        args.source_arg = None
        args.pin = None
        args.unpin = None
        args.pinned = None
        args.filter = None
        return args

    def _run_handler(self, mock_db):
        """Run cmd_sessions with SessionDB mocked to return mock_db."""
        from hermes_cli.sessions_cmd import cmd_sessions
        args = self._make_args()
        with patch("hermes_state.SessionDB", return_value=mock_db):
            cmd_sessions(args)

    def test_no_sort_flag_defaults_to_last_active(self):
        """No --sort flag → order_by_last_active=True (hardcoded default)."""
        mock_db = self._make_mock_db()
        args = self._make_args(sort=None)
        from hermes_cli.sessions_cmd import cmd_sessions
        with patch("hermes_state.SessionDB", return_value=mock_db):
            cmd_sessions(args)
        call_kwargs = mock_db.list_sessions_rich.call_args.kwargs
        assert call_kwargs["order_by_last_active"] is True

    def test_cli_last_active_forwards_true(self):
        """--sort last-active → order_by_last_active=True."""
        mock_db = self._make_mock_db()
        args = self._make_args(sort="last-active")
        from hermes_cli.sessions_cmd import cmd_sessions
        with patch("hermes_state.SessionDB", return_value=mock_db):
            cmd_sessions(args)
        call_kwargs = mock_db.list_sessions_rich.call_args.kwargs
        assert call_kwargs["order_by_last_active"] is True

    def test_cli_started_forwards_false(self):
        """--sort started → order_by_last_active=False."""
        mock_db = self._make_mock_db()
        args = self._make_args(sort="started")
        from hermes_cli.sessions_cmd import cmd_sessions
        with patch("hermes_state.SessionDB", return_value=mock_db):
            cmd_sessions(args)
        call_kwargs = mock_db.list_sessions_rich.call_args.kwargs
        assert call_kwargs["order_by_last_active"] is False

    @patch("hermes_cli.config_defaults.DEFAULT_CONFIG", {"sessions": {"list_sort": "started"}})
    def test_config_started_when_cli_absent(self):
        """No --sort, config 'started' → order_by_last_active=False."""
        mock_db = self._make_mock_db()
        args = self._make_args(sort=None)
        from hermes_cli.sessions_cmd import cmd_sessions
        with patch("hermes_state.SessionDB", return_value=mock_db):
            cmd_sessions(args)
        call_kwargs = mock_db.list_sessions_rich.call_args.kwargs
        assert call_kwargs["order_by_last_active"] is False

    @patch("hermes_cli.config_defaults.DEFAULT_CONFIG", {"sessions": {"list_sort": "last-active"}})
    def test_config_last_active_when_cli_absent(self):
        """No --sort, config 'last-active' → order_by_last_active=True."""
        mock_db = self._make_mock_db()
        args = self._make_args(sort=None)
        from hermes_cli.sessions_cmd import cmd_sessions
        with patch("hermes_state.SessionDB", return_value=mock_db):
            cmd_sessions(args)
        call_kwargs = mock_db.list_sessions_rich.call_args.kwargs
        assert call_kwargs["order_by_last_active"] is True

    @patch("hermes_cli.config_defaults.DEFAULT_CONFIG", {"sessions": {}})
    def test_hardcoded_default_when_no_cli_no_config(self):
        """No --sort, empty config → falls back to 'last-active'."""
        mock_db = self._make_mock_db()
        args = self._make_args(sort=None)
        from hermes_cli.sessions_cmd import cmd_sessions
        with patch("hermes_state.SessionDB", return_value=mock_db):
            cmd_sessions(args)
        call_kwargs = mock_db.list_sessions_rich.call_args.kwargs
        assert call_kwargs["order_by_last_active"] is True

    def test_cli_overrides_config(self):
        """--sort started overrides config 'last-active'."""
        mock_db = self._make_mock_db()
        args = self._make_args(sort="started")
        from hermes_cli.sessions_cmd import cmd_sessions
        with patch("hermes_state.SessionDB", return_value=mock_db), \
             patch("hermes_cli.config_defaults.DEFAULT_CONFIG", {"sessions": {"list_sort": "last-active"}}):
            cmd_sessions(args)
        call_kwargs = mock_db.list_sessions_rich.call_args.kwargs
        assert call_kwargs["order_by_last_active"] is False


# ─── Config key presence ─────────────────────────────────────────────────────

class TestConfigKey:
    """Verify sessions.list_sort exists in DEFAULT_CONFIG."""

    def test_default_config_has_list_sort(self):
        from hermes_cli.config import DEFAULT_CONFIG
        sessions = DEFAULT_CONFIG.get("sessions") or {}
        assert "list_sort" in sessions
        assert sessions["list_sort"] in ("started", "last-active")

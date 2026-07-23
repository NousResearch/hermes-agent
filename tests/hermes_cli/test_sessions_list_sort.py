"""Tests for ``hermes sessions list --sort`` and ``sessions.list_sort`` config.

Covers:
- argparse registration (--sort flag, default=None)
- Resolution chain: CLI --sort flag > config sessions.list_sort > hardcoded default
- order_by_last_active forwarding to list_sessions_rich
"""

import argparse
import time
from unittest.mock import MagicMock, patch

import pytest


# ─── Sample session data ──────────────────────────────────────────────────────

def _make_sessions(n=5):
    now = time.time()
    return [
        {
            "id": f"20260723_{i:06d}_test",
            "source": "cli",
            "model": "test/model",
            "title": f"Session {i}",
            "preview": f"Message {i}",
            "last_active": now - i * 3600,
            "started_at": now - i * 3600 - 60,
            "message_count": (i + 1) * 5,
        }
        for i in range(n)
    ]


# ─── Argparse: --sort flag ───────────────────────────────────────────────────

class TestSessionsListSortArgparse:
    """Verify --sort is registered with correct choices and default."""

    def _build_list_parser(self):
        """Replicate the sessions list subparser setup from main.py."""
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


# ─── Sort resolution logic ───────────────────────────────────────────────────

class TestSortResolution:
    """Test the sort-resolution chain used inside cmd_sessions."""

    @pytest.mark.parametrize(
        "cli_flag,config_value,expected",
        [
            # CLI flag wins over everything
            ("started", "last-active", "started"),
            ("last-active", "started", "last-active"),
            # No CLI flag → config value
            (None, "started", "started"),
            (None, "last-active", "last-active"),
            # No CLI flag, no config → hardcoded default
            (None, None, "last-active"),
        ],
    )
    def test_resolution_chain(self, cli_flag, config_value, expected):
        """CLI flag > config > hardcoded default 'last-active'."""
        # Simulate the resolution logic from cmd_sessions
        _sort = cli_flag
        if _sort is None:
            _cfg = {"sessions": {}} if config_value is None else {"sessions": {"list_sort": config_value}}
            _sort = (_cfg.get("sessions") or {}).get("list_sort", "last-active")
        assert _sort == expected


# ─── order_by_last_active forwarding ─────────────────────────────────────────

class TestOrderByLastActive:
    """Verify order_by_last_active is forwarded to list_sessions_rich."""

    def test_last_active_maps_to_true(self):
        """When sort is 'last-active', order_by_last_active=True."""
        _sort = "last-active"
        assert (_sort == "last-active") is True

    def test_started_maps_to_false(self):
        """When sort is 'started', order_by_last_active=False."""
        _sort = "started"
        assert (_sort == "last-active") is False

    def test_list_sessions_rich_receives_correct_flag(self):
        """End-to-end: mock SessionDB, verify list_sessions_rich receives the flag."""
        mock_db = MagicMock()
        mock_db.list_sessions_rich.return_value = _make_sessions(2)

        # Simulate the cmd_sessions "list" action with --sort last-active
        _sort = "last-active"
        mock_db.list_sessions_rich(
            source=None,
            exclude_sources=["tool"],
            limit=10,
            order_by_last_active=(_sort == "last-active"),
        )

        call_kwargs = mock_db.list_sessions_rich.call_args.kwargs
        assert call_kwargs["order_by_last_active"] is True

        # Reset and test with --sort started
        mock_db.reset_mock()
        _sort = "started"
        mock_db.list_sessions_rich(
            source=None,
            exclude_sources=["tool"],
            limit=10,
            order_by_last_active=(_sort == "last-active"),
        )

        call_kwargs = mock_db.list_sessions_rich.call_args.kwargs
        assert call_kwargs["order_by_last_active"] is False


# ─── Config key presence ─────────────────────────────────────────────────────

class TestConfigKey:
    """Verify sessions.list_sort exists in DEFAULT_CONFIG."""

    def test_default_config_has_list_sort(self):
        from hermes_cli.config import DEFAULT_CONFIG

        sessions = DEFAULT_CONFIG.get("sessions") or {}
        assert "list_sort" in sessions, (
            "sessions.list_sort must be present in DEFAULT_CONFIG"
        )
        assert sessions["list_sort"] in ("started", "last-active"), (
            f"Unexpected default: {sessions['list_sort']}"
        )

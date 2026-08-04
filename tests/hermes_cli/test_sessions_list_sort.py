"""Tests for ``hermes sessions list --sort`` and ``sessions.list_sort`` config.

Covers:
- Real argparse parser: --sort registered on the sessions list subparser
- Real cmd_sessions handler: CLI --sort flag > runtime config > hardcoded default
- order_by_last_active forwarding to list_sessions_rich via mock SessionDB
"""

import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest


# ─── Real argparse: --sort flag via hermes sessions list --help ──────────────

class TestSessionsListSortArgparse:
    """Verify --sort is registered on the real sessions list subparser.

    Uses a subprocess driver that imports hermes_cli.main (heavy import)
    and parses ``sessions list --help`` exactly as the real CLI would.
    This avoids hand-rolling a local parser that can drift from production.
    """

    _DRIVER = r"""
import io, json, sys
from contextlib import redirect_stdout, redirect_stderr

import hermes_cli.main as main_mod

sys.argv = ["hermes", "sessions", "list", "--help"]
out, err = io.StringIO(), io.StringIO()
try:
    with redirect_stdout(out), redirect_stderr(err):
        main_mod.main()
except SystemExit as exc:
    pass
text = out.getvalue() + err.getvalue()
print(json.dumps(text))
"""

    @pytest.fixture(scope="class")
    def help_text(self):
        result = subprocess.run(
            [sys.executable, "-c", self._DRIVER],
            capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 0, (
            f"driver failed rc={result.returncode}\n"
            f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
        )
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_sort_flag_present(self, help_text):
        assert "--sort" in help_text

    def test_sort_choices_documented(self, help_text):
        assert "started" in help_text
        assert "last-active" in help_text

    def test_invalid_sort_choice_rejected(self):
        """The real parser must reject invalid --sort values."""
        driver = r"""
import io, json, sys
from contextlib import redirect_stdout, redirect_stderr
import hermes_cli.main as main_mod
sys.argv = ["hermes", "sessions", "list", "--sort", "bogus"]
out, err = io.StringIO(), io.StringIO()
code = 0
try:
    with redirect_stdout(out), redirect_stderr(err):
        main_mod.main()
except SystemExit as exc:
    code = int(exc.code or 0)
print(json.dumps({"code": code, "stderr": err.getvalue()[:300]}))
"""
        result = subprocess.run(
            [sys.executable, "-c", driver],
            capture_output=True, text=True, timeout=180,
        )
        entry = json.loads(result.stdout.strip().splitlines()[-1])
        assert entry["code"] != 0, "invalid --sort should trigger SystemExit"


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

    @staticmethod
    def _run(args, mock_db):
        from hermes_cli.sessions_cmd import cmd_sessions
        with patch("hermes_state.SessionDB", return_value=mock_db):
            cmd_sessions(args)

    def test_no_sort_flag_defaults_to_last_active(self):
        """No --sort flag → order_by_last_active=True (hardcoded default)."""
        mock_db = self._make_mock_db()
        self._run(self._make_args(sort=None), mock_db)
        assert mock_db.list_sessions_rich.call_args.kwargs["order_by_last_active"] is True

    def test_cli_last_active_forwards_true(self):
        """--sort last-active → order_by_last_active=True."""
        mock_db = self._make_mock_db()
        self._run(self._make_args(sort="last-active"), mock_db)
        assert mock_db.list_sessions_rich.call_args.kwargs["order_by_last_active"] is True

    def test_cli_started_forwards_false(self):
        """--sort started → order_by_last_active=False."""
        mock_db = self._make_mock_db()
        self._run(self._make_args(sort="started"), mock_db)
        assert mock_db.list_sessions_rich.call_args.kwargs["order_by_last_active"] is False

    def test_config_started_when_cli_absent(self, monkeypatch):
        """No --sort, runtime config 'started' → order_by_last_active=False.

        Patches load_config() to return a config where sessions.list_sort
        is 'started', exercising the real config-read path in cmd_sessions.
        """
        mock_db = self._make_mock_db()
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"sessions": {"list_sort": "started"}},
        )
        self._run(self._make_args(sort=None), mock_db)
        assert mock_db.list_sessions_rich.call_args.kwargs["order_by_last_active"] is False

    def test_config_last_active_when_cli_absent(self, monkeypatch):
        """No --sort, runtime config 'last-active' → order_by_last_active=True."""
        mock_db = self._make_mock_db()
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"sessions": {"list_sort": "last-active"}},
        )
        self._run(self._make_args(sort=None), mock_db)
        assert mock_db.list_sessions_rich.call_args.kwargs["order_by_last_active"] is True

    def test_hardcoded_default_when_no_cli_no_config(self, monkeypatch):
        """No --sort, config missing list_sort → falls back to 'last-active'."""
        mock_db = self._make_mock_db()
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"sessions": {}},
        )
        self._run(self._make_args(sort=None), mock_db)
        assert mock_db.list_sessions_rich.call_args.kwargs["order_by_last_active"] is True

    def test_cli_overrides_config(self, monkeypatch):
        """--sort started overrides config 'last-active'."""
        mock_db = self._make_mock_db()
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"sessions": {"list_sort": "last-active"}},
        )
        self._run(self._make_args(sort="started"), mock_db)
        assert mock_db.list_sessions_rich.call_args.kwargs["order_by_last_active"] is False


# ─── Config key presence ─────────────────────────────────────────────────────

class TestConfigKey:
    """Verify sessions.list_sort exists in DEFAULT_CONFIG."""

    def test_default_config_has_list_sort(self):
        from hermes_cli.config import DEFAULT_CONFIG
        sessions = DEFAULT_CONFIG.get("sessions") or {}
        assert "list_sort" in sessions
        assert sessions["list_sort"] in ("started", "last-active")

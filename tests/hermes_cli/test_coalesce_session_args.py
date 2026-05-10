"""Tests for _coalesce_session_name_args — multi-word session name merging."""

import pytest
from hermes_cli.main import _coalesce_session_name_args


class TestCoalesceSessionNameArgs:
    """Ensure unquoted multi-word session names are merged into one token."""

    # ── -c / --continue ──────────────────────────────────────────────────

    def test_continue_multiword_unquoted(self):
        """hermes -c Pokemon Agent Dev → -c 'Pokemon Agent Dev'"""
        assert _coalesce_session_name_args(
            ["-c", "Pokemon", "Agent", "Dev"]
        ) == ["-c", "Pokemon Agent Dev"]

    def test_continue_long_form_multiword(self):
        """hermes --continue Pokemon Agent Dev"""
        assert _coalesce_session_name_args(
            ["--continue", "Pokemon", "Agent", "Dev"]
        ) == ["--continue", "Pokemon Agent Dev"]

    def test_continue_single_word(self):
        """hermes -c MyProject (no merging needed)"""
        assert _coalesce_session_name_args(["-c", "MyProject"]) == [
            "-c",
            "MyProject",
        ]

    def test_continue_already_quoted(self):
        """hermes -c 'Pokemon Agent Dev' (shell already merged)"""
        assert _coalesce_session_name_args(
            ["-c", "Pokemon Agent Dev"]
        ) == ["-c", "Pokemon Agent Dev"]

    def test_continue_bare_flag(self):
        """hermes -c (no name — means 'continue latest')"""
        assert _coalesce_session_name_args(["-c"]) == ["-c"]

    def test_continue_followed_by_flag(self):
        """hermes -c -w (no name consumed, -w stays separate)"""
        assert _coalesce_session_name_args(["-c", "-w"]) == ["-c", "-w"]

    def test_continue_multiword_then_flag(self):
        """hermes -c my project -w"""
        assert _coalesce_session_name_args(
            ["-c", "my", "project", "-w"]
        ) == ["-c", "my project", "-w"]

    def test_continue_multiword_then_subcommand(self):
        """hermes -c my project chat -q hello"""
        assert _coalesce_session_name_args(
            ["-c", "my", "project", "chat", "-q", "hello"]
        ) == ["-c", "my project", "chat", "-q", "hello"]

    # ── -r / --resume ────────────────────────────────────────────────────

    def test_resume_multiword(self):
        """hermes -r My Session Name"""
        assert _coalesce_session_name_args(
            ["-r", "My", "Session", "Name"]
        ) == ["-r", "My Session Name"]

    def test_resume_long_form_multiword(self):
        """hermes --resume My Session Name"""
        assert _coalesce_session_name_args(
            ["--resume", "My", "Session", "Name"]
        ) == ["--resume", "My Session Name"]

    def test_resume_multiword_then_flag(self):
        """hermes -r My Session -w"""
        assert _coalesce_session_name_args(
            ["-r", "My", "Session", "-w"]
        ) == ["-r", "My Session", "-w"]

    # ── combined flags ───────────────────────────────────────────────────

    def test_worktree_and_continue_multiword(self):
        """hermes -w -c Pokemon Agent Dev (the original failing case)"""
        assert _coalesce_session_name_args(
            ["-w", "-c", "Pokemon", "Agent", "Dev"]
        ) == ["-w", "-c", "Pokemon Agent Dev"]

    def test_continue_multiword_and_worktree(self):
        """hermes -c Pokemon Agent Dev -w (order reversed)"""
        assert _coalesce_session_name_args(
            ["-c", "Pokemon", "Agent", "Dev", "-w"]
        ) == ["-c", "Pokemon Agent Dev", "-w"]

    # ── passthrough (no session flags) ───────────────────────────────────

    def test_no_session_flags_passthrough(self):
        """hermes -w chat -q hello (nothing to merge)"""
        result = _coalesce_session_name_args(["-w", "chat", "-q", "hello"])
        assert result == ["-w", "chat", "-q", "hello"]

    def test_empty_argv(self):
        assert _coalesce_session_name_args([]) == []

    # ── subcommand boundary ──────────────────────────────────────────────

    def test_stops_at_sessions_subcommand(self):
        """hermes -c my project sessions list → stops before 'sessions'"""
        assert _coalesce_session_name_args(
            ["-c", "my", "project", "sessions", "list"]
        ) == ["-c", "my project", "sessions", "list"]

    def test_stops_at_setup_subcommand(self):
        """hermes -c my setup → 'setup' is a subcommand, not part of name"""
        assert _coalesce_session_name_args(
            ["-c", "my", "setup"]
        ) == ["-c", "my", "setup"]

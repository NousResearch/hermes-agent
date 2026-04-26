"""Tests for /plan slash command path generation (fix for Errno 36 ENAMETOOLONG)."""

import os
from datetime import datetime
from pathlib import Path

import pytest


def test_plan_path_short_instruction():
    from agent.skill_commands import build_plan_path

    path = build_plan_path("weekly sync", now=datetime(2024, 1, 15, 9, 30, 0))
    assert str(path) == ".hermes/plans/2024-01-15_093000-weekly-sync.md"


def test_plan_path_long_instruction_stays_within_name_max():
    """A 1000-char instruction must produce a filename ≤ 255 bytes (NAME_MAX)."""
    from agent.skill_commands import build_plan_path

    long_instruction = "a " * 500  # 1000 chars
    path = build_plan_path(long_instruction, now=datetime(2024, 1, 15, 9, 0, 0))
    filename = path.name
    assert len(filename.encode()) <= 255, f"filename too long: {len(filename.encode())} bytes"


def test_plan_path_empty_instruction_uses_default_slug():
    from agent.skill_commands import build_plan_path

    path = build_plan_path("", now=datetime(2024, 3, 1, 12, 0, 0))
    assert path.name.endswith("-conversation-plan.md")


def test_plan_path_timestamp_format():
    from agent.skill_commands import build_plan_path

    path = build_plan_path("test task", now=datetime(2024, 6, 5, 14, 22, 45))
    assert path.name.startswith("2024-06-05_142245-")


def test_plan_path_slug_capped_at_eight_words():
    from agent.skill_commands import build_plan_path

    instruction = "one two three four five six seven eight nine ten"
    path = build_plan_path(instruction, now=datetime(2024, 1, 1, 0, 0, 0))
    # After the timestamp prefix (YYYY-MM-DD_HHMMSS-), split on "-" to get slug
    # Timestamp is "2024-01-01_000000" = "2024", "01", "01_000000" parts when split by "-"
    # Full name: "2024-01-01_000000-one-two-three-four-five-six-seven-eight.md"
    # Split by "-" gives: ["2024", "01", "01_000000", "one", "two", ...]
    parts = path.stem.split("-")
    slug_parts = parts[3:]  # skip "2024", "01", "01_000000"
    assert len(slug_parts) <= 8


def test_plan_path_only_first_line_used():
    from agent.skill_commands import build_plan_path

    instruction = "first line summary\nsecond line details\nthird line more"
    path = build_plan_path(instruction, now=datetime(2024, 1, 1, 0, 0, 0))
    assert "second" not in path.name
    assert "third" not in path.name
    assert "first" in path.name


class TestCLIPlanCommand:
    def test_plan_command_includes_runtime_note_with_safe_path(self, monkeypatch):
        from agent.skill_commands import build_skill_invocation_message, scan_skill_commands

        commands = scan_skill_commands()
        if "/plan" not in commands:
            pytest.skip("plan skill not installed")

        msg = build_skill_invocation_message("/plan", "quick review", runtime_note="test-note")
        assert msg is not None
        assert "test-note" in msg

    def test_plan_command_long_instruction_safe_path(self):
        from agent.skill_commands import build_plan_path

        long_text = "analyze the entire codebase for performance bottlenecks across all modules"
        path = build_plan_path(long_text, now=datetime(2024, 5, 10, 8, 0, 0))
        assert len(path.name.encode()) <= 255

    def test_plan_command_no_args_still_sends_skill_message(self, monkeypatch):
        from agent.skill_commands import build_skill_invocation_message, scan_skill_commands

        commands = scan_skill_commands()
        if "/plan" not in commands:
            pytest.skip("plan skill not installed")

        msg = build_skill_invocation_message("/plan", "", runtime_note="")
        assert msg is not None

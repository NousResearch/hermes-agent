"""Tests for the command palette feature."""

import pytest
from hermes_cli.command_palette import fuzzy_match, CommandPalette


class TestFuzzyMatch:
    def test_empty_query_matches_everything(self):
        assert fuzzy_match("reset", "") is True
        assert fuzzy_match("anything", "") is True

    def test_exact_substring_match(self):
        assert fuzzy_match("reset", "res") is True
        assert fuzzy_match("reset", "reset") is True
        assert fuzzy_match("reset", "RES") is True  # case insensitive

    def test_fuzzy_match(self):
        assert fuzzy_match("reset", "rst") is True
        assert fuzzy_match("reset", "rse") is True
        assert fuzzy_match("background", "bg") is True
        assert fuzzy_match("background", "bgr") is True

    def test_no_match(self):
        assert fuzzy_match("reset", "xyz") is False
        assert fuzzy_match("reset", "resx") is False


class TestCommandPalette:
    def test_initialization(self):
        commands = [
            {
                "name": "reset",
                "description": "Reset conversation",
                "category": "Session",
                "aliases": (),
            },
            {
                "name": "clear",
                "description": "Clear screen",
                "category": "Session",
                "aliases": (),
            },
        ]
        selected = []
        closed = []

        palette = CommandPalette(
            commands=commands,
            on_select=lambda cmd: selected.append(cmd),
            on_close=lambda: closed.append(True),
        )

        assert palette is not None
        assert len(palette._filtered_commands) == 2

    def test_filter_by_name(self):
        commands = [
            {
                "name": "reset",
                "description": "Reset conversation",
                "category": "Session",
                "aliases": (),
            },
            {
                "name": "clear",
                "description": "Clear screen",
                "category": "Session",
                "aliases": (),
            },
        ]

        palette = CommandPalette(
            commands=commands,
            on_select=lambda cmd: None,
            on_close=lambda: None,
        )

        palette._query = "res"
        palette._filter_commands()

        assert len(palette._filtered_commands) == 1
        assert palette._filtered_commands[0]["name"] == "reset"

    def test_filter_by_alias(self):
        commands = [
            {
                "name": "background",
                "description": "Run in background",
                "category": "Session",
                "aliases": ("bg",),
            },
        ]

        palette = CommandPalette(
            commands=commands,
            on_select=lambda cmd: None,
            on_close=lambda: None,
        )

        palette._query = "bg"
        palette._filter_commands()

        assert len(palette._filtered_commands) == 1
        assert palette._filtered_commands[0]["name"] == "background"

    def test_on_select_called(self):
        commands = [
            {
                "name": "reset",
                "description": "Reset",
                "category": "Session",
                "aliases": (),
            },
        ]
        selected = []

        palette = CommandPalette(
            commands=commands,
            on_select=lambda cmd: selected.append(cmd),
            on_close=lambda: None,
        )

        palette._on_select("reset")
        assert selected == ["reset"]

    def test_on_close_called(self):
        commands = [
            {
                "name": "reset",
                "description": "Reset",
                "category": "Session",
                "aliases": (),
            },
        ]
        closed = []

        palette = CommandPalette(
            commands=commands,
            on_select=lambda cmd: None,
            on_close=lambda: closed.append(True),
        )

        palette._on_close()
        assert len(closed) == 1

    def test_empty_filter_returns_all(self):
        commands = [
            {
                "name": "reset",
                "description": "Reset",
                "category": "Session",
                "aliases": (),
            },
            {
                "name": "clear",
                "description": "Clear",
                "category": "Session",
                "aliases": (),
            },
        ]

        palette = CommandPalette(
            commands=commands,
            on_select=lambda cmd: None,
            on_close=lambda: None,
        )

        palette._query = ""
        palette._filter_commands()

        assert len(palette._filtered_commands) == 2

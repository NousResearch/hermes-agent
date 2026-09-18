"""Tests for /help declutter + Ctrl+P command palette (C-04 / C-05)."""

from types import SimpleNamespace

from cli import HermesCLI
from hermes_cli.commands import HELP_SESSION_SUBGROUPS


def _bare_cli():
    cli = HermesCLI.__new__(HermesCLI)
    return cli


class TestHelpSubgroups:
    def test_session_subgroups_defined(self):
        # The oversized Session category is split into readable sub-headers.
        assert "Context" in HELP_SESSION_SUBGROUPS
        assert "Background & Automation" in HELP_SESSION_SUBGROUPS
        # Representative members land in the right buckets.
        assert "compress" in HELP_SESSION_SUBGROUPS["Context"]
        assert "goal" in HELP_SESSION_SUBGROUPS["Background & Automation"]


class TestCommandPaletteFilter:
    def _state(self, entries, query):
        cli = _bare_cli()
        cli._command_palette_state = {
            "entries": entries, "filter": query, "selected": 0, "_scroll_offset": 0,
        }
        return cli

    ENTRIES = [
        ("/model", "Model", "Switch model"),
        ("/compress", "Context", "Compress the conversation"),
        ("/goal", "Background", "Set an autonomous goal"),
        ("/status", "Context", "Show session status"),
    ]

    def test_empty_query_returns_all(self):
        cli = self._state(self.ENTRIES, "")
        assert cli._command_palette_visible_entries() == self.ENTRIES

    def test_filter_by_command_name(self):
        cli = self._state(self.ENTRIES, "model")
        rows = cli._command_palette_visible_entries()
        assert [r[0] for r in rows] == ["/model"]

    def test_filter_by_description(self):
        cli = self._state(self.ENTRIES, "autonomous")
        rows = cli._command_palette_visible_entries()
        assert [r[0] for r in rows] == ["/goal"]

    def test_subsequence_match(self):
        # "cmp" is a subsequence of "/compress"
        cli = self._state(self.ENTRIES, "cmprs")
        rows = cli._command_palette_visible_entries()
        assert "/compress" in [r[0] for r in rows]

    def test_no_match(self):
        cli = self._state(self.ENTRIES, "zzznope")
        assert cli._command_palette_visible_entries() == []

    def test_command_name_matches_rank_above_description(self):
        # "status" matches /status by name (rank<=2). A row that only matches
        # in its description must rank strictly after any name match.
        entries = [
            ("/goal", "Background", "check the status of things"),  # desc-only match
            ("/status", "Context", "Show session status"),          # name match
        ]
        cli = self._state(entries, "status")
        rows = cli._command_palette_visible_entries()
        assert rows[0][0] == "/status"  # name match ranks first

    def test_query_does_not_overmatch_via_description(self):
        # A short query must not match a row solely because its letters appear
        # as a scattered subsequence of the description (the "steer" bug).
        entries = [
            ("/new", "Session", "Start a totally separate resettable session"),
        ]
        cli = self._state(entries, "steer")
        # "steer" is NOT a subsequence of "/new" and not a substring of the
        # description, so the row is dropped.
        assert cli._command_palette_visible_entries() == []


class TestCommandPaletteSelectionInserts:
    def test_selection_prefills_exact_command(self):
        cli = _bare_cli()
        cli._command_palette_state = {
            "entries": [("/model", "Model", "Switch model")],
            "filter": "", "selected": 0, "_scroll_offset": 0,
        }
        # Fake prompt_toolkit app + buffer.
        buf = SimpleNamespace(text="", cursor_position=0)
        cli._app = SimpleNamespace(current_buffer=buf)
        cli._invalidate = lambda **k: None
        cli._restore_modal_input_snapshot = lambda: None
        cli._handle_command_palette_selection()
        # Exact command string inserted (with trailing space), never executed.
        assert buf.text == "/model "
        assert cli._command_palette_state is None  # palette closed

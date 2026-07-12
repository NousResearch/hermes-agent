"""Tests for skill subcommand tab-completions via metadata.hermes.ui.

Covers: hermes_cli/commands.py SlashCommandCompleter yielding skill subcommands
with display_meta descriptions from SKILL.md frontmatter metadata.hermes.ui.subcommands.
"""

from unittest.mock import MagicMock

from prompt_toolkit.formatted_text import to_plain_text

from hermes_cli.commands import SlashCommandCompleter


def _display_names(completions):
    """Extract plain-text display names from a list of Completion objects."""
    return [to_plain_text(c.display) for c in completions]


def _display_metas(completions):
    """Extract plain-text display_meta from a list of Completion objects."""
    return [to_plain_text(c.display_meta) if c.display_meta else "" for c in completions]


class TestSkillSubcommandCompletions:
    """Skill subcommand completions via metadata.hermes.ui.subcommands dict.

    The completer yields subcommand completions when completing after a skill
    command that has declared subcommands in its SKILL.md frontmatter under
    metadata.hermes.ui.subcommands (a dict of {subcommand: description}).
    """

    def _completer_with_skills(self, skill_commands):
        """Build a SlashCommandCompleter with an injected skill_commands provider."""
        def provider():
            return skill_commands

        completer = SlashCommandCompleter()
        completer._skill_commands_provider = provider
        return completer

    def test_skill_with_subcommands_yields_completions(self):
        skills = {
            "/proto-idea": {
                "name": "proto-idea",
                "description": "Manage ideas.",
                "subcommands": {
                    "list": "Show all ideas in a table",
                    "find": "Search by name or premise",
                },
            }
        }
        completer = self._completer_with_skills(skills)
        doc = MagicMock()
        doc.text_before_cursor = "/proto-idea "
        doc.cursor_position = len("/proto-idea ")
        event = MagicMock()
        completions = list(completer.get_completions(doc, event))
        names = _display_names(completions)
        assert "list" in names
        assert "find" in names

    def test_skill_without_subcommands_yields_nothing(self):
        skills = {
            "/plain-skill": {
                "name": "plain-skill",
                "description": "A plain skill.",
                "subcommands": {},
            }
        }
        completer = self._completer_with_skills(skills)
        doc = MagicMock()
        doc.text_before_cursor = "/plain-skill "
        doc.cursor_position = len("/plain-skill ")
        event = MagicMock()
        completions = list(completer.get_completions(doc, event))
        names = _display_names(completions)
        assert names == []

    def test_skill_with_no_ui_at_all_yields_nothing(self):
        """Skill with no metadata.hermes.ui section — subcommands is {}."""
        skills = {
            "/no-ui-skill": {
                "name": "no-ui-skill",
                "description": "No UI metadata.",
                "subcommands": {},
            }
        }
        completer = self._completer_with_skills(skills)
        doc = MagicMock()
        doc.text_before_cursor = "/no-ui-skill "
        doc.cursor_position = len("/no-ui-skill ")
        event = MagicMock()
        completions = list(completer.get_completions(doc, event))
        names = _display_names(completions)
        assert names == []

    def test_subcommand_description_shown_in_display_meta(self):
        skills = {
            "/test-skill": {
                "name": "test-skill",
                "description": "Test.",
                "subcommands": {"run": "Run the thing"},
            }
        }
        completer = self._completer_with_skills(skills)
        doc = MagicMock()
        doc.text_before_cursor = "/test-skill "
        doc.cursor_position = len("/test-skill ")
        event = MagicMock()
        completions = list(completer.get_completions(doc, event))
        metas = _display_metas(completions)
        assert "Run the thing" in metas

    def test_description_truncated_at_50_chars_with_ellipsis(self):
        long_desc = "A" * 60  # 60 chars, over the 50-char cap
        skills = {
            "/long-desc": {
                "name": "long-desc",
                "description": "Long.",
                "subcommands": {"cmd": long_desc},
            }
        }
        completer = self._completer_with_skills(skills)
        doc = MagicMock()
        doc.text_before_cursor = "/long-desc "
        doc.cursor_position = len("/long-desc ")
        event = MagicMock()
        completions = list(completer.get_completions(doc, event))
        metas = _display_metas(completions)
        assert metas[0] == "A" * 50 + "..."

    def test_description_under_50_chars_shown_as_is(self):
        short_desc = "B" * 45  # under 50
        skills = {
            "/short-desc": {
                "name": "short-desc",
                "description": "Short.",
                "subcommands": {"cmd": short_desc},
            }
        }
        completer = self._completer_with_skills(skills)
        doc = MagicMock()
        doc.text_before_cursor = "/short-desc "
        doc.cursor_position = len("/short-desc ")
        event = MagicMock()
        completions = list(completer.get_completions(doc, event))
        metas = _display_metas(completions)
        assert metas[0] == short_desc
        assert "..." not in metas[0]

    def test_prefix_filter_still_works(self):
        """Only subcommands matching the typed prefix are yielded."""
        skills = {
            "/multi": {
                "name": "multi",
                "description": "Multi.",
                "subcommands": {
                    "list": "Show all",
                    "load": "Load a file",
                    "log": "Show logs",
                },
            }
        }
        completer = self._completer_with_skills(skills)
        doc = MagicMock()
        doc.text_before_cursor = "/multi lo"
        doc.cursor_position = len("/multi lo")
        event = MagicMock()
        completions = list(completer.get_completions(doc, event))
        names = _display_names(completions)
        assert "load" in names
        assert "log" in names
        assert "list" not in names

    def test_trailing_spaces_still_yields_completions(self):
        """Extra spaces after command name should not break completion."""
        skills = {
            "/trailing": {
                "name": "trailing",
                "description": "Trailing.",
                "subcommands": {"list": "List all", "find": "Find"},
            }
        }
        completer = self._completer_with_skills(skills)
        doc = MagicMock()
        doc.text_before_cursor = "/trailing  "
        doc.cursor_position = len("/trailing  ")
        event = MagicMock()
        completions = list(completer.get_completions(doc, event))
        names = _display_names(completions)
        assert "list" in names
        assert "find" in names

    def test_partial_subcommand_prefix_filter(self):
        """Partial subcommand typed — prefix filter applies to second token."""
        skills = {
            "/proto": {
                "name": "proto",
                "description": "Proto.",
                "subcommands": {
                    "list": "Show all",
                    "load": "Load",
                    "log": "Log",
                },
            }
        }
        completer = self._completer_with_skills(skills)
        doc = MagicMock()
        doc.text_before_cursor = "/proto l"
        doc.cursor_position = len("/proto l")
        event = MagicMock()
        completions = list(completer.get_completions(doc, event))
        names = _display_names(completions)
        # "list", "load", "log" all start with "l"
        assert "list" in names
        assert "load" in names
        assert "log" in names
        assert "find" not in names  # "find" does not start with "l"

    def test_subcommand_wrong_type_list_falls_back_to_empty(self):
        """subcommands declared as list (not dict) yields no completions."""
        skills = {
            "/list-type": {
                "name": "list-type",
                "description": "List type.",
                "subcommands": ["list", "find"],  # wrong type
            }
        }
        completer = self._completer_with_skills(skills)
        doc = MagicMock()
        doc.text_before_cursor = "/list-type "
        doc.cursor_position = len("/list-type ")
        event = MagicMock()
        completions = list(completer.get_completions(doc, event))
        names = _display_names(completions)
        assert names == []
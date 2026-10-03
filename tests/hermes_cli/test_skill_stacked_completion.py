"""Interactive skill spelling must agree with slash dispatch."""

from hermes_cli.commands_completion import SlashCommandCompleter


def test_stacked_completion_prefers_exact_underscored_plugin_key():
    commands = {
        "/plugin:my_skill": {"description": "Underscored"},
        "/other:guide": {"description": "Other"},
    }
    completer = SlashCommandCompleter(skill_commands_provider=lambda: commands)
    assert completer._is_skill_command("/plugin:my_skill")
    results = list(completer._stacked_skill_completions("/plugin:my_skill /other:"))
    assert [result.display_text for result in results] == ["/other:guide"]

def test_stacked_completion_matches_underscored_current_prefix():
    commands = {
        "/other:guide": {"description": "Other"},
        "/plugin:my_skill": {"description": "Underscored"},
    }
    completer = SlashCommandCompleter(skill_commands_provider=lambda: commands)
    results = list(completer._stacked_skill_completions("/other:guide /plugin:my_"))
    assert [result.display_text for result in results] == ["/plugin:my_skill"]
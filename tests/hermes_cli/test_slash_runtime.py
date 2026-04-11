"""Tests for shared slash-command runtime resolution."""

from hermes_cli.commands import (
    resolve_cli_slash_command,
    resolve_gateway_slash_command,
)


def test_cli_builtin_prefix_resolves_to_unique_shortest_match():
    result = resolve_cli_slash_command("/qui", config={}, skill_commands={})

    assert result.status == "matched"
    assert result.entry is not None
    assert result.entry.kind == "builtin"
    assert result.entry.canonical_name == "quit"


def test_cli_exact_builtin_alias_uses_central_command_resolution():
    result = resolve_cli_slash_command("/q", config={}, skill_commands={})

    assert result.status == "matched"
    assert result.entry is not None
    assert result.entry.kind == "builtin"
    assert result.entry.canonical_name == "quit"


def test_cli_skill_prefix_resolution_preserved():
    result = resolve_cli_slash_command(
        "/test-skill-xy",
        config={},
        skill_commands={
            "/test-skill-xyz": {
                "name": "Test Skill",
                "description": "Skill description",
            }
        },
    )

    assert result.status == "matched"
    assert result.entry is not None
    assert result.entry.kind == "skill"
    assert result.entry.slash_name == "/test-skill-xyz"


def test_cli_quick_commands_do_not_participate_in_prefix_matching():
    result = resolve_cli_slash_command(
        "/foo",
        config={"quick_commands": {"foobar": {"type": "exec", "command": "echo hi"}}},
        skill_commands={},
    )

    assert result.status == "unknown"
    assert result.entry is None


def test_gateway_builtin_underscore_alias_resolves():
    result = resolve_gateway_slash_command(
        "reload_mcp",
        config={},
        skill_commands={},
    )

    assert result.status == "matched"
    assert result.entry is not None
    assert result.entry.kind == "builtin"
    assert result.entry.canonical_name == "reload-mcp"


def test_gateway_skill_underscore_form_resolves_to_hyphenated_skill():
    result = resolve_gateway_slash_command(
        "claude_code",
        config={},
        skill_commands={
            "/claude-code": {
                "name": "Claude Code",
                "description": "Skill description",
            }
        },
        plugin_commands={
            "claude_code": {
                "description": "Plugin command",
                "handler": lambda _args: None,
            }
        },
    )

    assert result.status == "matched"
    assert result.entry is not None
    assert result.entry.kind == "skill"
    assert result.entry.canonical_name == "claude-code"


def test_gateway_plugin_underscore_form_resolves_to_hyphenated_plugin_command():
    result = resolve_gateway_slash_command(
        "design_sync",
        config={},
        skill_commands={},
        plugin_commands={
            "design-sync": {
                "description": "Plugin command",
                "handler": lambda _args: None,
            }
        },
    )

    assert result.status == "matched"
    assert result.entry is not None
    assert result.entry.kind == "plugin"
    assert result.entry.canonical_name == "design-sync"

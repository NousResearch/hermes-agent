import pytest

from hermes_cli.skills_command_request import (
    SkillsCommandRequest,
    _SkillsArgumentParser,
    parse_skills_slash_command,
    register_skills_subcommands,
    request_from_namespace,
)


def _make_parser():
    parser = _SkillsArgumentParser(prog="skills", add_help=False)
    register_skills_subcommands(parser)
    return parser


def test_request_from_namespace_preserves_cli_confirmation_policy():
    parser = _make_parser()
    args = parser.parse_args(["install", "openai/skills/skill-creator", "--yes"])

    request = request_from_namespace(args, command_source="cli")

    assert request == SkillsCommandRequest(
        action="install",
        command_source="cli",
        identifier="openai/skills/skill-creator",
        skip_confirm=True,
        invalidate_cache=True,
    )


def test_request_from_namespace_preserves_cli_cache_invalidation_behavior():
    parser = _make_parser()
    args = parser.parse_args(["install", "openai/skills/skill-creator"])

    request = request_from_namespace(args, command_source="cli")

    assert request.invalidate_cache is True


def test_parse_skills_slash_command_uses_same_parser_tree():
    request = parse_skills_slash_command(
        "/skills snapshot import my.json --force"
    )

    assert request.action == "snapshot"
    assert request.snapshot_action == "import"
    assert request.path == "my.json"
    assert request.force is True


def test_parse_skills_slash_command_accepts_multiword_search_without_quotes():
    request = parse_skills_slash_command("/skills search foo bar --limit 5")

    assert request.action == "search"
    assert request.query == "foo bar"
    assert request.limit == 5


def test_parse_skills_slash_command_marks_config_unavailable():
    request = parse_skills_slash_command("/skills config")

    assert request.action == "config"
    assert request.config_available is False


def test_parse_skills_slash_command_tap_defaults_to_list():
    request = parse_skills_slash_command("/skills tap")

    assert request.action == "tap"
    assert request.tap_action == "list"


def test_parse_skills_slash_command_returns_error_request_on_invalid_input():
    request = parse_skills_slash_command("/skills install")

    assert request.action == "error"
    assert request.error


def test_parse_skills_slash_command_returns_error_request_on_malformed_quotes():
    request = parse_skills_slash_command('/skills search "unterminated')

    assert request.action == "error"
    assert request.error

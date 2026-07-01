"""Characterization tests for personality parsing across runtime surfaces."""

from dataclasses import FrozenInstanceError
import threading
import types
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agent.personality import (
    PersonalityConfigError,
    PersonalityDefinition,
    PersonalityNotFoundError,
    is_personality_clear_request,
    resolve_personality,
)


PERSONALITY_CASES = [
    pytest.param("You are helpful.", "You are helpful.", id="simple-string"),
    pytest.param(
        "Você é uma assistente gentil ✨",
        "Você é uma assistente gentil ✨",
        id="unicode-string",
    ),
    pytest.param(
        {"system_prompt": "You are precise."},
        "You are precise.",
        id="minimal-mapping",
    ),
    pytest.param(
        {
            "description": "Expert programmer",
            "system_prompt": "You are an expert programmer.",
            "tone": "technical and precise",
            "style": "use code examples",
        },
        "You are an expert programmer.\n"
        "Tone: technical and precise\n"
        "Style: use code examples",
        id="complete-mapping",
    ),
    pytest.param(
        {"description": "Display text only"},
        "",
        id="description-only",
    ),
    pytest.param(
        {"tone": "calm"},
        "Tone: calm",
        id="tone-only",
    ),
    pytest.param(
        {"style": "brief"},
        "Style: brief",
        id="style-only",
    ),
    pytest.param(
        {"system_prompt": "", "tone": "", "style": ""},
        "",
        id="empty-fields",
    ),
]

INVALID_PERSONALITY_CASES = [
    pytest.param(None, "got null", id="null"),
    pytest.param([], "got list", id="list"),
    pytest.param(42, "got int", id="integer"),
    pytest.param(
        {"description": 42},
        "field 'description'",
        id="non-text-description",
    ),
    pytest.param(
        {"system_prompt": 42},
        "field 'system_prompt'",
        id="non-text-system-prompt",
    ),
    pytest.param(
        {"tone": 42},
        "field 'tone'",
        id="non-text-tone",
    ),
    pytest.param(
        {"style": 42},
        "field 'style'",
        id="non-text-style",
    ),
]


@pytest.mark.parametrize(("value", "expected"), PERSONALITY_CASES)
def test_personality_definition_parses_and_renders_legacy_formats(value, expected):
    definition = PersonalityDefinition.parse("characterized", value)

    assert definition.render() == expected


def test_personality_definition_is_immutable():
    definition = PersonalityDefinition.parse("helper", "You are helpful.")

    with pytest.raises(FrozenInstanceError):
        definition.tone = "different"  # type: ignore[misc]


def test_personality_definition_retains_all_mapping_fields():
    definition = PersonalityDefinition.parse(
        "expert",
        {
            "description": "Expert programmer",
            "system_prompt": "You are an expert programmer.",
            "tone": "technical and precise",
            "style": "use code examples",
        },
    )

    assert definition == PersonalityDefinition(
        name="expert",
        description="Expert programmer",
        system_prompt="You are an expert programmer.",
        tone="technical and precise",
        style="use code examples",
    )


@pytest.mark.parametrize(("value", "detail"), INVALID_PERSONALITY_CASES)
def test_personality_definition_rejects_invalid_configuration(value, detail):
    with pytest.raises(PersonalityConfigError) as exc_info:
        PersonalityDefinition.parse("broken", value)

    assert exc_info.value.personality_name == "broken"
    assert "Invalid personality 'broken'" in str(exc_info.value)
    assert detail in str(exc_info.value)


@pytest.mark.parametrize("alias", ["none", "default", "neutral"])
def test_resolve_personality_centralizes_clear_aliases(alias):
    assert is_personality_clear_request(alias)
    assert resolve_personality(alias, {"helper": "You are helpful."}) == ("", "")


def test_resolve_personality_reports_missing_name():
    with pytest.raises(PersonalityNotFoundError) as exc_info:
        resolve_personality("missing", {"helper": "You are helpful."})

    assert exc_info.value.normalized_name == "missing"
    assert exc_info.value.available_names == ("helper",)


def test_resolve_personality_ignores_invalid_unselected_definition():
    assert resolve_personality(
        "helper",
        {
            "helper": "You are helpful.",
            "broken": {"tone": 42},
        },
    ) == ("helper", "You are helpful.")


def _cli_applied_overlay(personalities, requested_name) -> str:
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    cli = CLICommandsMixin.__new__(CLICommandsMixin)
    cli.personalities = personalities
    cli.system_prompt = "before"
    cli.agent = object()

    with patch("cli.save_config_value", return_value=True):
        cli._handle_personality_command(f"/personality {requested_name}")

    return cli.system_prompt


async def _gateway_applied_overlay(tmp_path, personalities, requested_name) -> str:
    from gateway.slash_commands import GatewaySlashCommandsMixin

    runner = GatewaySlashCommandsMixin.__new__(GatewaySlashCommandsMixin)
    runner._ephemeral_system_prompt = "before"
    event = MagicMock()
    event.get_command_args.return_value = requested_name
    config = {
        "agent": {
            "personalities": personalities,
            "system_prompt": "before",
        }
    }
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True),
        encoding="utf-8",
    )

    with patch("gateway.run._hermes_home", tmp_path):
        await runner._handle_personality_command(event)

    return runner._ephemeral_system_prompt


def _tui_applied_overlay(personalities, requested_name) -> str:
    from tui_gateway import server

    sid = "personality-parity"
    agent = types.SimpleNamespace(
        ephemeral_system_prompt="before",
        _cached_system_prompt="cached",
    )
    session = {
        "agent": agent,
        "session_key": "personality-parity-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
    }
    server._sessions[sid] = session
    cfg = {"agent": {"personalities": personalities}}
    available_patch = patch.object(
        server,
        "_available_personalities",
        return_value=personalities,
    )

    try:
        with (
            patch.object(server, "_load_cfg", return_value=cfg),
            available_patch as available_personalities,
            patch.object(server, "_write_config_key"),
            patch.object(server, "_session_info", return_value={"model": "stub"}),
            patch.object(server, "_emit"),
        ):
            response = server.handle_request(
                {
                    "id": "personality-parity",
                    "method": "config.set",
                    "params": {
                        "session_id": sid,
                        "key": "personality",
                        "value": requested_name,
                    },
                }
            )
        assert "error" not in response
        if is_personality_clear_request(requested_name):
            available_personalities.assert_not_called()
        else:
            available_personalities.assert_called_once()
        return agent.ephemeral_system_prompt or ""
    finally:
        server._sessions.pop(sid, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(("definition", "expected"), PERSONALITY_CASES)
async def test_runtime_surfaces_apply_identical_overlay(
    tmp_path, definition, expected
):
    personalities = {"characterized": definition}

    cli_prompt = _cli_applied_overlay(personalities, "characterized")
    gateway_prompt = await _gateway_applied_overlay(
        tmp_path,
        personalities,
        "characterized",
    )
    tui_prompt = _tui_applied_overlay(personalities, "characterized")

    assert cli_prompt == gateway_prompt == tui_prompt == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["none", "default", "neutral"])
async def test_existing_clear_aliases_resolve_to_empty_prompt(tmp_path, alias):
    personalities = {
        "helper": "You are helpful.",
        "broken": {"tone": 42},
    }

    cli_prompt = _cli_applied_overlay(personalities, alias)
    gateway_prompt = await _gateway_applied_overlay(
        tmp_path,
        personalities,
        alias,
    )
    tui_prompt = _tui_applied_overlay(personalities, alias)

    assert cli_prompt == gateway_prompt == tui_prompt == ""


@pytest.mark.asyncio
async def test_existing_unknown_personality_does_not_mutate_cli_or_gateway(
    tmp_path,
):
    from cli import HermesCLI
    from gateway.run import GatewayRunner

    active_agent = MagicMock()
    cli = HermesCLI.__new__(HermesCLI)
    cli.personalities = {"helper": "You are helpful."}
    cli.system_prompt = "before"
    cli.agent = active_agent
    with patch("cli.save_config_value") as save_cli:
        cli._handle_personality_command("/personality missing")
    assert cli.system_prompt == "before"
    assert cli.agent is active_agent
    save_cli.assert_not_called()

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._ephemeral_system_prompt = "before"
    event = MagicMock()
    event.get_command_args.return_value = "missing"
    config = {
        "agent": {
            "personalities": {"helper": "You are helpful."},
            "system_prompt": "before",
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    before = config_path.read_text(encoding="utf-8")
    with patch("gateway.run._hermes_home", tmp_path):
        result = await runner._handle_personality_command(event)

    assert "unknown personality" in result.lower()
    assert runner._ephemeral_system_prompt == "before"
    assert config_path.read_text(encoding="utf-8") == before


@pytest.mark.asyncio
@pytest.mark.parametrize(("definition", "_detail"), INVALID_PERSONALITY_CASES)
async def test_selected_invalid_personality_does_not_mutate_runtime_surfaces(
    tmp_path,
    capsys,
    definition,
    _detail,
):
    from gateway.slash_commands import GatewaySlashCommandsMixin
    from hermes_cli.cli_commands_mixin import CLICommandsMixin
    from tui_gateway import server

    active_agent = MagicMock()
    cli = CLICommandsMixin.__new__(CLICommandsMixin)
    cli.personalities = {"broken": definition}
    cli.system_prompt = "before"
    cli.agent = active_agent
    with patch("cli.save_config_value") as save_cli:
        cli._handle_personality_command("/personality broken")

    assert "Invalid personality 'broken'" in capsys.readouterr().out
    assert cli.system_prompt == "before"
    assert cli.agent is active_agent
    save_cli.assert_not_called()

    runner = GatewaySlashCommandsMixin.__new__(GatewaySlashCommandsMixin)
    runner._ephemeral_system_prompt = "before"
    event = MagicMock()
    event.get_command_args.return_value = "broken"
    config = {
        "agent": {
            "personalities": {"broken": definition},
            "system_prompt": "before",
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True),
        encoding="utf-8",
    )
    before = config_path.read_text(encoding="utf-8")

    with patch("gateway.run._hermes_home", tmp_path):
        result = await runner._handle_personality_command(event)

    assert "Invalid personality 'broken'" in result
    assert runner._ephemeral_system_prompt == "before"
    assert config_path.read_text(encoding="utf-8") == before

    sid = "invalid-personality"
    tui_agent = types.SimpleNamespace(
        ephemeral_system_prompt="before",
        _cached_system_prompt="cached",
    )
    session = {
        "agent": tui_agent,
        "session_key": "invalid-personality-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
    }
    writes = []
    server._sessions[sid] = session
    try:
        with (
            patch.object(server, "_load_cfg", return_value=config),
            patch.object(
                server,
                "_available_personalities",
                return_value={"broken": definition},
            ),
            patch.object(
                server,
                "_write_config_key",
                side_effect=lambda path, value: writes.append((path, value)),
            ),
        ):
            response = server.handle_request(
                {
                    "id": "invalid-personality",
                    "method": "config.set",
                    "params": {
                        "session_id": sid,
                        "key": "personality",
                        "value": "broken",
                    },
                }
            )

        assert "Invalid personality 'broken'" in response["error"]["message"]
        assert writes == []
        assert server._sessions[sid] is session
        assert session["agent"] is tui_agent
        assert tui_agent.ephemeral_system_prompt == "before"
        assert tui_agent._cached_system_prompt == "cached"
        assert session["history"] == []
    finally:
        server._sessions.pop(sid, None)


@pytest.mark.asyncio
async def test_invalid_unselected_personality_does_not_break_catalog_listing(
    tmp_path,
    capsys,
):
    from cli import HermesCLI
    from gateway.run import GatewayRunner

    personalities = {
        "helper": "You are helpful.",
        "broken": {"style": 42},
    }
    cli = HermesCLI.__new__(HermesCLI)
    cli.personalities = personalities
    cli._handle_personality_command("/personality")
    cli_output = capsys.readouterr().out

    assert "helper" in cli_output
    assert "broken" in cli_output
    assert "invalid" in cli_output

    runner = GatewayRunner.__new__(GatewayRunner)
    event = MagicMock()
    event.get_command_args.return_value = ""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"agent": {"personalities": personalities}}),
        encoding="utf-8",
    )
    with patch("gateway.run._hermes_home", tmp_path):
        gateway_output = await runner._handle_personality_command(event)

    assert "helper" in gateway_output
    assert "broken" in gateway_output
    assert "invalid" in gateway_output

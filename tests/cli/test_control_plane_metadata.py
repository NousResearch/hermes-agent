"""Wave 7 control-plane metadata acceptance tests."""

from pathlib import Path
from unittest.mock import patch


class TestUnifiedCommandMetadata:
    def test_builtin_plugin_and_skill_share_metadata_envelope(self, tmp_path):
        from hermes_cli.commands import command_metadata_dicts

        skill_dir = tmp_path / "skills" / "local-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: local-skill\ndescription: Local skill.\n---\n\nBody.\n"
        )

        plugin_commands = {
            "plug": {
                "description": "Plugin command",
                "args_hint": "[x]",
                "category": "Plugins",
                "aliases": ["p"],
                "permissions": ["network"],
                "platforms": ["cli"],
            }
        }

        with (
            patch("tools.skills_tool.SKILLS_DIR", tmp_path / "empty-global"),
            patch("agent.skill_utils.get_external_skills_dirs", return_value=[tmp_path / "skills"]),
            patch("hermes_cli.plugins.get_plugin_commands", return_value=plugin_commands),
        ):
            metadata = command_metadata_dicts(include_plugins=True, include_skills=True)

        by_name = {entry["name"]: entry for entry in metadata}
        for name in ("help", "plug", "local-skill"):
            entry = by_name[name]
            assert set(
                [
                    "name",
                    "aliases",
                    "source",
                    "category",
                    "schema",
                    "permissions",
                    "platforms",
                    "hidden",
                    "deprecated",
                    "help",
                    "args_hint",
                ]
            ).issubset(entry)

        assert by_name["help"]["source"] == "builtin"
        assert by_name["plug"]["source"] == "plugin"
        assert by_name["plug"]["aliases"] == ["p"]
        assert by_name["plug"]["permissions"] == ["network"]
        assert by_name["local-skill"]["source"] == "skill"


def test_acp_available_commands_derive_from_unified_metadata():
    import pytest
    pytest.importorskip("acp")

    from acp_adapter.server import HermesACPAgent
    from hermes_cli.commands import command_metadata_dicts

    advertised = {cmd.name for cmd in HermesACPAgent._available_commands()}
    metadata = {
        entry["name"]: entry
        for entry in command_metadata_dicts(include_plugins=False, include_skills=False)
    }

    assert "help" in advertised
    assert "model" in advertised
    assert HermesACPAgent._ACP_COMMAND_ALIASES["reset"] == "new"
    assert metadata["help"]["help"] == "Show available commands"
    assert metadata["model"]["args_hint"]


def test_tui_catalog_includes_unified_metadata_records():
    import tui_gateway.server as server

    response = server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "commands.catalog", "params": {}})
    assert "error" not in response
    records = response["result"]["commands"]
    help_record = next(item for item in records if item["name"] == "help")
    assert help_record["source"] == "builtin"
    assert help_record["help"] == "Show available commands"
    assert help_record["category"] == "Info"

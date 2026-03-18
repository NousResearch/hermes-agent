from rich.console import Console

import hermes_cli.banner as banner
import model_tools
import tools.mcp_tool


def test_display_toolset_name_strips_legacy_suffix():
    assert banner._display_toolset_name("homeassistant_tools") == "homeassistant"
    assert banner._display_toolset_name("honcho_tools") == "honcho"
    assert banner._display_toolset_name("browser") == "browser"


def test_build_welcome_banner_uses_normalized_toolset_names(monkeypatch):
    monkeypatch.setattr(
        model_tools,
        "check_tool_availability",
        lambda quiet=False: (
            ["web"],
            [
                {"name": "homeassistant", "tools": ["ha_call_service"]},
                {"name": "honcho", "tools": ["honcho_conclude"]},
            ],
        ),
    )
    monkeypatch.setattr(banner, "get_available_skills", lambda: {})
    monkeypatch.setattr(banner, "get_update_result", lambda timeout=0.5: None)
    monkeypatch.setattr(tools.mcp_tool, "get_mcp_status", lambda: [])

    console = Console(record=True, force_terminal=False, color_system=None, width=160)
    banner.build_welcome_banner(
        console=console,
        model="anthropic/test-model",
        cwd="/tmp/project",
        tools=[
            {"function": {"name": "web_search"}},
            {"function": {"name": "read_file"}},
        ],
        get_toolset_for_tool=lambda name: {
            "web_search": "web_tools",
            "read_file": "file",
        }.get(name),
    )

    output = console.export_text()
    assert "homeassistant:" in output
    assert "honcho:" in output
    assert "web:" in output
    assert "homeassistant_tools:" not in output
    assert "honcho_tools:" not in output

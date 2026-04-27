"""Tests for banner toolset name normalization, skin color usage, and Code Mode console."""

import os
from unittest.mock import patch

from rich.console import Console

import hermes_cli.banner as banner
import model_tools
import tools.mcp_tool


# ---------------------------------------------------------------------------
# Toolset name normalization
# ---------------------------------------------------------------------------

def test_display_toolset_name_strips_legacy_suffix():
    assert banner._display_toolset_name("homeassistant_tools") == "homeassistant"
    assert banner._display_toolset_name("honcho_tools") == "honcho"
    assert banner._display_toolset_name("web_tools") == "web"


def test_display_toolset_name_preserves_clean_names():
    assert banner._display_toolset_name("browser") == "browser"
    assert banner._display_toolset_name("file") == "file"
    assert banner._display_toolset_name("terminal") == "terminal"


def test_display_toolset_name_handles_empty():
    assert banner._display_toolset_name("") == "unknown"
    assert banner._display_toolset_name(None) == "unknown"


def test_build_welcome_banner_uses_normalized_toolset_names():
    """Unavailable toolsets should not have '_tools' appended in banner output."""
    with (
        patch.object(
            model_tools,
            "check_tool_availability",
            return_value=(
                ["web"],
                [
                    {"name": "homeassistant", "tools": ["ha_call_service"]},
                    {"name": "honcho", "tools": ["honcho_conclude"]},
                ],
            ),
        ),
        patch.object(banner, "get_available_skills", return_value={}),
        patch.object(banner, "get_update_result", return_value=None),
        patch.object(tools.mcp_tool, "get_mcp_status", return_value=[]),
    ):
        console = Console(
            record=True, force_terminal=False, color_system=None, width=160
        )
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
    assert "web_tools:" not in output


# ---------------------------------------------------------------------------
# Code Mode console — rendering
# ---------------------------------------------------------------------------

def _make_console(width: int = 120) -> Console:
    return Console(record=True, force_terminal=False, color_system=None, width=width)


def test_build_hermes_code_console_renders_without_crash():
    """Banner renders without errors when all params are provided."""
    console = _make_console()
    banner.build_hermes_code_console(console=console, model="claude-opus-4.7", profile="test-profile")
    output = console.export_text()
    assert len(output) > 0


def test_build_hermes_code_console_shows_title():
    """Title must reference 'Hermes Code Mode'."""
    console = _make_console()
    banner.build_hermes_code_console(console=console, model="test-model")
    output = console.export_text()
    assert "Hermes Code Mode" in output


def test_build_hermes_code_console_shows_ai_development_subtitle():
    """Subtitle 'AI Development Console' must appear in the output."""
    console = _make_console()
    banner.build_hermes_code_console(console=console, model="test-model")
    output = console.export_text()
    assert "AI Development Console" in output or "Development Console" in output


def test_build_hermes_code_console_shows_model_info():
    """Model name passed in must appear in the output."""
    console = _make_console()
    banner.build_hermes_code_console(console=console, model="my-special-model")
    output = console.export_text()
    assert "my-special-model" in output


def test_build_hermes_code_console_shows_provider_label():
    """'Provider:' label must appear in the output."""
    console = _make_console()
    banner.build_hermes_code_console(console=console, model="test-model")
    output = console.export_text()
    assert "Provider" in output


def test_build_hermes_code_console_shows_workspace_label():
    """'Workspace:' label must appear in the output."""
    console = _make_console()
    banner.build_hermes_code_console(console=console, model="test-model")
    output = console.export_text()
    assert "Workspace" in output


def test_build_hermes_code_console_shows_branch_label():
    """'Branch:' label must appear in the output."""
    console = _make_console()
    banner.build_hermes_code_console(console=console, model="test-model")
    output = console.export_text()
    assert "Branch" in output


def test_build_hermes_code_console_shows_backend_label():
    """'Backend:' label must appear in the output."""
    console = _make_console()
    banner.build_hermes_code_console(console=console, model="test-model")
    output = console.export_text()
    assert "Backend" in output


def test_build_hermes_code_console_shows_web_cockpit_label():
    """'Web Cockpit:' label and URL must appear in the output."""
    console = _make_console()
    banner.build_hermes_code_console(console=console, model="test-model")
    output = console.export_text()
    assert "Web Cockpit" in output
    assert "localhost:3001" in output


def test_build_hermes_code_console_shows_quick_actions():
    """'Quick Actions' section must appear."""
    console = _make_console()
    banner.build_hermes_code_console(console=console, model="test-model")
    output = console.export_text()
    assert "Quick Actions" in output


def test_build_hermes_code_console_slash_commands_present():
    """Core slash commands must be listed in the output."""
    console = _make_console()
    banner.build_hermes_code_console(console=console, model="test-model")
    output = console.export_text()
    assert "/code" in output
    assert "/web" in output
    assert "/workspace" in output
    assert "/session" in output
    assert "/help" in output


def test_build_hermes_code_console_skills_code_command():
    """/skills-code must appear in the quick actions."""
    console = _make_console()
    banner.build_hermes_code_console(console=console, model="test-model")
    output = console.export_text()
    assert "/skills-code" in output


def test_build_hermes_code_console_approvals_command():
    """/approvals must appear in the quick actions."""
    console = _make_console()
    banner.build_hermes_code_console(console=console, model="test-model")
    output = console.export_text()
    assert "/approvals" in output


def test_build_hermes_code_banner_preserves_classic_impact_with_code_identity():
    """The Code banner keeps the old large logo/caduceus feel with new naming."""
    data = {
        "provider": "MiniMax",
        "model": "MiniMax-M2.7-highspeed",
        "profile": "arkeon-cto",
        "workspace": "/home/andrey/dev/hermes-agent",
        "branch": "feature/hermes-code-mode",
        "session_id": "20260427_150000",
        "backend_status": "online",
        "backend_port": "9119",
        "web_url": "http://localhost:3001/code",
        "db_schema": "v17",
        "tools_available": 28,
        "skills_available": 94,
        "code_skills": ["fix_build", "review_diff", "implement_feature"],
        "active_sessions": 1,
        "pending_approvals": 0,
    }

    console = _make_console(width=150)
    with (
        patch.object(banner, "_get_code_mode_data", return_value=data),
        patch.object(
            banner,
            "format_banner_version_label",
            return_value="Hermes Agent v0.9.0 (2026.4.13) · upstream abc12345",
        ),
        patch("hermes_cli.banner.shutil.get_terminal_size", return_value=os.terminal_size((150, 40))),
    ):
        banner.build_hermes_code_console(
            console=console,
            model="MiniMax-M2.7-highspeed",
            provider="MiniMax",
            profile="arkeon-cto",
            session_id="20260427_150000",
            tools_count=28,
            skills_count=94,
        )

    output = console.export_text()
    assert "HERMES CODE" in output
    assert "AI Development Console" in output
    assert "Hermes Agent Code v0.9.0 (2026.4.13) · upstream abc12345" in output
    assert "⣿" in output
    assert "Provider" in output and "MiniMax" in output
    assert "Model" in output and "MiniMax-M2.7-highspeed" in output
    assert "Profile" in output and "arkeon-cto" in output
    assert "Workspace" in output and "/home/andrey/dev/hermes-agent" in output
    assert "Branch" in output and "feature/hermes-code-mode" in output
    assert "Session" in output and "20260427_150000" in output
    assert "Backend" in output and "online :9119" in output
    assert "Web Cockpit" in output and "http://localhost:3001/code" in output
    assert "DB" in output and "v17" in output
    assert "Tools: 28 available" in output
    assert "Skills: 94 available" in output
    assert "Code Skills: fix_build · review_diff · implement_feature" in output
    assert "Available Tools" not in output
    assert "Available Skills" not in output


def test_build_hermes_code_console_compact_layout_matches_required_shape():
    data = {
        "provider": "MiniMax",
        "model": "MiniMax-M2.7-highspeed",
        "profile": "arkeon-cto",
        "workspace": "/home/andrey/dev/hermes-agent",
        "branch": "feature/hermes-code-mode",
        "session_id": "20260427_150000",
        "backend_status": "online",
        "backend_port": "9119",
        "web_url": "http://localhost:3001/code",
        "db_schema": "v17",
        "tools_available": 28,
        "skills_available": 94,
        "code_skills": ["fix_build", "review_diff", "implement_feature"],
        "active_sessions": 1,
        "pending_approvals": 0,
    }

    console = _make_console(width=78)
    with (
        patch.object(banner, "_get_code_mode_data", return_value=data),
        patch.object(
            banner,
            "format_banner_version_label",
            return_value="Hermes Agent v0.9.0 (2026.4.13)",
        ),
        patch("hermes_cli.banner.shutil.get_terminal_size", return_value=os.terminal_size((78, 40))),
    ):
        banner.build_hermes_code_console(console=console, model="MiniMax-M2.7-highspeed")

    output = console.export_text()
    assert "HERMES CODE · AI Development Console" in output
    assert "Hermes Agent Code v0.9.0 (2026.4.13)" in output
    assert "Backend: online :9119 | Web: http://localhost:3001/code | DB: v17" in output
    assert "Quick: /code /web /workspace /session /approvals /skills-code /help" in output
    assert "Tools: 28 — /tools | Skills: 94 — /skills" in output


def test_build_hermes_code_console_compact_layout():
    """Banner renders in compact mode (narrow terminal) without crash."""
    console = _make_console(width=80)
    banner.build_hermes_code_console(console=console, model="test-model")
    output = console.export_text()
    assert len(output) > 0
    assert "Hermes Code Mode" in output or "Status" in output


def test_build_hermes_code_console_no_model_param():
    """Banner renders without model parameter (uses default from config)."""
    console = _make_console()
    banner.build_hermes_code_console(console=console)
    output = console.export_text()
    assert len(output) > 0


# ---------------------------------------------------------------------------
# Code Mode data helper
# ---------------------------------------------------------------------------

def test_get_code_mode_data_returns_dict_with_required_keys():
    """_get_code_mode_data must return a dict with all required keys."""
    data = banner._get_code_mode_data()
    required_keys = {
        "provider", "model", "profile", "workspace", "branch",
        "backend_status", "db_schema", "active_sessions", "pending_approvals",
    }
    assert required_keys.issubset(data.keys())


def test_get_code_mode_data_has_safe_fallbacks():
    """Fallback values must be non-empty strings or zero for numeric fields."""
    data = banner._get_code_mode_data()
    assert isinstance(data["provider"], str) and data["provider"]
    assert isinstance(data["model"], str) and data["model"]
    assert isinstance(data["workspace"], str) and data["workspace"]
    assert isinstance(data["branch"], str) and data["branch"]
    assert isinstance(data["backend_status"], str) and data["backend_status"]
    assert isinstance(data["active_sessions"], int)
    assert isinstance(data["pending_approvals"], int)


def test_get_code_mode_data_uses_local_schema_version_when_backend_offline():
    """DB/schema should still show the real local schema version offline."""
    from hermes_state import SCHEMA_VERSION

    with patch("requests.get", side_effect=ConnectionRefusedError("offline")):
        data = banner._get_code_mode_data()

    assert data["db_schema"] == f"v{SCHEMA_VERSION}"

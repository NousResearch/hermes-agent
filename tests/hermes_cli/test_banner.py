"""Tests for banner toolset name normalization and skin color usage."""

import json

from unittest.mock import patch

from rich.console import Console

import hermes_cli.banner as banner
import model_tools
import tools.mcp_tool


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


def test_build_welcome_banner_title_is_hyperlinked_to_release():
    """Panel title (version label) is wrapped in an OSC-8 hyperlink to the GitHub release."""
    import io
    from unittest.mock import patch as _patch
    import hermes_cli.banner as _banner
    import model_tools as _mt
    import tools.mcp_tool as _mcp

    _banner._latest_release_cache = None
    tag_url = ("v2026.4.23", "https://github.com/NousResearch/hermes-agent/releases/tag/v2026.4.23")

    buf = io.StringIO()
    with (
        _patch.object(_mt, "check_tool_availability", return_value=(["web"], [])),
        _patch.object(_banner, "get_available_skills", return_value={}),
        _patch.object(_banner, "get_update_result", return_value=None),
        _patch.object(_mcp, "get_mcp_status", return_value=[]),
        _patch.object(_banner, "get_latest_release_tag", return_value=tag_url),
    ):
        console = Console(file=buf, force_terminal=True, color_system="truecolor", width=160)
        _banner.build_welcome_banner(
            console=console, model="x", cwd="/tmp",
            session_id="abc123",
            tools=[{"function": {"name": "read_file"}}],
            get_toolset_for_tool=lambda n: "file",
        )

    raw = buf.getvalue()
    # The existing version label must still be present in the title
    assert "Hermes Agent v" in raw, "Version label missing from title"
    # OSC-8 hyperlink escape sequence present with the release URL
    assert "\x1b]8;" in raw, "OSC-8 hyperlink not emitted"
    assert "releases/tag/v2026.4.23" in raw, "Release URL missing from banner output"


def test_build_welcome_banner_title_falls_back_when_no_tag():
    """Without a resolvable tag, the panel title renders as plain text (no hyperlink escape)."""
    import io
    from unittest.mock import patch as _patch
    import hermes_cli.banner as _banner
    import model_tools as _mt
    import tools.mcp_tool as _mcp

    _banner._latest_release_cache = None
    buf = io.StringIO()
    with (
        _patch.object(_mt, "check_tool_availability", return_value=(["web"], [])),
        _patch.object(_banner, "get_available_skills", return_value={}),
        _patch.object(_banner, "get_update_result", return_value=None),
        _patch.object(_mcp, "get_mcp_status", return_value=[]),
        _patch.object(_banner, "get_latest_release_tag", return_value=None),
    ):
        console = Console(file=buf, force_terminal=True, color_system="truecolor", width=160)
        _banner.build_welcome_banner(
            console=console, model="x", cwd="/tmp",
            session_id="abc123",
            tools=[{"function": {"name": "read_file"}}],
            get_toolset_for_tool=lambda n: "file",
        )

    raw = buf.getvalue()
    assert "Hermes Agent v" in raw, "Version label missing from title"
    assert "\x1b]8;" not in raw, "OSC-8 hyperlink should not be emitted without a tag"


def test_check_for_updates_uses_cached_result_when_local_head_matches(tmp_path, monkeypatch):
    import hermes_cli.banner as _banner

    monkeypatch.delenv("HERMES_REVISION", raising=False)
    monkeypatch.setattr(_banner, "get_hermes_home", lambda: tmp_path)
    (tmp_path / ".update_check").write_text(json.dumps({"ts": 1000, "behind": 764, "rev": "head1234"}))

    with (
        patch.object(_banner.time, "time", return_value=1001),
        patch.object(_banner, "_git_short_hash", return_value="head1234"),
        patch.object(_banner, "_check_via_local_git") as check_local,
    ):
        behind = _banner.check_for_updates()

    assert behind == 764
    check_local.assert_not_called()


def test_check_for_updates_invalidates_cache_when_local_head_changes(tmp_path, monkeypatch):
    import hermes_cli.banner as _banner

    monkeypatch.delenv("HERMES_REVISION", raising=False)
    monkeypatch.setattr(_banner, "get_hermes_home", lambda: tmp_path)
    cache_file = tmp_path / ".update_check"
    cache_file.write_text(json.dumps({"ts": 1000, "behind": 764, "rev": "oldhead"}))

    with (
        patch.object(_banner.time, "time", return_value=1001),
        patch.object(_banner, "_git_short_hash", return_value="newhead"),
        patch.object(_banner, "_check_via_local_git", return_value=0) as check_local,
    ):
        behind = _banner.check_for_updates()

    assert behind == 0
    check_local.assert_called_once()
    cached = json.loads(cache_file.read_text())
    assert cached["behind"] == 0
    assert cached["rev"] == "newhead"

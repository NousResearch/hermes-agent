"""Tests for banner toolset name normalization and skin color usage."""

from unittest.mock import patch

from rich.console import Console

import hermes_cli.banner as banner
import model_tools
import tools.mcp_tool_discovery


def test_cprint_falls_back_to_plain_print_when_prompt_toolkit_has_no_console(capsys):
    with patch(
        "prompt_toolkit.print_formatted_text",
        side_effect=RuntimeError("no console screen buffer"),
    ):
        banner.cprint("fallback text")

    assert capsys.readouterr().out == "fallback text\n"








def test_build_welcome_banner_title_falls_back_when_no_tag():
    """Without a resolvable tag, the panel title renders as plain text (no hyperlink escape)."""
    import io
    from unittest.mock import patch as _patch
    import hermes_cli.banner as _banner
    import model_tools as _mt
    import tools.mcp_tool as _mcp
    from tools import mcp_tool_discovery as _mcp_discovery

    _banner._latest_release_cache = None
    buf = io.StringIO()
    with (
        _patch.object(_mt, "check_tool_availability", return_value=(["web"], [])),
        _patch.object(_banner, "get_available_skills", return_value={}),
        _patch.object(_banner, "get_update_result", return_value=None),
        _patch.object(_mcp_discovery, "get_mcp_status", return_value=[]),
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






def test_build_welcome_banner_non_moa_unchanged(tmp_path, monkeypatch):
    """A normal provider still renders the bare model slug, no MoA prefix."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()

    with (
        patch.object(model_tools, "check_tool_availability", return_value=([], [])),
        patch.object(banner, "get_available_skills", return_value={}),
        patch.object(banner, "get_update_result", return_value=None),
        patch.object(tools.mcp_tool_discovery, "get_mcp_status", return_value=[]),
    ):
        console = Console(record=True, force_terminal=False, color_system=None, width=160)
        banner.build_welcome_banner(
            console=console,
            model="anthropic/claude-opus-4.8",
            cwd="/tmp/project",
            tools=[],
            enabled_toolsets=[],
            provider="openrouter",
        )

    out = console.export_text()
    assert "claude-opus-4.8" in out
    assert "MoA:" not in out


def test_empty_model_shows_the_free_tier_route_when_it_carries_inference(tmp_path, monkeypatch):
    """The banner prints before credentials resolve, so ``model`` is empty on a fresh install. On the
    free tier the route is known locally (identity on disk + tier on): the banner shows its model.
    When nothing resolves the red "no model configured" line stays."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    import hermes_cli.anon_auth as anon_auth

    def render(carries: bool) -> str:
        with (
            patch.object(model_tools, "check_tool_availability", return_value=([], [])),
            patch.object(banner, "get_available_skills", return_value={}),
            patch.object(banner, "get_update_result", return_value=None),
            patch.object(tools.mcp_tool_discovery, "get_mcp_status", return_value=[]),
            patch.object(anon_auth, "guest_carries_inference", return_value=carries),
        ):
            console = Console(record=True, force_terminal=False, color_system=None, width=160)
            banner.build_welcome_banner(console=console, model="", cwd="/tmp/project", tools=[],
                                        enabled_toolsets=[], provider="auto")
        return console.export_text()

    assert "welcome" in render(True) and "no model configured" not in render(True)
    assert "no model configured" in render(False)


def test_build_welcome_banner_does_not_center_pad_hero_art():
    """A braille hero relies on its own U+2800 padding for symmetry; Rich centering inserts
    ASCII spaces around the left column and distorts the silhouette (#9879). The hero line
    must start flush at the column start."""
    import io
    from types import SimpleNamespace
    from tools import mcp_tool_discovery as _mcp_discovery

    skin = SimpleNamespace(banner_hero="[green]\u2800X[/]", banner_logo="")
    buf = io.StringIO()
    with (
        patch.object(model_tools, "check_tool_availability", return_value=([], [])),
        patch.object(banner, "get_available_skills", return_value={}),
        patch.object(banner, "get_update_result", return_value=None),
        patch.object(banner, "get_latest_release_tag", return_value=None),
        patch.object(_mcp_discovery, "get_mcp_status", return_value=[]),
        patch.object(banner, "_active_skin", return_value=skin),
    ):
        console = Console(file=buf, force_terminal=False, color_system=None, width=80)
        banner.build_welcome_banner(console=console, model="m", cwd="/tmp", tools=[],
                                    get_toolset_for_tool=lambda _: None)

    hero_line = next(line for line in buf.getvalue().splitlines() if "\u2800X" in line)
    assert hero_line.startswith("\u2502  \u2800X"), repr(hero_line)


def test_build_welcome_banner_renders_lazy_mcp_server_as_not_failed(tmp_path, monkeypatch):
    """A lazily registered MCP server (tools from the schema cache, process not
    yet spawned) must render with its cached tool count — not fall into the
    red "failed" branch that catches every unknown status."""
    import yaml

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        yaml.safe_dump({"mcp_servers": {"playwright": {"command": "npx", "lazy": True}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    lazy_entry = {
        "name": "playwright", "transport": "stdio", "tools": 3,
        "connected": False, "disabled": False, "status": "lazy",
    }
    with (
        patch.object(model_tools, "check_tool_availability", return_value=([], [])),
        patch.object(banner, "get_available_skills", return_value={}),
        patch.object(banner, "get_update_result", return_value=None),
        patch.object(banner, "get_latest_release_tag", return_value=None),
        patch.object(tools.mcp_tool_discovery, "get_mcp_status", return_value=[lazy_entry]),
    ):
        console = Console(record=True, force_terminal=False, color_system=None, width=160)
        banner.build_welcome_banner(
            console=console, model="x", cwd="/tmp", tools=[], enabled_toolsets=[],
        )
    out = console.export_text()

    assert "playwright" in out
    assert "3 tool(s)" in out
    assert "lazy, starts on first use" in out
    assert "failed" not in out

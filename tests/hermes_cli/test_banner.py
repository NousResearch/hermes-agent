"""Tests for banner toolset name normalization and skin color usage."""

from unittest.mock import patch

from rich.console import Console

import hermes_cli.banner as banner
import model_tools
import tools.mcp_tool


def test_cprint_falls_back_to_plain_print_when_prompt_toolkit_has_no_console(capsys):
    with patch(
        "prompt_toolkit.print_formatted_text",
        side_effect=RuntimeError("no console screen buffer"),
    ):
        banner.cprint("fallback text")

    assert capsys.readouterr().out == "fallback text\n"


def test_render_notice_ansi_emits_escape_codes():
    """The deferred-notice renderer expands Rich markup to real ANSI escapes."""
    ansi = banner._render_notice_ansi("[bold yellow]⚠ 3 commits behind[/]")
    assert "\x1b[1;33m" in ansi, "bold-yellow notice should carry a SGR 1;33 sequence"
    assert ansi.endswith("\x1b[0m"), "notice should reset styling"
    assert "3 commits behind" in ansi


def test_defer_update_notice_routes_through_cprint_not_raw_console():
    """A deferred notice must print via cprint (prompt_toolkit-safe ANSI path),
    not console.print (which writes raw escapes into patch_stdout and shows up
    as literal '\\x1b[1;33m…' garbage once the TUI owns the terminal)."""
    import threading
    import time as _time
    from unittest.mock import patch as _patch

    from rich.console import Console

    captured = {}

    class _FakeEvent:
        def wait(self, timeout=None):
            return True

    def _fake_cprint(text):
        captured["text"] = text

    with (
        _patch.object(banner, "_update_check_done", _FakeEvent()),
        _patch.object(banner, "_update_result", 3),
        _patch.object(banner, "cprint", side_effect=_fake_cprint),
    ):
        banner._deferred_update_notice_started = False
        console = Console(force_terminal=True, color_system="truecolor")
        banner._defer_update_notice(console, max_wait=5)

        # Wait for the background thread to run.
        deadline = _time.time() + 3
        while "text" not in captured and _time.time() < deadline:
            _time.sleep(0.05)

    assert "text" in captured, "deferred notice should have printed"
    assert "3 commits behind" in captured["text"]
    assert "\x1b[1;33m" in captured["text"], (
        "cprint must receive the ANSI-rendered notice, not raw Rich markup"
    )








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






def test_build_welcome_banner_non_moa_unchanged(tmp_path, monkeypatch):
    """A normal provider still renders the bare model slug, no MoA prefix."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()

    with (
        patch.object(model_tools, "check_tool_availability", return_value=([], [])),
        patch.object(banner, "get_available_skills", return_value={}),
        patch.object(banner, "get_update_result", return_value=None),
        patch.object(tools.mcp_tool, "get_mcp_status", return_value=[]),
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

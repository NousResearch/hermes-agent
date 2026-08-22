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


def test_deferred_update_notice_uses_prompt_toolkit_cprint(monkeypatch):
    """Late update notices must not write raw Rich ANSI through patch_stdout."""
    calls = []
    monkeypatch.setattr(banner, "cprint", calls.append)

    notice = banner._format_update_notice_ansi(89)
    banner.cprint(notice)

    assert calls == [notice]
    assert "\x1b[1;33m⚠ 89 commits behind\x1b[0m" in calls[0]
    assert "?[" not in calls[0]
    assert "[bold yellow]" not in calls[0]


def test_deferred_update_notice_path_does_not_use_rich_console(monkeypatch):
    console = Console()

    def fail_print(*_args, **_kwargs):
        raise AssertionError("deferred update notice used Rich Console.print")

    monkeypatch.setattr(console, "print", fail_print)
    banner._deferred_update_notice_started = False
    monkeypatch.setattr(banner, "_update_result", 89)
    banner._update_check_done.set()

    calls = []
    monkeypatch.setattr(banner, "cprint", calls.append)

    banner._defer_update_notice(console, max_wait=0.01)
    import time
    deadline = time.time() + 1
    while not calls and time.time() < deadline:
        time.sleep(0.01)

    assert calls
    assert "?[" not in calls[0]
    banner._deferred_update_notice_started = False






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

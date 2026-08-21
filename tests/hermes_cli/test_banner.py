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


# ---------------------------------------------------------------------------
# Deferred update notice (#87444)
# ---------------------------------------------------------------------------

_NOTICE_MARKUP = (
    "[bold yellow]⚠ 35 commits behind[/]"
    "[dim yellow] — run [bold]hermes update[/bold] to update[/]"
)


def test_emit_update_notice_uses_plain_console_when_no_app_is_running():
    """Outside the interactive loop the notice goes straight to Rich as before."""
    from types import SimpleNamespace

    printed = []
    console = SimpleNamespace(print=lambda markup: printed.append(markup))

    with patch.object(banner, "_prompt_toolkit_app_running", return_value=False):
        banner._emit_update_notice(console, _NOTICE_MARKUP)

    assert printed == [_NOTICE_MARKUP]


def test_emit_update_notice_routes_through_cprint_while_app_is_running():
    """With a live prompt_toolkit app the notice is pre-rendered and cprint-ed.

    Regression for #87444: a bare ``console.print()`` here reaches
    prompt_toolkit's ``StdoutProxy``, whose ``Vt100_Output.write()`` rewrites
    every ESC to ``?``, so the warning displayed as literal ``?[1;33m…`` text.
    """
    from types import SimpleNamespace

    emitted = []
    console = SimpleNamespace(
        print=lambda markup: emitted.append(("console", markup))
    )

    with (
        patch.object(banner, "_prompt_toolkit_app_running", return_value=True),
        patch.object(banner, "cprint", lambda line: emitted.append(("cprint", line))),
    ):
        banner._emit_update_notice(console, _NOTICE_MARKUP)

    assert emitted, "notice was not emitted at all"
    assert all(channel == "cprint" for channel, _ in emitted), (
        f"expected the prompt_toolkit-safe path only, got {emitted}"
    )

    rendered = "\n".join(line for _, line in emitted)
    assert "\x1b[" in rendered, "no real ANSI escapes — colors would be lost"
    assert "[bold yellow]" not in rendered, "Rich markup leaked out unrendered"
    assert "35 commits behind" in rendered


def test_emit_update_notice_falls_back_to_console_when_cprint_fails():
    """A broken prompt_toolkit output must not swallow the notice entirely."""
    from types import SimpleNamespace

    emitted = []
    console = SimpleNamespace(
        print=lambda markup: emitted.append(("console", markup))
    )

    def _boom(_line):
        raise RuntimeError("no console screen buffer")

    with (
        patch.object(banner, "_prompt_toolkit_app_running", return_value=True),
        patch.object(banner, "cprint", _boom),
    ):
        banner._emit_update_notice(console, _NOTICE_MARKUP)

    assert emitted == [("console", _NOTICE_MARKUP)]


def test_defer_update_notice_emits_through_the_ansi_safe_path():
    """The deferred thread must not bypass ``_emit_update_notice``."""
    import threading
    import time
    from types import SimpleNamespace

    calls = []
    prev_started = banner._deferred_update_notice_started
    prev_result = banner._update_result
    prev_done = banner._update_check_done

    banner._deferred_update_notice_started = False
    banner._update_result = 35
    banner._update_check_done = threading.Event()
    banner._update_check_done.set()

    try:
        with (
            patch.object(
                banner, "_format_update_notice", lambda behind: f"MARKUP-{behind}"
            ),
            patch.object(
                banner, "_emit_update_notice", lambda _c, markup: calls.append(markup)
            ),
        ):
            banner._defer_update_notice(
                SimpleNamespace(print=lambda *_a, **_k: None), max_wait=5.0
            )
            deadline = time.monotonic() + 5.0
            while not calls and time.monotonic() < deadline:
                time.sleep(0.01)
    finally:
        banner._deferred_update_notice_started = prev_started
        banner._update_result = prev_result
        banner._update_check_done = prev_done

    assert calls == ["MARKUP-35"]

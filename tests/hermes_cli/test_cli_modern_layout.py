"""The glyph CLI layout (``display.cli_layout: modern``): scrollback rows never depend on the
terminal width, and the status line renders under the composer instead of above it."""
import re

import pytest


def _plain(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _run_turn(monkeypatch, width: int, *, modern: bool) -> list[str]:
    from cli import HermesCLI
    import cli as climod

    cli = HermesCLI.__new__(HermesCLI)
    cli._modern_layout = modern
    cli.show_reasoning, cli.reasoning_full = True, False
    cli.final_response_markdown, cli.show_timestamps = "raw", False
    cli._reset_stream_state()
    emitted: list[str] = []
    monkeypatch.setattr(climod, "_cprint", lambda s: emitted.append(_plain(s)))
    monkeypatch.setattr(climod, "_terminal_width_for_streaming", lambda: width)
    monkeypatch.setattr(HermesCLI, "_scrollback_box_width", lambda self: width)
    cli._stream_reasoning_delta("**Planning the change**\nI will read calc.py first.\n")
    cli._stream_delta("Added multiply to calc.py.\n")
    cli._flush_stream()
    return emitted


@pytest.mark.parametrize("modern", [True, False])
def test_glyph_layout_scrollback_is_width_independent(monkeypatch, modern):
    narrow, wide = _run_turn(monkeypatch, 60, modern=modern), _run_turn(monkeypatch, 120, modern=modern)
    boxed = any(ch in line for line in wide for ch in "╭╮╰╯┌┐└┘")
    if not modern:  # control: the classic layout's frames are sized to the terminal
        assert boxed and narrow != wide
        return
    assert narrow == wide and not boxed, wide
    rows = "\n".join(wide).split("\n")
    thought = [line for line in rows if line.startswith("∴")]
    assert len(thought) == 1 and "Planning the change" in thought[0], rows
    # Reasoning streams in full like classic, under a fixed-width dim bar (never re-wraps on resize)
    # and closed by the ∴ summary; the answer opens under its own bar.
    start = rows.index("**Planning the change**")
    assert rows[start - 1] == "─" * 40 and rows[start + 1] == "I will read calc.py first.", rows
    assert rows[start + 2] == thought[0], rows
    assert rows[next(i for i, r in enumerate(rows) if r.startswith("☤ Hermes")) - 1] == "─" * 40, rows
    assert any("Added multiply to calc.py." in line for line in rows)


@pytest.mark.parametrize("modern", [True, False])
def test_status_line_sits_under_the_composer_only_in_the_glyph_layout(modern):
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli._modern_layout = modern
    names = ("sudo_widget", "secret_widget", "approval_widget", "clarify_widget", "spacer", "status_bar",
             "input_rule_top", "image_bar", "input_area", "input_rule_bot", "voice_status_bar", "completions_menu")
    order = cli._build_tui_layout_children(**{name: name for name in names})
    if modern:
        assert order.index("status_bar") > order.index("input_rule_bot")
        assert order[-1] == "status_bar"
    else:
        assert order.index("status_bar") < order.index("input_rule_top")


def test_panel_rows_never_carry_a_newline():
    """A multi-line command (heredoc) in the approval panel wraps line by line, so no row breaks
    the panel's right border."""
    from cli import _wrap_panel_text_keep_ws

    rows = _wrap_panel_text_keep_ws("python3 - <<'PY'\nprint(sum(range(10)))\nPY", 30)
    assert rows == ["python3 - <<'PY'", "print(sum(range(10)))", "PY"]


@pytest.mark.parametrize("display, modern, busy", [
    ({}, True, "interrupt"),
    ({"cli_layout": "classic"}, False, "interrupt"),
    ({"busy_input_mode": "queue"}, True, "queue"),
    ({"cli_layout": "classic", "busy_input_mode": "steer"}, False, "steer"),
])
def test_layout_and_busy_default_through_the_config_loader(tmp_path, monkeypatch, display, modern, busy):
    import hermes_yaml as yaml
    import cli
    from hermes_cli.cli_modern_layout import busy_input_mode_from_config, layout_from_config

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(cli, "_hermes_home", home)
    (home / "config.yaml").write_text(yaml.safe_dump({"display": display} if display else {"model": {}}))
    loaded = cli.load_cli_config()["display"]
    assert layout_from_config(loaded) is modern
    assert busy_input_mode_from_config(loaded) == busy


def test_status_line_keeps_every_classic_status_bar_field(monkeypatch):
    """The glyph footer renders the classic bar's segments: nothing (tps, cache, latency, context
    detail, duration...) is dropped by a layout-private default subset."""
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli._modern_layout, cli._status_bar_visible = True, True
    snapshot = {"model_short": "m", "duration": "6h 43m", "context_percent": 7, "context_length": 1_000_000,
                "context_tokens": 73_200, "context_estimated": True, "avg_latency_label": "9.0s",
                "avg_velocity_label": "61 t/s", "cache_hit_rate": 28.9, "cache_read_tokens": 1, "prompt_tokens": 1,
                "active_background_subagents": 3, "prompt_elapsed": "⏱ 10s"}
    monkeypatch.setattr(HermesCLI, "_get_status_bar_snapshot", lambda self: snapshot)
    monkeypatch.setattr(HermesCLI, "_get_tui_terminal_width", lambda self: 400)
    monkeypatch.setattr(HermesCLI, "_get_status_bar_field_set", lambda self: None)
    monkeypatch.setattr(HermesCLI, "_is_session_yolo_active", lambda self: False)
    monkeypatch.setattr(HermesCLI, "_vim_mode_label", lambda self: "")
    monkeypatch.setattr(HermesCLI, "_cache_hit_rate", lambda self, snap, precision=1: (28.9, "◎ 28.9%"))
    monkeypatch.setattr("hermes_cli.cli_modern_layout.hint_pairs", lambda cli: [])
    monkeypatch.setattr("hermes_cli.cli_modern_layout._cwd_label", lambda: "~")
    line = "".join(text for _, text in cli._get_status_bar_fragments())
    for piece in ("73.2K/1M", "7%", "◎ 28.9%", "◷ 9.0s", "↑ 61 t/s", "⛓ 3", "6h 43m", "⏱ 10s"):
        assert piece in line, (piece, line)


@pytest.mark.parametrize("prompt", ["_connection_state", "_secret_state", "_sudo_state", "_clarify_state"])
def test_tab_never_queues_what_a_prompt_owns(prompt):
    """A connection form's masked field (or any prompt) owns the composer while a turn runs: Tab must
    not queue its text as a chat message, which would print and save a secret in plain text."""
    from types import SimpleNamespace
    from prompt_toolkit.application import Application
    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.layout import BufferControl, Layout, Window
    from hermes_cli.cli_modern_layout import modern_input_rules

    cli = SimpleNamespace(_agent_running=True, _tui_input_rule_height=lambda pos: 1)
    kb = KeyBindings()
    modern_input_rules(cli, kb)
    tab = next(b for b in kb.bindings if b.keys == (Keys.ControlI,))
    buf = Buffer(document=None)
    buf.text = "REVIEW_CANARY_NOT_A_CREDENTIAL"
    app = Application(layout=Layout(Window(BufferControl(buffer=buf))))
    with set_app(app):
        assert tab.filter()  # control: a plain busy composer does queue on Tab
        setattr(cli, prompt, {"phase": "form"})
        assert not tab.filter(), prompt


def test_long_user_message_soft_wraps_so_a_resize_reflows_it(monkeypatch):
    """The glyph layout's echo of a long prompt is one logical line per source line: the terminal
    wraps it, so widening the window re-flows it instead of keeping old-width breaks."""
    from cli import HermesCLI
    import cli as climod
    import shutil

    cli = HermesCLI.__new__(HermesCLI)
    cli._modern_layout, cli.show_timestamps = True, False
    long_line = "word " * 60
    for text in (long_line.strip(), f"{long_line.strip()}\nsecond line"):
        out = {}
        for width in (60, 140):
            emitted: list[str] = []
            monkeypatch.setattr(climod, "_cprint", lambda s: emitted.append(_plain(s)))
            monkeypatch.setattr(shutil, "get_terminal_size", lambda *_a, w=width: __import__("os").terminal_size((w, 40)))
            cli._print_user_message_preview(text)
            out[width] = [line for line in "\n".join(emitted).split("\n") if line.strip()]
        assert out[60] == out[140], out
        assert any(line.rstrip().endswith("word") and line.count("word") == 60 for line in out[60]), out[60]


def test_tool_rows_keep_the_classic_tool_emoji():
    """Each finished tool row leads with the same emoji the classic ``┊`` row shows (📖 read, 💻
    terminal…); a failed entry swaps it for the red ✗ so failures stay obvious."""
    from agent.display import get_cute_tool_message
    from hermes_cli.cli_modern_layout import tool_row

    # clarify has no dedicated renderer: both layouts take its registered emoji from one source.
    for name, args in (("read_file", {"path": "/etc/hostname"}), ("terminal", {"command": "echo ok"}),
                       ("execute_code", {"code": "import math"}), ("clarify", {"question": "Which?"})):
        classic_emoji = get_cute_tool_message(name, args, 0.2).split()[1]
        assert _plain(tool_row(name, args, 0.2)).startswith(classic_emoji + " "), (name, classic_emoji)
    failed = _plain(tool_row("terminal", {"command": "false"}, 0.2, result='{"exit_code": 1, "output": ""}'))
    assert failed.startswith("✗ ") and "[exit 1]" in failed, failed


def test_a_blank_line_separates_the_user_message_from_the_first_tool_row(monkeypatch):
    """Reasoning and answers open under a blank line + bar; the first tool row of a turn gets the
    same blank line, so the user's message never runs straight into the tool calls. Later rows
    stay tight."""
    from cli import HermesCLI
    import cli as climod
    from hermes_cli import cli_modern_layout as layout

    emitted: list[str] = []
    monkeypatch.setattr(climod, "_cprint", lambda s: emitted.append(_plain(s)))
    cli = HermesCLI.__new__(HermesCLI)
    cli._modern_layout, cli.show_timestamps = True, False
    cli._print_user_message_preview("read the hostname")
    first = layout.completed_tool_line(cli, "read_file", {"path": "/etc/hostname"}, 0.2)
    second = layout.completed_tool_line(cli, "terminal", {"command": "echo hi"}, 0.1)
    assert first.startswith("\n") and _plain(first).lstrip("\n").startswith("📖 "), repr(first)
    assert not second.startswith("\n"), repr(second)
    # A reasoning block opened first already printed its own blank line + bar: no double gap.
    cli._print_user_message_preview("again")
    layout.separator(False)
    assert not layout.completed_tool_line(cli, "read_file", {"path": "/etc/hostname"}, 0.2).startswith("\n")

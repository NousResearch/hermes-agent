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


@pytest.mark.parametrize("reasoning", [
    "scripted reasoning about gamma zeta-reason",
    # Longer than the 72-char title cap: a shortened title would still repeat it.
    "scripted reasoning about gamma zeta-reason that keeps going well past the title length cap here",
])
def test_one_line_reasoning_is_shown_once(monkeypatch, reasoning):
    """Reasoning streams above its ∴ summary, so a one-sentence block must not reappear as the title."""
    from cli import HermesCLI
    import cli as climod

    cli = HermesCLI.__new__(HermesCLI)
    cli._modern_layout, cli.show_reasoning, cli.reasoning_full = True, True, False
    cli.final_response_markdown, cli.show_timestamps = "raw", False
    cli._reset_stream_state()
    emitted: list[str] = []
    monkeypatch.setattr(climod, "_cprint", lambda s: emitted.append(_plain(s)))
    monkeypatch.setattr(climod, "_terminal_width_for_streaming", lambda: 100)
    monkeypatch.setattr(HermesCLI, "_scrollback_box_width", lambda self: 100)
    cli._stream_reasoning_delta(reasoning)
    cli._stream_delta("GAMMA answered.\n")
    cli._flush_stream()

    rows = "\n".join(emitted).split("\n")
    assert sum(row.count(reasoning) for row in rows) == 1, emitted
    thought = [row for row in rows if row.startswith("∴")]
    assert len(thought) == 1 and "·" not in thought[0], thought


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


def test_refused_skill_write_reads_neutral_not_failed():
    """A skill_manage write the agent will re-plan (size cap, stale old_string) keeps the tool's own
    emoji and a dim reason: no red ✗ for the agent's routine retry loop."""
    from agent.display import get_tool_emoji
    from hermes_cli.cli_modern_layout import _RED, tool_row

    refused = ('{"success": false, "error": "operations[0] (patch on \'hermes-agent-dev\') failed: '
               'SKILL.md content is 100,193 characters (limit: 100,000).", '
               '"file_preview": "Never ignore a ROLLBACK FAILED note."}')
    row = tool_row("skill_manage", {"operations": [{"action": "patch", "name": "hermes-agent-dev"}]}, 1.1, refused)
    assert _plain(row).startswith(get_tool_emoji("skill_manage") + " "), row
    assert "[operations[0]" in _plain(row) and _RED not in row, row
    # A failed rollback left partial changes on disk: that one stays a red failure.
    broken = refused.replace('(limit: 100,000)."', '(limit: 100,000). — batch aborted, ROLLBACK FAILED for demo."')
    assert _plain(tool_row("skill_manage", {"operations": []}, 1.1, broken)).startswith("✗ ")


def test_every_block_is_framed_by_bars_and_blank_lines(monkeypatch):
    """User message, reasoning and answer each read as: blank line, bar, content, bar, blank line.
    The blank line after a closing bar is paid by whatever prints next (a tool row, the footer)
    and never doubled when the next block opens with its own blank line + bar. Tool rows inside a
    run stay tight."""
    from cli import HermesCLI
    import cli as climod
    from hermes_cli import cli_modern_layout as layout

    out: list[str] = []
    monkeypatch.setattr(climod, "_cprint", lambda s: out.extend(_plain(s).split("\n")))
    cli = HermesCLI.__new__(HermesCLI)
    cli._modern_layout, cli.show_timestamps = True, False
    bar = "─" * 40

    cli._print_user_message_preview("read the hostname")
    for row in layout.thought_rows(2.0, "Plan the read.", full=True):
        climod._cprint(row)
    climod._cprint(layout.completed_tool_line(cli, "read_file", {"path": "/etc/hostname"}, 0.2))
    climod._cprint(layout.completed_tool_line(cli, "terminal", {"command": "echo hi"}, 0.1))
    climod._cprint(layout.answer_label())
    climod._cprint("It is LINUX.")
    climod._cprint(layout.closing_separator(True))
    climod._cprint(layout.turn_footer("1.0s"))

    rows = [r.rstrip() for r in out]
    start = rows.index("❯ read the hostname") - 1
    got = ["BAR" if r == bar else ("" if not r else r.split()[0]) for r in rows[start - 1:]]
    assert got == ["", "BAR", "❯", "BAR",          # user message
                   "", "BAR", "Plan", "∴", "BAR",  # reasoning
                   "", "📖", "💻",                  # tool rows, tight
                   "", "BAR", "☤", "It", "BAR",    # answer
                   "", "─"], got                   # footer


# ── display.answer_wrap: "word" ─────────────────────────────────────────────────────────────

_PARAGRAPH = ("The factory conveyor moves each job from the research bay to the design bay and finally "
              "to the listing bay, where a verifier checks every artifact before the gate opens.")


def test_wrap_prose_breaks_on_words_within_width():
    from hermes_cli.cli_modern_layout import wrap_prose

    rows = wrap_prose(_PARAGRAPH, 40)
    assert len(rows) > 1
    assert all(len(row) <= 40 for row in rows), rows
    assert " ".join(rows) == _PARAGRAPH
    # No row may start with the fragment of a word the previous row began.
    assert all(row and not row.endswith(" ") for row in rows)


def test_wrap_prose_measures_display_cells_and_never_splits_a_token():
    from hermes_cli.cli_modern_layout import wrap_prose

    cjk = "注入" * 12  # 24 chars, 48 cells
    rows = wrap_prose(cjk, 20)
    assert all(len(wrap_prose(row, 20)[0]) <= len(row) for row in rows)
    assert "".join(rows) == cjk

    url = "see https://example.com/" + "a" * 80
    rows = wrap_prose(url, 40)
    assert any(("a" * 80) in row for row in rows), rows  # a URL wider than the row is left whole


def test_wrap_prose_leaves_code_and_tables_alone():
    from hermes_cli.cli_modern_layout import wrap_prose

    long_code = "    result = compute(" + ", ".join(f"argument_{i}" for i in range(12)) + ")"
    assert wrap_prose(long_code, 40) == [long_code]
    table = "| " + " | ".join(f"column_{i}" for i in range(12)) + " |"
    assert wrap_prose(table, 40) == [table]


def test_wrap_stream_row_holds_a_fenced_block(monkeypatch):
    from hermes_cli.cli_modern_layout import wrap_stream_row

    rows, fence = wrap_stream_row("```python", 30)
    assert rows == ["```python"] and fence == ("`", 3)
    long_code = "print(" + ", ".join(f"value_{i}" for i in range(10)) + ")"
    rows, fence = wrap_stream_row(long_code, 30, fence=fence)
    assert rows == [long_code] and fence == ("`", 3)
    rows, fence = wrap_stream_row("```", 30, fence=fence)
    assert rows == ["```"] and fence is None
    rows, _ = wrap_stream_row(_PARAGRAPH, 40, fence=None)
    assert len(rows) > 1


def test_a_longer_fence_is_not_closed_by_a_shorter_inner_run():
    from hermes_cli.cli_modern_layout import wrap_stream_row

    rows, fence = wrap_stream_row("````markdown", 30)
    assert fence == ("`", 4)
    # An inner ``` inside a four-backtick block is content, not a close.
    rows, fence = wrap_stream_row("```", 30, fence=fence)
    assert rows == ["```"] and fence == ("`", 4)
    # A different fence character does not close it either.
    rows, fence = wrap_stream_row("~~~", 30, fence=fence)
    assert fence == ("`", 4)
    rows, fence = wrap_stream_row("````", 30, fence=fence)
    assert fence is None


def test_strip_mode_still_recognises_code(monkeypatch):
    """final_response_markdown: strip removes the backticks; the source row must still say 'code'."""
    from hermes_cli.cli_modern_layout import wrap_stream_row

    long_code = "print(" + ", ".join(f"value_{i}" for i in range(10)) + ")"
    rows, fence = wrap_stream_row("python", 30, source_row="```python")
    assert rows == ["python"] and fence == ("`", 3)
    rows, fence = wrap_stream_row(long_code, 30, source_row=long_code, fence=fence)
    assert rows == [long_code], rows
    rows, fence = wrap_stream_row("", 30, source_row="```", fence=fence)
    assert fence is None


def test_ansi_escapes_do_not_skew_the_width():
    from hermes_cli.cli_modern_layout import _cells, wrap_prose

    esc, reset = chr(27) + "[38;2;255;191;0m", chr(27) + "[0m"
    colored = esc + "word " * 11 + "word" + reset
    # A row carrying inline escapes is left whole for the terminal: splitting it would strand the
    # escape and lose the style on the continuations.
    assert wrap_prose(colored, 30) == [colored]
    # Measurement itself ignores the escapes.
    assert _cells(colored) == len("word " * 11 + "word")


def test_a_nested_bullet_keeps_its_indent_and_wraps():
    from hermes_cli.cli_modern_layout import wrap_prose

    bullet = "  - " + " ".join(f"word{i}" for i in range(14))
    rows = wrap_prose(bullet, 30)
    assert len(rows) > 1
    assert rows[0].startswith("  - word0")
    assert all(row.startswith("    ") for row in rows[1:]), rows
    assert all(len(row) <= 30 for row in rows), rows


def test_a_leading_pipeless_table_row_is_not_wrapped():
    from hermes_cli.cli_modern_layout import wrap_prose

    row = "alpha | beta | " + " | ".join(f"column_{i}" for i in range(8))
    assert wrap_prose(row, 30) == [row]
    # A two-column row with a single pipe is still a table row (the stream's detector misses it).
    two_col = "a very long first column | a very long second column"
    assert wrap_prose(two_col, 30) == [two_col]


def test_a_hanging_indent_row_never_overflows():
    from hermes_cli.cli_modern_layout import _cells, wrap_prose

    rows = wrap_prose("  - " + "abcdef " * 12, 30)
    assert all(_cells(row) <= 30 for row in rows), rows


def test_an_osc_link_measures_only_its_text():
    from hermes_cli.cli_modern_layout import _cells

    link = "\x1b]8;;https://example.com/very/long/target\x1b\\hello world\x1b]8;;\x1b\\"
    assert _cells(link) == len("hello world")


def test_a_fence_line_with_an_info_string_does_not_close_the_block():
    from hermes_cli.cli_modern_layout import wrap_stream_row

    _, fence = wrap_stream_row("~~~text", 30)
    assert fence == ("~", 3)
    # ~~~literal is an opening fence in markdown, not a close: the block stays open.
    _, fence = wrap_stream_row("~~~literal", 30, fence=fence)
    assert fence == ("~", 3)
    _, fence = wrap_stream_row("~~~", 30, fence=fence)
    assert fence is None


def test_wrap_answer_rows_honours_source_rows_when_stripping(monkeypatch):
    from hermes_cli.cli_modern_layout import wrap_answer_rows

    code = "print(" + ", ".join(f"value_{i}" for i in range(10)) + ")"
    text = "\n".join(["```python", code, "```", _PARAGRAPH])
    rows = wrap_answer_rows(text, 40, strip=False)
    assert rows[0] == "```python"
    assert code in rows, rows                       # fenced code stays whole
    assert rows.count("```") == 1                   # the closing fence survives
    assert any(row.startswith("The factory conveyor") for row in rows), rows
    assert len(wrap_answer_rows(_PARAGRAPH, 40)) > 1


def _run_wrapped_turn(monkeypatch, width: int, *, wrap: bool) -> list[str]:
    from cli import HermesCLI
    import cli as climod

    cli = HermesCLI.__new__(HermesCLI)
    cli._modern_layout, cli._answer_wrap_word = True, wrap
    cli.show_reasoning, cli.reasoning_full = False, False
    cli.final_response_markdown, cli.show_timestamps = "raw", False
    cli._reset_stream_state()
    emitted: list[str] = []
    monkeypatch.setattr(climod, "_cprint", lambda s: emitted.append(_plain(s)))
    monkeypatch.setattr(climod, "_terminal_width_for_streaming", lambda: width)
    cli._stream_delta(_PARAGRAPH + "\n")
    cli._flush_stream()
    pad = climod._STREAM_PAD
    # _STREAM_PAD is "" in this process, so every row matches it: drop the block framing (the
    # accent bar and the "☤ Hermes" label) and keep only the answer prose.
    rows = [line[len(pad):] for row in emitted for line in row.split("\n") if line.startswith(pad)]
    return [row for row in rows
            if row.strip() and set(row.strip()) != {"─"} and not row.strip().startswith("☤")]


def test_answer_wrap_word_wraps_the_streamed_answer(monkeypatch):
    from cli import _terminal_width_for_streaming
    import cli as climod

    wrapped = _run_wrapped_turn(monkeypatch, 40, wrap=True)
    assert len(wrapped) > 1
    assert all(len(row) <= 40 for row in wrapped), wrapped
    assert " ".join(wrapped) == _PARAGRAPH

    monkeypatch.undo()
    unwrapped = _run_wrapped_turn(monkeypatch, 40, wrap=False)
    assert unwrapped == [_PARAGRAPH], unwrapped


def test_answer_wrap_defaults_to_the_terminal():
    from hermes_cli.cli_modern_layout import answer_wrap_from_config

    assert answer_wrap_from_config({}) is False
    assert answer_wrap_from_config({"answer_wrap": "terminal"}) is False
    assert answer_wrap_from_config({"answer_wrap": "word"}) is True
    assert answer_wrap_from_config({"answer_wrap": "WORD "}) is True


def test_a_row_that_fits_keeps_its_whitespace():
    from hermes_cli.cli_modern_layout import wrap_prose

    row = "Use `a  b` — two spaces there are content."
    assert wrap_prose(row, 80) == [row]


def test_a_quoted_row_keeps_its_markers_on_continuations():
    from hermes_cli.cli_modern_layout import wrap_prose

    rows = wrap_prose("> " + " ".join(f"word{i}" for i in range(16)), 40)
    assert len(rows) > 1
    assert all(row.startswith("> ") for row in rows), rows
    assert all(len(row) <= 40 for row in rows), rows


def test_wrap_answer_rows_strips_the_whole_answer(monkeypatch):
    import hermes_cli.cli_modern_layout as layout

    em = "**first line\nsecond line** tail"
    # The real stripper removes both markers of a span that crosses the newline; per-row stripping
    # would leave them. Substitute it so the test does not import the CLI facade.
    monkeypatch.setattr(layout, "_strip_markdown", lambda text: text.replace("**", ""))
    rows = layout.wrap_answer_rows(em, 80, strip=True)
    assert not any("*" in row for row in rows), rows
    assert any("first line" in row for row in rows), rows


def test_cjk_tokens_are_measured_in_cells_not_characters():
    from hermes_cli.cli_modern_layout import _cells, wrap_prose

    row = "注入 " * 10  # each token is 2 chars / 4 cells
    rows = wrap_prose(row, 20)
    assert len(rows) > 1
    assert all(_cells(c) <= 20 for c in rows), rows


def test_a_quoted_fence_opens_and_closes():
    from hermes_cli.cli_modern_layout import wrap_stream_row

    rows, fence = wrap_stream_row("> ```python", 30)
    assert rows == ["> ```python"] and fence == ("`", 3)
    code = "> print(" + ", ".join(f"value_{i}" for i in range(8)) + ")"
    rows, fence = wrap_stream_row(code, 30, fence=fence)
    assert rows == [code], rows                      # quoted code is not wrapped
    rows, fence = wrap_stream_row("> ```", 30, fence=fence)
    assert rows == ["> ```"] and fence is None
    # Ordinary prose wraps again once the quoted block is closed.
    assert len(wrap_stream_row(_PARAGRAPH, 40, fence=fence)[0]) > 1

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
    thought = [line for line in wide if line.startswith("∴")]
    assert len(thought) == 1 and "Planning the change" in thought[0], wide
    assert "I will read calc.py first." not in "\n".join(wide)  # collapsed unless /reasoning full
    assert any("Added multiply to calc.py." in line for line in wide)


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
    ({}, True, "steer"),
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

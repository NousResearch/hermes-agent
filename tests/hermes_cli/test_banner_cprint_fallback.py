"""cprint()'s fallback and the update-notice renderer must never put raw ANSI on screen (#87444)."""

from hermes_cli import banner


def test_cprint_fallback_strips_ansi(monkeypatch, capsys):
    import prompt_toolkit

    def _no_console(*_a, **_k):
        raise RuntimeError("no console")

    monkeypatch.setattr(prompt_toolkit, "print_formatted_text", _no_console)
    banner.cprint("\x1b[1;33m⚠ 12 commits behind\x1b[0m — run \x1b[1mhermes update\x1b[0m")
    out = capsys.readouterr().out
    assert "\x1b" not in out
    assert "⚠ 12 commits behind — run hermes update" in out


def test_ansi_regex_strips_csi_and_osc_only():
    text = "a\x1b[38;2;1;2;3mb\x1b]8;;https://x\x07c\x1b]8;;\x07d\x1b[0m"
    assert banner._ANSI_ESCAPE_RE.sub("", text) == "abcd"
    assert banner._ANSI_ESCAPE_RE.sub("", "plain [1;33m text") == "plain [1;33m text"


def test_rendered_notice_is_not_folded_at_80_columns():
    long_cmd = "hermes update --channel stable --with-a-deliberately-long-option-name"
    rendered = banner._render_markup_to_ansi(
        f"[bold yellow]⚠ 1234 commits behind[/][dim yellow] — run [bold]{long_cmd}[/bold] to update[/]")
    assert "\n" not in rendered
    assert long_cmd in banner._ANSI_ESCAPE_RE.sub("", rendered)

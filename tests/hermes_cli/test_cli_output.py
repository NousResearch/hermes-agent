from hermes_cli import cli_output, colors
from hermes_cli.colors import Colors


def _force_color(monkeypatch, enabled=True):
    monkeypatch.setattr(colors, "should_use_color", lambda: enabled)


def test_print_helpers_use_semantic_palette(monkeypatch, capsys):
    """Each helper renders its semantic role's color + prefix."""
    _force_color(monkeypatch, True)

    cli_output.print_success("saved")
    cli_output.print_warning("careful")
    cli_output.print_error("boom")
    cli_output.print_info("note")
    out = capsys.readouterr().out

    assert Colors.GREEN in out and "✓ saved" in out
    assert Colors.YELLOW in out and "⚠ careful" in out
    assert Colors.RED in out and "✗ boom" in out
    assert Colors.DIM in out and "note" in out


def test_header_uses_design_system_heading_style(monkeypatch, capsys):
    """Headers render cyan+bold, not the old one-off yellow."""
    _force_color(monkeypatch, True)

    cli_output.print_header("Section")
    out = capsys.readouterr().out

    assert Colors.CYAN in out
    assert Colors.BOLD in out
    assert "Section" in out
    assert Colors.YELLOW not in out


def test_truncated_and_prompt_keep_their_roles(monkeypatch, capsys):
    """Muted footer and the input prompt stay dim and yellow."""
    _force_color(monkeypatch, True)
    monkeypatch.setattr(cli_output, "t", lambda key, **_kw: key)

    cli_output.print_truncated(3, hint="use --limit 40")
    out = capsys.readouterr().out
    assert out.startswith(Colors.DIM)
    assert "cli.shared.n_more" in out

    seen = {}

    def fake_input(prompt_text):
        seen["prompt"] = prompt_text
        return "x"

    monkeypatch.setattr(cli_output.sys, "stdin", _NotTTY())
    monkeypatch.setattr(cli_output.sys, "stdout", _TTY())
    monkeypatch.setattr("builtins.input", fake_input)
    assert cli_output.prompt("Name") == "x"
    assert seen["prompt"].startswith(Colors.YELLOW)
    assert "Name:" in seen["prompt"]


class _TTY:
    def isatty(self):
        return True


class _NotTTY:
    def isatty(self):
        return False


def test_line_input_supports_cursor_and_emacs_navigation(monkeypatch):
    from prompt_toolkit.application.current import create_app_session
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    monkeypatch.setattr(cli_output.sys, "stdin", _TTY())
    monkeypatch.setattr(cli_output.sys, "stdout", _TTY())

    with create_pipe_input() as pipe_input:
        # Type "ac", move left, insert "b", move right, append "d", then
        # exercise Ctrl+A/Ctrl+E before accepting the line.
        pipe_input.send_text("ac\x1b[Db\x1b[Cd\x01X\x05Y\r")
        with create_app_session(input=pipe_input, output=DummyOutput()):
            assert cli_output.line_input("Enter model name: ") == "XabcdY"


def test_line_input_preserves_builtin_input_for_redirected_stdin(monkeypatch):
    seen = {}

    def fake_input(prompt_text):
        seen["prompt"] = prompt_text
        return "piped-model"

    monkeypatch.setattr(cli_output.sys, "stdin", _NotTTY())
    monkeypatch.setattr(cli_output.sys, "stdout", _TTY())
    monkeypatch.setattr("builtins.input", fake_input)

    assert cli_output.line_input("Enter model name: ") == "piped-model"
    assert seen["prompt"] == "Enter model name: "


def test_line_input_falls_back_to_input_on_any_prompt_toolkit_failure(monkeypatch):
    """Any prompt_toolkit runtime failure — not just OSError — degrades to
    input() so the wizard keeps running.  ValueError and RuntimeError can
    arise from exotic stream wrappers or event-loop issues that share the
    same root cause: prompt_toolkit cannot attach stdin on this terminal.
    """
    seen = {}

    def raising_prompt(*_args, **_kwargs):
        raise ValueError("Invalid file descriptor")

    def fake_input(prompt_text):
        seen["prompt"] = prompt_text
        return "fallback-value"

    monkeypatch.setattr(cli_output.sys, "stdin", _TTY())
    monkeypatch.setattr(cli_output.sys, "stdout", _TTY())
    monkeypatch.setattr("prompt_toolkit.prompt", raising_prompt)
    monkeypatch.setattr("builtins.input", fake_input)

    assert cli_output.line_input("Enter token: ") == "fallback-value"
    assert seen["prompt"] == "Enter token: "


def test_password_prompt_uses_masked_secret_prompt(monkeypatch):
    seen = {}

    def fake_masked_secret_prompt(display):
        seen["display"] = display
        return " secret "

    monkeypatch.setattr(cli_output, "masked_secret_prompt", fake_masked_secret_prompt)

    assert cli_output.prompt("API key", default="old", password=True) == "secret"
    assert "API key [old]" in seen["display"]


def test_empty_password_prompt_returns_default(monkeypatch):
    monkeypatch.setattr(cli_output, "masked_secret_prompt", lambda _display: "")

    assert cli_output.prompt("API key", default="old", password=True) == "old"

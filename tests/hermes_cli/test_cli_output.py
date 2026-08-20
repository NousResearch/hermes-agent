from hermes_cli import cli_output


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

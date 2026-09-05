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


def test_line_input_falls_back_to_input_when_prompt_toolkit_raises_oserror(monkeypatch):
    """isatty() can report True while the asyncio selector still rejects stdin.

    Under a `curl ... | bash` install the setup wizard reattaches stdin from
    /dev/tty, so isatty() is True and the guard above passes. prompt_toolkit
    then fails to register fd 0 with the event-loop selector (observed on
    macOS, where kqueue raises OSError EINVAL / "Invalid argument"). line_input
    must degrade to the built-in reader instead of letting the OSError abort
    the whole wizard.
    """
    seen = {}

    def raising_prompt(*_args, **_kwargs):
        raise OSError(22, "Invalid argument")

    def fake_input(prompt_text):
        seen["prompt"] = prompt_text
        return "fallback-value"

    monkeypatch.setattr(cli_output.sys, "stdin", _TTY())
    monkeypatch.setattr(cli_output.sys, "stdout", _TTY())
    monkeypatch.setattr("prompt_toolkit.prompt", raising_prompt)
    monkeypatch.setattr("builtins.input", fake_input)

    assert cli_output.line_input("Choice [1/2]: ") == "fallback-value"
    assert seen["prompt"] == "Choice [1/2]: "


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

import pytest

from hermes_cli import secret_prompt
from hermes_cli.secret_prompt import _collect_masked_input, masked_secret_prompt


class _TTY:
    def isatty(self):
        return True


class _Pipe:
    def isatty(self):
        return False


def _run_collect(chars: str):
    output: list[str] = []
    iterator = iter(chars)

    def read_char() -> str:
        return next(iterator, "")

    def write(text: str) -> None:
        output.append(text)

    value = _collect_masked_input(
        read_char,
        write,
        "API key: ",
    )
    return value, "".join(output)


def test_collect_masked_input_shows_feedback_without_echoing_secret():
    value, output = _run_collect("secret\n")

    assert value == "secret"
    assert output == "API key: ******\r\n"
    assert "secret" not in output


def test_collect_masked_input_handles_backspace():
    value, output = _run_collect("sec\x7fret\r")

    assert value == "seret"
    assert output == "API key: ***\b \b***\r\n"
    assert "secret" not in output


def test_collect_masked_input_raises_keyboard_interrupt():
    output: list[str] = []

    with pytest.raises(KeyboardInterrupt):
        _collect_masked_input(
            lambda: "\x03",
            output.append,
            "API key: ",
        )

    assert "".join(output) == "API key: \r\n"


def test_masked_secret_prompt_falls_back_to_getpass_for_non_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin", _Pipe())
    monkeypatch.setattr("sys.stdout", _Pipe())
    monkeypatch.setattr("getpass.getpass", lambda prompt: f"value from {prompt}")

    assert masked_secret_prompt("API key: ") == "value from API key: "


def test_masked_secret_prompt_sanitizes_getpass_fallback(monkeypatch):
    monkeypatch.setattr("sys.stdin", _Pipe())
    monkeypatch.setattr("sys.stdout", _Pipe())
    monkeypatch.setattr(
        "getpass.getpass",
        lambda _prompt: "\x1b[200~secret-token\x1b[201~",
    )

    assert masked_secret_prompt("API key: ") == "secret-token"


def test_masked_secret_prompt_uses_prompt_toolkit_on_windows_tty(monkeypatch):
    captured = {}

    def fake_prompt(message, is_password=False, key_bindings=None):
        captured["message"] = message
        captured["is_password"] = is_password
        captured["key_bindings"] = key_bindings
        return "\x1b[200~secret-token\x1b[201~"

    monkeypatch.setattr(secret_prompt.sys, "platform", "win32", raising=False)
    monkeypatch.setattr("sys.stdin", _TTY())
    monkeypatch.setattr("sys.stdout", _TTY())
    monkeypatch.setattr("prompt_toolkit.shortcuts.prompt", fake_prompt)

    value = masked_secret_prompt("API key: ")

    assert value == "secret-token"
    assert captured["is_password"] is True
    assert captured["key_bindings"] is not None

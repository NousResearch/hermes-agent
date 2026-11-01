import os

import pytest

from hermes_cli.secret_prompt import _collect_masked_input, _masked_secret_prompt_posix, masked_secret_prompt


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




def test_collect_masked_input_raises_keyboard_interrupt():
    output: list[str] = []

    with pytest.raises(KeyboardInterrupt):
        _collect_masked_input(
            lambda: "\x03",
            output.append,
            "API key: ",
        )

    assert "".join(output) == "API key: \r\n"


class _ScriptedTerminal:
    """stdin whose termios calls hit a real pty and whose reads replay scripted keystrokes."""

    def __init__(self, keys: str, fd: int):
        self._keys = iter(keys)
        self._fd = fd

    def fileno(self) -> int:
        return self._fd

    def read(self, _n: int = -1) -> str:
        return next(self._keys, "")


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("keys, secret", [
    ("sk-abcd\x1b[D\x1b[C\r", "sk-abcd"),  # Left, Right
    ("sk-ab\x1b[3~cd\r", "sk-abcd"),  # Delete
    ("\x1b[200~sk-pasted\x1b[201~\r", "sk-pasted"),  # bracketed paste
    ("sk-abcd\x1bOD\x1b[1;5C\r", "sk-abcd"),  # application-mode Left, Ctrl+Right
    ("sk-abcd\x1b[[A\x1b[[E\r", "sk-abcd"),  # Linux console F1, F5
    ("sk-\x1bxy\r", "sk-xy"),  # a lone ESC / Alt+x: the next key is still text
    ("sk-ab\x1bO\r", "sk-ab"),  # Alt+O (an unfinished SS3 prefix), then Enter still submits
])
def test_terminal_key_sequences_never_become_secret_text(monkeypatch, capsys, keys, secret):
    import pty

    master, slave = pty.openpty()
    try:
        monkeypatch.setattr("sys.stdin", _ScriptedTerminal(keys, slave))
        assert _masked_secret_prompt_posix("API key: ", mask="*") == secret
    finally:
        os.close(master)
        os.close(slave)
    # One mask char per secret char: the feedback the user saw matches what gets saved.
    assert capsys.readouterr().out == "API key: " + "*" * len(secret) + "\r\n"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("keys", ["sk-\x1bO\x03", "sk-\x1b[1;\x03", "sk-\x1b[[\x03"])
def test_ctrl_c_inside_an_unfinished_key_sequence_still_cancels(monkeypatch, keys):
    import pty

    master, slave = pty.openpty()
    try:
        monkeypatch.setattr("sys.stdin", _ScriptedTerminal(keys, slave))
        with pytest.raises(KeyboardInterrupt):
            _masked_secret_prompt_posix("API key: ", mask="*")
    finally:
        os.close(master)
        os.close(slave)


def test_masked_secret_prompt_falls_back_to_getpass_for_non_tty(monkeypatch):
    class NonTty:
        def isatty(self):
            return False

    monkeypatch.setattr("sys.stdin", NonTty())
    monkeypatch.setattr("sys.stdout", NonTty())
    monkeypatch.setattr("getpass.getpass", lambda prompt: f"value from {prompt}")

    assert masked_secret_prompt("API key: ") == "value from API key: "

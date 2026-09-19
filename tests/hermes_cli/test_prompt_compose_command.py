"""Tests for the CLI `/prompt` editor-compose command.

`/prompt` opens `$VISUAL`/`$EDITOR` on a temp markdown file so the user can
hand-edit a multi-line prompt, then queues the saved buffer as the next
agent turn via the one-shot `_pending_agent_seed` (same path `/blueprint`
uses). These drive a fake editor subprocess to verify read-back, header
stripping, seeding, and the empty-buffer cancel path.
"""

import os
import stat
import tempfile

import pytest

from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.commands import resolve_command


class _Stub(CLICommandsMixin):
    def __init__(self):
        self._pending_agent_seed = None


def _fake_editor(body: str, mode: str = "append") -> str:
    """Write a tiny shell 'editor' that mutates the file it is handed."""
    f = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False)
    if mode == "append":
        f.write("#!/usr/bin/env bash\n")
        f.write(f"cat >> \"$1\" <<'EOF'\n{body}\nEOF\n")
    else:  # clear
        f.write("#!/usr/bin/env bash\n: > \"$1\"\n")
    f.close()
    os.chmod(f.name, os.stat(f.name).st_mode | stat.S_IEXEC)
    return f.name


@pytest.fixture(autouse=True)
def _no_visual(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)


def test_command_registered():
    cd = resolve_command("prompt")
    assert cd and cd.name == "prompt"
    assert resolve_command("compose").name == "prompt"


def test_compose_reads_and_strips_header(monkeypatch):
    monkeypatch.setenv("EDITOR", _fake_editor("Refactor the auth module.\nUse pytest."))
    out = _Stub()._compose_in_editor("")
    assert "Refactor the auth module." in out
    assert "Use pytest." in out
    assert "#!" not in out  # the instructional header is stripped


def test_empty_buffer_does_not_seed(monkeypatch):
    monkeypatch.setenv("EDITOR", _fake_editor("", mode="clear"))
    s = _Stub()
    s._handle_prompt_compose_command("/prompt")
    assert s._pending_agent_seed is None


def test_compose_uses_shell_on_windows(monkeypatch):
    """On Windows, subprocess.call must use shell=True so .CMD/.BAT shims
    (e.g. VS Code's `code` is `code.CMD`) resolve. CreateProcess only
    appends .exe, not .cmd, so without shell=True the editor never opens.
    """
    monkeypatch.setenv("EDITOR", "echo")
    calls = []

    def spy_call(*args, **kwargs):
        calls.append(kwargs.get("shell", False))
        return 0

    monkeypatch.setattr("subprocess.call", spy_call)
    monkeypatch.setattr("os.name", "nt")
    _Stub()._compose_in_editor("")
    assert calls, "subprocess.call was never invoked"
    assert calls[0] is True, f"shell=True expected on Windows, got {calls[0]}"


def test_compose_no_shell_on_posix(monkeypatch):
    """On POSIX, subprocess.call must NOT use shell=True to avoid shell
    injection via $EDITOR. The editor command is invoked directly.
    """
    monkeypatch.setenv("EDITOR", _fake_editor("test"))
    calls = []

    def spy_call(*args, **kwargs):
        calls.append(kwargs.get("shell", False))
        return 0

    monkeypatch.setattr("subprocess.call", spy_call)
    monkeypatch.setattr("os.name", "posix")
    _Stub()._compose_in_editor("")
    assert calls, "subprocess.call was never invoked"
    assert calls[0] is False, f"shell=False expected on POSIX, got {calls[0]}"

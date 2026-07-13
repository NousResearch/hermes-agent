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


def test_prompt_sets_pending_seed(monkeypatch):
    monkeypatch.setenv("EDITOR", _fake_editor("Write a haiku about caching."))
    s = _Stub()
    s._handle_prompt_compose_command("/prompt")
    assert s._pending_agent_seed
    assert "haiku about caching" in s._pending_agent_seed


def test_initial_text_is_seeded(monkeypatch):
    # The fake editor appends, so the initial text leads the buffer.
    monkeypatch.setenv("EDITOR", _fake_editor("rest of prompt"))
    out = _Stub()._compose_in_editor("DRAFT: ")
    assert out.startswith("DRAFT:")


def test_empty_buffer_does_not_seed(monkeypatch):
    monkeypatch.setenv("EDITOR", _fake_editor("", mode="clear"))
    s = _Stub()
    s._handle_prompt_compose_command("/prompt")
    assert s._pending_agent_seed is None


def test_compose_does_not_retry_failed_editor_through_shell(monkeypatch):
    """An editor launch failure must not reinterpret the value as shell code."""
    monkeypatch.setenv("EDITOR", "editor --safe-mode")
    calls = []

    def fail_once(command, *args, **kwargs):
        calls.append((command, kwargs))
        raise OSError("editor failed")

    monkeypatch.setattr("subprocess.call", fail_once)

    with pytest.raises(OSError, match="editor failed"):
        _Stub()._compose_in_editor("")

    assert len(calls) == 1
    assert isinstance(calls[0][0], list)
    assert calls[0][1].get("shell") is not True

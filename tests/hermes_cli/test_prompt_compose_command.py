"""Tests for the CLI `/prompt` editor-compose command.

`/prompt` opens `$VISUAL`/`$EDITOR` on a temp markdown file so the user can
hand-edit a multi-line prompt, then queues the saved buffer as the next
agent turn via the one-shot `_pending_agent_seed` (same path `/blueprint`
uses). These drive a fake editor subprocess to verify read-back, header
stripping, seeding, and the empty-buffer cancel path.
"""

import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.cli_prompt_editor import _read_editor_file_when_settled


class _Stub(CLICommandsMixin):
    def __init__(self):
        self._pending_agent_seed = None


def _fake_editor(tmp_path, body: str, mode: str = "append", exit_code: int = 0) -> str:
    """Write a tiny python 'editor' that mutates the file it is handed."""
    script = tmp_path / "editor.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\n"
        "path = Path(sys.argv[1])\n"
        + (f"path.write_text(path.read_text(encoding='utf-8') + {body!r}, encoding='utf-8')\n"
           if mode == "append" else "path.write_text('', encoding='utf-8')\n")
        + f"sys.exit({exit_code})\n", encoding="utf-8")
    return f'"{Path(sys.executable).as_posix()}" "{script.as_posix()}"'


@pytest.fixture(autouse=True)
def _no_visual(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)


@pytest.mark.platforms("linux")
def test_compose_reads_and_strips_header(tmp_path, monkeypatch):
    monkeypatch.setenv("EDITOR", _fake_editor(tmp_path, "Refactor the auth module.\nUse pytest."))
    out = _Stub()._compose_in_editor("")
    assert "Refactor the auth module." in out
    assert "Use pytest." in out
    assert "#!" not in out  # the instructional header is stripped


@pytest.mark.platforms("linux")
def test_empty_buffer_does_not_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("EDITOR", _fake_editor(tmp_path, "", mode="clear"))
    s = _Stub()
    s._handle_prompt_compose_command("/prompt")
    assert s._pending_agent_seed is None


def test_compose_waits_for_save_visible_after_editor_exit(monkeypatch, tmp_path):
    prompt_path = tmp_path / "prompt.md"
    writer = None

    def fake_mkstemp(*_args, **_kwargs):
        fd = os.open(prompt_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        return fd, str(prompt_path)

    def fake_editor_call(*_args, **_kwargs):
        nonlocal writer

        def delayed_save():
            time.sleep(0.1)
            prompt_path.write_text("edited prompt", encoding="utf-8-sig")

        writer = threading.Thread(target=delayed_save)
        writer.start()
        return 0

    monkeypatch.setattr(tempfile, "mkstemp", fake_mkstemp)
    monkeypatch.setattr(subprocess, "call", fake_editor_call)
    monkeypatch.setenv("EDITOR", "fake-editor")
    try:
        out = _Stub()._compose_in_editor("initial draft")
    finally:
        if writer is not None:
            writer.join()

    assert out == "edited prompt"


def test_unchanged_editor_file_returns_without_full_timeout(tmp_path, monkeypatch):
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("initial draft", encoding="utf-8")

    elapsed = 0.0

    def advance(seconds):
        nonlocal elapsed
        elapsed += seconds

    monkeypatch.setattr(time, "monotonic", lambda: elapsed)
    monkeypatch.setattr(time, "sleep", advance)
    out = _read_editor_file_when_settled(str(prompt_path), "initial draft")

    assert out == "initial draft"
    assert 0.3 <= elapsed < 2.0


def test_editor_failure_never_falls_back_to_shell(monkeypatch):
    """An editor the argv path cannot run must not retry through the shell.

    The old fallback re-invoked ``$EDITOR`` via ``shell=True`` with the editor
    string unquoted, so a crafted EDITOR value executed as shell code. Now a
    failed editor simply cancels the compose (#81364).
    """
    import subprocess as _sp

    calls = []

    def _record(args, **kwargs):
        calls.append((args, kwargs))
        raise OSError("editor not runnable")

    monkeypatch.setattr(_sp, "call", _record)
    monkeypatch.setenv("EDITOR", "nonexistent-editor; touch /tmp/pwned")
    assert _Stub()._compose_in_editor("") == ""
    assert calls, "argv invocation should have been attempted"
    for argv, kwargs in calls:
        assert kwargs.get("shell") is not True
    monkeypatch.setenv("EDITOR", '"unterminated')
    assert _Stub()._compose_in_editor("seed") == ""


def test_nonzero_editor_exit_cancels_even_with_buffer_content(tmp_path, monkeypatch):
    """A failed editor may leave seeded or abandoned text in the buffer — cancel."""
    monkeypatch.setenv("EDITOR", _fake_editor(tmp_path, "typed but abandoned\n", exit_code=3))
    assert _Stub()._compose_in_editor("seed text from the command line") == ""
    stub = _Stub()
    stub._handle_prompt_compose_command("/prompt draft this")
    assert stub._pending_agent_seed is None

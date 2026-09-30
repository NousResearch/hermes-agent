"""Tests for the CLI `/prompt` editor-compose command.

`/prompt` opens `$VISUAL`/`$EDITOR` on a temp markdown file so the user can
hand-edit a multi-line prompt, then queues the saved buffer as the next
agent turn via the one-shot `_pending_agent_seed` (same path `/blueprint`
uses). These drive a fake editor subprocess to verify read-back, header
stripping, seeding, and the empty-buffer cancel path.
"""

import os
import stat
import subprocess
import tempfile
import threading
import time

import pytest

from hermes_cli.cli_commands_mixin import CLICommandsMixin, _read_editor_file_when_settled


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
        f.write('#!/usr/bin/env bash\n: > "$1"\n')
    f.close()
    os.chmod(f.name, os.stat(f.name).st_mode | stat.S_IEXEC)
    return f.name


@pytest.fixture(autouse=True)
def _no_visual(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)




@pytest.mark.platforms("linux")
def test_compose_reads_and_strips_header(monkeypatch):
    monkeypatch.setenv("EDITOR", _fake_editor("Refactor the auth module.\nUse pytest."))
    out = _Stub()._compose_in_editor("")
    assert "Refactor the auth module." in out
    assert "Use pytest." in out
    assert "#!" not in out  # the instructional header is stripped


@pytest.mark.platforms("linux")
def test_empty_buffer_does_not_seed(monkeypatch):
    monkeypatch.setenv("EDITOR", _fake_editor("", mode="clear"))
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

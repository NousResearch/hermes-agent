"""Tests for the CLI `/prompt` editor-compose command.

`/prompt` opens `$VISUAL`/`$EDITOR` on a temp markdown file so the user can
hand-edit a multi-line prompt, then queues the saved buffer as the next agent
turn. These drive fake app/editor boundaries to verify the managed terminal
handoff, read-back, header stripping, seeding, and the empty-buffer cancel
path.
"""

import contextvars
import os
import queue
import stat
import tempfile

import pytest

from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.commands import resolve_command


class _Stub(CLICommandsMixin):
    def __init__(self):
        self._pending_agent_seed = None


class _FakeLoop:
    def __init__(self):
        self.callbacks = []

    def is_running(self):
        return True

    def call_soon_threadsafe(self, callback):
        self.callbacks.append(callback)


class _FakeApp:
    def __init__(self):
        self.context = contextvars.copy_context()
        self.is_running = True
        self.loop = _FakeLoop()


class _FakeTask:
    def __init__(self):
        self.callback = None
        self.editor_call = None
        self.editor_result = None

    def add_done_callback(self, callback):
        self.callback = callback

    def result(self):
        return self.editor_result

    def finish(self):
        assert self.callback is not None
        self.callback(self)


class _InteractiveStub(_Stub):
    def __init__(self):
        super().__init__()
        self._app = _FakeApp()
        self._pending_input = queue.Queue()
        self.compose_calls = []

    def _compose_in_editor(self, initial_text=""):
        self.compose_calls.append(initial_text)
        return "Composed in managed editor"


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


def test_interactive_prompt_suspends_app_while_editor_runs(monkeypatch):
    task = _FakeTask()

    def fake_run_in_terminal(editor_call, *, in_executor):
        task.editor_call = editor_call
        assert in_executor is True
        return task

    monkeypatch.setattr(
        "prompt_toolkit.application.run_in_terminal", fake_run_in_terminal
    )
    s = _InteractiveStub()

    s._handle_prompt_compose_command("/prompt Initial draft")

    assert s.compose_calls == []
    assert s._pending_agent_seed is None
    assert len(s._app.loop.callbacks) == 1

    s._app.loop.callbacks[0]()
    assert task.editor_call is not None
    task.editor_result = task.editor_call()
    task.finish()

    assert s.compose_calls == ["Initial draft"]
    assert s._pending_input.get_nowait() == "Composed in managed editor"


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

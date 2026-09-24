"""Tests for the CLI `/prompt` editor-compose command.

`/prompt` opens `$VISUAL`/`$EDITOR` on a temp markdown file so the user can
hand-edit a multi-line prompt, then queues the saved buffer as the next
agent turn via the one-shot `_pending_agent_seed` (same path `/blueprint`
uses). These drive a fake editor subprocess to verify read-back, header
stripping, seeding, and the empty-buffer cancel path.
"""

import os
import shlex
import sys
import tempfile

import pytest

from hermes_cli.cli_commands_mixin import CLICommandsMixin


class _Stub(CLICommandsMixin):
    def __init__(self):
        self._pending_agent_seed = None


def _fake_editor(body: str, mode: str = "append") -> str:
    """Return an ``$EDITOR`` command that mutates the file it is handed.

    A Python script driven by this interpreter, not a ``#!/usr/bin/env bash``
    script: production launches the editor as
    ``subprocess.call([*shlex.split(editor), path])``, and a shebang script is
    only self-executing where the kernel honours shebangs. Where it isn't, the
    launch fails and ``_compose_in_editor`` maps that onto "empty buffer"
    (its documented behaviour) — which silently turns the append test red and
    the *cancel* test vacuously green. Returning a shlex-quoted
    ``<python> <script>`` command keeps the editor genuinely running, and
    ``shlex.split`` round-trips it unchanged on both platforms.
    """
    fd, script = tempfile.mkstemp(suffix=".py", prefix="hermes_fake_editor_")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("import sys\n")
        fh.write("target = sys.argv[1]\n")
        if mode == "append":
            fh.write(f"body = {body!r}\n")
            fh.write("with open(target, 'a', encoding='utf-8') as f:\n")
            fh.write("    f.write(body + '\\n')\n")
        else:  # clear
            fh.write("open(target, 'w', encoding='utf-8').close()\n")
    return shlex.join([sys.executable, script])


@pytest.fixture(autouse=True)
def _no_visual(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)




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

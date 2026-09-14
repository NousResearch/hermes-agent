"""Tests for the CLI `/prompt` editor-compose command.

`/prompt` opens `$VISUAL`/`$EDITOR` on a temp markdown file so the user can
hand-edit a multi-line prompt, then queues the saved buffer as the next
agent turn via the one-shot `_pending_agent_seed` (same path `/blueprint`
uses). These drive a fake editor subprocess to verify read-back, header
stripping, seeding, and the empty-buffer cancel path.
"""

import sys
from pathlib import Path

import pytest

from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.commands import resolve_command


class _Stub(CLICommandsMixin):
    def __init__(self):
        self._pending_agent_seed = None


def _fake_editor(tmp_path, body: str, mode: str = "append", exit_code: int = 0) -> str:
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


def test_command_registered():
    cd = resolve_command("prompt")
    assert cd and cd.name == "prompt"
    assert resolve_command("compose").name == "prompt"


def test_compose_reads_and_strips_header(tmp_path, monkeypatch):
    monkeypatch.setenv("EDITOR", _fake_editor(tmp_path, "Refactor the auth module.\nUse pytest."))
    out = _Stub()._compose_in_editor("")
    assert "Refactor the auth module." in out
    assert "Use pytest." in out
    assert "#!" not in out  # the instructional header is stripped


def test_empty_buffer_does_not_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("EDITOR", _fake_editor(tmp_path, "", mode="clear"))
    s = _Stub()
    s._handle_prompt_compose_command("/prompt")
    assert s._pending_agent_seed is None


def test_editor_failure_never_falls_back_to_shell(monkeypatch):
    """An editor the argv path cannot run must not retry through the shell.

    The old fallback re-invoked `$EDITOR` via `shell=True` with the editor
    string unquoted, so a malicious or malformed EDITOR value executed as
    shell code. Now a failed editor simply cancels the compose.
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
        assert not Path(argv[-1]).exists()
    monkeypatch.setenv("EDITOR", '"unterminated')
    assert _Stub()._compose_in_editor("seed") == ""


def test_nonzero_editor_exit_cancels_even_with_buffer_content(tmp_path, monkeypatch):
    monkeypatch.setenv("EDITOR", _fake_editor(tmp_path, "typed but abandoned\n", exit_code=3))
    assert _Stub()._compose_in_editor("seed text from the command line") == ""
    stub = _Stub()
    stub._handle_prompt_compose_command("/prompt draft this")
    assert stub._pending_agent_seed is None

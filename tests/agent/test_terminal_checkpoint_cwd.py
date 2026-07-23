"""Pre-destructive-command checkpoints must snapshot the task/session cwd.

The terminal tool resolves its execution cwd per task/session (explicit
``workdir`` arg -> per-task override -> live session cwd -> ``TERMINAL_CWD`` /
process cwd). Sessions created with an explicit ``cwd``
(``session.create(cwd=...)``) or re-anchored via ``session.cwd.set`` therefore
run commands in a directory that can differ from the Hermes process cwd.

Before this fix the checkpoint-before-destructive-terminal-command path
resolved ``TERMINAL_CWD``/``os.getcwd()`` directly, snapshotting the WRONG
tree for such sessions — the later rollback would silently restore nothing.
``_ensure_file_checkpoint`` was already fixed to honor the task cwd (#68195);
this pins the terminal twin to the same pipeline.
"""

import os

from agent.tool_executor import _terminal_checkpoint_cwd
from tools import terminal_tool


def test_explicit_workdir_wins(tmp_path):
    assert _terminal_checkpoint_cwd({"workdir": str(tmp_path)}, "t-any") == str(tmp_path)


def test_session_cwd_override_beats_process_cwd(tmp_path, monkeypatch):
    """A per-task cwd override (gateway workspace tracking / session.cwd.set)
    must win over the process cwd and a stale TERMINAL_CWD."""
    task_id = "chk-terminal-cwd-task"
    session_dir = tmp_path / "session-ws"
    session_dir.mkdir()
    process_dir = tmp_path / "process-cwd"
    process_dir.mkdir()
    monkeypatch.chdir(process_dir)
    monkeypatch.setenv("TERMINAL_CWD", str(process_dir))

    terminal_tool.register_task_env_overrides(task_id, {"cwd": str(session_dir)})
    try:
        got = _terminal_checkpoint_cwd({}, task_id)
    finally:
        terminal_tool.clear_task_env_overrides(task_id)

    assert os.path.realpath(got) == os.path.realpath(str(session_dir))


def test_falls_back_to_terminal_cwd_env(tmp_path, monkeypatch):
    """Without an override the historical behavior is preserved (TERMINAL_CWD)."""
    anchored = tmp_path / "anchored"
    anchored.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("TERMINAL_CWD", str(anchored))

    got = _terminal_checkpoint_cwd({}, "chk-no-override-task")

    assert os.path.realpath(got) == os.path.realpath(str(anchored))

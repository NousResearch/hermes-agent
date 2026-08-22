"""Regression tests for session-cwd aligned subdirectory context discovery."""

from pathlib import Path

from agent.subdirectory_hints import SubdirectoryHintTracker
from gateway.session_context import clear_session_vars, set_session_vars


def test_session_cwd_prevents_project_root_agents_hint_when_launch_cwd_differs(
    tmp_path, monkeypatch
):
    """Desktop/TUI session cwd should be the subdirectory hint root.

    Regression: startup context can load HERMES.md from the session cwd while
    SubdirectoryHintTracker is rooted at the gateway/process cwd. Then the
    project root looks like a newly visited subdirectory and its AGENTS.md is
    appended even though startup context already handled that directory.
    """
    project = tmp_path / "pharos"
    project.mkdir()
    (project / "HERMES.md").write_text("Hermes project context")
    (project / "AGENTS.md").write_text("Codex project context")

    launch_cwd = tmp_path / "gateway-launch"
    launch_cwd.mkdir()
    monkeypatch.chdir(launch_cwd)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    from agent.agent_init import _subdirectory_hint_working_dir

    tokens = set_session_vars(session_key="desktop-session", cwd=str(project))
    try:
        working_dir = _subdirectory_hint_working_dir()
        assert Path(working_dir).resolve() == project

        tracker = SubdirectoryHintTracker(working_dir=working_dir)
        hint = tracker.check_tool_call(
            "terminal", {"command": "pwd", "workdir": str(project)}
        )
    finally:
        clear_session_vars(tokens)

    assert hint is None


def test_subdirectory_hint_working_dir_falls_back_to_process_cwd_for_plain_cli(
    tmp_path, monkeypatch
):
    """Plain CLI sessions without a configured cwd keep existing fallback behavior."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    clear_session_vars([])

    from agent.agent_init import _subdirectory_hint_working_dir

    assert _subdirectory_hint_working_dir() is None
    tracker = SubdirectoryHintTracker(working_dir=_subdirectory_hint_working_dir())
    assert tracker.working_dir == tmp_path.resolve()

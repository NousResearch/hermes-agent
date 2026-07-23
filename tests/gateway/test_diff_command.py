"""End-to-end tests for the gateway ``/diff`` command.

Exercises the real handler against a real checkpoint store and git, proving
the messaging surface returns the cumulative working-tree diff (and degrades
to friendly messages when checkpoints are off or nothing has changed).
"""

import shutil

import pytest

import gateway.run as gateway_run
import tools.checkpoint_manager as cpm
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git required for checkpoint diffs"
)


def _runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = None
    runner.config = None
    return runner


def _event(text: str) -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user-1",
        chat_id="chat-1",
        user_name="tester",
        chat_type="dm",
    )
    return MessageEvent(text=text, source=source)


def _enable_checkpoints(tmp_path, monkeypatch, enabled=True):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"checkpoints:\n  enabled: {str(enabled).lower()}\n", encoding="utf-8"
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", home, raising=False)
    monkeypatch.setattr(cpm, "CHECKPOINT_BASE", tmp_path / "checkpoints")


@pytest.mark.asyncio
async def test_diff_reports_cumulative_changes(tmp_path, monkeypatch):
    _enable_checkpoints(tmp_path, monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("print('hello')\n", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(project))

    # Baseline checkpoint (pre-edit) then an edit, so a diff exists.
    mgr = cpm.CheckpointManager(enabled=True, max_snapshots=50)
    assert mgr.ensure_checkpoint(str(project), "baseline") is True
    (project / "main.py").write_text("print('changed')\n", encoding="utf-8")

    result = await _runner()._handle_diff_command(_event("/diff"))

    assert "-print('hello')" in result
    assert "+print('changed')" in result


@pytest.mark.asyncio
async def test_diff_stat_only_omits_body(tmp_path, monkeypatch):
    _enable_checkpoints(tmp_path, monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("a = 1\n", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(project))

    mgr = cpm.CheckpointManager(enabled=True, max_snapshots=50)
    mgr.ensure_checkpoint(str(project), "baseline")
    (project / "main.py").write_text("a = 2\n", encoding="utf-8")

    result = await _runner()._handle_diff_command(_event("/diff --stat"))

    assert "main.py" in result
    assert "+a = 2" not in result  # body suppressed


@pytest.mark.asyncio
async def test_diff_no_changes_message(tmp_path, monkeypatch):
    _enable_checkpoints(tmp_path, monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(project))

    result = await _runner()._handle_diff_command(_event("/diff"))

    assert "No changes" in result


@pytest.mark.asyncio
async def test_diff_disabled_message(tmp_path, monkeypatch):
    _enable_checkpoints(tmp_path, monkeypatch, enabled=False)
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    result = await _runner()._handle_diff_command(_event("/diff"))

    assert "not enabled" in result.lower()

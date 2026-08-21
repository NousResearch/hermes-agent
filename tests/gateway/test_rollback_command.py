"""End-to-end tests for the gateway ``/rollback`` command.

Bare ``/rollback`` (no args) previously reported "no checkpoints found" for
the current directory even when checkpoints existed under a DIFFERENT
directory (a stale ``TERMINAL_CWD``, or a checkpoint written from a
different session cwd). ``hermes_cli.cli_commands_mixin``'s CLI ``/rollback``
already fell back to a cross-project "all directories" view in this case
(#10505, reapply of PR #10633 by @nightq) — the gateway's own independent
``_handle_rollback_command`` implementation (used by Discord/Slack/Telegram/
etc, and by TUI via ``GatewaySlashCommandsMixin``) never got the same fix.
"""

import pytest

import gateway.run as gateway_run
import tools.checkpoint_manager as cpm
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


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
async def test_bare_rollback_falls_back_to_all_directories(tmp_path, monkeypatch):
    _enable_checkpoints(tmp_path, monkeypatch)

    other_project = tmp_path / "other-project"
    other_project.mkdir()
    (other_project / "main.py").write_text("print('hi')\n", encoding="utf-8")
    mgr = cpm.CheckpointManager(enabled=True, max_snapshots=50)
    assert mgr.ensure_checkpoint(str(other_project), "baseline") is True

    # The session's own cwd has zero checkpoints.
    empty_project = tmp_path / "empty-project"
    empty_project.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(empty_project))

    result = await _runner()._handle_rollback_command(_event("/rollback"))

    assert f"No checkpoints for {empty_project}" in result
    assert "showing all directories" in result
    assert "baseline" in result


@pytest.mark.asyncio
async def test_bare_rollback_no_checkpoints_anywhere(tmp_path, monkeypatch):
    """No fallback claim when there is truly nothing to show anywhere."""
    _enable_checkpoints(tmp_path, monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(project))

    result = await _runner()._handle_rollback_command(_event("/rollback"))

    assert "showing all directories" not in result
    assert "no checkpoints" in result.lower() or "0" in result


@pytest.mark.asyncio
async def test_bare_rollback_uses_own_directory_when_present(tmp_path, monkeypatch):
    """The fallback must not fire when the session's own cwd already has
    checkpoints — same-directory checkpoints always take priority."""
    _enable_checkpoints(tmp_path, monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("print('own')\n", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(project))
    mgr = cpm.CheckpointManager(enabled=True, max_snapshots=50)
    assert mgr.ensure_checkpoint(str(project), "own baseline") is True

    other_project = tmp_path / "other-project"
    other_project.mkdir()
    (other_project / "main.py").write_text("print('other')\n", encoding="utf-8")
    assert mgr.ensure_checkpoint(str(other_project), "other baseline") is True

    result = await _runner()._handle_rollback_command(_event("/rollback"))

    assert "showing all directories" not in result
    assert "own baseline" in result

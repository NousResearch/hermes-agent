"""Smoke test for the /rollback CLI message fix.

``CLICommandsMixin._handle_rollback_command`` used to print "A pre-rollback
snapshot was saved automatically." unconditionally on a successful restore,
even when ``CheckpointManager.restore()`` could not actually take that
snapshot (surfaced via the ``result["warning"]`` key added alongside this
fix). This test exercises the real branching logic with a minimal stand-in
host object, without needing a full HermesCLI instance.
"""

from unittest.mock import patch

import pytest

from hermes_cli.cli_commands_mixin import CLICommandsMixin
from tools.checkpoint_manager import CheckpointManager


class _FakeAgent:
    def __init__(self, mgr):
        self._checkpoint_mgr = mgr


class _FakeHost(CLICommandsMixin):
    """Minimal object exposing just what _handle_rollback_command touches."""

    def __init__(self, mgr):
        self.agent = _FakeAgent(mgr)
        self.conversation_history = []  # skip the undo_last branch

    def _resolve_checkpoint_ref(self, ref, checkpoints):
        idx = int(ref) - 1
        return checkpoints[idx]["hash"] if 0 <= idx < len(checkpoints) else None


@pytest.fixture()
def work_dir(tmp_path):
    d = tmp_path / "project"
    d.mkdir()
    (d / "main.py").write_text("v1\n")
    return d


@pytest.fixture()
def mgr(work_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", tmp_path / "checkpoints")
    monkeypatch.setenv("TERMINAL_CWD", str(work_dir))
    m = CheckpointManager(enabled=True, max_snapshots=50)
    m.ensure_checkpoint(str(work_dir), "v1")
    m.new_turn()
    (work_dir / "main.py").write_text("v2\n")
    return m


def test_normal_restore_prints_saved_message(mgr, capsys):
    host = _FakeHost(mgr)
    host._handle_rollback_command("/rollback 1")
    out = capsys.readouterr().out
    assert "A pre-rollback snapshot was saved automatically." in out
    assert "⚠️" not in out


def test_failed_snapshot_prints_warning_not_false_claim(mgr, capsys):
    host = _FakeHost(mgr)
    with patch.object(CheckpointManager, "_take", return_value=False):
        host._handle_rollback_command("/rollback 1")
    out = capsys.readouterr().out
    assert "A pre-rollback snapshot was saved automatically." not in out
    assert "⚠️" in out
    assert "pre-rollback" in out.lower()

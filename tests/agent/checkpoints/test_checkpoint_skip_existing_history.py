"""A later file-cap skip must not hide an earlier rollback point (#154876)."""

import argparse

from agent.checkpoints import manager as cm
from agent.checkpoints.maintenance import store_status
from hermes_cli.checkpoints import cmd_status
from hermes_constants import get_hermes_home


def test_existing_history_survives_file_cap_skip_in_status(tmp_path, monkeypatch, capsys):
    workdir = tmp_path / "project"
    workdir.mkdir()
    source = workdir / "main.py"
    source.write_text("print('first')\n")
    checkpoint_base = get_hermes_home() / "checkpoints"
    manager = cm.CheckpointManager(enabled=True)

    assert manager.ensure_checkpoint(str(workdir), "before edit") is True
    assert store_status(checkpoint_base)["projects"][0]["commits"] == 1

    source.write_text("print('second')\n")
    manager.new_turn()
    # Preserve the shipped 50,000-file cap; only model the counted workload.
    monkeypatch.setattr(cm, "_dir_file_count", lambda _: cm._MAX_FILES + 1)
    assert manager.ensure_checkpoint(str(workdir), "after edit") is False

    rows = store_status(checkpoint_base)["projects"]
    assert rows[0]["commits"] == 1
    assert rows[0]["skip_reason"] == f"file_count>{cm._MAX_FILES}"
    assert cmd_status(argparse.Namespace(limit=None)) == 0

    output = capsys.readouterr().out
    project_line = next(line for line in output.splitlines() if "file_count>" in line)
    assert "new snapshots skipped: file_count>" in project_line
    assert "no snapshots" not in project_line

"""Regression coverage for checkpoint CLI failure reporting."""

import argparse


def test_clear_legacy_returns_nonzero_when_deletion_fails(monkeypatch, capsys):
    import hermes_cli.checkpoints as checkpoints_cli
    import tools.checkpoint_manager as checkpoint_manager

    monkeypatch.setattr(
        checkpoint_manager,
        "store_status",
        lambda: {"legacy_archives": [{"name": "legacy-20200101-000000", "size_bytes": 1000}]},
    )
    monkeypatch.setattr(
        checkpoint_manager,
        "clear_legacy",
        lambda: {"deleted": 0, "bytes_freed": 0, "errors": 1},
    )

    result = checkpoints_cli.cmd_clear_legacy(argparse.Namespace(force=True))

    assert result == 2
    assert "Could not delete 1 archive(s) (see logs)." in capsys.readouterr().out

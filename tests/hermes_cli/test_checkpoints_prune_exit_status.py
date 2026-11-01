"""An explicit prune command must tell automation when maintenance did not run."""

import argparse

import pytest


@pytest.mark.platforms("posix")
def test_busy_store_fails_without_deleting_then_succeeds_after_unlock(capsys):
    from hermes_constants import get_hermes_home
    from hermes_cli.subcommands.checkpoints import build_checkpoints_parser
    from tools.checkpoint_pruning import store_lock

    base = get_hermes_home() / "checkpoints"
    base.mkdir()
    retained = base / "keep.txt"
    retained.write_text("checkpoint data must survive a busy maintenance pass")
    parser = argparse.ArgumentParser()
    build_checkpoints_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args([
        "checkpoints", "prune", "--keep-orphans", "--force", "--max-size-mb", "0",
    ])

    with store_lock(base):
        rc = args.func(args)
        assert "Errors:          1" in capsys.readouterr().out
        assert retained.read_text() == "checkpoint data must survive a busy maintenance pass"
        assert rc == 2

    assert args.func(args) == 0
    assert "Errors:          0" in capsys.readouterr().out
    assert retained.read_text() == "checkpoint data must survive a busy maintenance pass"

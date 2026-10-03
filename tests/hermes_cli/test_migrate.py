"""Tests for hermes_cli/migrate.py — cmd_migrate dispatcher."""

from unittest.mock import MagicMock


def test_dispatcher_unknown_subtype():
    from hermes_cli.migrate import cmd_migrate
    args = MagicMock(migrate_type=None)
    assert cmd_migrate(args) == 2  # usage exit code


def test_dispatcher_xai_subtype():
    from hermes_cli.migrate import cmd_migrate
    args = MagicMock(migrate_type="xai", apply=False, no_backup=False)
    with __import__("unittest").mock.patch("hermes_cli.migrate.cmd_migrate_xai", return_value=0):
        assert cmd_migrate(args) == 0

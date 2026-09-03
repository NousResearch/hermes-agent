"""Tests for hermes_cli/update_cmd.py — update command helpers."""


def test_update_module_imports():
    from hermes_cli.update_cmd import read_latest_receipt
    assert callable(read_latest_receipt)

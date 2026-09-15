"""Tests for hermes_cli/_early_recovery.py — early-boot recovery helpers."""


def test_import_works():
    from hermes_cli._early_recovery import run_early_recovery
    assert callable(run_early_recovery)

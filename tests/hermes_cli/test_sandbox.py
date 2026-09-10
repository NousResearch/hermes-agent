"""Tests for hermes_cli/sandbox.py — sandbox config helper."""


def test_sandbox_config_default():
    from hermes_cli.sandbox import get_sandbox_config
    cfg = get_sandbox_config()
    assert isinstance(cfg, dict)

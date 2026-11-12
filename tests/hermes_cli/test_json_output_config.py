"""Tests for display.json_output config default applied to --json flag (#147088).

When display.json_output is true in config.yaml, every subcommand that declares
--json should default to JSON output without the user repeating the flag.
"""

# hermes_bootstrap runs module-level code (update-pull settlement) that probes
# the git root — under ~/.hermes paths the home_io_guard refuses it. We import
# a no-op mock before any test touches hermes_cli.main so the module-level
# side effects are skipped; _apply_json_output_default needs no bootstrap state.
import sys as _sys
from unittest.mock import MagicMock as _Mock
if "hermes_bootstrap" not in _sys.modules:
    _sys.modules["hermes_bootstrap"] = _Mock()

from types import SimpleNamespace

import pytest


@pytest.fixture
def _args():
    """An args namespace with the --json attribute, False by default (argparse store_true)."""
    return SimpleNamespace(json=False, command="status")


@pytest.mark.allow_real_home_io
class TestApplyJsonOutputDefault:
    """_apply_json_output_default promotes args.json from False to True when config says so."""

    def test_promotes_when_config_says_true(self, _args, monkeypatch):
        """display.json_output: true → args.json becomes True."""
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"display": {"json_output": True}},
        )
        from hermes_cli.main import _apply_json_output_default
        _apply_json_output_default(_args)
        assert _args.json is True

    def test_does_not_promote_when_config_says_false(self, _args, monkeypatch):
        """display.json_output: false (or absent) → args.json stays False."""
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"display": {"json_output": False}},
        )
        from hermes_cli.main import _apply_json_output_default
        _apply_json_output_default(_args)
        assert _args.json is False

    def test_explicit_json_wins_over_config(self, _args, monkeypatch):
        """Explicit --json on CLI already sets args.json=True; config does not override."""
        _args.json = True
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"display": {"json_output": True}},
        )
        from hermes_cli.main import _apply_json_output_default
        _apply_json_output_default(_args)
        assert _args.json is True

    def test_subcommand_without_json_attribute_is_skipped(self, monkeypatch):
        """A subcommand that does not declare --json (no json attr) is untouched."""
        args_no_json = SimpleNamespace(command="some-cmd")  # no .json attribute
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"display": {"json_output": True}},
        )
        from hermes_cli.main import _apply_json_output_default
        _apply_json_output_default(args_no_json)
        assert not hasattr(args_no_json, "json")

    def test_config_error_does_not_raise(self, _args, monkeypatch):
        """If load_config_readonly raises, the function is silently a no-op (no crash)."""
        def _boom():
            raise RuntimeError("config unreadable")
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", _boom)
        from hermes_cli.main import _apply_json_output_default
        _apply_json_output_default(_args)
        assert _args.json is False

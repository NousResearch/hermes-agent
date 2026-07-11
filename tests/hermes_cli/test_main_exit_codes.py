"""CLI entrypoint exit-code propagation tests."""

from __future__ import annotations

import pytest


def test_main_returns_integer_status_from_dispatched_command(monkeypatch):
    """Console-script wrappers must receive integer command failures."""
    from hermes_cli import main as main_mod

    def fake_cmd_kanban(args):
        return 7

    monkeypatch.setattr(main_mod, "cmd_kanban", fake_cmd_kanban)
    monkeypatch.setattr("sys.argv", ["hermes", "kanban", "list"])

    assert main_mod.main() == 7


def test_main_preserves_none_return_as_success(monkeypatch):
    """Commands that historically returned None should still mean success."""
    from hermes_cli import main as main_mod

    def fake_cmd_config(args):
        return None

    monkeypatch.setattr(main_mod, "cmd_config", fake_cmd_config)
    monkeypatch.setattr("sys.argv", ["hermes", "config"])

    assert main_mod.main() == 0


def test_main_treats_boolean_handler_result_as_success(monkeypatch):
    """Plugin handlers returning True/False must not become exit status 1/0.

    ``bool`` is a subclass of ``int``, so a naive ``isinstance(result, int)``
    check would map ``True`` to exit status ``1`` (a spurious failure).
    """
    from hermes_cli import main as main_mod

    def fake_cmd_config(args):
        return True

    monkeypatch.setattr(main_mod, "cmd_config", fake_cmd_config)
    monkeypatch.setattr("sys.argv", ["hermes", "config"])

    assert main_mod.main() == 0


def test_normalize_exit_status_excludes_bool():
    """The helper accepts only exact ints; booleans map to success."""
    from hermes_cli import main as main_mod

    assert main_mod._normalize_exit_status(7) == 7
    assert main_mod._normalize_exit_status(0) == 0
    assert main_mod._normalize_exit_status(True) == 0
    assert main_mod._normalize_exit_status(False) == 0
    assert main_mod._normalize_exit_status(None) == 0
    assert main_mod._normalize_exit_status("2") == 0


def test_main_preserves_system_exit_from_dispatched_command(monkeypatch):
    """Commands that already raise SystemExit should keep owning their exit."""
    from hermes_cli import main as main_mod

    def fake_cmd_config(args):
        raise SystemExit(3)

    monkeypatch.setattr(main_mod, "cmd_config", fake_cmd_config)
    monkeypatch.setattr("sys.argv", ["hermes", "config"])

    with pytest.raises(SystemExit) as exc:
        main_mod.main()

    assert exc.value.code == 3

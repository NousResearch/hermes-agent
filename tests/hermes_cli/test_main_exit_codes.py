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

"""CLI entrypoint exit-code propagation tests."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "fast_path",
    ["_try_termux_fast_tui_launch", "_try_termux_fast_cli_launch"],
)
def test_main_fast_launch_success_returns_zero(monkeypatch, fast_path):
    """Successful fast-launch shortcuts should return an explicit zero status."""
    from hermes_cli import main as main_mod

    monkeypatch.setattr(main_mod, "_try_termux_fast_tui_launch", lambda: False)
    monkeypatch.setattr(main_mod, "_try_termux_fast_cli_launch", lambda: False)
    monkeypatch.setattr(main_mod, fast_path, lambda: True)
    monkeypatch.setattr("sys.argv", ["hermes"])

    assert main_mod.main() == 0


def test_main_version_success_returns_zero(monkeypatch):
    """Top-level version handling should return an explicit zero status."""
    from hermes_cli import main as main_mod

    monkeypatch.setattr(main_mod, "cmd_version", lambda args: None)
    monkeypatch.setattr("sys.argv", ["hermes", "--version"])

    assert main_mod.main() == 0


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


def test_main_preserves_boolean_return_as_success(monkeypatch):
    """Truth-valued handler results are not process exit codes."""
    from hermes_cli import main as main_mod

    def fake_cmd_config(args):
        return True

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

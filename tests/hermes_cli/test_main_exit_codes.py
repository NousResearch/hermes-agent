"""CLI entrypoint exit-code propagation tests."""

from __future__ import annotations

import pytest


_FAST_PATHS = (
    "_try_termux_fast_tui_launch",
    "_try_termux_fast_cli_launch",
    "_try_fast_serve_launch",
    "_try_fast_chat_launch",
)


@pytest.mark.parametrize("fast_path", _FAST_PATHS)
def test_main_fast_launch_success_returns_zero(monkeypatch, fast_path):
    """Successful fast-launch shortcuts should return an explicit zero status."""
    from hermes_cli import main as main_mod

    for name in _FAST_PATHS:
        monkeypatch.setattr(main_mod, name, lambda: False)
    monkeypatch.setattr(main_mod, fast_path, lambda: True)
    monkeypatch.setattr("sys.argv", ["hermes"])

    assert main_mod.main() == 0


def test_main_version_success_returns_zero(monkeypatch):
    """Top-level version handling should return an explicit zero status."""
    from hermes_cli import main as main_mod

    monkeypatch.setattr(main_mod, "cmd_version", lambda args: None)
    monkeypatch.setattr("sys.argv", ["hermes", "--version"])

    assert main_mod.main() == 0


def test_main_fails_the_process_on_a_dispatched_command_failure(monkeypatch):
    """A handler's non-zero return must fail the process, even via a bare main()."""
    from hermes_cli import main as main_mod

    def fake_cmd_kanban(args):
        return 7

    monkeypatch.setattr(main_mod, "cmd_kanban", fake_cmd_kanban)
    monkeypatch.setattr("sys.argv", ["hermes", "kanban", "list"])

    with pytest.raises(SystemExit) as exc:
        main_mod.main()

    assert exc.value.code == 7


def test_main_propagates_chat_failure_when_no_subcommand_is_given(monkeypatch):
    """The default-to-chat path must not swallow chat's failure status."""
    from hermes_cli import main as main_mod

    for name in _FAST_PATHS:
        monkeypatch.setattr(main_mod, name, lambda: False)
    monkeypatch.setattr(main_mod, "_prepare_agent_startup", lambda args: None)
    monkeypatch.setattr(main_mod, "cmd_chat", lambda args: 5)
    monkeypatch.setattr("sys.argv", ["hermes"])

    with pytest.raises(SystemExit) as exc:
        main_mod.main()

    assert exc.value.code == 5


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

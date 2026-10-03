"""Global version introspection must precede updater service-scope isolation."""

import importlib
from pathlib import Path
import sys

import pytest


@pytest.fixture
def canonical_dispatch(monkeypatch):
    """Keep the canonical module, parser, version handler and scope guard real."""
    main = importlib.import_module("hermes_cli.main")
    for name in (
        "_set_process_title", "_warn_if_unsupervised_pid1", "_advertise_agent_env",
        "_cleanup_quarantined_exes", "_sweep_stale_bytecode_if_checkout_changed",
        "_prepare_agent_startup",
    ):
        monkeypatch.setattr(main, name, lambda *a, **kw: None)
    for name in (
        "_try_termux_fast_tui_launch", "_try_termux_fast_cli_launch",
        "_try_fast_serve_launch", "_try_fast_chat_launch",
    ):
        monkeypatch.setattr(main, name, lambda: False)
    monkeypatch.setattr("agent.ssl_verify.install_truststore", lambda: False)
    monkeypatch.setattr("hermes_cli.config.get_container_exec_info", lambda: None)
    monkeypatch.setattr("hermes_cli.venv_sync.check_runtime", lambda *a: None)
    monkeypatch.setattr("hermes_cli.source_check.check_for_updates", lambda **kw: {"behind": 0})

    def forbidden_update(args):
        pytest.fail("version introspection invoked the updater")

    monkeypatch.setattr(main, "cmd_update", forbidden_update)
    monkeypatch.setenv("INVOCATION_ID", "test-owned-dashboard")
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        if str(path) == "/proc/self/cgroup":
            return "0::/user.slice/user@1000.service/app.slice/test-dashboard.service\n"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr("tools.process_registry._systemd_run_user_scope_available", lambda: False)
    argv = [sys.executable, "-m", "hermes_cli.main", "--version", "update"]
    monkeypatch.setattr(sys, "orig_argv", argv)
    monkeypatch.setattr(sys, "argv", ["hermes", *argv[3:]])
    return main


@pytest.mark.platforms("linux")
def test_version_update_succeeds_without_available_scope(canonical_dispatch, capsys):
    assert canonical_dispatch.main() is None
    output = capsys.readouterr()
    assert "Install directory:" in output.out
    assert "Python:" in output.out
    assert "Up to date" in output.out
    assert output.err == ""

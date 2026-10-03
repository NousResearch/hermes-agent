"""Updater ownership must move before the lock, receipt or source mutation."""
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys

import pytest

from hermes_cli import main
from tools import process_registry


@pytest.fixture
def dispatch(monkeypatch):
    # Exercise the real CLI dispatch; only unrelated startup and supervisor I/O
    # are seams. The handler stands for the mutating update boundary.
    for name in ("_set_process_title", "_warn_if_unsupervised_pid1", "_advertise_agent_env",
                 "_cleanup_quarantined_exes", "_sweep_stale_bytecode_if_checkout_changed",
                 "_prepare_agent_startup"):
        monkeypatch.setattr(main, name, lambda *a, **kw: None)
    for name in ("_try_termux_fast_tui_launch", "_try_termux_fast_cli_launch",
                 "_try_fast_serve_launch", "_try_fast_chat_launch"):
        monkeypatch.setattr(main, name, lambda: False)
    monkeypatch.setattr("hermes_cli.config.get_container_exec_info", lambda: None)
    monkeypatch.setattr("hermes_cli.venv_sync.check_runtime", lambda *a: None)
    monkeypatch.setenv("INVOCATION_ID", "test-owned-dashboard")
    original_read = Path.read_text
    def read(path, *args, **kwargs):
        if str(path) == "/proc/self/cgroup":
            return "0::/user.slice/user@1000.service/app.slice/test-dashboard.service\n"
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    argv = [sys.executable, "-u", "-m", "hermes_cli.main", "-p", "alpha", "update", "--yes"]
    monkeypatch.setattr(sys, "orig_argv", argv)
    monkeypatch.setattr(sys, "argv", ["hermes", *argv[4:]])
    calls = []
    args = SimpleNamespace(command="update", version=False, func=lambda args: calls.append("mutated"))
    monkeypatch.setattr(main, "_parse_cli_args", lambda *a: args)
    return calls, argv


@pytest.mark.platforms("linux")
def test_dashboard_cli_moves_whole_update_before_mutating_handler(dispatch, monkeypatch, tmp_path):
    calls, argv = dispatch
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", lambda: True)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout="systemd 255 (255.4)\n", stderr=""))
    def exec_scope(binary, wrapped, env):
        assert wrapped[:4] == [binary, "--user", "--scope", "--quiet"]
        assert wrapped[wrapped.index("--") + 1:] == argv
        assert env["HERMES_ACTION_ID"] == "a" * 32
        assert env["HERMES_HOME"] == str(tmp_path / "profile alpha")
        calls.append("isolated")
        raise SystemExit(0)
    monkeypatch.setenv("HERMES_ACTION_ID", "a" * 32)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile alpha"))
    monkeypatch.setattr(subprocess, "Popen", lambda wrapped, env: SimpleNamespace(
        wait=lambda: exec_scope(wrapped[0], wrapped, env)))
    try:
        main.main()
    except SystemExit as exc:
        assert exc.code == 0
    assert calls == ["isolated"]


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("version_output,version_rc", [
    ("systemd 254 (254.1)\n", 0),
    ("systemd 255 (255.4-1ubuntu8.17)\n", 0),
    ("systemd 258 (258.1)\n", 0),
    ("unknown version\n", 0),
    ("systemd 253 (253.9)\n", 1),
    (None, "timeout"),
    (None, "oserror"),
])
def test_updater_scope_explicitly_disables_modern_environment_expansion(
        dispatch, monkeypatch, version_output, version_rc):
    calls, argv = dispatch
    argv.extend(["${HOME}", "$HOME", "two ${HOME} spaces", "$$", ""])
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", lambda: True)
    def version_probe(command, **kwargs):
        assert command[-1] == "--version"
        assert kwargs["timeout"] == 3
        if version_rc == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if version_rc == "oserror":
            raise OSError("test-owned unavailable version probe")
        return SimpleNamespace(returncode=version_rc, stdout=version_output, stderr="")

    monkeypatch.setattr("subprocess.run", version_probe)

    def exec_scope(binary, wrapped, env):
        options = wrapped[:wrapped.index("--")]
        assert "--expand-environment=no" in options
        assert wrapped[wrapped.index("--") + 1:] == argv
        calls.append("literal scoped exec")
        raise SystemExit(0)

    monkeypatch.setattr(subprocess, "Popen", lambda wrapped, env: SimpleNamespace(
        wait=lambda: exec_scope(wrapped[0], wrapped, env)))
    with pytest.raises(SystemExit) as result:
        main.main()
    assert result.value.code == 0
    assert calls == ["literal scoped exec"]


@pytest.mark.platforms("linux")
def test_updater_scope_preserves_literal_argv_on_old_systemd(dispatch, monkeypatch):
    calls, argv = dispatch
    argv.extend(["${HOME}", "$HOME", "$$", ""])
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", lambda: True)
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout="systemd 253 (253.9)\n", stderr=""))

    def exec_scope(binary, wrapped, env):
        options = wrapped[:wrapped.index("--")]
        # Before 254 scope argv was literal and the new option was unsupported.
        assert "--expand-environment=no" not in options
        assert wrapped[wrapped.index("--") + 1:] == argv
        assert "--scope" in options
        calls.append("legacy literal scoped exec")
        raise SystemExit(0)

    monkeypatch.setattr(subprocess, "Popen", lambda wrapped, env: SimpleNamespace(
        wait=lambda: exec_scope(wrapped[0], wrapped, env)))
    with pytest.raises(SystemExit) as result:
        main.main()
    assert result.value.code == 0
    assert calls == ["legacy literal scoped exec"]


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("scope_failure", ["unavailable", "disappeared"])
def test_dashboard_update_refuses_without_safe_scope(dispatch, monkeypatch, scope_failure):
    calls, argv = dispatch
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available",
                        lambda: scope_failure != "unavailable")
    if scope_failure == "disappeared":
        monkeypatch.setattr(process_registry, "_build_systemd_scope_argv", lambda command, **kw: command)
    original_popen = subprocess.Popen

    def launch(command, *args, **kwargs):
        # Receipt code identity may run harmless git probes; only an updater
        # or its scope launcher is forbidden on this refusal path.
        if command == argv or "--scope" in command:
            calls.append("unsafe launch")
            pytest.fail("refused update attempted a launcher")
        return original_popen(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setenv("HERMES_ACTION_ID", "b" * 32)
    with pytest.raises(SystemExit) as error:
        main.main()
    assert error.value.code == 1
    assert calls == []
    from hermes_cli.update_receipt import read_latest_receipt
    receipt = read_latest_receipt()
    assert receipt["update_id"] == "b" * 32
    assert receipt["finished_at"] and receipt["outcome"] == "failed"


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("action_id", ["d" * 32, "../bad", "A" * 32])
@pytest.mark.parametrize("exec_error", [FileNotFoundError("owned executable disappeared"),
                                       PermissionError("owned executable denied")])
def test_scope_exec_error_persists_correlated_failure(dispatch, monkeypatch, exec_error, action_id):
    calls, _ = dispatch
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", lambda: True)
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout="systemd 255 (255.4)\n", stderr=""))
    monkeypatch.setenv("HERMES_ACTION_ID", action_id)
    attempted = []

    def exec_scope(binary, command, env):
        attempted.append(command)
        raise exec_error

    monkeypatch.setattr(subprocess, "Popen", lambda wrapped, env: exec_scope(wrapped[0], wrapped, env))
    with pytest.raises(SystemExit) as result:
        main.main()
    assert result.value.code == 1
    assert calls == []
    assert len(attempted) == 1 and "--scope" in attempted[0]
    from hermes_cli.update_receipt import read_latest_receipt
    receipt = read_latest_receipt()
    assert len(receipt["update_id"]) == 32 and all(char in "0123456789abcdef" for char in receipt["update_id"])
    if action_id == "d" * 32:
        assert receipt["update_id"] == action_id
    assert receipt["finished_at"] and receipt["outcome"] == "failed"
    assert str(exec_error) in receipt["stop_reason"]


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("flag,value", [("plan", True), ("check", True), ("list_venv_holders", True),
                                       ("install_id", True), ("set_channel", "stable")])
def test_non_updating_commands_do_not_require_scope(dispatch, monkeypatch, flag, value):
    calls, _ = dispatch
    setattr(main._parse_cli_args(None, None, None), flag, value)
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", lambda: False)
    main.main()
    assert calls == ["mutated"]  # The fixture handler is harmless; no real update runs.


@pytest.mark.platforms("linux")
def test_real_dispatch_allows_explicit_restart_deferral_without_scope(monkeypatch):
    calls = []
    # Keep the public parser and dispatch real; only startup and handler are inert.
    for name in ("_set_process_title", "_warn_if_unsupervised_pid1", "_advertise_agent_env",
                 "_cleanup_quarantined_exes", "_sweep_stale_bytecode_if_checkout_changed",
                 "_prepare_agent_startup"):
        monkeypatch.setattr(main, name, lambda *a, **kw: None)
    for name in ("_try_termux_fast_tui_launch", "_try_termux_fast_cli_launch",
                 "_try_fast_serve_launch", "_try_fast_chat_launch"):
        monkeypatch.setattr(main, name, lambda: False)
    monkeypatch.setattr("hermes_cli.config.get_container_exec_info", lambda: None)
    monkeypatch.setattr("hermes_cli.venv_sync.check_runtime", lambda *a: None)
    monkeypatch.setattr(main, "cmd_update", lambda args: calls.append(args.no_gateway_restart))
    monkeypatch.setenv("INVOCATION_ID", "test-owned-dashboard")
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        if str(path) == "/proc/self/cgroup":
            return "0::/user.slice/user@1000.service/app.slice/test-dashboard.service\n"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", lambda: False)
    argv = [sys.executable, "-m", "hermes_cli.main", "update", "--yes", "--no-gateway-restart"]
    monkeypatch.setattr(sys, "orig_argv", argv)
    monkeypatch.setattr(sys, "argv", ["hermes", *argv[3:]])
    main.main()
    assert calls == [True]
    # Negative control: removing deferral must still stop before the same handler.
    monkeypatch.setattr(sys, "argv", ["hermes", "update", "--yes"])
    monkeypatch.setattr(sys, "orig_argv", argv[:-1])
    with pytest.raises(SystemExit) as result:
        main.main()
    assert result.value.code == 1
    assert calls == [True]

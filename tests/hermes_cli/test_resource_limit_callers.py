"""Integration tests for resource-limit caller wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import dashboard_procs
from hermes_cli import main_dashboard
from runtime import resource_limits


def test_cron_reclaim_passes_canonical_loaded_config_to_runtime(monkeypatch, tmp_path):
    """The caller loads canonical Hermes config and hands the mapping to runtime."""
    from cron import scheduler

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "runtime:\n  nofile_soft_limit: 2048\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    seen: list[object] = []
    monkeypatch.setattr(
        resource_limits,
        "apply_nofile_soft_limit",
        lambda config: seen.append(config) or False,
    )

    scheduler._reclaim_fds_best_effort()

    assert len(seen) == 1
    assert seen[0]["runtime"]["nofile_soft_limit"] == 2048


@pytest.mark.anyio
async def test_gateway_startup_applies_limit_before_gateway_initialization(monkeypatch):
    import gateway.code_skew
    import gateway.run as gateway_run

    calls: list[str] = []

    monkeypatch.setattr(
        resource_limits,
        "apply_nofile_soft_limit",
        lambda config: calls.append("limit"),
    )

    class _StopStartup(Exception):
        pass

    def stop_after_limit():
        calls.append("gateway-init")
        raise _StopStartup

    monkeypatch.setattr(gateway.code_skew, "record_boot_fingerprint", stop_after_limit)

    with pytest.raises(_StopStartup):
        await gateway_run.start_gateway()

    assert calls == ["limit", "gateway-init"]


def test_serve_startup_applies_limit_before_web_server(monkeypatch):
    from hermes_cli import main as cli_main
    import hermes_cli.main_web_build as main_web_build
    import hermes_cli.plugins
    import hermes_cli.web_server

    # cmd_dashboard(headless_backend=True) exports HERMES_SERVE_HEADLESS=1 into
    # this process's environment (main.py serve path). Touch the key through
    # monkeypatch FIRST so teardown restores the pre-test state — otherwise the
    # leaked flag flips later web-server tests (mount_spa) into the headless
    # 404 path.
    monkeypatch.setenv("HERMES_SERVE_HEADLESS", "0")

    calls: list[str] = []
    monkeypatch.setattr(
        resource_limits,
        "apply_nofile_soft_limit",
        lambda config: calls.append("limit"),
    )
    monkeypatch.setattr(cli_main, "_sync_bundled_skills_quietly", lambda: None)
    monkeypatch.setattr(cli_main, "_build_web_ui", lambda *args, **kwargs: True)
    monkeypatch.setattr(main_web_build, "_build_web_ui", lambda *args, **kwargs: True)
    monkeypatch.setattr(cli_main, "_maybe_setup_dashboard_auth_interactively", lambda args: None)
    monkeypatch.setattr(hermes_cli.plugins, "discover_plugins", lambda: None)
    monkeypatch.setattr(
        hermes_cli.web_server,
        "start_server",
        lambda **kwargs: calls.append("server"),
    )

    args = SimpleNamespace(
        status=False,
        stop=False,
        headless_backend=True,
        ssh_owner_nonce=None,
        ssh_session_token_file=None,
        host="127.0.0.1",
        port=0,
        no_open=True,
        insecure=False,
        open_profile="",
        isolated=True,
        skip_build=False,
    )

    cli_main.cmd_dashboard(args)

    assert calls == ["limit", "server"]


def test_named_profile_reroute_defers_limit_to_final_process(monkeypatch, tmp_path):
    """The launcher profile must not leak its limit across machine re-exec."""
    from hermes_cli import main as cli_main
    import profiles.current
    import hermes_constants
    from tools.environments import local as local_environment

    calls: list[str] = []
    exec_call: dict[str, object] = {}

    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.setattr(
        resource_limits,
        "apply_nofile_soft_limit",
        lambda config: calls.append("limit"),
    )
    monkeypatch.setattr(
        profiles.current,
        "get_active_profile_name",
        lambda: "worker",
    )
    monkeypatch.setattr(main_dashboard, "_dashboard_listening", lambda *args: False)
    monkeypatch.setattr(
        local_environment,
        "build_subprocess_env",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        hermes_constants,
        "get_default_hermes_root",
        lambda: tmp_path,
    )

    class _ExecCalled(Exception):
        pass

    def stop_at_exec(executable, argv, env):
        exec_call.update(executable=executable, argv=argv, env=env)
        raise _ExecCalled

    def stop_at_popen(argv, env):
        exec_call.update(executable=argv[0], argv=argv, env=env)

        class _Proc:
            def wait(self):
                raise _ExecCalled

        return _Proc()

    monkeypatch.setattr(cli_main.os, "execvpe", stop_at_exec)
    monkeypatch.setattr(main_dashboard.subprocess, "Popen", stop_at_popen)

    args = SimpleNamespace(
        status=False,
        stop=False,
        headless_backend=True,
        ssh_owner_nonce=None,
        ssh_session_token_file=None,
        host="127.0.0.1",
        port=0,
        no_open=True,
        insecure=False,
        open_profile="",
        isolated=False,
        skip_build=False,
    )

    with pytest.raises(_ExecCalled):
        cli_main.cmd_dashboard(args)

    assert calls == []
    assert exec_call["argv"][1:5] == ["-m", "hermes_cli.main", "-p", "default"]
    assert exec_call["env"]["HERMES_HOME"] == str(tmp_path)


@pytest.mark.parametrize("lifecycle_flag", ["status", "stop"])
def test_dashboard_lifecycle_flags_skip_limit_adjustment(monkeypatch, lifecycle_flag):
    """Informational/stop-only commands must not mutate process limits."""
    from hermes_cli import main as cli_main
    import hermes_cli.main_dashboard as hermes_cli_main_dashboard

    calls: list[str] = []
    monkeypatch.setattr(
        resource_limits,
        "apply_nofile_soft_limit",
        lambda config: calls.append("limit"),
    )
    monkeypatch.setattr(dashboard_procs, "_scan_dashboard_processes", lambda: [])
    monkeypatch.setattr(cli_main, "_find_stale_dashboard_pids", lambda **_: [])
    monkeypatch.setattr(hermes_cli_main_dashboard, "_find_stale_dashboard_pids", lambda **_: [])

    args = SimpleNamespace(
        status=lifecycle_flag == "status",
        stop=lifecycle_flag == "stop",
        headless_backend=False,
        ssh_owner_nonce=None,
        ssh_session_token_file=None,
        host="127.0.0.1",
        port=0,
        no_open=True,
        insecure=False,
        open_profile="",
        isolated=False,
        skip_build=False,
    )

    with pytest.raises(SystemExit):
        cli_main.cmd_dashboard(args)

    assert calls == []
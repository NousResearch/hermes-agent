"""Tests for configurable RLIMIT_NOFILE startup handling."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from hermes_cli import resource_limits


class _FakeResource:
    RLIMIT_NOFILE = 7
    RLIM_INFINITY = 2**63 - 1

    def __init__(self, soft: int, hard: int) -> None:
        self.limits = (soft, hard)
        self.set_calls: list[tuple[int, tuple[int, int]]] = []

    def getrlimit(self, resource: int) -> tuple[int, int]:
        assert resource == self.RLIMIT_NOFILE
        return self.limits

    def setrlimit(self, resource: int, limits: tuple[int, int]) -> None:
        assert resource == self.RLIMIT_NOFILE
        self.set_calls.append((resource, limits))
        self.limits = limits


def test_real_config_loader_reads_runtime_nofile_setting(monkeypatch, tmp_path):
    """The helper uses the canonical config loader, not a second YAML parser."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "runtime:\n  nofile_soft_limit: 2048\n",
        encoding="utf-8",
    )
    fake_resource = _FakeResource(soft=256, hard=4096)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(resource_limits, "_resource", fake_resource)

    assert resource_limits.apply_nofile_soft_limit() is True
    assert fake_resource.set_calls == [
        (fake_resource.RLIMIT_NOFILE, (2048, 4096)),
    ]


def test_default_is_clamped_to_hard_limit(monkeypatch):
    fake_resource = _FakeResource(soft=256, hard=1024)
    monkeypatch.setattr(resource_limits, "_resource", fake_resource)

    assert resource_limits.apply_nofile_soft_limit({}) is True
    assert fake_resource.limits == (1024, 1024)


def test_finite_soft_limit_raises_when_hard_limit_is_infinite(monkeypatch):
    fake_resource = _FakeResource(soft=256, hard=_FakeResource.RLIM_INFINITY)
    monkeypatch.setattr(resource_limits, "_resource", fake_resource)

    assert resource_limits.apply_nofile_soft_limit({}) is True
    assert fake_resource.set_calls == [
        (
            fake_resource.RLIMIT_NOFILE,
            (4096, fake_resource.RLIM_INFINITY),
        ),
    ]


def test_never_lowers_an_already_higher_soft_limit(monkeypatch):
    fake_resource = _FakeResource(soft=8192, hard=16384)
    monkeypatch.setattr(resource_limits, "_resource", fake_resource)

    assert resource_limits.apply_nofile_soft_limit(
        {"runtime": {"nofile_soft_limit": 4096}}
    ) is False
    assert fake_resource.set_calls == []
    assert fake_resource.limits == (8192, 16384)


@pytest.mark.parametrize("disabled", [0, False, None])
def test_explicit_values_disable(monkeypatch, disabled):
    fake_resource = _FakeResource(soft=256, hard=4096)
    monkeypatch.setattr(resource_limits, "_resource", fake_resource)

    assert resource_limits.apply_nofile_soft_limit(
        {"runtime": {"nofile_soft_limit": disabled}}
    ) is False
    assert fake_resource.set_calls == []


def test_unsupported_platform_is_a_safe_noop(monkeypatch):
    monkeypatch.setattr(resource_limits, "_resource", None)

    assert resource_limits.apply_nofile_soft_limit({}) is False


def test_fresh_process_import_without_posix_resource_is_a_safe_noop():
    code = textwrap.dedent(
        """
        import importlib.util
        import pathlib
        import sys

        sys.modules["resource"] = None
        module_path = pathlib.Path(sys.argv[1])
        spec = importlib.util.spec_from_file_location(
            "hermes_cli._resource_limits_without_posix_resource",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module._resource is None
        assert module.apply_nofile_soft_limit({}) is False
        """
    )

    subprocess.run(
        [sys.executable, "-c", code, resource_limits.__file__],
        check=True,
        cwd=Path(resource_limits.__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("invalid", [True, -1, 4096.0, "4096", object()])
def test_invalid_values_are_safe_noops(monkeypatch, invalid):
    fake_resource = _FakeResource(soft=256, hard=4096)
    monkeypatch.setattr(resource_limits, "_resource", fake_resource)

    assert resource_limits.apply_nofile_soft_limit(
        {"runtime": {"nofile_soft_limit": invalid}}
    ) is False
    assert fake_resource.set_calls == []


def test_setrlimit_denial_is_a_safe_noop(monkeypatch):
    class _DeniedResource(_FakeResource):
        def setrlimit(self, resource: int, limits: tuple[int, int]) -> None:
            raise PermissionError("simulated EPERM")

    fake_resource = _DeniedResource(soft=256, hard=4096)
    monkeypatch.setattr(resource_limits, "_resource", fake_resource)

    assert resource_limits.apply_nofile_soft_limit({}) is False
    assert fake_resource.limits == (256, 4096)


def test_getrlimit_failure_is_a_safe_noop(monkeypatch):
    class _BrokenResource(_FakeResource):
        def getrlimit(self, resource: int) -> tuple[int, int]:
            raise OSError("simulated getrlimit failure")

    fake_resource = _BrokenResource(soft=256, hard=4096)
    monkeypatch.setattr(resource_limits, "_resource", fake_resource)

    assert resource_limits.apply_nofile_soft_limit({}) is False
    assert fake_resource.set_calls == []


def test_never_lowers_an_unlimited_soft_limit(monkeypatch):
    fake_resource = _FakeResource(soft=-1, hard=-1)
    fake_resource.RLIM_INFINITY = -1
    monkeypatch.setattr(resource_limits, "_resource", fake_resource)

    assert resource_limits.apply_nofile_soft_limit({}) is False
    assert fake_resource.set_calls == []
    assert fake_resource.limits == (-1, -1)


@pytest.mark.anyio
async def test_gateway_startup_applies_limit_before_gateway_initialization(monkeypatch):
    import gateway.code_skew
    import gateway.run as gateway_run

    calls: list[str] = []

    monkeypatch.setattr(
        resource_limits,
        "apply_nofile_soft_limit",
        lambda: calls.append("limit"),
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
    import hermes_cli.plugins
    import hermes_cli.web_server

    calls: list[str] = []
    monkeypatch.setattr(
        resource_limits,
        "apply_nofile_soft_limit",
        lambda: calls.append("limit"),
    )
    monkeypatch.setattr(cli_main, "_sync_bundled_skills_quietly", lambda: None)
    monkeypatch.setattr(cli_main, "_build_web_ui", lambda *args, **kwargs: True)
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
    import hermes_cli.profiles
    import hermes_constants
    from tools.environments import local as local_environment

    calls: list[str] = []
    exec_call: dict[str, object] = {}

    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.setattr(
        resource_limits,
        "apply_nofile_soft_limit",
        lambda: calls.append("limit"),
    )
    monkeypatch.setattr(
        hermes_cli.profiles,
        "get_active_profile_name",
        lambda: "worker",
    )
    monkeypatch.setattr(cli_main, "_dashboard_listening", lambda *args: False)
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

    monkeypatch.setattr(cli_main.os, "execvpe", stop_at_exec)

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

    calls: list[str] = []
    monkeypatch.setattr(
        resource_limits,
        "apply_nofile_soft_limit",
        lambda: calls.append("limit"),
    )
    monkeypatch.setattr(cli_main, "_scan_dashboard_processes", lambda: [])
    monkeypatch.setattr(cli_main, "_find_stale_dashboard_pids", lambda: [])

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

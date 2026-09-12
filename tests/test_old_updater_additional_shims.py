"""Retired import names suppress old updater work without claiming success."""

import importlib
import os
from pathlib import Path
import socket
import subprocess
import urllib.request

import pytest


@pytest.fixture
def no_external_work(monkeypatch):
    import pm

    def forbidden(*args, **kwargs):
        pytest.fail("old-updater shim attempted external work")

    for name in ("sync_venv", "ensure_environment", "build_environment", "ensure_import"):
        monkeypatch.setattr(pm, name, forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    monkeypatch.setattr(os, "kill", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(urllib.request, "urlretrieve", forbidden)
    before_env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    home = Path(os.environ["HERMES_HOME"])
    before_files = {p: p.read_bytes() for p in home.rglob("*") if p.is_file()}
    yield forbidden
    assert {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"} == before_env
    assert {p: p.read_bytes() for p in home.rglob("*") if p.is_file()} == before_files


@pytest.mark.parametrize("command", ["", "hermes -p ops gateway run", "hermes --profile=ops gateway run"])
def test_retired_profile_probe_returns_unknown(command, no_external_work):
    from gateway.status import profile_flag_value

    # The old scanner compares this value to a profile before selecting a PID.
    assert profile_flag_value(command) is None


@pytest.mark.parametrize("refresh", [False, True])
def test_retired_code_identity_is_unknown(refresh, no_external_work, monkeypatch):
    from hermes_cli import version_info
    from hermes_cli.build_info import get_code_identity

    monkeypatch.setattr(version_info, "get_code_identity", no_external_work)
    identity = get_code_identity(refresh=refresh)
    assert identity == {"sha": None, "short_sha": None, "version": None, "source": "unknown"}
    identity["sha"] = "caller mutation"
    assert get_code_identity(refresh)["sha"] is None


def test_retired_constants_reload_does_not_replace_live_module(no_external_work, monkeypatch):
    import hermes_constants
    from hermes_cli.managed_uv import _reload_hermes_constants

    before = dict(vars(hermes_constants))
    monkeypatch.setattr(importlib, "reload", no_external_work)
    assert _reload_hermes_constants() is None
    assert vars(hermes_constants) == before


@pytest.mark.parametrize("kwargs", [{}, {"timeout": 120, "capture_output": False}])
def test_retired_pip_install_stops_before_reporting_success(kwargs, no_external_work, capsys):
    from hermes_cli.tools_config import _pip_install

    with pytest.raises(SystemExit) as exc:
        result = _pip_install(["--quiet", "honcho-ai"], **kwargs)
        pytest.fail(f"retired installer returned a result: {result}")
    assert exc.value.code == 0
    assert "relaunch" in capsys.readouterr().err.lower()


def test_retired_root_stops_before_inventing_portable_git_path(no_external_work, capsys):
    from hermes_cli.update_cmd import get_default_hermes_root

    with pytest.raises(SystemExit) as exc:
        get_default_hermes_root() / "git" / "mingw64" / "libexec" / "git-core" / "git.exe"
    assert exc.value.code == 0
    assert "relaunch" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("prompt", [True, False])
def test_retired_ensure_reports_unavailable_without_installing(prompt, no_external_work):
    from tools.lazy_deps import ensure

    # Dependency-unavailable callers must not mistake a no-op for readiness.
    with pytest.raises(ImportError, match="relaunch"):
        ensure("memory.honcho", prompt=prompt)


def test_live_dingtalk_dependencies_use_pm_not_retired_installer(monkeypatch):
    from plugins.platforms.dingtalk import adapter
    from pm import extras

    requested = []

    def unavailable(extra):
        requested.append(extra)
        raise ImportError("dependency unavailable")

    monkeypatch.setattr(adapter, "DINGTALK_STREAM_AVAILABLE", False)
    monkeypatch.setattr(extras, "ensure_import", unavailable)
    assert adapter.ensure_dingtalk_deps() is False
    assert requested == ["dingtalk"]


@pytest.mark.parametrize("specs", [[], ["honcho-ai"]])
def test_retired_install_specs_stops_before_reporting_success(specs, no_external_work, capsys):
    from tools.lazy_deps import install_specs

    with pytest.raises(SystemExit) as exc:
        result = install_specs(specs, timeout=120)
        pytest.fail(f"retired installer returned a result: {result}")
    assert exc.value.code == 0
    assert "relaunch" in capsys.readouterr().err.lower()


def test_retired_subprocess_run_stops_powershell_installer(no_external_work, capsys):
    from hermes_cli import _subprocess_compat

    with pytest.raises(SystemExit) as exc:
        _subprocess_compat.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-c", "irm https://astral.sh/uv/install.ps1 | iex"],
            env=dict(os.environ), check=True, capture_output=True,
        )
    assert exc.value.code == 0
    assert "relaunch" in capsys.readouterr().err.lower()

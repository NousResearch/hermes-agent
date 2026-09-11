"""The post-swap import boundary must never revive retired installers."""

import importlib
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import sys
import urllib.request

import pytest


@pytest.fixture
def no_external_work(monkeypatch):
    """Fail at real I/O and PM boundaries, not at the shim under test."""
    import pm

    def forbidden(*args, **kwargs):
        pytest.fail("old-updater shim attempted external work")

    for name in ("sync_venv", "ensure_environment", "build_environment"):
        monkeypatch.setattr(pm, name, forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    monkeypatch.setattr(os, "kill", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(urllib.request, "urlretrieve", forbidden)
    return forbidden


@pytest.mark.parametrize(
    "name,args,kwargs",
    [
        ("ensure_uv", (), {}),
        ("ensure_uv", (), {"repair_observer": lambda result: pytest.fail("repair observer ran")}),
        ("update_managed_uv", (), {}),
        ("update_managed_uv", (), {"force": True}),
        ("resolve_uv", (), {}),
        ("managed_python_env", (), {}),
        ("managed_python_env", (Path("checkout"),), {"install_dir": Path("python"), "base_env": {}}),
        ("rebuild_venv", ("uv", Path("venv")), {}),
        ("rebuild_venv", ("uv", Path("venv"), "3.11"), {}),
    ],
)
def test_retired_managed_uv_stops_before_fallback(name, args, kwargs, no_external_work, capsys):
    module = importlib.import_module("hermes_cli.managed_uv")
    before_env = dict(os.environ)
    before_home = set(Path(os.environ["HERMES_HOME"]).rglob("*"))
    with pytest.raises(SystemExit) as exc:
        getattr(module, name)(*args, **kwargs)
    assert exc.value.code == 0
    output = capsys.readouterr()
    assert "relaunch" in (output.out + output.err).lower()
    assert dict(os.environ) == before_env
    assert set(Path(os.environ["HERMES_HOME"]).rglob("*")) == before_home


@pytest.mark.parametrize("unpack", [False, True], ids=["path-era", "tuple-era"])
def test_ensure_uv_stops_both_historical_return_contracts(unpack, no_external_work, capsys):
    from hermes_cli.managed_uv import ensure_uv

    with pytest.raises(SystemExit) as exc:
        if unpack:
            uv, fresh_bootstrap = ensure_uv()
        else:
            uv = ensure_uv()
        # A falsy result is NOT inert: old callers install through pip instead.
        subprocess.run([uv, "pip", "install"] if uv else [sys.executable, "-m", "pip", "install"])
    assert exc.value.code == 0
    assert "relaunch" in capsys.readouterr().err.lower()


@pytest.mark.parametrize(
    "module,name,args,kwargs",
    [
        ("hermes_cli.psutil_android", "prepare_patched_psutil_sdist", (Path("psutil.tar.gz"), Path("src")), {}),
        ("hermes_cli.update_cmd", "_ensure_uv_for_termux", (["python", "-m", "pip"],), {}),
        ("hermes_cli.update_cmd", "_ensure_venv_pip", (["python", "-m", "pip"], "python"), {}),
        ("hermes_cli.update_cmd", "_pip_install_prefix", (None,), {}),
        ("hermes_cli.update_cmd", "_pip_install_prefix", ("uv",), {}),
        ("hermes_cli.update_cmd", "_refuse_update_for_contended_shims", (RuntimeError("locked"),), {}),
    ],
)
def test_other_dependency_entrypoints_stop_cleanly(module, name, args, kwargs, no_external_work, capsys):
    before_home = set(Path(os.environ["HERMES_HOME"]).rglob("*"))
    with pytest.raises(SystemExit) as exc:
        shim = getattr(importlib.import_module(module), name)
        shim(*args, **kwargs)
    assert exc.value.code == 0
    assert "relaunch" in capsys.readouterr().err.lower()
    assert set(Path(os.environ["HERMES_HOME"]).rglob("*")) == before_home


def test_retired_probes_and_refreshes_do_no_work(no_external_work, tmp_path):
    from hermes_cli import _install_repair, backup, banner, config, main, update_cmd
    from tools import browser_tool

    assert _install_repair._sync_windows_cli_launchers(tmp_path) == []
    # Unknown, not an invented "no live holders" result.
    assert backup._foreign_db_holder_pids(tmp_path / "state.db") is None
    assert banner._check_via_pypi() is None
    assert banner.check_via_pypi() is None
    assert config.is_uv_tool_install() is False
    assert config.is_unsupported_install_method("pip") is False
    assert config.format_unsupported_install_warning("pip") == ""
    assert main._detect_venv_python_processes() == []
    assert main._detect_venv_python_processes(exclude_pids={123}) == []
    assert browser_tool.warm_agent_browser_npx_cache() is False
    assert browser_tool.warm_agent_browser_npx_cache(timeout=0.1) is False
    # A private, never-raised type keeps historical `except helper():` valid
    # without swallowing real errors or resolving the removed quarantine code.
    error_type = update_cmd._shim_quarantine_error_type()
    assert issubclass(error_type, Exception)
    with pytest.raises(RuntimeError, match="not a quarantine"):
        try:
            raise RuntimeError("not a quarantine")
        except error_type:
            pytest.fail("retired quarantine caught an unrelated exception")


@pytest.fixture
def old_updater():
    path = Path(__file__).parent / "compat" / "old_updater_dependencies.py"
    spec = importlib.util.spec_from_file_location("old_updater_dependencies", path)
    assert spec is not None and spec.loader is not None
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    return old


def test_old_android_updater_stops_before_download(old_updater, no_external_work, capsys):
    with pytest.raises(SystemExit) as exc:
        old_updater._install_psutil_android_compat(["uv", "pip"])
    assert exc.value.code == 0
    assert "relaunch" in capsys.readouterr().err.lower()


def test_old_updater_retains_its_code_but_loads_new_managed_uv(old_updater, no_external_work, capsys, tmp_path):
    """The real pre-PM post-pull function must stop at its new lazy import."""
    from types import SimpleNamespace

    old = old_updater
    # These are the already-imported pre-swap collaborators. Record the prefix
    # without touching an installation; do not replace any lazy imports in the
    # frozen function (the managed_uv import is the boundary being exercised).
    prefix = []
    old._refuse_update_if_venv_foreign_owned = lambda root: prefix.append("ownership")
    old_main = SimpleNamespace(
        PROJECT_ROOT=tmp_path,
        _abort_dependency_sync_if_self_locked=lambda token: prefix.append("self-lock"),
    )
    old._m = lambda: old_main
    old._write_update_incomplete_marker = lambda: prefix.append("old-marker")
    old._editable_install_is_current = lambda *args: False
    old._ensure_venv_pip = no_external_work
    old._ensure_uv_for_termux = no_external_work
    before = set(tmp_path.rglob("*"))

    with pytest.raises(SystemExit) as exc:
        old._sync_python_dependencies_after_pull(
            ["git"], "main", "before-pull", active_lazy_features=[],
            active_tool_dependencies=[], _windows_gateway_resume=None,
        )
    assert exc.value.code == 0
    assert prefix == ["ownership", "self-lock", "old-marker"]
    assert "relaunch" in capsys.readouterr().err.lower()
    assert set(tmp_path.rglob("*")) == before


def test_live_windows_scan_does_not_use_the_retired_main_alias(monkeypatch, no_external_work):
    from hermes_cli import main, process_identity, update_cmd_windows

    # This is routing, not OS emulation: the lifecycle fallback accepts holder
    # rows on any host. The real scanner remains owned by update_cmd_windows.
    monkeypatch.setattr(main, "_detect_venv_python_processes", no_external_work)
    monkeypatch.setattr(process_identity, "ledger_entries", lambda: [])
    monkeypatch.setattr(update_cmd_windows, "_psutil", lambda: None)
    monkeypatch.setattr(
        update_cmd_windows, "_detect_venv_python_processes",
        lambda: [(123, "python", "python -m hermes_cli.main serve")],
    )
    assert update_cmd_windows._desktop_owns_gateway_lifecycle() is True

"""The post-swap import boundary must never revive retired installers."""

from copy import deepcopy
import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.compat.old_updater_support import (
    fresh_child as fresh_child,
    no_external_work as no_external_work,
)


@pytest.mark.parametrize(
    "module,name,args,kwargs",
    [
        ("managed_uv", "ensure_uv", (), {}),
        ("managed_uv", "ensure_uv", (), {"repair_observer": lambda result: pytest.fail("repair observer ran")}),
        ("managed_uv", "update_managed_uv", (), {}),
        ("managed_uv", "update_managed_uv", (), {"force": True}),
        ("managed_uv", "resolve_uv", (), {}),
        ("managed_uv", "managed_python_env", (), {}),
        ("managed_uv", "managed_python_env", (Path("checkout"),), {"install_dir": Path("python"), "base_env": {}}),
        ("managed_uv", "rebuild_venv", ("uv", Path("venv")), {}),
        ("managed_uv", "rebuild_venv", ("uv", Path("venv"), "3.11"), {}),
        ("psutil_android", "prepare_patched_psutil_sdist", (Path("psutil.tar.gz"), Path("src")), {}),
        ("update_cmd", "_ensure_uv_for_termux", (["python", "-m", "pip"],), {}),
        ("update_cmd", "_ensure_venv_pip", (["python", "-m", "pip"], "python"), {}),
        ("update_cmd", "_pip_install_prefix", (None,), {}),
        ("update_cmd", "_pip_install_prefix", ("uv",), {}),
        ("update_cmd", "_refuse_update_for_contended_shims", (RuntimeError("locked"),), {}),
        ("tools_config", "install_cua_driver", (),
         {"upgrade": True, "require_confirmed_update": True, "show_installer_progress": False}),
        ("update_cmd", "_capture_active_lazy_features", (), {}),
        ("update_cmd", "_refresh_active_lazy_features", (), {}),
        ("update_cmd", "_refresh_active_lazy_features", (["browser"],), {}),
        ("update_cmd", "_refresh_active_lazy_features", (["uv", "pip"],),
         {"env": {"VIRTUAL_ENV": "venv"}, "features": ["browser"]}),
        ("update_cmd", "_refresh_active_memory_provider_dependencies", (), {}),
        ("update_cmd", "_npm_lockfile_changed", (Path("checkout"),), {}),
        ("update_cmd", "_update_node_dependencies", (), {}),
        ("update_cmd", "_rebuild_desktop_after_update", (Path("desktop"),),
         {"had_desktop_app_before_update": True}),
        ("update_cmd", "_rebuild_desktop_after_update", (Path("desktop"),),
         {"had_desktop_app_before_update": False}),
        ("update_cmd", "_path_uid", (Path("venv"),), {}),
        ("update_cmd", "_write_update_incomplete_marker", (), {}),
        ("update_cmd", "_write_lazy_refresh_incomplete_marker", (), {}),
        ("update_cmd", "_reload_updated_runtime_modules", (), {}),
        ("update_cmd_maint", "_reload_updated_runtime_modules", (), {}),
    ],
)
def test_retired_dependency_entrypoints_handoff_without_fallback(module, name, args, kwargs, fresh_child):
    # Some boundaries (notably psutil_android) hand off during import itself.
    # Resolve ordinary modules before the guard: their CLI startup is not a shim.
    if module != "psutil_android":
        importlib.import_module(f"hermes_cli.{module}")
    # Exceptions have identity equality; preserve the caller's instance too.
    memo = {id(arg): arg for arg in args if isinstance(arg, BaseException)}
    before = deepcopy((args, kwargs), memo)
    with fresh_child.exits():
        getattr(importlib.import_module(f"hermes_cli.{module}"), name)(*args, **kwargs)
    assert (args, kwargs) == before


@pytest.mark.parametrize("unpack", [False, True], ids=["path-era", "tuple-era"])
@pytest.mark.parametrize("status", [0, 19])
def test_ensure_uv_stops_both_historical_return_contracts(unpack, status, fresh_child):
    from hermes_cli.managed_uv import ensure_uv

    fresh_child.returncode = status
    with fresh_child.exits():
        if unpack:
            uv, fresh_bootstrap = ensure_uv()
        else:
            uv = ensure_uv()
        # A falsy result is NOT inert: old callers install through pip instead.
        subprocess.run([uv, "pip", "install"] if uv else [sys.executable, "-m", "pip", "install"])


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


def test_old_android_updater_handoffs_before_download(old_updater, fresh_child):
    with fresh_child.exits():
        old_updater._install_psutil_android_compat(["uv", "pip"])


def test_old_updater_retains_its_code_but_loads_new_managed_uv(old_updater, fresh_child, no_external_work, tmp_path):
    """The real pre-PM post-pull function must hand off at its new lazy import."""
    old = old_updater
    # Already-imported pre-swap collaborators. Leave frozen lazy imports real.
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
    resume = {"resume_needed": True, "profiles": {"work": "old-pid"}}
    before = deepcopy(resume)
    before_files = set(tmp_path.rglob("*"))
    with fresh_child.exits():
        old._sync_python_dependencies_after_pull(
            ["git"], "main", "before-pull", active_lazy_features=[],
            active_tool_dependencies=[], _windows_gateway_resume=resume,
        )
    assert prefix == ["ownership", "self-lock", "old-marker"]
    assert fresh_child.requests[0]["windows_resume"] == before
    # No acknowledgement means the historical caller still owns recovery.
    assert resume == before
    assert set(tmp_path.rglob("*")) == before_files


def test_live_windows_scan_does_not_use_the_retired_main_alias(monkeypatch, no_external_work):
    from hermes_cli import main, process_identity, update_cmd_windows

    # Routing, not OS emulation: lifecycle fallback accepts rows on any host.
    monkeypatch.setattr(main, "_detect_venv_python_processes", no_external_work)
    monkeypatch.setattr(process_identity, "ledger_entries", lambda: [])
    monkeypatch.setattr(update_cmd_windows, "_psutil", lambda: None)
    monkeypatch.setattr(
        update_cmd_windows, "_detect_venv_python_processes",
        lambda: [(123, "python", "python -m hermes_cli.main serve")],
    )
    assert update_cmd_windows._desktop_owns_gateway_lifecycle() is True

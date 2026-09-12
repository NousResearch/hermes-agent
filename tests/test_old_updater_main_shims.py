"""Historical main imports must not restart pre-PM updater work after a swap."""

from copy import deepcopy
from pathlib import Path

import pytest

from tests.compat.old_updater_support import (
    fresh_child as fresh_child,
    no_external_work as no_external_work,
)


@pytest.fixture
def historical_main(no_external_work):
    from hermes_cli import main

    return main


def test_historical_main_data_and_skipped_probes_preserve_caller_shapes(historical_main, tmp_path):
    main = historical_main
    from hermes_cli import main_web_build

    # Old recorders compose this name with PROJECT_ROOT. It remains data only.
    assert tmp_path / main._BYTECODE_FINGERPRINT_FILE == (
        tmp_path / main_web_build._BYTECODE_FINGERPRINT_FILE
    )
    failed = ["hermes.exe"]
    try:
        raise main.ShimQuarantineError(failed)
    except main.ShimQuarantineError as exc:
        assert isinstance(exc, RuntimeError)
        assert exc.failed_shims == failed
        assert exc.failed_shims is not failed
        assert failed[0] in str(exc)

    before_files = set(tmp_path.rglob("*"))
    prefix = ["uv", "pip"]
    env = {"VIRTUAL_ENV": str(tmp_path)}
    # None means indeterminate to the historical repair caller, NOT healthy [].
    assert main._detect_broken_lazy_refresh_imports(prefix, env=env) is None
    assert main._resolve_install_target_python(prefix, env) is None
    moved = [(tmp_path / "hermes.exe", tmp_path / "hermes.exe.old")]
    before = list(moved)
    assert main._restore_quarantined_exes(moved) is None
    assert moved == before
    assert main._write_web_ui_build_stamp(tmp_path, tmp_path / "web") is None
    assert prefix == ["uv", "pip"]
    assert env == {"VIRTUAL_ENV": str(tmp_path)}
    assert set(tmp_path.rglob("*")) == before_files


def test_historical_marker_cleanup_preserves_path_and_is_idempotent(historical_main, tmp_path, monkeypatch):
    # The retired writer now hands off. Cleanup of an existing legacy marker
    # remains supported; PM's recovery lifecycle is covered in test_early_recovery.
    monkeypatch.setattr(historical_main, "PROJECT_ROOT", tmp_path)
    marker = historical_main._update_marker_path()
    assert marker == tmp_path / ".update-incomplete"
    marker.write_text("started=1\npid=0\n", encoding="utf-8")
    assert historical_main._clear_update_incomplete_marker() is None
    assert not marker.exists()
    assert historical_main._clear_update_incomplete_marker() is None


@pytest.mark.parametrize("cached", [False, True], ids=["cold-lookup", "cached-export"])
@pytest.mark.parametrize(
    "name,args,kwargs",
    [
        ("_capture_active_lazy_features", (), {}),
        ("_refresh_active_lazy_features", (), {}),
        ("_refresh_active_lazy_features", (["browser"],), {}),
        ("_refresh_active_lazy_features", (["uv", "pip"],),
         {"env": {"VIRTUAL_ENV": "venv"}, "features": ["browser"]}),
        ("_refresh_active_memory_provider_dependencies", (), {}),
        ("_npm_lockfile_changed", (Path("checkout"),), {}),
        ("_write_update_incomplete_marker", (), {}),
        ("_reload_updated_runtime_modules", (), {}),
    ],
)
def test_historical_main_lazy_hooks_handoff(name, args, kwargs, cached, historical_main, fresh_child, monkeypatch):
    main = historical_main
    before = deepcopy((args, kwargs))
    # Exercise PEP 562 even if an earlier test cached this export. Register the
    # temporary slot with monkeypatch so it also restores an absent attribute.
    monkeypatch.setitem(main.__dict__, name, None)
    monkeypatch.delitem(main.__dict__, name)
    if cached:
        getattr(main, name)
    with fresh_child.exits():
        getattr(main, name)(*args, **kwargs)
    assert (args, kwargs) == before


@pytest.mark.parametrize(
    "name,args,kwargs",
    [
        ("_desktop_stamp_path", (), {}),
        ("_expected_windows_pe_machines", (), {}),
        ("_hermes_exe_shims", (Path("venv"),), {}),
        ("_insert_python_pin", (["uv", "pip", "install", "-e", "."],), {}),
        ("_interpreter_scripts_dir", (), {}),
        ("_load_installable_optional_extras", (), {"group": "termux-all"}),
        ("_parse_pe_machine", (Path("Hermes.exe"),), {}),
        ("_quarantine_running_hermes_exe", (Path("venv"),), {"max_attempts": 1, "failed_out": []}),
        ("_repair_broken_lazy_refresh_imports", (["uv", "pip"], ["certifi"]), {"env": {"VIRTUAL_ENV": "venv"}}),
        ("_run_install_with_heartbeat", (["uv", "pip", "install", "-e", "."],),
         {"env": {"VIRTUAL_ENV": "venv"}, "heartbeat_interval_seconds": 1}),
        ("_run_package_only_install", (["uv", "pip", "install", "-e", "."],), {"env": {"VIRTUAL_ENV": "venv"}}),
        ("_run_quarantined_install", (["uv", "pip", "install", "-e", "."],),
         {"env": {"VIRTUAL_ENV": "venv"}, "scripts_dir": Path("venv"), "strict_quarantine": True}),
        ("_run_quarantined_install", (["uv", "pip", "install", "-e", "."],), {}),
        ("_run_with_idle_timeout", (["uv", "pip", "install", "-e", "."], Path("venv")),
         {"env": {"VIRTUAL_ENV": "venv"}, "idle_timeout_seconds": 1, "indent": ""}),
        ("_self", (), {}),
        ("_verify_console_scripts_installed", (["uv", "pip"],), {"env": {"VIRTUAL_ENV": "venv"}}),
        ("_verify_core_dependencies_installed", (["uv", "pip"],), {"env": {"VIRTUAL_ENV": "venv"}, "group": "all"}),
        ("_web_ui_build_needed", (Path("web"),), {}),
        ("_windows_native_machine", (), {}),
        ("_windows_shim_in_process_chain", (), {}),
    ],
)
def test_historical_main_entrypoints_handoff_without_install_or_success_fallback(
    name, args, kwargs, historical_main, fresh_child,
):
    before = deepcopy((args, kwargs))
    with fresh_child.exits():
        getattr(historical_main, name)(*args, **kwargs)
    assert (args, kwargs) == before

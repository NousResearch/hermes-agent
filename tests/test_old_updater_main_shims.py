"""Historical main imports must not restart pre-PM updater work after a swap."""

import os
from pathlib import Path
import socket
import subprocess
import urllib.request

import pytest


@pytest.fixture
def inert_main(monkeypatch):
    # Import real current modules before guarding work: CLI startup is not a shim.
    from hermes_cli import main
    import pm

    def forbidden(*args, **kwargs):
        pytest.fail("historical main shim attempted updater work")

    for name in ("sync_venv", "ensure_environment", "build_environment"):
        monkeypatch.setattr(pm, name, forbidden)
    for name in ("Popen", "run"):
        monkeypatch.setattr(subprocess, name, forbidden)
    for name in ("system", "kill", "rename", "replace", "unlink", "mkdir"):
        monkeypatch.setattr(os, name, forbidden)
    for name in ("write_text", "write_bytes", "touch", "rename", "replace", "unlink", "mkdir"):
        monkeypatch.setattr(Path, name, forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(urllib.request, "urlretrieve", forbidden)
    return main, forbidden


def test_historical_main_data_and_skipped_probes_preserve_caller_shapes(inert_main, tmp_path):
    main, _ = inert_main
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


def test_historical_main_entrypoints_stop_before_install_or_success_fallback(
    inert_main, tmp_path, capsys,
):
    main, forbidden = inert_main
    cmd = ["uv", "pip", "install", "-e", "."]
    env = {"VIRTUAL_ENV": str(tmp_path)}
    failed = []
    calls = [
        ("_desktop_stamp_path", (), {}),
        ("_expected_windows_pe_machines", (), {}),
        ("_hermes_exe_shims", (tmp_path,), {}),
        ("_insert_python_pin", (cmd,), {}),
        ("_interpreter_scripts_dir", (), {}),
        ("_load_installable_optional_extras", (), {"group": "termux-all"}),
        ("_parse_pe_machine", (tmp_path / "Hermes.exe",), {}),
        ("_quarantine_running_hermes_exe", (tmp_path,), {"max_attempts": 1, "failed_out": failed}),
        ("_repair_broken_lazy_refresh_imports", (cmd[:2], ["certifi"]), {"env": env}),
        ("_run_install_with_heartbeat", (cmd,), {"env": env, "heartbeat_interval_seconds": 1}),
        ("_run_package_only_install", (cmd,), {"env": env}),
        ("_run_quarantined_install", (cmd,), {"env": env, "scripts_dir": tmp_path, "strict_quarantine": True}),
        ("_run_quarantined_install", (cmd,), {}),
        ("_run_with_idle_timeout", (cmd, tmp_path), {"env": env, "idle_timeout_seconds": 1, "indent": ""}),
        ("_self", (), {}),
        ("_verify_console_scripts_installed", (cmd[:2],), {"env": env}),
        ("_verify_core_dependencies_installed", (cmd[:2],), {"env": env, "group": "all"}),
        ("_web_ui_build_needed", (tmp_path / "web",), {}),
        ("_windows_native_machine", (), {}),
        ("_windows_shim_in_process_chain", (), {}),
    ]
    before_env = dict(os.environ)
    for name, args, kwargs in calls:
        shim = getattr(main, name)
        with pytest.raises(SystemExit) as exc:
            try:
                shim(*args, **kwargs)
            except Exception:
                # Historical installers catch ordinary failures to retry pip/uv.
                forbidden()
            # Returning also lets old callers claim completion or try a fallback.
            forbidden()
        assert exc.value.code == 0, name
        assert "relaunch" in capsys.readouterr().err.lower(), name
    assert cmd == ["uv", "pip", "install", "-e", "."]
    assert env == {"VIRTUAL_ENV": str(tmp_path)}
    assert failed == []
    assert dict(os.environ) == before_env

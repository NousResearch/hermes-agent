"""Windows gateway interpreter-startup encoding contract (regression #127633)."""

import json
import os
import subprocess
import sys

import pytest

from hermes_cli import gateway_windows


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("kind", ["launch", "restart"])
def test_gateway_overlay_enables_utf8_before_interpreter_start(kind, monkeypatch, tmp_path):
    import hermes_cli.gateway as gateway

    monkeypatch.setattr(gateway, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(gateway_windows, "_launcher_settings", lambda home=None: (
        sys.executable, str(tmp_path), str(tmp_path), ""))
    monkeypatch.setattr(gateway_windows, "_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(gateway_windows, "_stable_gateway_working_dir", lambda root: str(tmp_path))
    if kind == "launch":
        argv, cwd, overlay = gateway_windows._build_gateway_argv()
    else:
        argv, cwd, overlay = gateway_windows.windowless_gateway_restart_spec(
            [sys.executable, "-m", "hermes_cli.main", "gateway", "run"])
    env = os.environ.copy()
    env.pop("PYTHONUTF8", None)
    env.update(overlay)
    probe = (
        "import json,sys,subprocess; "
        "r=subprocess.run([sys.executable,'-c',"
        "\"import sys;sys.stdout.buffer.write(bytes.fromhex('e4b8ade69687'))\"],"
        "capture_output=True,text=True); "
        "print(json.dumps([sys.flags.utf8_mode,r.stdout]))"
    )
    result = subprocess.run([argv[0], "-c", probe], cwd=cwd, env=env,
                            capture_output=True, encoding="utf-8", timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [1, "中文"]


@pytest.mark.platforms("windows")
def test_update_refreshes_existing_launchers_with_utf8(monkeypatch, tmp_path):
    from hermes_cli.update_cmd_windows import _refresh_windows_gateway_launchers

    script = tmp_path / "gateway.cmd"
    script.write_text("old launcher", encoding="utf-8")
    script.with_suffix(".vbs").write_text("old launcher", encoding="utf-16")
    monkeypatch.setattr(gateway_windows, "is_installed", lambda: True)
    monkeypatch.setattr(gateway_windows, "is_task_registered", lambda: False)
    monkeypatch.setattr(gateway_windows, "_legacy_startup_entry_path", lambda: tmp_path / "absent.cmd")
    monkeypatch.setattr(gateway_windows, "get_task_script_path", lambda: script)
    monkeypatch.setattr(gateway_windows, "_launcher_settings", lambda: (
        sys.executable, str(tmp_path), str(tmp_path), "--profile work"))
    _refresh_windows_gateway_launchers()
    cmd = script.read_text(encoding="utf-8-sig")
    vbs = script.with_suffix(".vbs").read_text(encoding="utf-8")
    assert 'set "PYTHONUTF8=1"' in cmd
    assert 'env.Item("PYTHONUTF8") = "1"' in vbs
    assert 'set "PYTHONIOENCODING=utf-8"' in cmd
    assert 'env.Item("PYTHONIOENCODING") = "utf-8"' in vbs
    assert "--profile work gateway run" in cmd
    assert "--profile work gateway run" in vbs

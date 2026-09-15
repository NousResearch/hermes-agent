"""Execute generated launchers against harmless children, never a gateway."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hermes_cli import gateway_windows as gw


def _run_launcher(monkeypatch, tmp_path, child_source, *, max_failures=3):
    child = tmp_path / "child.py"
    child.write_text(child_source, encoding="utf-8")
    monkeypatch.setattr(gw, "_resolve_detached_python", lambda _: (sys.executable, Path(sys.prefix), []))
    monkeypatch.setattr(gw, "_gateway_run_argv", lambda *_: [sys.executable, str(child)])
    text = gw._build_gateway_vbs_script(
        sys.executable,
        str(tmp_path),
        str(tmp_path),
        "",
        max_failures=max_failures,
        stable_run_seconds=3600,
        restart_base_delay_ms=10,
        restart_max_delay_ms=20,
    )
    script = tmp_path / "launcher.vbs"
    script.write_bytes(text.encode("utf-8"))
    result = subprocess.run(["cscript.exe", "//nologo", str(script)], capture_output=True, timeout=20)
    count = int((tmp_path / "runs.txt").read_text(encoding="utf-8"))
    return result, count, text


@pytest.mark.skipif(os.name != "nt", reason="Windows Script Host contract")
def test_generated_launcher_stops_after_clean_child_exit(monkeypatch, tmp_path):
    result, count, _ = _run_launcher(
        monkeypatch,
        tmp_path,
        'from pathlib import Path\np=Path("runs.txt")\np.write_text("1")\n',
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert count == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows Script Host contract")
def test_generated_launcher_retries_failure_then_stops_on_success(monkeypatch, tmp_path):
    result, count, _ = _run_launcher(
        monkeypatch,
        tmp_path,
        'from pathlib import Path\np=Path("runs.txt")\nn=int(p.read_text())+1 if p.exists() else 1\np.write_text(str(n))\nraise SystemExit(7 if n == 1 else 0)\n',
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert count == 2


@pytest.mark.skipif(os.name != "nt", reason="Windows Script Host contract")
def test_generated_launcher_bounds_rapid_failure_retries(monkeypatch, tmp_path):
    result, count, text = _run_launcher(
        monkeypatch,
        tmp_path,
        'from pathlib import Path\np=Path("runs.txt")\nn=int(p.read_text())+1 if p.exists() else 1\np.write_text(str(n))\nraise SystemExit(7)\n',
        max_failures=3,
    )
    assert result.returncode == 7, result.stderr.decode(errors="replace")
    assert count == 3
    assert "WScript.Sleep CLng(delay_ms)" in text
    assert "If result = 0 Then WScript.Quit 0" in text


def test_startup_wrapper_remains_async():
    text = gw._build_startup_launcher(Path("gateway.cmd"))
    assert ", 0, False" in text

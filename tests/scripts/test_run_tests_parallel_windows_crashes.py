"""Exit-status reporting contracts for #137198; no Windows host is simulated."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import run_tests_parallel as runner


@pytest.mark.parametrize("status,name", [
    (0xC000012D, "STATUS_COMMITMENT_LIMIT"),
    (0xC0000142, "STATUS_DLL_INIT_FAILED"),
    (0xC0000135, "STATUS_DLL_NOT_FOUND"),
    (0xC0000138, "STATUS_ORDINAL_NOT_FOUND"),
    (0xC0000139, "STATUS_ENTRYPOINT_NOT_FOUND"),
    (0xC000007B, "STATUS_INVALID_IMAGE_FORMAT"),
])
@pytest.mark.parametrize("signed", [False, True])
def test_windows_startup_status_reaches_crash_report(status, name, signed, tmp_path, monkeypatch, capsys):
    rc = status - (1 << 32) if signed else status
    child = SimpleNamespace(pid=123, returncode=rc, communicate=lambda **kwargs: ("", None))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(runner, "_kill_tree", lambda *args, **kwargs: None)
    file = tmp_path / "test_probe.py"

    path, code, output, counts, _ = runner._run_one_file_once(file, [], tmp_path, 30)

    assert code == rc
    assert counts.get("crashed") == 1
    assert name in output
    assert f"0x{status:08x}" in output
    runner._print_failure_report([(path, output, counts)], tmp_path)
    report = capsys.readouterr().out
    assert "interpreter CRASHED" in report
    assert "where no tests ran" not in report


@pytest.mark.parametrize("rc,output", [
    (1, ""),
    (2, "ERROR collecting tests/test_probe.py"),
    (5, "no tests ran in 0.01s"),
    (23, ""),
    (124, "file timeout exceeded"),
    (1, "bash dofork: 0xc0000142\n1 failed in 0.01s"),
    (0xC000013A, ""),
])
def test_non_crash_exits_and_embedded_ntstatus_text_are_not_native_crashes(rc, output):
    assert runner._describe_interpreter_crash(rc, output) is None


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("status", [0xC000012D, 0xC0000142])
def test_native_windows_child_status_is_reported_as_crashed(tmp_path, monkeypatch, status):
    """Exercise real DWORD exit transport, without exhausting host resources."""
    probe = tmp_path / "test_startup_exit.py"
    probe.write_text(
        f"import ctypes\nctypes.windll.kernel32.ExitProcess(ctypes.c_uint32({status}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    script = Path(runner.__file__).resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--paths", str(probe), "-j", "1",
         "--file-retries", "0", "--file-timeout", "30"],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=60, check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 file CRASHED" in result.stdout
    assert f"0x{status:08x}" in result.stdout
    assert "where no tests ran" not in result.stdout

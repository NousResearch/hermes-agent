"""Windows Store python alias detection (#129102, #129121).

``windows_store_python_stubs`` takes resolved paths plus a filesystem size
probe (mocked here; live on Windows). Host-independent except the two
platform-gated tests: silence off Windows and the live ``which`` receipt.
"""
import pytest

from hermes_cli import doctor_platform
from hermes_cli.doctor_report import Finding


def test_both_aliases_reported(monkeypatch):
    import os

    monkeypatch.setattr(os.path, "getsize", lambda _p: 0)
    assert doctor_platform.windows_store_python_stubs(
        r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\python.exe",
        r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\python3.exe",
    ) == ["python", "python3"]


def test_msys_spelling_counts_as_alias(monkeypatch):
    import os

    monkeypatch.setattr(os.path, "getsize", lambda _p: 0)
    assert doctor_platform.windows_store_python_stubs(
        r"C:\Python314\python.exe",
        "/c/Users/u/AppData/Local/Microsoft/WindowsApps/python3",
    ) == ["python3"]


def test_nonzero_windowsapps_shim_is_not_reported(monkeypatch):
    """#129121: Python Install Manager-style shim under WindowsApps has
    non-zero size — it is a real interpreter and must not be flagged."""
    import os

    monkeypatch.setattr(os.path, "getsize", lambda _p: 4096)
    assert doctor_platform.windows_store_python_stubs(
        r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\python.exe",
        r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\python3.exe",
    ) == []


def test_real_interpreters_are_not_reported():
    assert doctor_platform.windows_store_python_stubs(
        r"C:\Python314\python.exe",
        r"C:\Python314\python3.exe",
    ) == []
    assert doctor_platform.windows_store_python_stubs(None, None) == []


@pytest.mark.platforms("not windows")
def test_check_is_silent_off_windows(capsys):
    """The win32-gated check stays silent off Windows — skipped on win32
    where the live probe runs (see the windows-only live test below)."""
    f = Finding()
    doctor_platform._check_windows_store_python_aliases(f)
    assert capsys.readouterr().out == ""
    assert f.manual_issues == []


@pytest.mark.platforms("windows")
def test_live_which_results_agree_with_alias_helper():
    """Live Windows receipt: the helper's verdict on the real ``which``
    results matches a direct ``is_windows_app_alias`` probe, and the check
    never raises on the live host."""
    import shutil

    from pm.shell import is_windows_app_alias

    python_path = shutil.which("python")
    python3_path = shutil.which("python3")
    expected = [
        name for name, path in (("python", python_path), ("python3", python3_path))
        if path and is_windows_app_alias(path)
    ]
    assert doctor_platform.windows_store_python_stubs(python_path, python3_path) == expected
    doctor_platform._check_windows_store_python_aliases(Finding())

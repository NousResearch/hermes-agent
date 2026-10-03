"""Doctor flags `python`/`python3` Microsoft Store alias stubs (#129102)."""

from hermes_cli import doctor_platform

STUB_PY = r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\python.exe"
STUB_PY3 = r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\python3.exe"


def _which_stub(name):
    return {"python": STUB_PY, "python3": STUB_PY3}.get(name)


def _which_real(name):
    return {"python": r"C:\Python311\python.exe", "python3": None}.get(name)


def test_windows_store_alias_pythons_classifies_stubs_vs_real():
    assert doctor_platform._windows_store_alias_pythons(_which_stub) == ["python", "python3"]
    assert doctor_platform._windows_store_alias_pythons(_which_real) == []


def test_doctor_python_environment_flags_store_alias_python(monkeypatch, capsys):
    monkeypatch.setattr(doctor_platform.shutil, "which", _which_stub)
    finding = doctor_platform._check_python_environment(False)
    out = capsys.readouterr().out
    assert "Microsoft Store" in out
    assert finding.manual_issues, "store-alias warning must surface as a manual issue"


def test_doctor_python_environment_quiet_without_store_alias(monkeypatch, capsys):
    monkeypatch.setattr(doctor_platform.shutil, "which", _which_real)
    doctor_platform._check_python_environment(False)
    assert "Microsoft Store" not in capsys.readouterr().out

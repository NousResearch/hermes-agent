"""Doctor flags the `python3` Microsoft Store alias stub (#129102/#129156).

Only the `python3` family under WindowsApps is an always-stub POSIX-compat
alias; a properly Store-installed Python legitimately lives at
`...\\WindowsApps\\python.exe`, so `python.exe` there must never warn by
path alone.
"""

from hermes_cli import doctor_platform

STUB_PY = r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\python.exe"
STUB_PY3 = r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\python3.exe"


def _which_both_aliases(name):
    return {"python": STUB_PY, "python3": STUB_PY3}.get(name)


def _which_store_python_plus_stub_py3(name):
    # Store-installed interpreter: `python` legitimately resolves under
    # WindowsApps, while `python3` remains the always-stub compat alias.
    return {"python": STUB_PY, "python3": STUB_PY3}.get(name)


def _which_real(name):
    return {"python": r"C:\Python311\python.exe", "python3": None}.get(name)


def test_windows_store_alias_pythons_flags_only_python3():
    # Both names resolve under WindowsApps, yet only python3 is a stub.
    assert doctor_platform._windows_store_alias_pythons(_which_both_aliases) == ["python3"]
    assert doctor_platform._windows_store_alias_pythons(_which_real) == []


def test_windows_store_alias_pythons_spares_store_installed_python():
    # `python.exe` under WindowsApps alone (Store-installed) never flags.
    assert doctor_platform._windows_store_alias_pythons(lambda name: STUB_PY if name == "python" else None) == []


def test_doctor_python_environment_flags_store_alias_python3(monkeypatch, capsys):
    monkeypatch.setattr(doctor_platform.shutil, "which", _which_store_python_plus_stub_py3)
    finding = doctor_platform._check_python_environment(False)
    out = capsys.readouterr().out
    assert "Microsoft Store" in out
    assert "`python3`" in out
    assert finding.manual_issues, "store-alias warning must surface as a manual issue"


def test_doctor_python_environment_quiet_without_store_alias(monkeypatch, capsys):
    monkeypatch.setattr(doctor_platform.shutil, "which", _which_real)
    doctor_platform._check_python_environment(False)
    assert "Microsoft Store" not in capsys.readouterr().out


def test_doctor_python_environment_quiet_for_store_installed_python(monkeypatch, capsys):
    # P1 regression: a Store-installed `python.exe` under WindowsApps with no
    # `python3` on PATH must not warn — there IS a real interpreter behind it.
    monkeypatch.setattr(
        doctor_platform.shutil, "which",
        lambda name: STUB_PY if name == "python" else None)
    doctor_platform._check_python_environment(False)
    assert "Microsoft Store" not in capsys.readouterr().out

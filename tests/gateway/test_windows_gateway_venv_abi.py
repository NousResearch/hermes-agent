"""Windows gateway venv selection: use the PM generation, never a stale venv.

Regression test for https://github.com/NousResearch/hermes-agent/issues/122183:
on PM-managed installs the gateway runs under a newer Python while
``_ensure_windows_gateway_venv_imports`` forced the legacy in-tree venv first,
so binary extensions (pydantic_core) failed to import.
"""

import sys

import hermes_cli._launchers as launchers
import pm.environments as env

import gateway.run as run


def _write_cfg(path, version, key="version_info"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyvenv.cfg").write_text(
        "home = /nonexistent\nimplementation = CPython\n%s = %s\n" % (key, version),
        encoding="utf-8",
    )
    (path / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)


def _other_version():
    return "2.7.18" if sys.version_info[:2] != (2, 7) else "3.99.0"


def test_abi_matches_current_interpreter(tmp_path):
    current = "%d.%d.0" % sys.version_info[:2]
    match = tmp_path / "match"
    _write_cfg(match, current)
    assert run._venv_abi_matches(match) is True


def test_abi_rejects_other_interpreter(tmp_path):
    other = tmp_path / "other"
    _write_cfg(other, _other_version())
    assert run._venv_abi_matches(other) is False


def test_abi_stdlib_version_key(tmp_path):
    """CPython's `python -m venv` writes `version`, not `version_info`."""
    current = "%d.%d.%d" % sys.version_info[:3]
    match = tmp_path / "stdlib-match"
    _write_cfg(match, current, key="version")
    assert run._venv_abi_matches(match) is True
    mismatch = tmp_path / "stdlib-mismatch"
    _write_cfg(mismatch, _other_version(), key="version")
    assert run._venv_abi_matches(mismatch) is False


def test_abi_missing_or_broken_cfg_keeps_legacy_behavior(tmp_path):
    assert run._venv_abi_matches(tmp_path / "absent") is True
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "pyvenv.cfg").write_text("home = /nonexistent\n", encoding="utf-8")
    assert run._venv_abi_matches(broken) is True
    garbage = tmp_path / "garbage"
    garbage.mkdir()
    (garbage / "pyvenv.cfg").write_text("version_info = nonsense\n", encoding="utf-8")
    assert run._venv_abi_matches(garbage) is True


def _force_non_pm(monkeypatch):
    monkeypatch.setattr(run.sys, "platform", "win32")
    monkeypatch.setattr(launchers, "resolve_store_python", lambda root: None)


def test_abi_sort_demotes_mismatch(tmp_path, monkeypatch):
    """A venv built for another Python sorts behind the matching one."""
    _force_non_pm(monkeypatch)
    bad = tmp_path / "bad"
    good = tmp_path / "good"
    _write_cfg(bad, _other_version())
    _write_cfg(good, "%d.%d.0" % sys.version_info[:2])
    assert run._order_venv_candidates([bad, good]) == [good, bad]
    assert run._order_venv_candidates([good, bad]) == [good, bad]


def test_non_pm_venv_still_added(tmp_path, monkeypatch):
    """The legacy path keeps working when no PM runtime is present."""
    _force_non_pm(monkeypatch)
    venv = tmp_path / "venv"
    _write_cfg(venv, "%d.%d.0" % sys.version_info[:2])
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))
    old_path = list(sys.path)
    old_pythonpath = run.os.environ.get("PYTHONPATH")
    try:
        run.os.environ.pop("PYTHONPATH", None)
        run._ensure_windows_gateway_venv_imports()
        entries = [entry.lower() for entry in sys.path]
        assert str(venv / "Lib" / "site-packages").lower() in entries
    finally:
        sys.path[:] = old_path
        if old_pythonpath is None:
            run.os.environ.pop("PYTHONPATH", None)
        else:
            run.os.environ["PYTHONPATH"] = old_pythonpath


def test_pm_already_on_selected_env_touches_nothing(tmp_path, monkeypatch):
    """A gateway booted on the committed generation keeps its environment."""
    monkeypatch.setattr(run.sys, "platform", "win32")
    monkeypatch.setattr(launchers, "resolve_store_python", lambda root: tmp_path / "python.exe")
    monkeypatch.setattr(env, "running_from_selected_environment", lambda root: True)
    old_path = list(sys.path)
    old_venv = run.os.environ.get("VIRTUAL_ENV")
    old_pythonpath = run.os.environ.get("PYTHONPATH")
    try:
        run._ensure_windows_gateway_venv_imports()
        assert sys.path == old_path
        assert run.os.environ.get("VIRTUAL_ENV") == old_venv
        assert run.os.environ.get("PYTHONPATH") == old_pythonpath
    finally:
        sys.path[:] = old_path


def test_pm_uses_selected_venv_only(tmp_path, monkeypatch):
    """On a PM install the stale venv never reaches sys.path."""
    selected = tmp_path / "selected"
    stale = tmp_path / "stale"
    _write_cfg(selected, "%d.%d.0" % sys.version_info[:2])
    _write_cfg(stale, _other_version())
    monkeypatch.setattr(run.sys, "platform", "win32")
    monkeypatch.setattr(launchers, "resolve_store_python", lambda root: tmp_path / "python.exe")
    monkeypatch.setattr(env, "running_from_selected_environment", lambda root: False)
    monkeypatch.setattr(env, "selected_venv", lambda root: selected)
    monkeypatch.setenv("VIRTUAL_ENV", str(stale))
    old_path = list(sys.path)
    old_pythonpath = run.os.environ.get("PYTHONPATH")
    try:
        run.os.environ.pop("PYTHONPATH", None)
        run._ensure_windows_gateway_venv_imports()
        entries = [entry.lower() for entry in sys.path]
        assert str(selected / "Lib" / "site-packages").lower() in entries
        assert not any(str(stale).lower() in entry for entry in entries)
    finally:
        sys.path[:] = old_path
        if old_pythonpath is None:
            run.os.environ.pop("PYTHONPATH", None)
        else:
            run.os.environ["PYTHONPATH"] = old_pythonpath

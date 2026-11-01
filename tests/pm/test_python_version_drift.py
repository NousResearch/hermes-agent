"""#85356: a venv built with the wrong Python minor must read as a wrong-Python
venv, not a mystery install error.

``pinned_python_version`` is the interpreter minor PM's lock pins;
``venv_python_version`` is what the venv actually holds. A user-built venv on
the wrong minor imports pure-Python modules fine and dies on the first C
extension (``import _ssl`` → os error 32) — the report's undiagnosable chain.
The drift report must name the mismatch and the remedy.
"""
from __future__ import annotations

import json
import importlib

import pytest


def _write_facts(path, packages):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"packages": packages}), encoding="utf-8")


class _FakeVenvPackage:
    """Stands in for the ``venv`` package so drift() enters its venv branch."""

    optional = False
    internal = False

    def missing_reason(self, target):
        return None


class _FakeLockfile:
    def __init__(self):
        self._names = ["python"]

    def names(self):
        return list(self._names)


def _install_drift_fixture(tmp_path, monkeypatch, venv_cfg_text):
    engine = importlib.import_module("pm.install")
    environments = importlib.import_module("pm.environments")
    pm_paths = importlib.import_module("pm.paths")
    registry = importlib.import_module("pm.registry")

    monkeypatch.setattr(pm_paths, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(pm_paths, "facts_path", lambda: tmp_path / "facts.json")
    monkeypatch.setattr(pm_paths, "runtime_facts_path", lambda: tmp_path / "runtime-facts.json")
    monkeypatch.setattr(engine, "_lockfile", lambda: _FakeLockfile())
    _write_facts(tmp_path / "runtime-facts.json", {"venv": {"version": "recorded"}})
    monkeypatch.setattr(engine, "venv_is_current", lambda **_: True)
    monkeypatch.setattr(registry, "get_package", lambda name: _FakeVenvPackage())

    venv_dir = tmp_path / "venv"
    venv_dir.mkdir(exist_ok=True)
    if venv_cfg_text is not None:
        (venv_dir / "pyvenv.cfg").write_text(venv_cfg_text, encoding="utf-8")
    monkeypatch.setattr(environments, "selected_venv", lambda _root: venv_dir)

    return engine, environments


def test_pinned_python_version_reads_the_committed_pin() -> None:
    from pm.environments import pinned_python_version
    from pm.paths import repo_root

    pinned = pinned_python_version(repo_root())

    # Contract, not snapshot: the committed lock pins a parseable major.minor.
    assert pinned is not None, "the committed pm/lock.json must pin a python package"
    assert isinstance(pinned, tuple) and len(pinned) == 2 and pinned[0] > 0


def test_pinned_python_version_parses_the_lock_row_format(monkeypatch: pytest.MonkeyPatch) -> None:
    from pm.environments import pinned_python_version

    import pm.lock

    class FakeLockfile:
        def __init__(self, _path):
            pass

        def version(self, name):
            return "3.11.9+20260815" if name == "python" else None

    monkeypatch.setattr(pm.lock, "Lockfile", FakeLockfile)

    assert pinned_python_version("/any/project") == (3, 11)


def test_pinned_python_version_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    from pm.environments import pinned_python_version

    import pm.lock

    class ExplodingLockfile:
        def __init__(self, _path):
            raise RuntimeError("corrupt lock")

    monkeypatch.setattr(pm.lock, "Lockfile", ExplodingLockfile)

    # A corrupt or missing pin is a no-op, never a boot failure.
    assert pinned_python_version("/any/project") is None


def test_drift_names_a_wrong_python_venv_with_the_rebuild_hint(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The report's chain: a venv built for 3.11 against a 3.14 pin."""
    engine, environments = _install_drift_fixture(
        tmp_path, monkeypatch, "home = /tmp/host-python3.11\nversion = 3.11.9\n"
    )
    monkeypatch.setattr(environments, "pinned_python_version", lambda _root: (3, 14))

    problems = engine.drift(include_venv=True)

    assert "venv" in problems
    assert "built for Python 3.11" in problems["venv"]
    assert "pins" in problems["venv"] and "Python 3.14" in problems["venv"]
    assert "hermes pm install" in problems["venv"], "the remedy must name the rebuild command"


def test_drift_stays_silent_when_the_venv_matches_the_pin(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, environments = _install_drift_fixture(
        tmp_path, monkeypatch, "home = /tmp/host-python3.14\nversion = 3.14.7\n"
    )
    monkeypatch.setattr(environments, "pinned_python_version", lambda _root: (3, 14))

    assert engine.drift(include_venv=True) == {}


def test_drift_never_reports_when_either_version_is_unknown(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No lock pin (sealed/legacy payload) or no readable pyvenv.cfg: the
    wrong-Python check must degrade to silence, not guess."""
    engine, environments = _install_drift_fixture(tmp_path, monkeypatch, None)
    monkeypatch.setattr(environments, "pinned_python_version", lambda _root: (3, 14))

    # Unknown actual (no pyvenv.cfg): silent.
    assert engine.drift(include_venv=True) == {}

    # Unknown pin: silent too.
    monkeypatch.setattr(environments, "pinned_python_version", lambda _root: None)

    assert engine.drift(include_venv=True) == {}

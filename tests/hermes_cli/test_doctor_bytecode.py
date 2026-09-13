"""Behavior tests for stale bytecode diagnosis and repair."""

from __future__ import annotations

import os
import py_compile
from pathlib import Path

from hermes_cli import doctor as doctor_mod
from hermes_cli.doctor_bytecode import _check_stale_bytecode, _stale_bytecode_dirs


def _compile_cache(source: Path, *, stale: bool) -> Path:
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    cache = Path(py_compile.compile(str(source), doraise=True))
    if stale:
        newer = cache.stat().st_mtime_ns + 2_000_000_000
        os.utime(source, ns=(newer, newer))
    return cache.parent


def test_finds_stale_caches_across_checkout_and_profile_home(tmp_path):
    project = tmp_path / "checkout"
    home = tmp_path / "home"
    project_cache = _compile_cache(project / "hermes_cli" / "command.py", stale=True)
    profile_cache = _compile_cache(
        home / "profiles" / "researcher" / "plugins" / "sample" / "plugin.py",
        stale=True,
    )
    fresh_cache = _compile_cache(home / "skills" / "helper.py", stale=False)

    assert set(_stale_bytecode_dirs((project, home))) == {project_cache, profile_cache}
    assert fresh_cache.exists()


def test_doctor_fix_removes_only_stale_caches_and_requests_restart(
    monkeypatch, tmp_path, capsys
):
    project = tmp_path / "checkout"
    home = tmp_path / "home"
    stale_cache = _compile_cache(project / "gateway" / "worker.py", stale=True)
    fresh_cache = _compile_cache(home / "plugins" / "sample" / "plugin.py", stale=False)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)

    finding = _check_stale_bytecode(True)

    assert finding.fixed == 1
    assert finding.issues == []
    assert not stale_cache.exists()
    assert fresh_cache.exists()
    assert "hermes gateway restart" in capsys.readouterr().out
"""Windows runtime cutover repoints ``pyvenv.cfg`` instead of renaming the live venv (#93032).

Windows refuses to rename a directory while any handle is open under it (a process cwd, an open
file, a sync client); the POSIX park-rename then fails with ``WinError 5``. Replacing the venv's
``pyvenv.cfg`` redirects every fresh process to the new generation with no directory rename.
"""

from __future__ import annotations

import subprocess
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.sqlite_runtime import SQLiteRuntimeInfo


def _python(venv_dir: Path) -> Path:
    return venv_dir / "Scripts" / "python.exe"


def _info(exe: Path, version: tuple[int, int, int], sqlite: str) -> SQLiteRuntimeInfo:
    major, minor, patch = (int(x) for x in sqlite.split("."))
    return SQLiteRuntimeInfo(
        executable=exe, base_prefix=exe.parent, python_version=version,
        sqlite_version=(major, minor, patch),
        sqlite_version_string=sqlite, sqlite_source_id=sqlite)


@pytest.mark.windows_only
def test_runtime_config_cutover_repoints_a_running_venv(tmp_path: Path, monkeypatch) -> None:
    from hermes_cli import managed_uv

    live = tmp_path / "venv"
    candidate = tmp_path / "candidate"
    venv.EnvBuilder(with_pip=False).create(live)
    venv.EnvBuilder(with_pip=False).create(candidate)
    candidate_config = candidate / "pyvenv.cfg"
    replacement = candidate_config.read_bytes() + b"runtime-cutover = candidate\n"
    candidate_config.write_bytes(replacement)
    info = SimpleNamespace(sqlite_version_string="3.53.1")
    monkeypatch.setattr(managed_uv, "_smoke_candidate_venv", lambda target: (True, "", info))
    same = (3, 11, 15)
    current = _info(_python(live), same, "3.50.4")
    fixed = _info(candidate / "python.exe", same, "3.53.1")

    # A holder the updater cannot see or evict: a process whose cwd is inside the venv. (A process
    # merely executing from the venv does not block the rename; proven live on windows-latest.)
    # The child opens its cwd handle during its own startup, so wait for it to report ready.
    process = subprocess.Popen(
        [str(_python(live)), "-c", "import time; print('ready', flush=True); time.sleep(30)"],
        cwd=str(live / "Scripts"), stdout=subprocess.PIPE, text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        assert process.stdout.readline().strip() == "ready"
        # The symptom: the directory rename the POSIX cutover relies on fails with WinError 5.
        with pytest.raises(OSError):
            managed_uv._rename_with_retry(live, live.with_name("venv.parked"))
        assert live.is_dir()

        ok, generation_in_use, final_info, detail = managed_uv._cut_over_windows_runtime_config(
            candidate, live=live, current=current, candidate_info=fixed)
        assert process.poll() is None
        assert (ok, generation_in_use, final_info, detail) == (True, True, info, "")
        assert (live / "pyvenv.cfg").read_bytes() == replacement
        probe = subprocess.run(
            [str(_python(live)), "-I", "-c", "print('repointed')"],
            capture_output=True, text=True, check=False, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        assert probe.returncode == 0 and probe.stdout.strip() == "repointed"
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_runtime_config_cutover_rolls_back_failed_live_smoke(tmp_path: Path, monkeypatch) -> None:
    from hermes_cli import managed_uv

    live = tmp_path / "venv"
    candidate = tmp_path / "candidate"
    live.mkdir()
    candidate.mkdir()
    original = b"home = old-runtime\n"
    (live / "pyvenv.cfg").write_bytes(original)
    (candidate / "pyvenv.cfg").write_bytes(b"home = new-runtime\n")
    monkeypatch.setattr(
        managed_uv, "_smoke_candidate_venv", lambda target: (False, "core import smoke failed", None))
    same = (3, 11, 15)

    ok, generation_in_use, info, detail = managed_uv._cut_over_windows_runtime_config(
        candidate, live=live,
        current=_info(_python(live), same, "3.50.4"),
        candidate_info=_info(candidate / "python.exe", same, "3.53.1"))

    assert (ok, generation_in_use, info) == (False, False, None)
    assert detail == "post-cutover smoke failed: core import smoke failed"
    assert (live / "pyvenv.cfg").read_bytes() == original


def test_runtime_config_cutover_refuses_a_minor_line_jump(tmp_path: Path, monkeypatch) -> None:
    """The live venv keeps its cp311 site-packages, so a 3.12 generation must never be pointed at it."""
    from hermes_cli import managed_uv

    live = tmp_path / "venv"
    candidate = tmp_path / "candidate"
    live.mkdir()
    candidate.mkdir()
    original = b"home = old-runtime\n"
    (live / "pyvenv.cfg").write_bytes(original)
    (candidate / "pyvenv.cfg").write_bytes(b"home = new-runtime\n")
    smoked = []
    monkeypatch.setattr(managed_uv, "_smoke_candidate_venv", lambda target: smoked.append(target))

    ok, generation_in_use, info, detail = managed_uv._cut_over_windows_runtime_config(
        candidate, live=live,
        current=_info(_python(live), (3, 11, 15), "3.50.4"),
        candidate_info=_info(candidate / "python.exe", (3, 12, 4), "3.53.1"))

    assert (ok, generation_in_use, info) == (False, False, None)
    assert "3.12" in detail and "3.11" in detail
    assert (live / "pyvenv.cfg").read_bytes() == original
    assert smoked == []

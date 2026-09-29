"""The runtime's own interpreter/venv is not agent-deletable (#58748).

A Hermes session asked to clean up "older Pythons" removed the base
interpreter its own venv pointed at; the next boot died with ``uv trampoline
failed to spawn Python child process``. These tests pin both defense layers:

* the approval floor (``_floor_block``) must hard-block shell commands that
  delete the running interpreter/venv/base — even under yolo;
* the file-safety write classifier must deny writes/deletes to them.

The ``_protected_snapshot`` cache is keyed on (executable, prefix), so tests
swap ``sys`` attributes and let the cache key change with them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from agent import runtime_self_protection as rsp


@pytest.fixture
def fake_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A venv whose pyvenv.cfg points at a uv-managed base interpreter."""
    venv = tmp_path / "hermes-agent" / "venv"
    venv.mkdir(parents=True)
    exe_dir = venv / ("Scripts" if sys.platform == "win32" else "bin")
    exe_dir.mkdir()
    exe = exe_dir / ("python.exe" if sys.platform == "win32" else "python")
    exe.write_text("", encoding="utf-8")

    uv_base = tmp_path / "uv" / "python" / "cpython-3.11.9-windows-x86_64-none"
    uv_base.mkdir(parents=True)
    base_exe_dir = uv_base / ("Scripts" if sys.platform == "win32" else "bin")
    base_exe_dir.mkdir()
    (base_exe_dir / ("python.exe" if sys.platform == "win32" else "python")).write_text("", encoding="utf-8")

    home_dir = str(base_exe_dir)
    (venv / "pyvenv.cfg").write_text(
        f"home = {home_dir}\nversion = 3.11.9\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setattr(sys, "prefix", str(venv))
    rsp._protected_snapshot.cache_clear()
    yield {"exe": str(exe), "venv": str(venv), "uv_base": str(uv_base), "base_dir": home_dir}
    rsp._protected_snapshot.cache_clear()


def test_pyvenv_home_is_parsed(fake_runtime):
    assert rsp._pyvenv_home(fake_runtime["venv"]) == fake_runtime["base_dir"]


def test_rm_of_running_interpreter_is_detected(fake_runtime):
    assert rsp.command_deletes_runtime(f'rm "{fake_runtime["exe"]}"') is not None


def test_rm_rf_of_own_venv_is_detected(fake_runtime):
    assert rsp.command_deletes_runtime(f"rm -rf {fake_runtime['venv']}") is not None


def test_rm_of_base_interpreter_is_detected(fake_runtime):
    base_exe = os.path.join(fake_runtime["base_dir"], os.listdir(fake_runtime["base_dir"])[0])
    assert rsp.command_deletes_runtime(f"sudo rm -f '{base_exe}'") is not None


def test_rm_of_whole_uv_install_dir_is_detected(fake_runtime):
    assert rsp.command_deletes_runtime(f"rm -rf {fake_runtime['uv_base']}") is not None


def test_uv_python_uninstall_of_running_version_is_detected(fake_runtime):
    assert rsp.command_deletes_runtime("uv python uninstall 3.11") is not None


def test_uv_python_uninstall_all_is_detected(fake_runtime):
    assert rsp.command_deletes_runtime("uv python uninstall --all") is not None


def test_uv_python_uninstall_of_other_version_is_allowed(fake_runtime):
    assert rsp.command_deletes_runtime("uv python uninstall 3.9") is None


def test_find_delete_over_own_venv_is_detected(fake_runtime):
    # Both are deletes under the protected root: an unfiltered -delete removes
    # the interpreter itself, and a filtered one still deletes protected files.
    assert rsp.command_deletes_runtime(f"find '{fake_runtime['venv']}' -delete") is not None
    assert rsp.command_deletes_runtime(f"find '{fake_runtime['venv']}' -name __pycache__ -delete") is not None
    assert rsp.command_deletes_runtime(f"find /tmp -name __pycache__ -delete") is None


def test_rm_of_unrelated_venv_is_allowed(fake_runtime, tmp_path):
    other = tmp_path / "project" / ".venv"
    other.mkdir(parents=True)
    assert rsp.command_deletes_runtime(f"rm -rf {other}") is None


def test_windows_del_and_powershell_spellings(fake_runtime, monkeypatch):
    with mock.patch.object(os, "name", "nt"):
        exe = fake_runtime["exe"]
        assert rsp.command_deletes_runtime(f'del "{exe}"') is not None
        assert rsp.command_deletes_runtime(f'Remove-Item -Recurse -Force "{exe}"') is not None
        assert rsp.command_deletes_runtime(f"rd /s /q {fake_runtime['venv']}") is not None


def test_mkdir_over_protected_path_not_flagged_as_delete(fake_runtime):
    # mkdir touches the venv dir but deletes nothing — command layer says fine
    # (the file-safety layer still denies the write).
    assert rsp.command_deletes_runtime(f"mkdir -p {fake_runtime['venv']}") is None


def test_is_protected_path_covers_venv_children_and_base(fake_runtime):
    site = os.path.join(fake_runtime["venv"], "lib", "site-packages", "x.py")
    assert rsp.is_protected_path(site) is not None
    assert rsp.is_protected_path(fake_runtime["base_dir"]) is not None
    assert rsp.is_protected_path("/tmp/scratch.txt") is None


def test_file_safety_denies_write_to_running_interpreter(fake_runtime):
    from agent.file_safety import _classify_write_denial

    assert _classify_write_denial(fake_runtime["exe"]) == "credential"
    assert _classify_write_denial(os.path.join(fake_runtime["venv"], "pyvenv.cfg")) == "credential"


def test_file_safety_allows_unrelated_paths(fake_runtime, tmp_path):
    from agent.file_safety import _classify_write_denial

    scratch = tmp_path / "scratch.txt"
    assert _classify_write_denial(str(scratch)) is None


def test_approval_floor_blocks_runtime_delete_even_under_yolo(fake_runtime, monkeypatch):
    from tools import approval

    monkeypatch.setattr(approval, "_yolo_active", lambda: True)
    result = approval._floor_block(f'rm -rf "{fake_runtime["venv"]}"')
    assert result is not None
    assert result.get("approved") is False


def test_check_all_guards_blocks_runtime_delete(fake_runtime, monkeypatch):
    from tools import approval

    # Outside CLI/gateway/ask contexts with yolo on, the floor is the only
    # thing standing; it must still block.
    monkeypatch.setattr(approval, "_yolo_active", lambda: True)
    result = approval.check_all_command_guards(f'rm "{fake_runtime["exe"]}"', "local")
    assert result.get("approved") is False


def test_normal_command_unaffected(fake_runtime, monkeypatch):
    from tools import approval

    monkeypatch.setattr(approval, "_yolo_active", lambda: True)
    assert approval.check_all_command_guards("ls /tmp", "local").get("approved") is True

"""Windows launcher fallback ownership contracts."""

import io
from pathlib import Path
from zipfile import ZipFile

from hermes_cli import _launchers


def _cmd_fallback_fixture(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    root = tmp_path / "hermes-agent"
    out_dir = tmp_path / "bin"
    root.mkdir()
    out_dir.mkdir()
    store_python = tmp_path / "store" / "python.exe"
    store_python.parent.mkdir()
    store_python.write_bytes(b"MZ store python")

    exe = out_dir / "hermes.exe"
    cmd = out_dir / "hermes.cmd"

    def mint_cmd(*_args, **_kwargs):
        cmd.write_text("@echo off\r\n", encoding="utf-8")
        return cmd

    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _root: store_python)
    monkeypatch.setattr(_launchers, "mint_launcher", mint_cmd)
    return root, exe, cmd


def _stage_cmd_fallback(root: Path, exe: Path, cmd: Path) -> None:
    assert _launchers.stage_launcher("hermes", root, exe.parent) == cmd


def _launcher_exe_bytes(root: Path) -> bytes:
    payload = io.BytesIO()
    with ZipFile(payload, "w") as archive:
        archive.writestr("__main__.py", _launchers._launcher_script("hermes", root, None))
    return b"MZ old loader\n#!missing-python.exe -I\n" + payload.getvalue()


def test_cmd_fallback_preserves_unowned_executable(tmp_path, monkeypatch):
    root, exe, cmd = _cmd_fallback_fixture(tmp_path, monkeypatch)
    foreign_root = tmp_path / "another-install"
    foreign_root.mkdir()
    foreign = _launcher_exe_bytes(foreign_root)
    exe.write_bytes(foreign)

    _stage_cmd_fallback(root, exe, cmd)

    assert exe.read_bytes() == foreign


def test_cmd_fallback_retires_owned_stale_executable(tmp_path, monkeypatch):
    root, exe, cmd = _cmd_fallback_fixture(tmp_path, monkeypatch)
    exe.write_bytes(_launcher_exe_bytes(root))

    _stage_cmd_fallback(root, exe, cmd)

    assert not exe.exists()

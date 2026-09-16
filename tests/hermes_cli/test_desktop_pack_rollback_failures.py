"""Rollback must preserve both generations when Windows refuses a rename."""

from pathlib import Path

import pytest

from hermes_cli import main_desktop


@pytest.mark.parametrize("boundary", ["park", "promote"])
def test_failed_rollback_keeps_both_generations(tmp_path, monkeypatch, boundary):
    live = tmp_path / "win-unpacked"
    backup = tmp_path / "win-unpacked.bak"
    live.mkdir()
    backup.mkdir()
    (live / "Hermes.exe").write_bytes(b"candidate")
    (backup / "Hermes.exe").write_bytes(b"previous")
    monkeypatch.setattr(main_desktop, "_desktop_exe_integrity_error", lambda _: None)
    real_rename = Path.rename

    def rename(source, destination):
        if source == (live if boundary == "park" else backup):
            raise PermissionError(f"fixture: {boundary} denied")
        return real_rename(source, destination)

    monkeypatch.setattr(Path, "rename", rename)
    assert main_desktop._rollback_desktop_from_backup(live / "Hermes.exe") is None
    assert (live / "Hermes.exe").read_bytes() == b"candidate"
    assert (backup / "Hermes.exe").read_bytes() == b"previous"

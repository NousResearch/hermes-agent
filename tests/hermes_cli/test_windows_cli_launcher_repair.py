"""Regression coverage for Windows' dedicated Hermes launcher directory."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import _install_repair as install_repair
from hermes_cli import update_cmd


def _make_windows_launchers(tmp_path):
    root = tmp_path / "hermes-agent"
    scripts = root / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "hermes.exe").write_bytes(b"MZ-hermes")
    (scripts / "hermes-acp.exe").write_bytes(b"MZ-hermes-acp")
    return root


def _force_windows(monkeypatch, root):
    fake_main = SimpleNamespace(
        sys=SimpleNamespace(platform="win32"),
        PROJECT_ROOT=root,
    )
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_main)
    monkeypatch.setattr(install_repair, "_is_windows", lambda: True)


def test_update_restores_missing_dedicated_launchers(tmp_path, monkeypatch):
    root = _make_windows_launchers(tmp_path)
    _force_windows(monkeypatch, root)

    update_cmd._ensure_acp_launcher()

    assert (root / "bin" / "hermes.exe").read_bytes() == b"MZ-hermes"
    assert (root / "bin" / "hermes-acp.exe").read_bytes() == b"MZ-hermes-acp"


def test_update_does_not_overwrite_running_launcher(tmp_path, monkeypatch):
    root = _make_windows_launchers(tmp_path)
    bin_dir = root / "bin"
    bin_dir.mkdir()
    (bin_dir / "hermes.exe").write_bytes(b"MZ-running")
    _force_windows(monkeypatch, root)

    update_cmd._ensure_acp_launcher()

    assert (bin_dir / "hermes.exe").read_bytes() == b"MZ-running"
    assert (bin_dir / "hermes-acp.exe").read_bytes() == b"MZ-hermes-acp"


def test_missing_required_source_is_visible_and_does_not_create_bin(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "hermes-agent"
    (root / "venv" / "Scripts").mkdir(parents=True)
    _force_windows(monkeypatch, root)

    update_cmd._ensure_acp_launcher()

    assert "Could not restore Windows command launchers" in capsys.readouterr().out
    assert not (root / "bin").exists()


def test_sync_helper_is_noop_off_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(install_repair, "_is_windows", lambda: False)

    assert install_repair._sync_windows_cli_launchers(tmp_path) == []


def test_sync_helper_requires_hermes_source(tmp_path, monkeypatch):
    root = tmp_path / "hermes-agent"
    (root / "venv" / "Scripts").mkdir(parents=True)
    _force_windows(monkeypatch, root)

    with pytest.raises(FileNotFoundError, match="required Hermes launcher"):
        install_repair._sync_windows_cli_launchers(root)

    assert not (root / "bin").exists()

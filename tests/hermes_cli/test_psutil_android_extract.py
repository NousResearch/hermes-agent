"""Regression tests for the Android psutil compatibility installer."""

from __future__ import annotations

import io
import shutil
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.psutil_android import (
    MARKER,
    REPLACEMENT,
    PSUTIL_URL,
    PsutilAndroidInstallError,
    prepare_patched_psutil_sdist,
)


def _add_dir(tf: tarfile.TarFile, name: str) -> None:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    tf.addfile(info)


def _add_file(tf: tarfile.TarFile, name: str, content: str) -> None:
    payload = content.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    tf.addfile(info, io.BytesIO(payload))


def _build_psutil_archive(archive: Path, *, malicious_symlink: bool) -> None:
    with tarfile.open(archive, "w:gz") as tf:
        _add_dir(tf, "psutil-7.2.2")
        if malicious_symlink:
            link = tarfile.TarInfo("psutil-7.2.2/psutil")
            link.type = tarfile.SYMTYPE
            link.linkname = "../../outside"
            tf.addfile(link)
        else:
            _add_dir(tf, "psutil-7.2.2/psutil")
        _add_file(
            tf,
            "psutil-7.2.2/psutil/_common.py",
            f"{MARKER}\n",
        )


def test_prepare_patched_psutil_sdist_rejects_symlink_member(tmp_path):
    """A symlink member must be rejected before any file payload is written."""
    archive = tmp_path / "evil.tar.gz"
    _build_psutil_archive(archive, malicious_symlink=True)

    destination = tmp_path / "extract"
    with pytest.raises(PsutilAndroidInstallError, match="Unsupported archive member type"):
        prepare_patched_psutil_sdist(archive, destination)

    assert not (tmp_path / "outside" / "_common.py").exists()


def test_install_psutil_android_compat_uses_patched_tree(tmp_path):
    """Updater path should install from the patched temporary sdist tree."""
    archive = tmp_path / "psutil.tar.gz"
    _build_psutil_archive(archive, malicious_symlink=False)

    from hermes_cli import main as hermes_main

    captured: dict[str, object] = {}

    def fake_urlretrieve(url: str, dest: Path):
        assert url == PSUTIL_URL
        shutil.copyfile(archive, dest)
        return str(dest), None

    def fake_run_install(cmd: list[str], *, env=None):
        src_root = Path(cmd[-1])
        captured["cmd"] = cmd
        captured["env"] = env
        captured["common_py"] = (src_root / "psutil" / "_common.py").read_text(
            encoding="utf-8"
        )

    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve), \
         patch.object(hermes_main, "_run_install_with_heartbeat", side_effect=fake_run_install):
        hermes_main._install_psutil_android_compat(
            ["uv", "pip"],
            env={"HERMES_TEST": "1"},
        )

    assert captured["cmd"][:4] == ["uv", "pip", "install", "--no-build-isolation"]
    assert captured["env"] == {"HERMES_TEST": "1"}
    assert REPLACEMENT in str(captured["common_py"])


def test_install_psutil_android_script_uses_patched_tree(tmp_path, monkeypatch, capsys):
    """Standalone installer script should reuse the same safe patched tree."""
    archive = tmp_path / "psutil.tar.gz"
    _build_psutil_archive(archive, malicious_symlink=False)

    import scripts.install_psutil_android as installer

    def fake_urlretrieve(url: str, dest: Path):
        assert url == PSUTIL_URL
        shutil.copyfile(archive, dest)
        return str(dest), None

    def fake_subprocess_run(cmd: list[str]):
        src_root = Path(cmd[-1])
        patched = (src_root / "psutil" / "_common.py").read_text(encoding="utf-8")
        assert REPLACEMENT in patched
        return type("RunResult", (), {"returncode": 0})()

    monkeypatch.setattr(installer.sys, "argv", ["install_psutil_android.py"])
    monkeypatch.setattr(installer, "_resolve_install_cmd", lambda *_args: ["python", "-m", "pip"])

    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve), \
         patch.object(installer.subprocess, "run", side_effect=fake_subprocess_run):
        assert installer.main() == 0

    captured = capsys.readouterr()
    assert "psutil installed via Android compatibility shim" in captured.out

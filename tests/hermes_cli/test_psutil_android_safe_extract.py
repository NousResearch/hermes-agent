import io
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.psutil_android import (
    PsutilAndroidInstallError,
    _safe_extract_tar_gz,
)


def _write_tar(path: Path, member_name: str, data: bytes = b"payload") -> None:
    info = tarfile.TarInfo(member_name)
    info.size = len(data)
    with tarfile.open(path, "w:gz") as tar:
        tar.addfile(info, io.BytesIO(data))


def test_install_script_rejects_tar_path_traversal(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    _write_tar(archive, "../evil.txt")
    destination = tmp_path / "extract"
    destination.mkdir()

    with pytest.raises(PsutilAndroidInstallError, match="Unsafe archive member path"):
        _safe_extract_tar_gz(archive, destination)

    assert not (tmp_path / "evil.txt").exists()


def test_update_android_psutil_rejects_tar_path_traversal_before_install(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    _write_tar(archive, "../evil.txt")

    from hermes_cli.main import _install_psutil_android_compat

    def fake_urlretrieve(_url, dest):
        Path(dest).write_bytes(archive.read_bytes())
        return dest, None

    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve), \
         patch("hermes_cli.main._run_install_with_heartbeat") as install, \
         pytest.raises(PsutilAndroidInstallError, match="Unsafe archive member path"):
        _install_psutil_android_compat(["python", "-m", "pip"])

    install.assert_not_called()

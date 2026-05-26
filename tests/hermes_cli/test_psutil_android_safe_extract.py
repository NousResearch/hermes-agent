import importlib.util
import io
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest


def _write_tar(path: Path, member_name: str, data: bytes = b"payload") -> None:
    info = tarfile.TarInfo(member_name)
    info.size = len(data)
    with tarfile.open(path, "w:gz") as tar:
        tar.addfile(info, io.BytesIO(data))


def _load_install_script():
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "install_psutil_android.py"
    spec = importlib.util.spec_from_file_location("install_psutil_android", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_install_script_rejects_tar_path_traversal(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    _write_tar(archive, "../evil.txt")
    destination = tmp_path / "extract"
    destination.mkdir()

    _load_install_script()
    from hermes_cli.psutil_android import (
        PsutilAndroidInstallError,
        prepare_patched_psutil_sdist,
    )

    with pytest.raises(
        PsutilAndroidInstallError,
        match="Unsafe archive member path",
    ):
        prepare_patched_psutil_sdist(archive, destination)

    assert not (tmp_path / "evil.txt").exists()


def test_update_android_psutil_rejects_tar_path_traversal_before_install(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    _write_tar(archive, "../evil.txt")

    from hermes_cli.main import _install_psutil_android_compat

    def fake_urlretrieve(_url, dest):
        Path(dest).write_bytes(archive.read_bytes())
        return dest, None

    from hermes_cli.psutil_android import PsutilAndroidInstallError

    with (
        patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve),
        patch("hermes_cli.main._run_install_with_heartbeat") as install,
        pytest.raises(PsutilAndroidInstallError, match="Unsafe archive member path"),
    ):
        _install_psutil_android_compat(["python", "-m", "pip"])

    install.assert_not_called()

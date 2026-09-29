"""The wheel publisher never trusts an unpinned source or an incompatible archive."""

import hashlib
from pathlib import Path
from zipfile import ZipFile

import pytest

from pm.downloader import HashError
from tests.pm._range_server import RangeHandler, dl_server, url
from scripts.wheelhouse.build import fetch_locked_sdist, inspect_wheel


def test_locked_sdist_download_rejects_changed_bytes(tmp_path, dl_server):
    body = b"original sdist bytes"
    RangeHandler.payloads["/source.tar.gz"] = body
    lock = tmp_path / "uv.lock"
    lock.write_text(
        '[[package]]\nname = "cryptography"\nversion = "50.0.1"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'sdist = {{ url = "{url(dl_server, "/source.tar.gz")}", '
        f'hash = "sha256:{hashlib.sha256(body).hexdigest()}" }}\n',
        encoding="utf-8",
    )
    dest = tmp_path / "source.tar.gz"
    assert fetch_locked_sdist(lock, "cryptography", "50.0.1", dest) == dest
    assert dest.read_bytes() == body

    RangeHandler.payloads["/source.tar.gz"] = b"replaced bytes"
    dest.unlink()
    with pytest.raises(HashError):
        fetch_locked_sdist(lock, "cryptography", "50.0.1", dest)
    assert not dest.exists()


def test_wheel_admission_checks_native_tag_and_internal_metadata(tmp_path):
    def wheel(filename: str, *, distribution: str = "cryptography") -> Path:
        path = tmp_path / filename
        with ZipFile(path, "w") as archive:
            root = "cryptography-50.0.1.dist-info"
            archive.writestr(f"{root}/METADATA", f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 50.0.1\n")
            archive.writestr(f"{root}/WHEEL", "Wheel-Version: 1.0\nTag: cp314-cp314-win_arm64\n")
        return path

    good = wheel("cryptography-50.0.1-cp314-cp314-win_arm64.whl")
    assert inspect_wheel(good, "cryptography", "50.0.1") == hashlib.sha256(good.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="win_arm64"):
        inspect_wheel(wheel("cryptography-50.0.1-cp314-cp314-win_amd64.whl"), "cryptography", "50.0.1")
    with pytest.raises(ValueError, match="metadata"):
        inspect_wheel(wheel("cryptography-50.0.1-1-cp314-cp314-win_arm64.whl", distribution="other"),
                      "cryptography", "50.0.1")

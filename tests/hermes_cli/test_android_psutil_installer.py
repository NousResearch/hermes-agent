"""Tests for Android psutil archive verification and extraction guards."""

from __future__ import annotations

import hashlib
import io
import tarfile
from importlib import import_module

import pytest


def _write_tar(path, members):
    with tarfile.open(path, "w:gz") as tar:
        for name, content in members:
            data = content if isinstance(content, bytes) else content.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


@pytest.mark.parametrize(
    "module_name, helper_name",
    [
        ("scripts.install_psutil_android", "_safe_extract_tar"),
        ("hermes_cli.main", "_safe_extract_psutil_android_tar"),
    ],
)
def test_psutil_safe_extract_rejects_path_traversal(tmp_path, module_name, helper_name):
    archive = tmp_path / "bad.tar.gz"
    _write_tar(archive, [("../../escape.txt", "bad")])
    dest = tmp_path / "dest"
    dest.mkdir()

    helper = getattr(import_module(module_name), helper_name)

    with tarfile.open(archive) as tar, pytest.raises(RuntimeError, match="Unsafe path"):
        helper(tar, dest)

    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.parametrize(
    "module_name, helper_name",
    [
        ("scripts.install_psutil_android", "_safe_extract_tar"),
        ("hermes_cli.main", "_safe_extract_psutil_android_tar"),
    ],
)
def test_psutil_safe_extract_rejects_links(tmp_path, module_name, helper_name):
    archive = tmp_path / "link.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("psutil-7.2.2/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../escape.txt"
        tar.addfile(info)
    dest = tmp_path / "dest"
    dest.mkdir()

    helper = getattr(import_module(module_name), helper_name)

    with tarfile.open(archive) as tar, pytest.raises(RuntimeError, match="Unsafe link"):
        helper(tar, dest)


@pytest.mark.parametrize(
    "module_name, helper_name",
    [
        ("scripts.install_psutil_android", "_safe_extract_tar"),
        ("hermes_cli.main", "_safe_extract_psutil_android_tar"),
    ],
)
def test_psutil_safe_extract_allows_regular_members(tmp_path, module_name, helper_name):
    archive = tmp_path / "good.tar.gz"
    _write_tar(archive, [("psutil-7.2.2/psutil/_common.py", "payload")])
    dest = tmp_path / "dest"
    dest.mkdir()

    helper = getattr(import_module(module_name), helper_name)

    with tarfile.open(archive) as tar:
        helper(tar, dest)

    assert (dest / "psutil-7.2.2" / "psutil" / "_common.py").read_text() == "payload"


def test_script_checksum_verification_rejects_mismatch(tmp_path):
    from scripts.install_psutil_android import _verify_archive_checksum

    archive = tmp_path / "archive.tar.gz"
    archive.write_bytes(b"payload")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _verify_archive_checksum(archive, "0" * 64)


def test_main_checksum_verification_accepts_expected_hash(tmp_path, monkeypatch):
    main = import_module("hermes_cli.main")
    archive = tmp_path / "archive.tar.gz"
    archive.write_bytes(b"payload")
    expected = hashlib.sha256(b"payload").hexdigest()

    monkeypatch.setattr(main, "_PSUTIL_ANDROID_SHA256", expected)

    main._verify_psutil_android_archive(archive)

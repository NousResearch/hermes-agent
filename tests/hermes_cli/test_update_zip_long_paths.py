"""Native Windows ZIP updates must handle deep trees without an OS policy change."""
import os
from pathlib import Path
import shutil
import stat
import sys
from types import SimpleNamespace
import zipfile

import pytest

from hermes_cli import update_cmd_zip as zip_update


def _extended(path):
    value = os.path.abspath(path)
    return value if value.startswith("\\\\?\\") else "\\\\?\\" + value


def _member():
    return "hermes-agent-main/website/" + "/".join(["nested-docs-" + "x" * 30] * 8) + "/page.md"


@pytest.mark.platforms("windows")
def test_deep_zip_extracts_without_long_path_policy(tmp_path):
    archive = tmp_path / "source.zip"
    destination = tmp_path / "extract"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(_member(), b"new documentation")
    error = None
    try:
        try:
            zip_update._extract_zip_safely(str(archive), str(destination))
        except OSError as exc:
            error = exc
        assert error is None, f"valid deep ZIP extraction failed: {error}"
        assert Path(_extended(destination / _member())).read_bytes() == b"new documentation"
    finally:
        shutil.rmtree(_extended(destination), ignore_errors=True)


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("fail_swap", [False, True])
def test_deep_zip_download_stages_swaps_and_cleans(tmp_path, monkeypatch, fail_swap):
    from hermes_cli import update_cmd

    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(_member(), b"new documentation")
    project = tmp_path / "install"
    (project / "website").mkdir(parents=True)
    (project / "website" / "old.md").write_text("old documentation")
    download = tmp_path / "download"
    download.mkdir()
    monkeypatch.setattr(update_cmd, "_m", lambda: SimpleNamespace(PROJECT_ROOT=project, sys=sys))
    monkeypatch.setattr("tempfile.mkdtemp", lambda **kwargs: str(download))
    monkeypatch.setattr("urllib.request.urlretrieve", lambda url, dst: shutil.copyfile(archive, dst))
    rename = os.rename

    def rename_or_fail(src, dst):
        if fail_swap and str(src).endswith(".hermes-update-staging"):
            raise OSError("injected swap failure")
        return rename(src, dst)

    monkeypatch.setattr(os, "rename", rename_or_fail)
    exit_code = 0
    try:
        try:
            zip_update._download_and_swap_zip("main", "https://example.invalid/source.zip")
        except SystemExit as exc:
            exit_code = exc.code
        assert exit_code == int(fail_swap), "deep ZIP must commit or report rollback"
        relative = _member().split("/", 1)[1]
        if fail_swap:
            assert (project / "website" / "old.md").read_text() == "old documentation"
            assert not Path(_extended(project / relative)).exists()
        else:
            assert Path(_extended(project / relative)).read_bytes() == b"new documentation"
            assert not (project / "website" / "old.md").exists()
        assert not (project / "website.hermes-update-old").exists()
        assert not (project / "website.hermes-update-staging").exists()
        assert not download.exists()
    finally:
        shutil.rmtree(_extended(project), ignore_errors=True)
        shutil.rmtree(_extended(download), ignore_errors=True)


@pytest.mark.parametrize("name", ["../outside.txt", "/outside.txt", "link"])
def test_zip_still_rejects_unsafe_members_before_writing(tmp_path, name):
    archive = tmp_path / "unsafe.zip"
    destination = tmp_path / "extract"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("hermes-agent-main/README.md", b"must not be extracted")
        info = zipfile.ZipInfo(name)
        if name == "link":
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, b"outside")
    with pytest.raises(ValueError):
        zip_update._extract_zip_safely(str(archive), str(destination))
    assert not destination.exists()
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.parametrize("payload", [None, b"normal documentation"])
def test_empty_and_normal_archives(tmp_path, payload):
    archive = tmp_path / "source.zip"
    destination = tmp_path / "extract"
    with zipfile.ZipFile(archive, "w") as zf:
        if payload is not None:
            zf.writestr("hermes-agent-main/README.md", payload)
    zip_update._extract_zip_safely(str(archive), str(destination))
    if payload is None:
        assert not destination.exists()
    else:
        assert (destination / "hermes-agent-main" / "README.md").read_bytes() == payload


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("path, expected", [
    ("C:/docs/../tree", "\\\\?\\C:\\tree"),
    ("\\\\server\\share\\tree", "\\\\?\\UNC\\server\\share\\tree"),
    ("\\\\?\\C:\\tree", "\\\\?\\C:\\tree"),
    ("\\\\?\\UNC\\server\\share\\tree", "\\\\?\\UNC\\server\\share\\tree"),
])
def test_windows_root_conversion(path, expected):
    assert zip_update._zip_filesystem_path(path) == expected

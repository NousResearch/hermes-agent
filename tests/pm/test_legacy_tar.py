"""Containment and permissions must survive the pre-filter tarfile API."""

import io
import os
import stat
import tarfile

import pytest

from pm.store import extract_tar


@pytest.fixture
def legacy_api(monkeypatch):
    extract = tarfile.TarFile.extract

    def legacy_extract(self, member, path="", set_attrs=True, *, numeric_owner=False):
        return extract(self, member, path, set_attrs, numeric_owner=numeric_owner,
                       filter="fully_trusted")

    def legacy_extractall(self, path=".", members=None, *, numeric_owner=False):
        for member in members or self:
            legacy_extract(self, member, path, numeric_owner=numeric_owner)

    monkeypatch.delattr(tarfile, "data_filter")
    monkeypatch.setattr(tarfile.TarFile, "extract", legacy_extract)
    monkeypatch.setattr(tarfile.TarFile, "extractall", legacy_extractall)


def archive(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as output:
        for name, kind, target, mode in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.linkname = target
            member.mode = mode
            member.size = 1 if kind == tarfile.REGTYPE else 0
            output.addfile(member, io.BytesIO(b"x") if member.size else None)
    stream.seek(0)
    return stream


@pytest.mark.parametrize("case", ["member-alias", "hardlink-alias", "archive-pivot"])
def test_legacy_containment_tracks_existing_and_new_links(tmp_path, legacy_api, case):
    dest = tmp_path / "out"
    dest.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim"
    victim.write_bytes(b"untouched")
    if case == "archive-pivot":
        entries = [("pivot", tarfile.SYMTYPE, ".", 0o777),
                   ("escape", tarfile.SYMTYPE, "pivot/../outside", 0o777)]
    else:
        (dest / "alias").symlink_to(outside, target_is_directory=True)
        entries = ([("alias/victim", tarfile.REGTYPE, "", 0o600)] if case == "member-alias"
                   else [("escape", tarfile.LNKTYPE, "alias/victim", 0o600)])
    with pytest.raises(tarfile.TarError):
        extract_tar(archive(entries), dest)
    assert victim.read_bytes() == b"untouched"
    assert not (dest / "escape").exists()
    assert not (dest / "escape").is_symlink()


@pytest.mark.platforms("posix")
def test_legacy_permissions_preserve_readable_files_and_writable_directories(tmp_path, legacy_api):
    dest = tmp_path / "out"
    entries = [("dir", tarfile.DIRTYPE, "", 0),
               ("dir/data", tarfile.REGTYPE, "", 0o000),
               ("dir/executable", tarfile.REGTYPE, "", 0o7777),
               ("dir/link", tarfile.LNKTYPE, "dir/data", 0o600)]
    try:
        extract_tar(archive(entries), dest)
        assert stat.S_IMODE((dest / "dir/data").stat().st_mode) == 0o600
        assert stat.S_IMODE((dest / "dir/executable").stat().st_mode) == 0o755
        assert os.path.samefile(dest / "dir/link", dest / "dir/data")
        assert (dest / "dir/data").read_bytes() == b"x"
        (dest / "dir/new").write_bytes(b"writable")
    finally:
        # Keep a failing legacy implementation from making pytest's temp tree unreadable.
        (dest / "dir").chmod(0o700)


@pytest.mark.platforms("windows")
def test_legacy_other_drive_paths_raise_the_tar_refusal_contract(tmp_path, legacy_api):
    drive = "Z" if tmp_path.drive.upper() != "Z:" else "Y"
    for name in [f"{drive}:/outside", f"{drive}:outside"]:
        with pytest.raises(tarfile.TarError):
            extract_tar(archive([(name, tarfile.REGTYPE, "", 0o600)]), tmp_path / "out")

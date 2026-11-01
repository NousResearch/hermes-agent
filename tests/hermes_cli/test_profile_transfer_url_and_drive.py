"""Portable profile transfers reject drive components and scrub URL credentials."""

import io
import tarfile

import pytest

from hermes_cli.archive_safe import archive_root_dirs, safe_extract_targz
from hermes_cli.profiles import export_profile


@pytest.mark.parametrize("name", ["profile/D:sentinel.txt", "profile/sub/D:/sentinel.txt",
                                  "profile\\sub\\D:sentinel.txt"])
def test_archive_rejects_interior_drive_components(tmp_path, name):
    archive = tmp_path / "input.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo(name)
        member.size = 4
        stream.addfile(member, io.BytesIO(b"test"))
    destination = tmp_path / "staging"
    destination.mkdir()
    for operation in (lambda: archive_root_dirs(archive),
                      lambda: safe_extract_targz(archive, destination)):
        with pytest.raises(ValueError, match="Unsafe archive member"):
            operation()
    assert not list(destination.rglob("*"))


@pytest.mark.parametrize("filename", ["config.yaml", "SOUL.md"])
def test_export_scrubs_opaque_url_credentials_without_changing_source(tmp_path, monkeypatch, filename):
    profile = tmp_path / "profile"
    profile.mkdir()
    original = ('url: https://service.example.test/api?token=opaque-query-value-912&view=public\n'
                'url: https://user:opaque-password-813@service.example.test/api\n')
    (profile / filename).write_text(original, encoding="utf-8")
    monkeypatch.setattr("hermes_cli.profiles._existing_profile_dir", lambda name: ("sample", profile))
    result = export_profile("sample", str(tmp_path / "export.tar.gz"))
    with tarfile.open(result, "r:gz") as stream:
        member = stream.extractfile(f"sample/{filename}")
        assert member is not None
        exported = member.read().decode()
    assert "opaque-query-value-912" not in exported
    assert "opaque-password-813" not in exported
    assert "view=public" in exported
    assert "service.example.test/api" in exported
    assert (profile / filename).read_text(encoding="utf-8") == original

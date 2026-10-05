"""The prebuilt desktop update accepts only a CI zip for one commit."""
import io
import zipfile
from pathlib import Path

import pytest

from hermes_cli.prebuilt_desktop import (
    ASSET_NAME,
    _extract_unpacked,
    prebuilt_state,
    release_tag,
)


def test_release_tag_is_the_full_commit():
    sha = "a" * 40
    assert release_tag(sha) == f"desktop-{sha}"
    assert release_tag(f"  {sha.upper()}  ") == f"desktop-{sha}"


def test_missing_release_is_not_ready(monkeypatch):
    import urllib.error

    def missing(url, timeout=15):
        raise urllib.error.HTTPError(url, 404, "missing", hdrs=None, fp=io.BytesIO(b""))

    monkeypatch.setattr("hermes_cli.prebuilt_desktop.urllib.request.urlopen", missing)
    assert prebuilt_state("intelli-verse-x/IVX-desktop", "ab" * 20) is False


def test_lookup_failure_does_not_hide_the_update(monkeypatch):
    def down(url, timeout=15):
        raise TimeoutError("offline")

    monkeypatch.setattr("hermes_cli.prebuilt_desktop.urllib.request.urlopen", down)
    assert prebuilt_state("intelli-verse-x/IVX-desktop", "ab" * 20) is None


def test_published_asset_is_ready(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _limit):
            return (
                b'{"assets":[{"name":"desktop-win-unpacked.zip",'
                b'"browser_download_url":"https://example.test/app.zip"}]}'
            )

    monkeypatch.setattr("hermes_cli.prebuilt_desktop.urllib.request.urlopen", lambda *args, **kwargs: Response())
    assert prebuilt_state("intelli-verse-x/IVX-desktop", "ab" * 20) is True


def test_extract_rejects_a_path_outside_the_archive(tmp_path: Path):
    archive = tmp_path / ASSET_NAME
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../Hermes.exe", b"nope")
    with pytest.raises(OSError):
        _extract_unpacked(archive, tmp_path / "out")


def test_extract_accepts_the_unpacked_app(tmp_path: Path):
    archive = tmp_path / ASSET_NAME
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("win-unpacked/IVX-Agency.exe", b"app")
    unpacked = _extract_unpacked(archive, tmp_path / "out")
    assert unpacked is not None
    assert (unpacked / "IVX-Agency.exe").read_bytes() == b"app"

"""Ffmpeg pin resolution uses the advertised version and target, never a guessed URL."""
from __future__ import annotations

import pytest

from pm import update
from pm.package import InstallError
from pm.packages import Ffmpeg


@pytest.fixture
def indexes(monkeypatch):
    monkeypatch.setattr(update, "_btbn_cache", update._TTL())
    monkeypatch.setattr(update, "_martin_cache", update._TTL())
    tag = "autobuild-2026-09-01-12-00"
    version = "9.1.2"
    assets = [
        f"ffmpeg-n{version}-1-gabcdef-linux64-gpl-9.1.tar.xz",
        f"ffmpeg-n{version}-1-gabcdef-win64-gpl-shared-9.1.zip",
        f"ffmpeg-n{version}-1-gabcdef-win64-lgpl-9.1.zip",
        f"ffmpeg-n{version}-1-gabcdef-win64-gpl-9.1.zip",
        f"ffmpeg-n{version}-1-gabcdef-winarm64-gpl-9.1.zip",
    ]
    monkeypatch.setattr(update, "_get_json", lambda url: [
        {"tag_name": tag, "assets": [{"name": name} for name in assets]},
    ])
    paths = {
        f"{osname}-{arch}": f"{source_os}/{source_arch}/1788300000_{version}/ffmpeg.zip"
        for osname, source_os in (("linux", "linux"), ("darwin", "macos"))
        for arch, source_arch in (("x64", "amd64"), ("arm64", "arm64"))
    }
    monkeypatch.setattr(update, "_get_text", lambda url: "\n".join(
        f'<a href="/download/{path}">Ffmpeg</a>' for path in paths.values()
    ))
    return version, tag, paths


@pytest.mark.parametrize("target", [
    "win32-x64", "win32-arm64", "linux-x64", "linux-arm64", "darwin-x64", "darwin-arm64",
])
def test_resolved_version_fetches_its_exact_target_asset(indexes, target):
    version, tag, paths = indexes
    package = Ffmpeg()
    # URL resolution and update discovery must agree about target identities.
    url = package.fetch_url(version, target)
    assert version in package.latest_versions(target)
    if target.startswith("win32"):
        arch = "64" if target.endswith("-x64") else "arm64"
        expected = f"ffmpeg-n{version}-1-gabcdef-win{arch}-gpl-9.1.zip"
        assert url == f"https://github.com/BtbN/FFmpeg-Builds/releases/download/{tag}/{expected}"
    else:
        assert url == f"https://ffmpeg.martin-riedl.de/download/{paths[target]}"


@pytest.mark.parametrize("target", ["win32-x64", "win32-arm64", "linux-x64", "darwin-arm64"])
def test_unadvertised_version_refuses_instead_of_relabelling_old_bytes(indexes, target):
    with pytest.raises(InstallError, match="9.9.9"):
        Ffmpeg().fetch_url("9.9.9", target)

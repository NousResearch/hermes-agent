"""Native artifact metadata stays bound to tagged or tagless package inputs."""
from __future__ import annotations

import json
import plistlib
import subprocess
import zipfile

import pytest

from scripts.bundles import release_artifacts
from tests.scripts.test_channel_build_versions import channel_request


@pytest.mark.parametrize("arch", ["arm64", "x64"])
@pytest.mark.parametrize("channel", [False, True])
def test_macos_record_binds_filename_version_and_provenance(tmp_path, monkeypatch, arch, channel):
    commit = "a" * 40
    request = channel_request(commit, sequence=6) if channel else None
    tag = None if channel else "v1.2.3"
    version = request["version"] if request else "1.2.3"
    identity = request["identity"]["appId"] if request else "ai.hermes.test"
    stamp = ({"source": "channel-build", "channelBuild": request, "commit": commit, "tag": None}
             if request else {"commit": commit, "tag": tag, "baseVersion": version})
    root = tmp_path / "release"
    app = root / "mac" / "Product.app"
    app.mkdir(parents=True)
    package = root / f"Product-{version}-mac-{arch}.zip"
    out = root / "metadata.json"

    def write_package(app_version=version, package_stamp=stamp):
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr("Product.app/Contents/Info.plist", plistlib.dumps({
                "CFBundleIdentifier": identity, "CFBundleShortVersionString": app_version,
            }))
            archive.writestr("Product.app/Contents/Resources/install-stamp.json",
                             json.dumps(package_stamp))

    def codesign(argv, **kwargs):
        assert argv[0] == "codesign"
        return subprocess.CompletedProcess(argv, 0, stderr="TeamIdentifier=ABCDEFGHIJ\n")

    monkeypatch.setattr(release_artifacts.subprocess, "run", codesign)
    write_package()
    original = package.read_bytes()
    release_artifacts.record("macos", arch, root, tag, commit, out, channel_request=request)
    metadata = json.loads(out.read_text(encoding="utf-8-sig"))
    assert metadata["identity"] == identity
    assert metadata["version"] == version
    assert metadata["filename"] == package.name
    assert metadata["teamId"] == "ABCDEFGHIJ"
    assert metadata.get("request") == request
    assert package.read_bytes() == original

    wrong_name = package.with_name(f"Product-9.9.9-mac-{arch}.zip")
    package.rename(wrong_name)
    with pytest.raises(ValueError, match="filename"):
        release_artifacts.record("macos", arch, root, tag, commit, out, channel_request=request)
    wrong_name.rename(package)
    write_package(app_version="9.9.9")
    with pytest.raises(ValueError, match="version"):
        release_artifacts.record("macos", arch, root, tag, commit, out, channel_request=request)
    write_package(package_stamp={**stamp, "commit": "b" * 40})
    with pytest.raises(ValueError, match="provenance"):
        release_artifacts.record("macos", arch, root, tag, commit, out, channel_request=request)

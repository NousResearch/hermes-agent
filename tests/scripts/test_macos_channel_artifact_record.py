"""Native artifact metadata stays bound to tagged or tagless package inputs."""
from __future__ import annotations

import json
import plistlib
import stat
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


# Squirrel.Mac's ShipIt aborts every macOS update when one entry of the downloaded
# app is read-only (#138659); record() is the release step that opens the ZIP.
PAYLOAD = "Product.app/Contents/Resources/payload/"
UNIX = 16  # a ZIP entry's Unix mode is the high word of external_attr
OWNER_WRITABLE = [
    (PAYLOAD, (stat.S_IFDIR | 0o755) << UNIX, b""),
    (PAYLOAD + "hermes/__pycache__/cli.cpython-313.pyc", (stat.S_IFREG | 0o644) << UNIX, b"pyc"),
    (PAYLOAD + "hermes/cli.py", (stat.S_IFREG | 0o644) << UNIX, b"py"),
    (PAYLOAD + "bin/python3", (stat.S_IFLNK | 0o755) << UNIX, b"python3.13"),
    # Only an MS-DOS attribute (archive), as a Windows archiver writes it: no Unix mode.
    (PAYLOAD + "no-unix-mode.txt", 0x20, b"dos"),
]


def record_mac_update(tmp_path, monkeypatch, extra=(), size=None):
    """record() a tagged arm64 update ZIP: a valid app plus OWNER_WRITABLE plus `extra`, grown to `size`."""
    commit, root = "a" * 40, tmp_path / "release"
    (root / "mac" / "Product.app").mkdir(parents=True)
    package = root / "Product-1.2.3-mac-arm64.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("Product.app/Contents/Info.plist", plistlib.dumps({
            "CFBundleIdentifier": "ai.hermes.test", "CFBundleShortVersionString": "1.2.3",
        }))
        archive.writestr("Product.app/Contents/Resources/install-stamp.json",
                         json.dumps({"commit": commit, "tag": "v1.2.3", "baseVersion": "1.2.3"}))
        for name, external_attr, data in [*OWNER_WRITABLE, *extra]:
            entry = zipfile.ZipInfo(name)
            entry.external_attr = external_attr
            archive.writestr(entry, data)
    if size is not None:
        # Sparse padding ahead of the archive: ZIP offsets count from the end, so it still opens.
        data = package.read_bytes()
        with package.open("wb") as grown:
            grown.truncate(size - len(data))
            grown.seek(size - len(data))
            grown.write(data)
    monkeypatch.setattr(release_artifacts.subprocess, "run", lambda argv, **kwargs: subprocess.CompletedProcess(
        argv, 0, stderr="TeamIdentifier=ABCDEFGHIJ\n"))
    out = root / "metadata.json"
    release_artifacts.record("macos", "arm64", root, "v1.2.3", commit, out)
    return out


def test_macos_record_accepts_an_owner_writable_update_zip(tmp_path, monkeypatch):
    out = record_mac_update(tmp_path, monkeypatch)
    assert json.loads(out.read_text(encoding="utf-8"))["version"] == "1.2.3"


@pytest.mark.parametrize("locked", [
    pytest.param([(PAYLOAD + f"pkg/__pycache__/m{n}.cpython-313.pyc", (stat.S_IFREG | 0o444) << UNIX, b"pyc")
                  for n in range(5)], id="pycs-0444"),
    pytest.param([(PAYLOAD + "pkg/vendored.py", (stat.S_IFREG | 0o444) << UNIX, b"py"),
                  (PAYLOAD + "pkg/_speedups.cpython-313-darwin.so", (stat.S_IFREG | 0o555) << UNIX, b"so")],
                 id="source-0444-and-extension-0555"),
    pytest.param([(PAYLOAD + "tools/", (stat.S_IFDIR | 0o555) << UNIX, b"")], id="directory-0555"),
])
def test_macos_record_refuses_an_update_zip_entry_without_owner_write(tmp_path, monkeypatch, locked):
    with pytest.raises(ValueError, match=f"^{len(locked)} update ZIP entries lack owner write") as refused:
        record_mac_update(tmp_path, monkeypatch, locked)
    named = [name for name, _, _ in locked]
    assert all(name in str(refused.value) for name in named[:3])
    assert not any(name in str(refused.value) for name in named[3:])
    assert not (tmp_path / "release" / "metadata.json").exists()


def test_macos_record_refuses_an_update_zip_squirrel_cannot_download(tmp_path, monkeypatch):
    """v0.22.0's Squirrel.Mac (Electron 40) dies downloading an update ZIP of 1 GiB or more; a release
    that published one would crash every v0.22.0 Mac that clicks Update now."""
    with pytest.raises(ValueError, match="1,073,741,824 bytes"):
        record_mac_update(tmp_path, monkeypatch, size=release_artifacts.SQUIRREL_ZIP_LIMIT)
    assert not (tmp_path / "release" / "metadata.json").exists()


def test_macos_record_accepts_an_update_zip_just_under_one_gib(tmp_path, monkeypatch):
    out = record_mac_update(tmp_path, monkeypatch, size=release_artifacts.SQUIRREL_ZIP_LIMIT - 1)
    assert json.loads(out.read_text(encoding="utf-8"))["version"] == "1.2.3"

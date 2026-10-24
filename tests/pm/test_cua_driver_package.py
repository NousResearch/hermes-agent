"""CUA archive staging preserves the native host payload, without running it."""

import tarfile
import zipfile

import pytest

from pm import Lockfile, Store, get_package, paths


@pytest.mark.parametrize("target", ["darwin-arm64", "darwin-x64"])
def test_macos_cua_selects_bundle_binary_and_preserves_signature(tmp_path, target):
    package = get_package("cua-driver")
    lock = Lockfile(paths.lockfile_path())
    version = lock.version(package.name)
    assert version is not None
    payload = tmp_path / "payload"
    app = payload / "CuaDriver.app"
    members = {
        "cua-driver": b"standalone executable fixture",
        "CuaDriver.app/Contents/MacOS/cua-driver": b"bundle executable fixture",
        "CuaDriver.app/Contents/MacOS/cua-cursor-theme": b"cursor helper fixture",
        "CuaDriver.app/Contents/Info.plist": b"bundle metadata fixture",
        "CuaDriver.app/Contents/_CodeSignature/CodeResources": b"signature fixture",
    }
    for name, content in members.items():
        file = payload / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content)
    archive = tmp_path / "cua.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname="cua-release")
    staged = tmp_path / "staged"
    package.unpack(archive, staged, target)
    package.stage(Store(tmp_path / "store"), staged, version, target)

    assert package.binary(staged, target) == staged / app.relative_to(payload) / "Contents/MacOS/cua-driver"
    for name, content in members.items():
        assert (staged / name).read_bytes() == content
    assert lock.artifacts(package.name, target)[0]["url"] == package.fetch_url(version, target)
    assert "-binary.tar.gz" not in package.fetch_url(version, target)


def test_windows_cua_keeps_uiaccess_and_cursor_helpers(tmp_path):
    package = get_package("cua-driver")
    archive = tmp_path / "cua.zip"
    members = {name: name.encode() for name in (
        "cua-driver.exe", "cua-driver-uia.exe", "cua-cursor-theme.exe", "cua_driver_sdk.dll",
    )}
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, content in members.items():
            zipped.writestr(name, content)
    staged = tmp_path / "staged"
    package.unpack(archive, staged, "win32-x64")
    package.stage(Store(tmp_path / "store"), staged, "fixture", "win32-x64")
    assert package.binary(staged, "win32-x64") == staged / "cua-driver.exe"
    for name, content in members.items():
        assert (staged / name).read_bytes() == content
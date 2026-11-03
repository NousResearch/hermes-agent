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
    standalone = payload / "cua-driver"
    standalone.parent.mkdir(parents=True, exist_ok=True)
    standalone.write_bytes(members["cua-driver"])
    bundle_binary = payload / "CuaDriver.app/Contents/MacOS/cua-driver"
    bundle_binary.parent.mkdir(parents=True, exist_ok=True)
    bundle_binary.write_bytes(members["CuaDriver.app/Contents/MacOS/cua-driver"])
    cursor = payload / "CuaDriver.app/Contents/MacOS/cua-cursor-theme"
    cursor.write_bytes(members["CuaDriver.app/Contents/MacOS/cua-cursor-theme"])
    info = payload / "CuaDriver.app/Contents/Info.plist"
    info.write_bytes(members["CuaDriver.app/Contents/Info.plist"])
    signature = payload / "CuaDriver.app/Contents/_CodeSignature/CodeResources"
    signature.parent.mkdir(parents=True, exist_ok=True)
    signature.write_bytes(members["CuaDriver.app/Contents/_CodeSignature/CodeResources"])
    archive = tmp_path / "cua.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname="cua-release")
    staged = tmp_path / "staged"
    package.unpack(archive, staged, target)
    package.stage(Store(tmp_path / "store"), staged, version, target)

    assert package.binary(staged, target) == staged / app.relative_to(payload) / "Contents/MacOS/cua-driver"
    assert (staged / "cua-driver").read_bytes() == members["cua-driver"]
    assert (staged / "CuaDriver.app/Contents/MacOS/cua-driver").read_bytes() == members[
        "CuaDriver.app/Contents/MacOS/cua-driver"
    ]
    assert (staged / "CuaDriver.app/Contents/MacOS/cua-cursor-theme").read_bytes() == members[
        "CuaDriver.app/Contents/MacOS/cua-cursor-theme"
    ]
    assert (staged / "CuaDriver.app/Contents/Info.plist").read_bytes() == members[
        "CuaDriver.app/Contents/Info.plist"
    ]
    assert (staged / "CuaDriver.app/Contents/_CodeSignature/CodeResources").read_bytes() == members[
        "CuaDriver.app/Contents/_CodeSignature/CodeResources"
    ]
    assert lock.artifacts(package.name, target)[0]["url"] == package.fetch_url(version, target)
    assert "-binary.tar.gz" not in package.fetch_url(version, target)


def test_windows_cua_keeps_uiaccess_and_cursor_helpers(tmp_path):
    package = get_package("cua-driver")
    archive = tmp_path / "cua.zip"
    members = {name: name.encode() for name in (
        "cua-driver.exe", "cua-driver-uia.exe", "cua-cursor-theme.exe", "cua_driver_sdk.dll",
    )}
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("cua-driver.exe", members["cua-driver.exe"])
        zipped.writestr("cua-driver-uia.exe", members["cua-driver-uia.exe"])
        zipped.writestr("cua-cursor-theme.exe", members["cua-cursor-theme.exe"])
        zipped.writestr("cua_driver_sdk.dll", members["cua_driver_sdk.dll"])
    staged = tmp_path / "staged"
    package.unpack(archive, staged, "win32-x64")
    package.stage(Store(tmp_path / "store"), staged, "fixture", "win32-x64")
    assert package.binary(staged, "win32-x64") == staged / "cua-driver.exe"
    assert (staged / "cua-driver.exe").read_bytes() == members["cua-driver.exe"]
    assert (staged / "cua-driver-uia.exe").read_bytes() == members["cua-driver-uia.exe"]
    assert (staged / "cua-cursor-theme.exe").read_bytes() == members["cua-cursor-theme.exe"]
    assert (staged / "cua_driver_sdk.dll").read_bytes() == members["cua_driver_sdk.dll"]

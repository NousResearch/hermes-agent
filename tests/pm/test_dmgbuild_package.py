"""The build-only DMG supplier preserves its paired runtime without executing it."""
import struct
import tarfile

import pytest

from pm import Lockfile, Store, get_package, paths
from pm.store import ALL_TARGETS
from scripts.bundles.native import _bundle_package_names


@pytest.mark.parametrize("target,cpu", [("darwin-arm64", 0x0100000C), ("darwin-x64", 0x01000007)])
def test_dmgbuild_stages_paired_runtime_and_stays_out_of_payload(tmp_path, monkeypatch, target, cpu):
    package = get_package("dmgbuild")
    lock = Lockfile(paths.lockfile_path())
    version = lock.version(package.name)
    assert version is not None
    assert lock.artifacts(package.name, target)[0]["url"] == package.fetch_url(version, target)
    assert package.latest_versions(target, locked=version) == []
    assert package.internal and not package.on_path
    assert package.name not in _bundle_package_names()
    assert all(package.missing_reason(t) for t in ALL_TARGETS if not t.startswith("darwin-"))
    payload = tmp_path / "payload"
    python = payload / "python/bin/python3"
    python.parent.mkdir(parents=True)
    python.write_bytes(struct.pack("<II", 0xFEEDFACF, cpu) + bytes(56))
    launcher = payload / "dmgbuild"
    launcher.write_text('#!/usr/bin/env bash\nexec "$(dirname "$0")/python/bin/python3" -m dmgbuild "$@"\n', encoding="utf-8")
    archive = tmp_path / "dmgbuild.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname=".")

    def forbidden(*args, **kwargs):
        pytest.fail("staging must not execute the foreign supplier")

    monkeypatch.setattr("subprocess.run", forbidden)
    staged = tmp_path / "staged"
    package.unpack(archive, staged, target)
    package.stage(Store(tmp_path / "store"), staged, version, target)
    assert package.binary(staged, target) == staged / "dmgbuild"
    assert package.verify(staged, target) == ""
    assert (staged / "dmgbuild").read_bytes() == launcher.read_bytes()
    wrong = "darwin-x64" if target == "darwin-arm64" else "darwin-arm64"
    assert "not a" in package.verify(staged, wrong)
    (staged / "python/bin/python3").unlink()
    assert "python/bin/python3 missing" in package.verify(staged, target)

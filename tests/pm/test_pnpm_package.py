"""pnpm stages from two archives into one entry: the native executable and its launcher's dist/.

pnpm 12's executable looks for the bundled node-gyp beside itself (``dist/node-gyp-bin``) to build
dependencies that ship a binding.gyp; the launcher package carries that dist/. The executable's own
manifest must survive the merge: node-deps reads pnpm's version from the package.json beside it.
"""

from __future__ import annotations

import errno
import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from pm import Lockfile, get_package, paths
from pm.install import _prepare_artifacts
from pm.store import ALL_TARGETS


def _tarball(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(f"package/{name}")
            info.size = len(content)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(content))
    return path


class _LocalArchives:
    """A store whose downloads are the given local archives, in artifact order."""

    def __init__(self, archives: list[Path]):
        self.archives = archives

    def fetch_many(self, artifacts, scratch, **_kwargs):
        return self.archives[:len(artifacts)]


def test_every_target_with_a_pinned_node_can_install_the_js_workspaces():
    """The dashboard, TUI and desktop source builds all install through pnpm on PM's node, so a
    target with a node pin but no pnpm pin cannot build them (Termux: "unavailable on
    linux-arm64-bionic" from `hermes dashboard`)."""
    package = get_package("pnpm")
    lock = Lockfile(paths.lockfile_path())
    version = lock.version("pnpm")
    for target in ALL_TARGETS:
        if not lock.artifacts("node", target):
            continue
        assert package.missing_reason(target) is None, target
        assert [a["url"] for a in lock.artifacts("pnpm", target)] == package.fetch_urls(version, target), target


@pytest.mark.parametrize("target", ["linux-x64", "win32-x64", "linux-arm64-bionic"])
def test_pnpm_entry_holds_the_executable_and_the_launchers_dist(tmp_path, target):
    package = get_package("pnpm")
    lock = Lockfile(paths.lockfile_path())
    version = lock.version("pnpm")
    artifacts = lock.artifacts("pnpm", target)
    # Extraction follows the lock's artifact order: the executable first, then the launcher.
    assert [artifact["url"] for artifact in artifacts] == package.fetch_urls(version, target)
    executable = package.binary_rel["win32" if target.startswith("win32") else "posix"]
    exe_name = artifacts[0]["url"].rsplit("/-/", 1)[0].removeprefix("https://registry.npmjs.org/")
    exe = _tarball(tmp_path / "exe.tgz", {
        executable: b"native pnpm", "package.json": json.dumps({"name": exe_name, "version": version}).encode(),
        "LICENSE": b"license"})
    launcher = _tarball(tmp_path / "launcher.tgz", {
        "package.json": json.dumps({"name": "pnpm", "version": version}).encode(), "bin/pnpm.cjs": b"launcher",
        "dist/pnpm.cjs": b"bundle", "dist/node-gyp-bin/node-gyp": b"#!/bin/sh\n",
        "dist/node_modules/node-gyp/package.json": b'{"name": "node-gyp"}'})
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    staged = _prepare_artifacts(package, _LocalArchives([exe, launcher]), scratch, artifacts, version, target)

    assert package.binary(staged, target).read_bytes() == b"native pnpm"
    assert (staged / "dist/node-gyp-bin/node-gyp").is_file()
    assert (staged / "dist/node_modules/node-gyp/package.json").is_file()
    assert json.loads((staged / "package.json").read_text(encoding="utf-8"))["name"] == exe_name
    # Only the launcher's dist/ is kept: its JS entry point never shadows the executable.
    assert not (staged / "bin").exists()
    assert not (staged / "package").exists()


def test_pnpm_dirs_replace_every_case_twin_of_their_names():
    """Windows hands Python upper-cased env names, and a child given both spellings picks one
    arbitrarily: the Hermes-owned store must be the only spelling left."""
    from pm.packages import pnpm_env

    env = pnpm_env({"PNPM_CONFIG_STORE_DIR": "/elsewhere/store", "Pnpm_Config_Cache_Dir": "/elsewhere/cache"})

    assert [key for key in env if key.upper().startswith("PNPM_CONFIG_")] == [
        "pnpm_config_store_dir", "pnpm_config_cache_dir", "pnpm_config_state_dir"]
    assert "/elsewhere" not in "".join(env.values())


class TestScannerHolds:
    """A Windows scanner opens what extraction just wrote, so moving or deleting it is refused for a
    moment ([WinError 32] on a file, [WinError 5] on a directory, both EACCES). The flatten_single_dir
    hoist waits that out (#131884); pnpm unpacks with its own hoist."""

    @pytest.fixture(autouse=True)
    def instant_backoff(self, monkeypatch):
        from pm import filesystem

        monkeypatch.setattr(filesystem, "_HELD_RETRY_DELAYS", (0,) * len(filesystem._HELD_RETRY_DELAYS))

    def _hold_after_extract(self, monkeypatch, name: str, times: float) -> list[str]:
        from pm import store

        refusals: list[str] = []
        extracted: list[Path] = []
        real_extract, real_rename, real_unlink = store.extract, os.rename, os.unlink

        def extract(archive, dest):
            real_extract(archive, dest)
            extracted.append(dest)

        def refuse(path):
            if extracted and os.path.basename(os.fspath(path)) == name and len(refusals) < times:
                refusals.append(name)
                raise PermissionError(errno.EACCES, "The process cannot access the file because it is being "
                                      "used by another process", os.fspath(path))

        def rename(src, dst, *args, **kwargs):
            refuse(src)
            return real_rename(src, dst, *args, **kwargs)

        def unlink(path, *args, **kwargs):
            refuse(path)
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(store, "extract", extract)
        monkeypatch.setattr(os, "rename", rename)
        monkeypatch.setattr(os, "unlink", unlink)
        return refusals

    def _stage(self, tmp_path: Path) -> Path:
        package = get_package("pnpm")
        lock = Lockfile(paths.lockfile_path())
        exe = _tarball(tmp_path / "exe.tgz", {"pnpm.exe": b"native pnpm", "package.json": b"{}"})
        launcher = _tarball(tmp_path / "launcher.tgz", {
            "package.json": b"{}", "bin/pnpm.cjs": b"launcher", "dist/pnpm.cjs": b"bundle"})
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        return _prepare_artifacts(package, _LocalArchives([exe, launcher]), scratch,
                                  lock.artifacts("pnpm", "win32-x64"), lock.version("pnpm"), "win32-x64")

    @pytest.mark.parametrize("held", ["pnpm.exe", "dist", "pnpm.cjs"],
                             ids=["executable-hoist", "launcher-dist-hoist", "launcher-leftover-removal"])
    def test_staging_waits_out_a_scanner_hold(self, tmp_path, monkeypatch, held):
        refusals = self._hold_after_extract(monkeypatch, held, times=2)

        staged = self._stage(tmp_path)

        assert refusals == [held, held]
        assert (staged / "pnpm.exe").read_bytes() == b"native pnpm"
        assert (staged / "dist" / "pnpm.cjs").read_bytes() == b"bundle"
        assert not (staged / "bin").exists() and not (staged / "package").exists()

    def test_a_hold_that_never_clears_still_fails_the_install(self, tmp_path, monkeypatch):
        self._hold_after_extract(monkeypatch, "dist", times=float("inf"))

        with pytest.raises(PermissionError):
            self._stage(tmp_path)

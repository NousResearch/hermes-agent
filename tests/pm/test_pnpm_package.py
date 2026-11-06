"""pnpm stages from two archives into one entry: the native executable and its launcher's dist/.

pnpm 12's executable looks for the bundled node-gyp beside itself (``dist/node-gyp-bin``) to build
dependencies that ship a binding.gyp; the launcher package carries that dist/. The executable's own
manifest must survive the merge: node-deps reads pnpm's version from the package.json beside it.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from pm import Lockfile, get_package, paths
from pm.install import _prepare_artifacts


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


@pytest.mark.parametrize("target", ["linux-x64", "win32-x64"])
def test_pnpm_entry_holds_the_executable_and_the_launchers_dist(tmp_path, target):
    package = get_package("pnpm")
    lock = Lockfile(paths.lockfile_path())
    version = lock.version("pnpm")
    artifacts = lock.artifacts("pnpm", target)
    # Extraction follows the lock's artifact order: the executable first, then the launcher.
    assert [artifact["url"] for artifact in artifacts] == package.fetch_urls(version, target)
    executable = package.binary_rel["win32" if target.startswith("win32") else "posix"]
    exe = _tarball(tmp_path / "exe.tgz", {
        executable: b"native pnpm", "package.json": json.dumps({"name": f"@pnpm/exe.{target}", "version": version}).encode(),
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
    assert json.loads((staged / "package.json").read_text(encoding="utf-8"))["name"] == f"@pnpm/exe.{target}"
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

"""agent-browser staging leaves the kept native binary executable.

The npm tarball stores every ``bin/agent-browser-*`` as 0644; agent-browser's own
postinstall sets the exec bit, and pm runs no postinstall.
"""

import io
import os
import stat
import tarfile

import pytest

from pm import Store, get_package


def _npm_tarball(path, names):
    with tarfile.open(path, "w:gz") as tar:
        for name in names:
            data = f"{name} fixture".encode()
            info = tarfile.TarInfo(f"package/{name}")
            info.size = len(data)
            info.mode = 0o755 if name.endswith(".js") else 0o644
            tar.addfile(info, io.BytesIO(data))


@pytest.mark.skipif(os.name == "nt", reason="POSIX exec bit")
@pytest.mark.parametrize("target", ["linux-arm64", "linux-x64", "darwin-arm64"])
def test_staged_native_binary_is_executable(tmp_path, target):
    package = get_package("agent-browser")
    archive = tmp_path / "agent-browser.tgz"
    _npm_tarball(archive, [
        "package.json",
        "bin/agent-browser.js",
        "bin/agent-browser-linux-arm64",
        "bin/agent-browser-linux-x64",
        "bin/agent-browser-darwin-arm64",
    ])
    staged = tmp_path / "staged"
    package.unpack(archive, staged, target)
    package.stage(Store(tmp_path / "store"), staged, "fixture", target)

    binary = package.binary(staged, target)
    assert binary == staged / "bin" / f"agent-browser-{target}"
    mode = binary.stat().st_mode
    assert mode & stat.S_IXUSR and mode & stat.S_IXGRP and mode & stat.S_IXOTH
    assert sorted(p.name for p in (staged / "bin").iterdir()) == sorted(
        ["agent-browser.js", f"agent-browser-{target}"])

"""The pinned Git for Windows (bash.exe + internal git calls on Windows)."""

from pm.lock import Lockfile
from pm.packages import Git
from pm.paths import lockfile_path


def test_locked_git_carries_the_2_56_security_fixes():
    """2.56.0.3 drops the compromised eMudhra CA chain, 2.56.0.4 ships brotli
    1.2.1, 2.56.1 carries the 2.56.1 git CVE fixes (#148835). Every locked
    artifact must be the URL the downloader builds for the pinned version."""
    package = Git()
    lock = Lockfile(lockfile_path())
    version = lock.version(package.name)
    assert version is not None
    tag, _, build = version.partition("+")
    assert (tuple(int(p) for p in tag.split(".")), int(build)) >= ((2, 56, 1), 1)
    for target in ("win32-x64", "win32-arm64"):
        rows = lock.artifacts(package.name, target)
        assert [row["url"] for row in rows] == [package.fetch_url(version, target)]

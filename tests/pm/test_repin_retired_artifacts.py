"""Re-pinning artifacts whose pinned archive the supplier retired (#122240)."""

from pm.install import _repin_retired_artifacts
from pm.lock import Lockfile

OLD = {"url": "https://supplier.example/retired.tar.xz", "sha256": "0" * 64}
OTHER = {"url": "https://supplier.example/win32.tar.xz", "sha256": "1" * 64}


class _FakePackage:
    """Duck type of the two Package hooks the repair consults."""

    name = "ffmpeg"

    def __init__(self, urls=None, known=None, raises=False):
        self._urls = urls
        self._known = known
        self._raises = raises

    def fetch_urls(self, version, target):
        if self._raises:
            raise RuntimeError("no advertised artifact")
        return self._urls

    def known_sha256(self, version, url):
        return self._known


def _seeded_lock(tmp_path):
    lockfile = Lockfile(tmp_path / "lock.json")
    lockfile.set_pin("ffmpeg", "9.0.1", {"linux-x64": dict(OLD), "win32-x64": dict(OTHER)})
    lockfile.save()
    return Lockfile(tmp_path / "lock.json")


def _rows(tmp_path):
    fresh = Lockfile(tmp_path / "lock.json")
    return fresh, fresh.artifacts("ffmpeg", "linux-x64")


def test_re_pins_the_retired_archive_preferring_the_index_hash(tmp_path, monkeypatch):
    lockfile = _seeded_lock(tmp_path)
    package = _FakePackage(["https://supplier.example/live.tar.xz"], known="b" * 64)

    def _fail(url):
        raise AssertionError("known_sha256 must be preferred over hashing the download")

    monkeypatch.setattr("pm.store.hash_url", _fail)
    assert _repin_retired_artifacts(package, lockfile, "9.0.1", "linux-x64") is True

    fresh, rows = _rows(tmp_path)
    assert rows == [{"url": "https://supplier.example/live.tar.xz", "sha256": "b" * 64}]
    assert fresh.version("ffmpeg") == "9.0.1"  # never a substituted version
    assert fresh.artifacts("ffmpeg", "win32-x64") == [dict(OTHER)]  # other targets untouched


def test_downloads_and_hashes_a_replacement_the_index_cannot_attest(tmp_path, monkeypatch):
    lockfile = _seeded_lock(tmp_path)
    package = _FakePackage(["https://supplier.example/live.tar.xz"])
    monkeypatch.setattr("pm.store.hash_url", lambda url: "c" * 64)
    assert _repin_retired_artifacts(package, lockfile, "9.0.1", "linux-x64") is True

    _, rows = _rows(tmp_path)
    assert rows == [{"url": "https://supplier.example/live.tar.xz", "sha256": "c" * 64}]


def test_keeps_the_pin_when_the_index_has_not_moved(tmp_path):
    lockfile = _seeded_lock(tmp_path)
    package = _FakePackage([OLD["url"]])
    assert _repin_retired_artifacts(package, lockfile, "9.0.1", "linux-x64") is False

    _, rows = _rows(tmp_path)
    assert rows == [dict(OLD)]


def test_keeps_the_original_failure_when_no_replacement_is_advertised(tmp_path):
    lockfile = _seeded_lock(tmp_path)
    package = _FakePackage(raises=True)
    assert _repin_retired_artifacts(package, lockfile, "9.0.1", "linux-x64") is False

    _, rows = _rows(tmp_path)
    assert rows == [dict(OLD)]

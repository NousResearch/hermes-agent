"""A supplier-retired archive is re-pinned on this machine, never in lock.json (#122240)."""

from __future__ import annotations

import hashlib
import importlib
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

from pm import paths
from pm.downloader import DownloadError, DownloadTransportError, HashError
from pm.lock import Lockfile
from pm.package import InstallError, Package
from pm.store import Store

install = importlib.import_module("pm.install")

TARGET = "linux-x64"
RETIRED = "https://supplier.example/autobuild-old/tool-n9.0.1-27.tar.xz"
LIVE = "https://supplier.example/autobuild-new/tool-n9.0.2-12.tar.xz"
MIRROR = "https://mirror.example/upstream/sha256/old"
PAYLOAD = b"live build bytes"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _RollingPackage(Package):
    """A minor-style package whose supplier only still builds 9.0.2."""

    name = "rolling-tool"
    version_style = "minor"

    def missing_reason(self, target):
        return None

    def latest_versions(self, target, locked=None):
        return ["9.1.0", "9.0.2", "8.1.3"]

    def fetch_urls(self, version, target):
        if version != "9.0.2":
            raise InstallError(self.name, f"no advertised {version} artifact for {target}")
        return [LIVE]

    def known_sha256(self, version, url):
        return _sha(PAYLOAD)

    def unpack(self, archive: Path, staged: Path, target: str) -> None:
        (staged / "bin").mkdir(parents=True)
        (staged / "bin" / "tool").write_bytes(archive.read_bytes())

    def verify(self, entry: Path, target: str) -> str:
        return "" if (entry / "bin" / "tool").is_file() else "bin/tool missing"


def _http_failure(url: str, status: int) -> DownloadTransportError:
    return DownloadTransportError(url, urllib.error.HTTPError(url, status, "", Message(), None))


def _combined(*failures: DownloadTransportError) -> DownloadError:
    error = DownloadError("every source failed")
    error.failures = failures
    return error


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    store = Store(tmp_path / "runtime")
    package = _RollingPackage()
    lock = Lockfile(tmp_path / "lock.json")
    lock.set_pin(package.name, "9.0.1", {TARGET: {"url": RETIRED, "sha256": "0" * 64}})
    lock.save()
    lock = Lockfile(tmp_path / "lock.json")
    entry = store.entry(f"fetch-{_sha(PAYLOAD)}")  # the live build is cached: no network
    entry.mkdir(parents=True)
    (entry / "tool.tar.xz").write_bytes(PAYLOAD)
    monkeypatch.setattr(install, "get_package", lambda name: package)
    monkeypatch.setattr(install, "_store", lambda: store)
    monkeypatch.setattr(install, "_lockfile", lambda: lock)
    return lock


def _fail_retired_fetch(monkeypatch, error: DownloadError) -> None:
    real = Store.fetch_many

    def fetch_many(self, artifacts, scratch, **kwargs):
        if any(row["url"] == RETIRED for row in artifacts):
            raise error
        return real(self, artifacts, scratch, **kwargs)

    monkeypatch.setattr(Store, "fetch_many", fetch_many)


def test_retired_archive_is_repinned_locally_until_the_shipped_row_moves(sandbox, monkeypatch):
    # Origin 404 and the mirror 403: the combined failure of a mirror-blocked region.
    _fail_retired_fetch(monkeypatch, _combined(_http_failure(RETIRED, 404), _http_failure(MIRROR, 403)))
    shipped = sandbox.path.read_bytes()

    entry = install.stage_only("rolling-tool", TARGET)

    assert (entry / "bin" / "tool").read_bytes() == PAYLOAD
    assert sandbox.path.read_bytes() == shipped  # the checkout stays clean
    live = [{"url": LIVE, "sha256": _sha(PAYLOAD)}]
    assert Lockfile(sandbox.path).artifacts("rolling-tool", TARGET) == live  # the next run reuses it
    retired = {"url": RETIRED, "sha256": "0" * 64}
    sandbox.set_pin("rolling-tool", "9.1.0", {TARGET: retired})  # label bump, same row
    sandbox.save()
    assert Lockfile(sandbox.path).artifacts("rolling-tool", TARGET) == [retired]  # re-pin was for 9.0.x
    moved = {"url": "https://supplier.example/autobuild-next/tool.tar.xz", "sha256": "1" * 64}
    sandbox.set_pin("rolling-tool", "9.0.1", {TARGET: moved})
    sandbox.save()
    assert Lockfile(sandbox.path).artifacts("rolling-tool", TARGET) == [moved]  # upstream wins again
    paths.repins_path().write_text('{"rolling-tool": {"linux-x64": {"replaces": ["%s"]}}}' % ("1" * 64), encoding="utf-8")
    assert Lockfile(sandbox.path).artifacts("rolling-tool", TARGET) == [moved]  # a broken entry is ignored


@pytest.mark.parametrize("error", [
    HashError("sha256 mismatch"),
    DownloadTransportError(RETIRED, TimeoutError("timed out")),
    # Origin 404 at probe, then the mirror probed fine and timed out mid-transfer.
    _combined(_http_failure(RETIRED, 404), DownloadTransportError(MIRROR, TimeoutError("timed out"))),
], ids=["integrity", "transient", "mirror-transient"])
def test_failures_that_do_not_prove_retirement_are_never_repinned(sandbox, monkeypatch, error):
    _fail_retired_fetch(monkeypatch, error)

    with pytest.raises(InstallError, match="install failed"):
        install.stage_only("rolling-tool", TARGET)

    assert Lockfile(sandbox.path).artifacts("rolling-tool", TARGET)[0]["url"] == RETIRED

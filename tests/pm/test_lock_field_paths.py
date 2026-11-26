"""Every lock.json field that names a store path stays one component under the
store root, and the plain-http loopback allowance names a real loopback host
(#138359)."""

import urllib.request

import pytest

from pm.downloader import Download, DownloadError, Source, _HttpsRedirectHandler
from pm.install import _install, _remove_entry
from pm.lock import Lockfile
from pm.package import Package
from pm.paths import lockfile_path
from pm.store import Store, entry_component

GOOD = "a" * 64


class Tool(Package):
    name = "faketool"


@pytest.mark.parametrize("version,target", [
    ("../../victim", "linux-x64"), ("1", "../../victim"), ("1/../../..", "linux-x64"),
    ("1\\..\\..", "linux-x64"),
])
def test_entry_built_from_lock_fields_is_never_removed_outside_the_root(tmp_path, version, target):
    store = Store(tmp_path / "a" / "b" / "store")
    store.root.mkdir(parents=True)
    victim = tmp_path / "a" / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("data")
    with pytest.raises(ValueError, match="single path component"):
        _remove_entry(store, Tool().store_entry(version, target))
    assert (victim / "keep.txt").read_text() == "data"


def test_install_refuses_a_traversing_lock_version(tmp_path):
    lock_path = tmp_path / "lock.json"
    lock_path.write_text('{"schema": 1, "packages": {"faketool": {"version": "../../victim", "artifacts": {}}}}')
    with pytest.raises(ValueError, match="single path component"):
        _install(Tool(), Lockfile(lock_path), None, Store(tmp_path / "s" / "store"), "linux-x64")


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b"])
def test_entry_component_rejects_non_component_names(name):
    with pytest.raises(ValueError, match="single path component"):
        entry_component(name)


def test_entry_component_keeps_real_entry_names():
    for name in ("python-3.14.9+20261021-linux-x64", f"fetch-{GOOD}", ".previous-stage-uv-1-any",
                 "chromium-1208-linux-x64"):
        assert entry_component(name) == name


@pytest.mark.parametrize("url", ["https://example.com/..", "https://example.com/", "https://example.com/a\\..\\x"])
def test_fetch_many_refuses_a_url_without_a_file_name(tmp_path, url):
    with pytest.raises(ValueError, match="does not end in a file name"):
        Store(tmp_path / "store").fetch_many([{"url": url, "sha256": GOOD}], tmp_path / "scratch")


def test_registry_digest_pin_without_sha256_still_reads_back():
    rows = Lockfile(lockfile_path()).artifacts("termux-docker", "linux-arm64-bionic")
    assert rows and rows[0]["url"].startswith("docker://")


@pytest.mark.parametrize("url", ["http://127.0.0.1:@evil.invalid/payload",
                                 "http://localhost:80@evil.invalid/payload",
                                 "http://localhost@evil.invalid/payload"])
def test_userinfo_does_not_pass_as_loopback(tmp_path, url):
    with pytest.raises(ValueError, match="non-https"):
        Download([Source(url, tmp_path / "m")], partials_dir=tmp_path / "p").run()
    local = urllib.request.Request("http://127.0.0.1:8000/a")
    with pytest.raises(DownloadError):
        _HttpsRedirectHandler().redirect_request(local, None, 302, "Found", {}, url)


@pytest.mark.parametrize("url", ["file://localhost/etc/hosts", "ftp://127.0.0.1:2121/payload"])
def test_only_plain_http_counts_as_loopback(tmp_path, url):
    with pytest.raises(ValueError, match="non-https"):
        Download([Source(url, tmp_path / "m")], partials_dir=tmp_path / "p").run()

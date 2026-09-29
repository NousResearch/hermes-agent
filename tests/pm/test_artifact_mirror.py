"""PM installation and cross-target staging consume the public pinned mirror."""
import hashlib
import importlib
import io
import os
import urllib.request
import zipfile

import pytest

from pm import paths
from pm.lock import Facts, Lockfile
from pm.package import Package
from pm.store import Store, current_target
from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401


@pytest.mark.parametrize("mode", ["install", "stage", "library"])
def test_cold_consumers_recover_after_upstream_removal(tmp_path, dl_server, monkeypatch, mode):
    from pm import artifact_mirror

    engine = importlib.import_module("pm.install")
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("tool.txt", b"pinned and preserved")
    body = data.getvalue()
    if mode == "library":
        from tests.scripts.test_termux_runtime_libs import _build_deb
        deb = tmp_path / "test.deb"
        _build_deb(deb, "libmirror.so", b"pinned and preserved")
        body = deb.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(artifact_mirror, "PUBLIC_PREFIX", url(dl_server, "/archive/"))
    monkeypatch.setattr(artifact_mirror, "github_asset_url", lambda _: url(dl_server, "/github/" + digest))
    RangeHandler.payloads["/archive/" + digest] = body
    row = {"url": url(dl_server, "/removed.zip"), "sha256": digest}
    monkeypatch.setattr(paths, "partials_root", lambda: tmp_path / "partials")
    if mode == "library":
        from scripts.termux.stage_runtime_libs import stage
        result = stage(tmp_path / "payload", {"libmirror": {**row, "version": "1.0"}})
        assert (result / "libmirror.so").read_bytes().endswith(b"pinned and preserved")
    else:
        package = Package()
        package.name = "mirror-tool"
        store = Store(tmp_path / "store")
        facts = Facts(store.root / "facts.json")
        lock = Lockfile(tmp_path / "lock.json")
        lock.set_pin(package.name, "1.0", {"any": row})
        if mode == "install":
            engine._install(package, lock, facts, store, current_target())
            result = store.entry(facts.get(package.name)["entry"])
        else:
            monkeypatch.setattr(engine, "_lockfile", lambda: lock)
            monkeypatch.setattr(engine, "_store", lambda: store)
            monkeypatch.setattr(engine, "get_package", lambda _: package)
            result = engine.stage_only(package.name, "linux-arm64-bionic")
        assert (result / "tool.txt").read_bytes() == b"pinned and preserved"
    assert any(path == "/archive/" + digest for path, *_ in RangeHandler.ranges_seen)


@pytest.mark.parametrize("available", ["github", "upstream", "archive"])
def test_pinned_download_tries_our_release_before_upstream_then_r2(tmp_path, dl_server, monkeypatch, available):
    from pm import artifact_mirror, downloader
    from pm.downloader import Download

    body = b"verified public release input"
    digest = hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(artifact_mirror, "PUBLIC_PREFIX", url(dl_server, "/archive/"))
    monkeypatch.setattr(artifact_mirror, "github_asset_url", lambda _: url(dl_server, "/github/" + digest))
    candidates = ["github", "upstream", "archive"]
    for candidate in candidates[candidates.index(available):]:
        RangeHandler.payloads[f"/{candidate}/{digest}"] = body
    upstream = f"https://upstream.example.invalid/upstream/{digest}"
    # Resolve only the simulated remote upstream to the loopback server; our
    # release and R2 URLs remain ordinary HTTP test endpoints.
    real_opener = downloader._OPENER

    class FixtureUpstream:
        def open(self, request, timeout):
            if request.full_url == upstream:
                request = urllib.request.Request(url(dl_server, "/upstream/" + digest),
                                                 headers=dict(request.header_items()))
            return real_opener.open(request, timeout=timeout)

    monkeypatch.setattr(downloader, "_OPENER", FixtureUpstream())
    source = artifact_mirror.pinned_source(upstream, tmp_path / "input", digest)
    assert (source.url, *source.fallbacks) == (url(dl_server, "/github/" + digest), upstream,
                                               url(dl_server, "/archive/" + digest))
    assert Download([source], partials_dir=tmp_path / "partials").run() == [source.dest]
    assert source.dest.read_bytes() == body
    assert RangeHandler.ranges_seen[0][0] == f"/{available}/{digest}"


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_loopback_pin_uses_local_source_without_contacting_public_release(tmp_path, scheme):
    from pm.artifact_mirror import pinned_source, mirror_url

    digest = "b" * 64
    local = f"{scheme}://127.0.0.1:12345/tool.tar.gz"
    source = pinned_source(local, tmp_path / "tool", digest)
    assert (source.url, *source.fallbacks) == (local, mirror_url(digest))


def test_npm_artifacts_follow_the_users_npm_registry(tmp_path, monkeypatch):
    """#123132: lock URLs name registry.npmjs.org; ~/.npmrc or npm_config_registry picks the mirror."""
    from pm.artifact_mirror import pinned_source

    digest = "a" * 64
    lock_url = "https://registry.npmjs.org/npm/-/npm-10.9.2.tgz"
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    for key in [k for k in os.environ if k.lower().startswith("npm_config_")]:
        monkeypatch.delenv(key)
    from pm.artifact_mirror import github_asset_url, mirror_url
    release, r2 = github_asset_url(digest), mirror_url(digest)
    source = pinned_source(lock_url, tmp_path / "npm.tgz", digest)
    assert (source.url, *source.fallbacks) == (release, lock_url, r2)
    (tmp_path / ".npmrc").write_text("; corp\nregistry = https://npm.corp.example/npm/\n", encoding="utf-8")
    source = pinned_source(lock_url, tmp_path / "npm.tgz", digest)
    assert (source.url, *source.fallbacks) == ("https://npm.corp.example/npm/npm/-/npm-10.9.2.tgz", release, lock_url, r2)
    monkeypatch.setenv("NPM_CONFIG_REGISTRY", "https://env.corp.example")
    assert pinned_source(lock_url, tmp_path / "npm.tgz", digest).url == "https://env.corp.example/npm/-/npm-10.9.2.tgz"
    other = "https://github.com/x/y/releases/download/v1/y.tgz"
    source = pinned_source(other, tmp_path / "y.tgz", digest)
    assert (source.url, *source.fallbacks) == (release, other, r2)

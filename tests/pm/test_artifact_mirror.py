"""PM installation and cross-target staging consume the public pinned mirror."""
import hashlib
import importlib
import io
import os
import zipfile

import pytest

from pm import paths
from pm.lock import Facts, Lockfile
from pm.package import Package
from pm.store import Store, current_target
from tests.pm._range_server import RangeHandler, dl_server, url


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


def test_npm_artifacts_follow_the_users_npm_registry(tmp_path, monkeypatch):
    """#123132: lock URLs name registry.npmjs.org; ~/.npmrc or npm_config_registry picks the mirror."""
    from pm.artifact_mirror import pinned_source

    digest = "a" * 64
    lock_url = "https://registry.npmjs.org/npm/-/npm-10.9.2.tgz"
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    for key in [k for k in os.environ if k.lower().startswith("npm_config_")]:
        monkeypatch.delenv(key)
    assert pinned_source(lock_url, tmp_path / "npm.tgz", digest).url == lock_url
    (tmp_path / ".npmrc").write_text("; corp\nregistry = https://npm.corp.example/npm/\n", encoding="utf-8")
    assert pinned_source(lock_url, tmp_path / "npm.tgz", digest).url == "https://npm.corp.example/npm/npm/-/npm-10.9.2.tgz"
    monkeypatch.setenv("NPM_CONFIG_REGISTRY", "https://env.corp.example")
    assert pinned_source(lock_url, tmp_path / "npm.tgz", digest).url == "https://env.corp.example/npm/-/npm-10.9.2.tgz"
    other = "https://github.com/x/y/releases/download/v1/y.tgz"
    assert pinned_source(other, tmp_path / "y.tgz", digest).url == other

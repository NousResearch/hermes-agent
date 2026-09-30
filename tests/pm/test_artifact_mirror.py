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


def test_pool_pins_ladder_through_the_historical_twin():
    """A pool archive the pool retires stays fetchable under the same hash
    gate: the fallback ladder extends past the content-addressed mirror to the
    Internet Archive twin, and only for Termux pool URLs."""
    from pathlib import Path as _Path

    from pm.artifact_mirror import historical_url, mirror_url as mirror_of, pinned_source

    digest = "b" * 64
    pool = "https://packages.termux.dev/apt/termux-main/pool/main/u/uv/uv_0.12.20_aarch64.deb"
    source = pinned_source(pool, _Path("uv.deb"), digest)
    assert source.fallbacks == (
        mirror_of(digest),
        "https://archive.org/download/termux_pkgs_archive_u/uv/uv_0.12.20_aarch64.deb",
    )
    # A GitHub pin keeps only the content-addressed mirror.
    github = "https://github.com/x/y/releases/download/v1/y.tgz"
    assert pinned_source(github, _Path("y.tgz"), digest).fallbacks == (mirror_of(digest),)
    # Non-pool hosts never produce a twin.
    assert historical_url("https://example.test/pool/main/a/b/c.deb") is None
    assert historical_url("https://packages.termux.dev/other/path.deb") is None


def test_historical_twin_is_single_sourced_from_the_mirror_module():
    """Both the CI archiver and the client ladder derive the twin from
    pm.artifact_mirror.historical_url: the seed source and the client rung
    cannot drift apart."""
    from pm.artifact_mirror import historical_url as canonical
    from scripts.ci import archive_inputs

    assert archive_inputs.historical_url is canonical


def test_historical_paths_encode_semantic_segments_once():
    from pm.artifact_mirror import historical_url

    pool = "https://packages.termux.dev/apt/termux-main/pool/main/l/"
    raw = pool + "libc++/libc++_1_aarch64.deb"
    encoded = pool + "libc%2B%2B/libc%2B%2B_1_aarch64.deb"
    assert historical_url(raw) == historical_url(encoded)
    assert "libc%2B%2B" in historical_url(raw)
    for part in ("%2f", "%5c", "%2e%2e", "%GG"):
        assert historical_url(pool + part + "/file.deb") is None

"""PM installation and cross-target staging consume the public pinned mirror."""
import contextlib
import hashlib
from http.server import HTTPServer
import importlib
import io
import os
import threading
import zipfile

import pytest

from pm import paths
from pm.lock import Facts, Lockfile
from pm.package import Package
from pm.store import Store, current_target
from tests.pm._range_server import RangeHandler, url
from tests.pm._range_server import dl_server as dl_server  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import


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


def test_both_pnpm_archives_follow_the_users_npm_registry(tmp_path, monkeypatch):
    """pnpm's executable (a scoped @pnpm/exe.* package) and its launcher both come from the mirror."""
    from pm.artifact_mirror import pinned_source
    from pm.lock import Lockfile
    from pm.paths import lockfile_path

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("NPM_CONFIG_REGISTRY", "https://npm.corp.example/npm/")
    artifacts = [a for target in ("linux-x64", "win32-arm64") for a in Lockfile(lockfile_path()).artifacts("pnpm", target)]
    assert artifacts
    for artifact in artifacts:
        path = artifact["url"].removeprefix("https://registry.npmjs.org/")
        assert path != artifact["url"]
        source = pinned_source(artifact["url"], tmp_path / "pnpm.tgz", artifact["sha256"])
        assert source.url == "https://npm.corp.example/npm/" + path


_PNPM_PATH = "/@pnpm/exe.linux-x64/-/exe.linux-x64-12.15.0.tgz"
_PNPM_LOCK_URL = "https://registry.npmjs.org" + _PNPM_PATH


@pytest.fixture
def no_proxy_env(monkeypatch):
    """urllib honours proxy variables in any case; a developer's proxy must not carry 127.0.0.2."""
    for key in [k for k in os.environ if k.lower() in ("http_proxy", "https_proxy", "all_proxy", "no_proxy")]:
        monkeypatch.delenv(key)


@contextlib.contextmanager
def _lan_mirror(handler=RangeHandler):
    """A plain-http npm registry on an address the downloader does not treat as loopback."""
    server = HTTPServer(("127.0.0.2", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.2:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.platforms("linux")  # all of 127/8 answers on Linux; macOS binds only 127.0.0.1
@pytest.mark.parametrize("tampered", [False, True])
def test_plain_http_npm_mirror_serves_pinned_bytes(tmp_path, monkeypatch, dl_server, no_proxy_env, tampered):
    """#150032: an http LAN registry in ~/.npmrc may serve a pinned artifact; the pin still decides."""
    from pm.artifact_mirror import pinned_source
    from pm.downloader import Download, HashError

    body = b"pinned pnpm tarball"
    RangeHandler.payloads[_PNPM_PATH] = b"tampered with mirror" if tampered else body
    with _lan_mirror() as lan_mirror:
        monkeypatch.setenv("NPM_CONFIG_REGISTRY", lan_mirror)
        source = pinned_source(_PNPM_LOCK_URL, tmp_path / "pnpm.tgz", hashlib.sha256(body).hexdigest())
        assert source.url.startswith(lan_mirror)
        download = Download([source], partials_dir=tmp_path / "partials")
        if tampered:
            with pytest.raises(HashError):
                download.run()
            assert not source.dest.exists()
        else:
            assert download.run() == [source.dest]
            assert source.dest.read_bytes() == body


@pytest.mark.parametrize("source", ["lock row", "fallback"])
def test_plain_http_is_only_for_the_users_npm_mirror(tmp_path, source):
    """A pinned lock row or archive on http stays refused: only the npmrc registry chose http."""
    from pm.downloader import Download, Source

    digest = "a" * 64
    http_url = "http://127.0.0.2:9/tool.tgz"  # refuses at once if the policy ever dials it
    plan = (Source(http_url, tmp_path / "t", digest) if source == "lock row"
            else Source("https://example.invalid/tool.tgz", tmp_path / "t", digest, fallbacks=(http_url,)))
    with pytest.raises(ValueError, match="non-https"):
        Download([plan], partials_dir=tmp_path / "p").run()


def test_plain_http_stays_refused_without_a_pin(tmp_path):
    """Unpinned downloads (model catalog) have no hash to fall back on, so http is still refused."""
    from pm.downloader import Download, Source

    with pytest.raises(ValueError, match="non-https"):
        Download([Source("http://192.168.1.20:4873/model.gguf", tmp_path / "m")], partials_dir=tmp_path / "p").run()

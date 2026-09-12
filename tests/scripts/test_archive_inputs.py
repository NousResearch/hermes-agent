"""CI archives exact pinned inputs and supplies the real staging consumers."""
import hashlib
import importlib
import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from pm.artifact_mirror import object_key
from pm.downloader import HashError
from scripts.ci import archive_inputs as inputs
from scripts.releases import r2
from tests.scripts.test_release_r2 import r2_server  # noqa: F401
from tests.test_termux_runtime_libs import _Server, _build_deb


def write_pins(repo, packages, libs=None):
    (repo / "pm").mkdir(parents=True, exist_ok=True)
    (repo / "pm/lock.json").write_text(json.dumps({"schema": 1, "packages": packages}), encoding="utf-8")
    (repo / "scripts/termux").mkdir(parents=True, exist_ok=True)
    (repo / "scripts/termux/runtime_libs.json").write_text(json.dumps(libs or {"libs": {}}), encoding="utf-8")


@pytest.fixture
def upstream(tmp_path):
    root = tmp_path / "upstream"
    root.mkdir()
    _build_deb(root / "lib.deb", "libarchive-proof.so", b"pinned library")
    server = _Server(root)
    try:
        yield server, root
    finally:
        server.stop()


def test_target_selection_preserves_multi_archive_and_any_fallback(tmp_path):
    def row(name):
        return {"url": f"https://upstream.test/{name}.zip", "sha256": hashlib.sha256(name.encode()).hexdigest()}
    packages = {
        "engine": {"version": "1", "artifacts": {
            "win32-x64": [row("engine"), row("cudart")], "linux-x64": row("linux"),
        }},
        "portable": {"version": "1", "artifacts": {"any": row("portable")}},
        "container": {"version": "1", "artifacts": {"linux-arm64-bionic": {
            "url": "docker://termux/termux-docker@sha256:" + "a" * 64,
        }}},
    }
    lib = row("lib")
    write_pins(tmp_path, packages, {"libs": {"lib": lib}, "licenses": row("license")})
    win = inputs.pinned_inputs(tmp_path, target="win32-x64")
    assert {p.sha256 for p in win} == {row(n)["sha256"] for n in ("engine", "cudart", "portable")}
    bionic = inputs.pinned_inputs(tmp_path, target="linux-arm64-bionic")
    assert {p.sha256 for p in bionic} == {row(n)["sha256"] for n in ("portable", "lib", "license")}
    all_pins = inputs.pinned_inputs(tmp_path)
    assert {p.sha256 for p in all_pins} == {row(n)["sha256"] for n in ("engine", "cudart", "portable", "linux", "lib", "license")}


@pytest.mark.parametrize("bad", [{}, {"url": "https://u/file.zip"}, {"url": "http://u/file.zip", "sha256": "a" * 64}, {"url": "https://u/file.zip", "sha256": " A "}])
def test_invalid_pins_fail_without_rewriting_authority(tmp_path, bad):
    write_pins(tmp_path, {"bad": {"version": "1", "artifacts": {"any": bad}}})
    before = (tmp_path / "pm/lock.json").read_bytes()
    with pytest.raises(ValueError):
        inputs.pinned_inputs(tmp_path)
    assert (tmp_path / "pm/lock.json").read_bytes() == before


def test_real_cli_miss_hit_and_staging_use_the_same_archived_bytes(tmp_path, upstream, r2_server, monkeypatch):
    from pm import paths
    from pm.store import Store
    from scripts.termux.stage_runtime_libs import stage
    from tests.pm.test_stage_only import _FakePackage

    server, root = upstream
    body = (root / "lib.deb").read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    row = {"url": server.url + "/lib.deb", "sha256": digest, "version": "1"}
    repo = tmp_path / "repo"
    write_pins(repo, {"stage-test": {"version": "1", "artifacts": {"linux-arm64-bionic": row}}}, {"libs": {"lib": row}})
    monkeypatch.setattr(paths, "repo_root", lambda: repo)
    monkeypatch.setattr(paths, "lockfile_path", lambda: repo / "pm/lock.json")
    assert inputs.main([]) == 0
    assert r2_server.store[object_key(digest)][0] == body
    puts = [r for r in r2_server.requests if r[0] == "PUT"]
    assert len(puts) == 1 and puts[0][2]["If-None-Match"] == "*"
    assert r2_server.requests[-1][0] == "GET"
    assert not (repo / ".archive-inputs").exists()
    (root / "lib.deb").unlink()
    r2_server.requests.clear()
    store = Store(tmp_path / "tools")
    payload = tmp_path / "payload"
    assert inputs.main(["--target", "linux-arm64-bionic", "--store", str(store.root), "--payload", str(payload)]) == 0
    assert not any(r[0] == "PUT" for r in r2_server.requests)
    assert (stage(payload, {"lib": row}) / "libarchive-proof.so").read_bytes().endswith(b"pinned library")
    engine = importlib.import_module("pm.ensure")
    monkeypatch.setattr(engine, "_store", lambda: store)
    monkeypatch.setattr(engine, "get_package", lambda _: _FakePackage())
    entry = engine.stage_only("stage-test", "linux-arm64-bionic")
    assert (entry / "bin/tool").read_bytes() == body
    assert not store.entry(f"fetch-{digest}").exists()
    assert r2.canary_doomed_keys([object_key(digest)], "99999999") == []


@pytest.mark.parametrize("failure", ["upstream-hash", "r2-hash", "r2-permission", "readback"])
def test_corrupt_or_denied_archive_never_publishes_a_destination(tmp_path, upstream, r2_server, monkeypatch, failure):
    server, root = upstream
    body = (root / "lib.deb").read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    pin = inputs.InputPin("lib", server.url + "/lib.deb", digest, "library")
    key = object_key(digest)
    if failure == "upstream-hash":
        (root / "lib.deb").write_bytes(b"corrupt")
    elif failure == "r2-hash":
        r2_server.store[key] = (b"corrupt", '"etag"')
    elif failure == "readback":
        original = r2.put_object
        def corrupt(*args, **kwargs):
            original(*args, **kwargs)
            r2_server.store[key] = (b"x" * len(body), '"etag"')
        monkeypatch.setattr(r2, "put_object", corrupt)
    else:
        original = r2.signed_request
        def deny(method, url, **kwargs):
            if method == "HEAD":
                raise r2.R2RequestError(method, url, 403)
            return original(method, url, **kwargs)
        monkeypatch.setattr(r2, "signed_request", deny)
    dest = tmp_path / "preserved"
    dest.write_bytes(b"old")
    with pytest.raises((HashError, ValueError, r2.R2RequestError)):
        inputs.Archive(*r2.credentials()).fetch(pin, dest)
    assert dest.read_bytes() == b"old"
    assert any(r[0] == "PUT" for r in r2_server.requests) == (failure == "readback")


def test_racing_misses_verify_the_immutable_winner(tmp_path, upstream, r2_server, monkeypatch):
    server, root = upstream
    body = (root / "lib.deb").read_bytes()
    pin = inputs.InputPin("lib", server.url + "/lib.deb", hashlib.sha256(body).hexdigest(), "library")
    barrier = threading.Barrier(2)
    original = r2.signed_request
    def race(method, url, **kwargs):
        try:
            return original(method, url, **kwargs)
        except r2.R2RequestError as exc:
            if method == "HEAD" and exc.status == 404:
                barrier.wait(timeout=10)
            raise
    monkeypatch.setattr(r2, "signed_request", race)
    archive = inputs.Archive(*r2.credentials())
    paths = [tmp_path / f"race-{i}" for i in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda p: archive.fetch(pin, p), paths)) == ["upstream", "upstream"]
    assert all(p.read_bytes() == body for p in paths)


def test_all_digests_start_together_and_seed_every_reference(tmp_path, upstream, r2_server, monkeypatch):
    from collections import Counter
    from pm.store import Store
    from scripts.termux.stage_runtime_libs import download_path

    server, root = upstream
    bodies = {f"input-{i}": f"distinct pinned bytes {i}".encode() for i in range(9)}
    pins = []
    for name, body in bodies.items():
        (root / f"{name}.deb").write_bytes(body)
        digest = hashlib.sha256(body).hexdigest()
        for kind, label in (("tool", name), ("library", name), ("library", f"{name}-alias")):
            pins.append(inputs.InputPin(label, f"{server.url}/{name}.deb", digest, kind))

    # Every unique input must reach the real HTTP path before any can finish.
    barrier = threading.Barrier(len(bodies))
    original = r2.signed_request
    def together(method, url, **kwargs):
        if method == "HEAD":
            barrier.wait(timeout=10)
        return original(method, url, **kwargs)
    monkeypatch.setattr(r2, "signed_request", together)
    store = Store(tmp_path / "tools")
    payload = tmp_path / "payload"
    assert inputs.stage_inputs(pins, archive=inputs.Archive(*r2.credentials()),
                               store=store, payload=payload) == len(bodies)
    puts = Counter(path for method, path, _ in r2_server.requests if method == "PUT")
    assert len(puts) == len(bodies) and set(puts.values()) == {1}
    for name, body in bodies.items():
        digest = hashlib.sha256(body).hexdigest()
        assert r2_server.store[object_key(digest)][0] == body
        assert (store.entry(f"fetch-{digest}") / f"{name}.deb").read_bytes() == body
        for label in (name, f"{name}-alias"):
            assert download_path(payload, label).read_bytes() == body


def test_parallel_readback_failure_reaches_cli_and_preserves_destination(tmp_path, upstream, r2_server, monkeypatch, capsys):
    from pm import paths
    from scripts.termux.stage_runtime_libs import download_path

    server, root = upstream
    packages, libs = {}, {}
    for name in ("good", "bad"):
        body = name.encode()
        (root / f"{name}.deb").write_bytes(body)
        row = {"url": f"{server.url}/{name}.deb", "sha256": hashlib.sha256(body).hexdigest()}
        libs[name] = row
        packages[name] = {"version": "1", "artifacts": {"any": row}}
        r2_server.store[object_key(row["sha256"])] = (body if name == "good" else b"xxx", '"etag"')
    repo = tmp_path / "repo"
    write_pins(repo, packages, {"libs": libs})
    monkeypatch.setattr(paths, "repo_root", lambda: repo)
    payload = tmp_path / "payload"
    preserved = download_path(payload, "bad")
    preserved.parent.mkdir(parents=True)
    preserved.write_bytes(b"existing destination")
    barrier = threading.Barrier(len(libs))
    original = r2.signed_request
    def together(method, url, **kwargs):
        if method == "HEAD":
            barrier.wait(timeout=10)
        return original(method, url, **kwargs)
    monkeypatch.setattr(r2, "signed_request", together)
    with pytest.raises(ValueError, match="checksum mismatch"):
        inputs.main(["--payload", str(payload), "--store", str(tmp_path / "tools")])
    assert preserved.read_bytes() == b"existing destination"
    assert not (tmp_path / "tools" / f"fetch-{libs['bad']['sha256']}").exists()
    assert download_path(payload, "good").read_bytes() == b"good"
    assert "Verified " not in capsys.readouterr().out


def test_historical_recovery_keeps_the_original_digest(tmp_path, upstream, r2_server, monkeypatch):
    server, root = upstream
    body = (root / "lib.deb").read_bytes()
    pin = inputs.InputPin("lib", server.url + "/gone.deb", hashlib.sha256(body).hexdigest(), "library")
    monkeypatch.setattr(inputs, "historical_url", lambda _: server.url + "/lib.deb")
    dest = tmp_path / "recovered"
    assert inputs.Archive(*r2.credentials()).fetch(pin, dest) == "historical archive"
    assert dest.read_bytes() == body


def test_committed_inventory_matches_every_http_pin():
    repo = Path(__file__).resolve().parents[2]
    lock = json.loads((repo / "pm/lock.json").read_text(encoding="utf-8"))
    expected = set()
    for package in lock["packages"].values():
        for artifact in package.get("artifacts", {}).values():
            for row in artifact if isinstance(artifact, list) else [artifact]:
                if row["url"].startswith("https://"):
                    expected.add(row["sha256"])
    table = json.loads((repo / "scripts/termux/runtime_libs.json").read_text(encoding="utf-8"))
    expected.update(row["sha256"] for row in table["libs"].values())
    expected.add(table["licenses"]["sha256"])
    assert {p.sha256 for p in inputs.pinned_inputs(repo)} == expected


def test_ci_toolchain_seed_runs_before_the_tool_installer(tmp_path, upstream, r2_server, monkeypatch):
    from types import SimpleNamespace
    from scripts.ci import setup_toolchain
    from pm import paths
    from pm.store import Store, current_target

    server, root = upstream
    body = (root / "lib.deb").read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    target = current_target()
    row = {"url": server.url + "/lib.deb", "sha256": digest}
    packages = {name: {"version": "1", "artifacts": {target: row}} for name in ("python", "uv")}
    repo = tmp_path / "repo"
    write_pins(repo, packages)
    monkeypatch.setattr(paths, "repo_root", lambda: repo)
    home = tmp_path / "ci-home"
    setup_toolchain.archive_inputs(SimpleNamespace(home=home, toolchain="python"))
    (root / "lib.deb").unlink()
    store = Store(home / "tools")
    with store.scratch() as scratch:
        assert store.fetch(row["url"], digest, scratch).read_bytes() == body

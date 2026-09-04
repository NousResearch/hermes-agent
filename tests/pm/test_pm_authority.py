"""pm authority spine: digest-bound facts, tree_digest + doctor re-hash,
adopt() verification gate, repair logging.

Same conventions as test_pm_core: real loopback server, real archives,
real store — no mocked stores. Assertions are relationships (identity
matches lock, digest matches bytes), not snapshots."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
import tarfile
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest

import pm.paths as paths
import pm.registry as registry
from pm.lock import Facts, Lockfile
from pm.package import Package
from pm.packages import BinaryPackage
from pm.store import Store, current_target, tree_digest


class FakeTool(BinaryPackage):
    name = "faketool"
    probe_version = False
    binary_rel = {"win32": "bin/faketool", "posix": "bin/faketool"}

    def fetch_url(self, version, target):
        return f"{FakeTool.base_url}/{self.name}-{version}.tar.gz"


def make_tar(docroot: Path, name: str, files: dict[str, str]) -> tuple[str, str]:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(data))
    payload = buf.getvalue()
    (docroot / name).write_bytes(payload)
    return name, hashlib.sha256(payload).hexdigest()


@pytest.fixture
def served(tmp_path):
    docroot = tmp_path / "www"
    docroot.mkdir()
    handler = partial(SimpleHTTPRequestHandler, directory=str(docroot))
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield docroot, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture
def pm_env(tmp_path, served, monkeypatch):
    docroot, base_url = served
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(runtime))
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    import importlib

    ensure_mod = importlib.import_module("pm.ensure")
    monkeypatch.setattr(ensure_mod, "lazy_installs_allowed", lambda: True)
    paths._stamp.cache_clear()

    saved = dict(registry._packages)
    registry._packages.clear()
    registry._packages[FakeTool.name] = FakeTool()
    FakeTool.base_url = base_url

    lock_dir = tmp_path / "repo-lock"
    lock_dir.mkdir()
    lockfile_path = lock_dir / "lock.json"
    monkeypatch.setattr(paths, "lockfile_path", lambda: lockfile_path)

    name, digest = make_tar(docroot, "faketool-1.0.tar.gz", {"bin/faketool": "#!x"})
    lockfile = Lockfile(lockfile_path)
    lockfile.set_pin(
        "faketool", "1.0",
        {"any": {"url": f"{base_url}/faketool-1.0.tar.gz", "sha256": digest}},
    )
    lockfile.save()

    yield {
        "lockfile_path": lockfile_path,
        "runtime": runtime,
        "docroot": docroot,
        "base_url": base_url,
        "digest": digest,
        "tmp_path": tmp_path,
        "monkeypatch": monkeypatch,
    }

    registry._packages.clear()
    registry._packages.update(saved)


def _pin(env, name: str, version: str, digest: str) -> None:
    lockfile = Lockfile(env["lockfile_path"])
    url = f"{env['base_url']}/{name}-{version}.tar.gz"
    lockfile.set_pin(name, version, {"any": {"url": url, "sha256": digest}})
    lockfile.save()


# ── item 1: digest-bound facts ────────────────────────────────────────


def test_same_version_different_sha_is_not_installed_and_repaired(pm_env):
    """The witness: same version, different artifact sha. Version/path
    matching cannot see this; identity matching must — check() reports
    it and ensure() replaces the entry bytes."""
    from pm.ensure import check, ensure, is_installed

    env = pm_env
    ensure("faketool", base_env={})
    assert is_installed("faketool")

    _, digest_b = make_tar(env["docroot"], "faketool-1.0-b.tar.gz", {"bin/faketool": "#!B"})
    # Same VERSION, different archive (different url + sha) — the
    # version-only check would accept this re-pin.
    lockfile = Lockfile(env["lockfile_path"])
    lockfile.set_pin(
        "faketool", "1.0",
        {"any": {"url": f"{env['base_url']}/faketool-1.0-b.tar.gz", "sha256": digest_b}},
    )
    lockfile.save()
    assert not is_installed("faketool")
    assert check() == ["faketool: not installed or outdated"]

    runner = ensure("faketool", base_env={})
    assert is_installed("faketool")
    fact = Facts(paths.facts_path()).get("faketool")
    entry_bin = paths.store_root() / fact["entry"] / "bin" / "faketool"
    with open(entry_bin, "rb") as f:
        assert f.read() == b"#!B"

    # The fact now binds the new identity, not the old one.
    fact = Facts(paths.facts_path()).get("faketool")
    assert fact["target"] == current_target()
    assert fact["artifacts"] == [digest_b]


def test_poisoned_fetch_cache_is_redownloaded(pm_env):
    """The fetch cache is keyed on the FULL sha and re-hashed before it is
    trusted; poisoned bytes are deleted and re-downloaded."""
    env = pm_env
    runtime = env["runtime"]
    store = Store(runtime / "scratch-store")
    url = f"{env['base_url']}/faketool-1.0.tar.gz"
    digest = env["digest"]

    first = store.fetch(url, digest, runtime / "scratch-store")
    assert sha_of(first) == digest

    # Poison the cached entry: same name, wrong bytes.
    with open(first, "wb") as f:
        f.write(b"poisoned")

    second = store.fetch(url, digest, runtime / "scratch-store")
    assert sha_of(second) == digest


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_legacy_fact_without_identity_is_not_installed(pm_env):
    """Facts written before identity existed read back fine but are NOT
    vouchable: installed() with identity returns False and forces one
    reinstall."""
    from pm.ensure import ensure, is_installed

    env = pm_env
    ensure("faketool", base_env={})

    facts_path = paths.facts_path()
    data = json.loads(facts_path.read_text(encoding="utf-8"))
    data["packages"]["faketool"] = {
        "entry": data["packages"]["faketool"]["entry"],
        "version": "1.0",
        "env": data["packages"]["faketool"]["env"],
    }
    facts_path.write_text(json.dumps(data), encoding="utf-8")

    assert not is_installed("faketool")
    # ensure() repairs it: reinstalls and records the full identity.
    ensure("faketool", base_env={})
    assert is_installed("faketool")
    fact = Facts(paths.facts_path()).get("faketool")
    assert fact["artifacts"] == [env["digest"]]


def test_doctor_flags_legacy_fact(pm_env, capsys):
    from pm.cli import cmd_doctor
    from pm.ensure import ensure

    env = pm_env
    ensure("faketool", base_env={})
    facts_path = paths.facts_path()
    data = json.loads(facts_path.read_text(encoding="utf-8"))
    for key in ("target", "artifacts", "digest"):
        data["packages"]["faketool"].pop(key, None)
    facts_path.write_text(json.dumps(data), encoding="utf-8")

    assert cmd_doctor(None) == 1
    assert "legacy fact: no recorded identity" in capsys.readouterr().out


# ── item 2: realized digest + doctor re-hash ──────────────────────────


def test_tree_digest_is_content_bound(pm_env):
    """Same content in different creation order digests identically; one
    changed byte digests differently; a symlink contributes its link
    TARGET TEXT, not the target's bytes."""
    import os

    a = pm_env["tmp_path"] / "tree-a"
    b = pm_env["tmp_path"] / "tree-b"
    (a / "sub").mkdir(parents=True)
    (b / "sub").mkdir(parents=True)
    (a / "sub" / "f.txt").write_text("hello")
    (b / "sub" / "f.txt").write_text("hello")
    (a / "top.txt").write_text("t")
    (b / "top.txt").write_text("t")
    assert tree_digest(a) == tree_digest(b)

    (b / "top.txt").write_text("u")
    assert tree_digest(a) != tree_digest(b)

    # Symlink: link text is the data — retargeting changes the digest
    # even though the pointed-at bytes never change.
    (a / "sub" / "f.txt").write_text("hello")
    try:
        (b / "link").symlink_to("sub/f.txt")
        d1 = tree_digest(b)
        (b / "link").unlink()
        (b / "link").symlink_to("top.txt")
        assert tree_digest(b) != d1
    except OSError:
        pytest.skip("symlinks unavailable on this host")


def test_doctor_flags_tampered_entry_bytes(pm_env, capsys):
    """Post-install tampering: doctor re-hashes the realized tree against
    the recorded digest and flags it; restoring the bytes clears it."""
    from pm.cli import cmd_doctor
    from pm.ensure import ensure

    env = pm_env
    ensure("faketool", base_env={})
    assert cmd_doctor(None) == 0

    fact = Facts(paths.facts_path()).get("faketool")
    binary = paths.store_root() / fact["entry"] / "bin" / "faketool"
    original = binary.read_bytes()
    binary.write_bytes(original + b"tampered")
    assert cmd_doctor(None) == 1
    assert "realized bytes do not match recorded digest" in capsys.readouterr().out

    binary.write_bytes(original)
    assert cmd_doctor(None) == 0


# ── item 4: adopt() verification gate ─────────────────────────────────


def _bundle_payload(env) -> Path:
    """Build a shipped payload the way cmd_bundle lays it out: a repo
    snapshot carrying pm/lock.json, and a store whose facts.json holds
    the installed state. adopt()'s paths point here via repo_root."""
    from pm.ensure import ensure

    monkeypatch = env["monkeypatch"]
    tmp_path = env["tmp_path"]
    shipped_repo = tmp_path / "shipped-repo"
    (shipped_repo / "pm").mkdir(parents=True)
    shutil.copy(env["lockfile_path"], shipped_repo / "pm" / "lock.json")
    monkeypatch.setattr(paths, "repo_root", lambda: shipped_repo)
    ensure("faketool", base_env={})
    return shipped_repo


def test_adopt_adopts_intact_payload(pm_env):
    from pm.ensure import adopt

    _bundle_payload(pm_env)
    marker = paths.store_root().parent / ".adopted"
    assert not marker.is_file()
    assert adopt() is True
    assert marker.is_file()
    # Idempotent: already adopted → False, marker untouched.
    assert adopt() is False


def test_adopt_refuses_on_missing_staged_binary(pm_env):
    from pm.ensure import adopt

    _bundle_payload(pm_env)
    fact = Facts(paths.facts_path()).get("faketool")
    binary = paths.store_root() / fact["entry"] / "bin" / "faketool"
    binary.unlink()

    marker = paths.store_root().parent / ".adopted"
    assert adopt() is False
    assert not marker.is_file()


def test_adopt_refuses_on_tampered_staged_binary(pm_env):
    """The adversarial witness: bytes substituted AFTER install fail even
    though facts + lock still agree — the digest is computed over the
    actual bytes, not read from the manifest."""
    from pm.ensure import adopt

    _bundle_payload(pm_env)
    fact = Facts(paths.facts_path()).get("faketool")
    binary = paths.store_root() / fact["entry"] / "bin" / "faketool"
    binary.write_bytes(b"#!substituted")

    marker = paths.store_root().parent / ".adopted"
    assert adopt() is False
    assert not marker.is_file()


def test_adopt_refuses_when_fact_lacks_identity(pm_env):
    from pm.ensure import adopt

    _bundle_payload(pm_env)
    facts_path = paths.facts_path()
    data = json.loads(facts_path.read_text(encoding="utf-8"))
    for key in ("target", "artifacts", "digest"):
        data["packages"]["faketool"].pop(key, None)
    facts_path.write_text(json.dumps(data), encoding="utf-8")

    assert adopt() is False
    assert not (paths.store_root().parent / ".adopted").is_file()


# ── item 6: repair logging ────────────────────────────────────────────


def test_gc_protects_in_flight_partials(pm_env):
    """gc sweeps the store root and the partials area, but partials an
    in-flight download still owns (fresh mtimes) survive; stale partials
    are swept."""
    from pm.cli import cmd_gc

    env = pm_env
    runtime = env["runtime"]
    runtime.mkdir(parents=True, exist_ok=True)
    partials = paths.partials_root()
    partials.mkdir(parents=True, exist_ok=True)
    fresh = partials / "abc.part"
    fresh.write_bytes(b"x")
    stale = partials / "old.part"
    stale.write_bytes(b"y")
    import os
    import time

    old = time.time() - 7 * 3600  # beyond the 6h gc grace window
    os.utime(stale, (old, old))

    orphan = runtime / "orphan-9.9-nowhere"
    orphan.mkdir()

    cmd_gc(None)
    assert fresh.is_file()
    assert not stale.exists()
    assert not orphan.exists()


def test_repair_log_line_on_reinstall(pm_env, caplog):
    """Established fact + missing entry + ensure() → one repair log line,
    and the install completes."""
    from pm.ensure import ensure, is_installed

    env = pm_env
    ensure("faketool", base_env={})

    fact = Facts(paths.facts_path()).get("faketool")
    shutil.rmtree(paths.store_root() / fact["entry"])

    with caplog.at_level(logging.INFO, logger="pm.ensure"):
        ensure("faketool", base_env={})
    assert is_installed("faketool")
    repairs = [r for r in caplog.records if "repair:" in r.getMessage()]
    assert any(
        "repair: faketool re-realized" in r.getMessage() for r in repairs
    )


def test_repair_log_line_on_same_version_repin(pm_env, caplog):
    from pm.ensure import ensure

    env = pm_env
    ensure("faketool", base_env={})
    _, digest_b = make_tar(env["docroot"], "faketool-1.0-b.tar.gz", {"bin/faketool": "#!B"})
    lockfile = Lockfile(env["lockfile_path"])
    lockfile.set_pin(
        "faketool", "1.0",
        {"any": {"url": f"{env['base_url']}/faketool-1.0-b.tar.gz", "sha256": digest_b}},
    )
    lockfile.save()

    with caplog.at_level(logging.INFO, logger="pm.ensure"):
        ensure("faketool", base_env={})
    repairs = [r.getMessage() for r in caplog.records if "repair:" in r.getMessage()]
    assert any("1.0" in msg and env["digest"][:12] in msg for msg in repairs)


def test_tree_digest_ignores_pycache(pm_env, monkeypatch):
    """Bytecode caches are runtime state, not package bytes: the staged
    python entry runs (uv venv/uv sync in a bundle build, first boot of a
    shipped app) and CPython writes __pycache__/*.pyc into it AFTER the
    digest was recorded. The digest must stay stable across that, while
    tampering a REAL file inside a dir that merely sits beside a
    __pycache__ is still caught. (Live field failure: '✗ python: realized
    bytes do not match recorded digest' on every bundled-release smoke.)"""
    tree = pm_env["tmp_path"] / "py-entry"
    (tree / "Lib").mkdir(parents=True)
    (tree / "Lib" / "os.py").write_text("print('stdlib')")

    before = tree_digest(tree)

    # The interpreter "ran": pyc caches appeared deep in the tree.
    cache = tree / "Lib" / "__pycache__"
    cache.mkdir()
    (cache / "os.cpython-311.pyc").write_bytes(b"\x00compiled-bytes\x00")
    nested = tree / "Lib" / "json" / "__pycache__"
    nested.mkdir(parents=True)
    (nested / "tool.cpython-311.pyc").write_bytes(b"\x00more\x00")

    assert tree_digest(tree) == before

    # A cache write is not a mask: changing a real .py still trips the digest.
    (tree / "Lib" / "os.py").write_text("print('tampered')")
    assert tree_digest(tree) != before

    # Restoring the .py (leaving the caches) restores the digest: caches
    # contribute nothing in either direction.
    (tree / "Lib" / "os.py").write_text("print('stdlib')")
    assert tree_digest(tree) == before

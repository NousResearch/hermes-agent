"""pm authority spine: digest-bound facts, tree_digest + doctor re-hash,
adopt() verification gate, repair logging.

Same conventions as test_pm_core: real loopback server, real archives,
real store — no mocked stores. Assertions are relationships (identity
matches lock, digest matches bytes), not snapshots."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

import pytest

import pm.paths as paths
import pm.registry as registry
from pm.lock import Facts, Lockfile
from pm.store import Store, current_target, tree_digest
from tests.pm._fixtures import make_tar, served as served
from tests.pm.test_pm_core import FakeTool, pm_env as core_env  # noqa: F401


@pytest.fixture
def pm_env(core_env, tmp_path, monkeypatch):
    lock, runtime, docroot, base_url = core_env
    return {"lockfile_path": lock, "runtime": runtime, "docroot": docroot,
            "base_url": base_url, "tmp_path": tmp_path, "monkeypatch": monkeypatch,
            "digest": Lockfile(lock).artifacts("faketool", current_target())[0]["sha256"]}


@pytest.mark.parametrize("route", ["install", "stage"])
def test_same_version_different_sha_is_not_installed_and_repaired(pm_env, caplog, route):
    """The witness: same version, different artifact sha. Version/path
    matching cannot see this; identity matching must — check() reports
    it and ensure() replaces the entry bytes."""
    from pm.ensure import check, ensure, is_installed

    env = pm_env
    from pm.ensure import stage_only
    realize = (lambda: ensure("faketool", explicit=True, base_env={})) if route == "install" else (
        lambda: stage_only("faketool", "linux-arm64-bionic"))
    realize()
    if route == "install":
        assert is_installed("faketool")
    else:
        entry = realize()
        (entry / "sentinel").write_text("published")
        assert realize() == entry
        assert (entry / "sentinel").read_text() == "published"

    _, digest_b = make_tar(env["docroot"], "faketool-1.0-b.tar.gz", {"bin/faketool": "#!B"})
    # Same VERSION, different archive (different url + sha) — the
    # version-only check would accept this re-pin.
    lockfile = Lockfile(env["lockfile_path"])
    lockfile.set_pin(
        "faketool", "1.0",
        {"any": {"url": f"{env['base_url']}/faketool-1.0-b.tar.gz", "sha256": digest_b}},
    )
    lockfile.save()
    if route == "stage":
        assert realize() == entry
        assert (entry / "bin/faketool").read_bytes() == b"#!B"
        assert not (entry / "sentinel").exists()
        assert realize() == entry
        assert not paths.facts_path().exists()
        return
    assert not is_installed("faketool")
    assert check() == ["faketool: not installed or outdated"]

    with caplog.at_level(logging.INFO, logger="pm.ensure"):
        realize()
    assert any("1.0" in r.message and env["digest"][:12] in r.message for r in caplog.records)
    assert is_installed("faketool")
    fact = Facts(paths.facts_path()).get("faketool")
    entry_bin = paths.store_root() / fact["entry"] / "bin" / "faketool"
    with open(entry_bin, "rb") as f:
        assert f.read() == b"#!B"

    # The fact now binds the new identity, not the old one.
    fact = Facts(paths.facts_path()).get("faketool")
    assert fact["target"] == current_target()
    assert fact["artifacts"] == [digest_b]


@pytest.mark.parametrize("matching_fact", [False, True])
@pytest.mark.parametrize("damage", ["missing-binary", "changed-bytes", "entry-is-file", "missing-entry"])
def test_install_repairs_corrupt_entry_from_verified_archive(pm_env, matching_fact, damage, caplog):
    from pm.cli import cmd_doctor
    from pm.ensure import ensure

    ensure("faketool", base_env={})
    facts_path = paths.facts_path()
    fact = Facts(facts_path).get("faketool")
    entry = paths.store_root() / fact["entry"]
    binary = entry / "bin" / "faketool"
    if damage == "missing-binary":
        binary.unlink()
    elif damage == "changed-bytes":
        binary.write_bytes(b"corrupted")
    else:
        shutil.rmtree(entry)
        if damage == "entry-is-file":
            entry.write_bytes(b"not a directory")
    if not matching_fact:
        data = json.loads(facts_path.read_text())
        data["packages"]["faketool"].pop("artifacts")
        facts_path.write_text(json.dumps(data))

    with caplog.at_level(logging.INFO, logger="pm.ensure"):
        ensure("faketool", explicit=True, base_env={})
    assert any("repair: faketool re-realized" in r.message for r in caplog.records)
    assert binary.read_bytes() == b"#!x"
    assert cmd_doctor(None) == 0
    assert Facts(facts_path).get("faketool")["digest"] == tree_digest(binary.parent.parent)


@pytest.mark.parametrize("route,failure", [
    (route, failure) for route in ("install", "stage")
    for failure in ("invalid", "fetch", "unpack", "verify", "publish", "published-verify", "interrupt", "facts")
    if route == "install" or failure != "facts"
])
@pytest.mark.parametrize("initial", [False, True], ids=["replacement", "first-install"])
def test_failed_replacement_preserves_entry_and_facts(pm_env, monkeypatch, route, initial, failure):
    from functools import partial
    from pm.ensure import ensure, stage_only
    from pm.package import InstallError

    env = pm_env
    target = current_target() if route == "install" else "linux-arm64-bionic"
    realize = partial(ensure, "faketool", explicit=True, base_env={}) if route == "install" else partial(
        stage_only, "faketool", target)
    if not initial:
        realize()
    facts_path = paths.facts_path()
    old_facts = facts_path.read_bytes() if facts_path.exists() else None
    entry = paths.store_root() / FakeTool().store_entry("1.0", target)
    marker = entry / ".pm-stage-pin.json"
    old_marker = marker.read_bytes() if marker.exists() else None
    files = {"bin/unrelated": "bad layout"} if failure == "invalid" else {"bin/faketool": "#!new"}
    _, digest = make_tar(env["docroot"], "replacement.tar.gz", files)
    _, extra_digest = make_tar(env["docroot"], "data.tar.gz", {"share/data": "second archive"})
    lockfile = Lockfile(env["lockfile_path"])
    lockfile.set_pin("faketool", "1.0", {"any": [
        {"url": f"{env['base_url']}/replacement.tar.gz", "sha256": digest},
        {"url": f"{env['base_url']}/data.tar.gz", "sha256": extra_digest},
    ]})
    lockfile.save()
    package = registry._packages["faketool"]

    def fail(*args, **kwargs):
        if failure == "interrupt":
            raise KeyboardInterrupt()
        if failure in ("publish", "facts"):
            raise OSError("injected failure")
        raise InstallError("faketool", "injected failure")

    with monkeypatch.context() as fault:
        if failure in ("publish", "published-verify", "interrupt"):
            original = Store.publish
            def publish(self, staged, name):
                if name != entry.name:
                    return original(self, staged, name)
                if failure == "published-verify":
                    published = original(self, staged, name)
                    (published / "bin/faketool").unlink()
                    return published
                fail()
            fault.setattr(Store, "publish", publish)
        elif failure == "facts":
            fault.setattr(Facts, "record", fail)
        elif failure == "fetch":
            fault.setattr(Store, "fetch_many", fail)
        elif failure == "verify":
            original_verify = package.verify
            def verify(path, target):
                if path != entry:
                    fail()
                return original_verify(path, target)
            fault.setattr(package, "verify", verify)
        elif failure == "unpack":
            original_unpack = package.unpack
            def unpack(archive, staged, target):
                original_unpack(archive, staged, target)
                if (staged / "share/data").exists():
                    fail()
            fault.setattr(package, "unpack", unpack)
        with pytest.raises(KeyboardInterrupt if failure == "interrupt" else InstallError):
            realize()
    if not initial:
        assert (entry / "bin/faketool").read_bytes() == b"#!x"
    assert (facts_path.read_bytes() if facts_path.exists() else None) == old_facts
    if not initial:
        assert (marker.read_bytes() if marker.exists() else None) == old_marker
    archives = [paths.store_root() / f"fetch-{sha}" for sha in (digest, extra_digest)]
    if failure != "fetch":
        assert all(archive.is_dir() for archive in archives)
    if failure not in ("invalid", "fetch"):
        for filename in ("replacement.tar.gz", "data.tar.gz"):
            (env["docroot"] / filename).unlink()
        realize()  # Publication failures must be retryable from verified cached bytes.
        assert (entry / "bin/faketool").read_bytes() == b"#!new"
        assert (entry / "share/data").read_bytes() == b"second archive"
        assert not any(archive.exists() for archive in archives)


@pytest.mark.parametrize("route", ["install", "stage"])
@pytest.mark.parametrize("interruption", ["before-publish", "after-publish"])
def test_killed_replacement_recovers_on_next_install(pm_env, route, interruption):
    """A killed publisher retains old bytes outside scratch until recovery."""
    import os
    import subprocess
    import sys
    import textwrap
    from functools import partial

    from pm.ensure import ensure, stage_only

    env = pm_env
    target = current_target() if route == "install" else "linux-arm64-bionic"
    realize = partial(ensure, "faketool", explicit=True, base_env={}) if route == "install" else partial(
        stage_only, "faketool", target)
    realize()
    old_fact = Facts(paths.facts_path()).get("faketool")
    entry = paths.store_root() / FakeTool().store_entry("1.0", target)
    old_digest = tree_digest(entry)
    _, digest = make_tar(env["docroot"], "replacement.tar.gz", {"bin/faketool": "#!new"})
    lockfile = Lockfile(env["lockfile_path"])
    lockfile.set_pin("faketool", "1.0", {"any": {
        "url": f"{env['base_url']}/replacement.tar.gz", "sha256": digest,
    }})
    lockfile.save()
    code = textwrap.dedent("""
        import os, sys
        from pathlib import Path
        import pm.paths as paths
        import pm.registry as registry
        from pm.store import Store
        from tests.pm.test_pm_authority import FakeTool
        from pm.ensure import ensure, stage_only
        paths.lockfile_path = lambda: Path(sys.argv[1])
        registry._packages[FakeTool.name] = FakeTool()
        publish = Store.publish
        def crash(self, staged, name):
            if not name.startswith('faketool-'):
                return publish(self, staged, name)
            if sys.argv[3] == 'after-publish':
                publish(self, staged, name)
            os._exit(17)
        Store.publish = crash
        if sys.argv[2] == 'install':
            ensure('faketool', explicit=True, base_env={})
        else:
            stage_only('faketool', 'linux-arm64-bionic')
    """)
    child = subprocess.run(
        [sys.executable, "-c", code, str(env["lockfile_path"]), route, interruption],
        env=dict(os.environ), capture_output=True, text=True, timeout=30,
    )
    assert child.returncode == 17, child.stderr
    assert Facts(paths.facts_path()).get("faketool") == old_fact
    previous = entry.with_name(f".previous-{'stage-' if route == 'stage' else ''}{entry.name}")
    assert tree_digest(previous) == old_digest
    realize()
    assert (entry / "bin/faketool").read_bytes() == b"#!new"
    assert not previous.exists()
    if route == "install":
        fact = Facts(paths.facts_path()).get("faketool")
        assert fact["digest"] == tree_digest(entry)
    else:
        assert not paths.facts_path().exists()
        assert json.loads((entry / ".pm-stage-pin.json").read_text())["sha256"] == [digest]


def test_failed_restore_preserves_both_interrupted_versions(pm_env, monkeypatch):
    from pm.ensure import ensure
    from pm.package import InstallError

    ensure("faketool", base_env={})
    fact = Facts(paths.facts_path()).get("faketool")
    entry = paths.store_root() / fact["entry"]
    previous = entry.with_name(".previous-" + entry.name)
    previous.mkdir()
    (previous / "saved").write_text("previous")
    (entry / "bin" / "faketool").write_text("uncommitted replacement")
    real_rename = Path.rename
    def fail_restore(self, target):
        if self == previous:
            raise PermissionError("restore refused")
        return real_rename(self, target)
    monkeypatch.setattr(Path, "rename", fail_restore)
    with pytest.raises((PermissionError, InstallError), match="restore refused"):
        ensure("faketool", explicit=True, base_env={})
    assert (entry / "bin" / "faketool").read_text() == "uncommitted replacement"
    assert (previous / "saved").read_text() == "previous"


@pytest.mark.parametrize("poisoned", [False, True], ids=["warm", "corrupt"])
def test_fetch_cache_verification_is_owned_by_downloader(pm_env, monkeypatch, poisoned):
    """Real cached bytes are checked in planning and under the partial lock;
    corrupt bytes must be replaced, without an extra Store-layer hash."""
    import pm.store as store_mod

    env = pm_env
    store = Store(env["runtime"] / "scratch-store")
    url = f"{env['base_url']}/faketool-1.0.tar.gz"
    digest = env["digest"]
    with store.scratch() as scratch:
        first = store.fetch(url, digest, scratch)
    assert sha_of(first) == digest

    if poisoned:
        first.write_bytes(b"poisoned")

    monkeypatch.setattr(store_mod, "sha256_file", lambda *args: pytest.fail("Store rehashed downloader-verified bytes"))
    with store.scratch() as scratch:
        second = store.fetch(url, digest, scratch)
    assert second == first
    assert sha_of(second) == digest


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_legacy_fact_without_identity_is_not_installed(pm_env, capsys):
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
    from pm.cli import cmd_doctor
    assert cmd_doctor(None) == 1
    assert "legacy fact: no recorded identity" in capsys.readouterr().out
    # ensure() repairs it: reinstalls and records the full identity.
    ensure("faketool", base_env={})
    assert is_installed("faketool")
    fact = Facts(paths.facts_path()).get("faketool")
    assert fact["artifacts"] == [env["digest"]]


def test_unproven_entry_without_facts_is_rebuilt(pm_env):
    from pm.ensure import ensure
    entry = paths.store_root() / FakeTool().store_entry("1.0", current_target())
    (entry / "bin").mkdir(parents=True)
    (entry / "bin/faketool").write_bytes(b"unproven")
    (entry / "sentinel").touch()
    ensure("faketool", explicit=True)
    assert (entry / "bin/faketool").read_bytes() == b"#!x"
    assert not (entry / "sentinel").exists()
    assert Facts(paths.facts_path()).get("faketool")["digest"] == tree_digest(entry)


def test_tree_digest_is_content_bound(pm_env):
    """Same content in different creation order digests identically; one
    changed byte digests differently; a symlink contributes its link
    TARGET TEXT, not the target's bytes."""

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


def test_adopt_refuses_on_missing_staged_binary(pm_env):
    from pm.ensure import adopt

    _bundle_payload(pm_env)
    fact = Facts(paths.facts_path()).get("faketool")
    binary = paths.store_root() / fact["entry"] / "bin" / "faketool"
    binary.unlink()

    marker = paths.runtime_facts_path().parent / ".adopted"
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

    marker = paths.runtime_facts_path().parent / ".adopted"
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

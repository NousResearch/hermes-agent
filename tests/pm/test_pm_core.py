"""pm end-to-end: install a fake tool from a loopback server, prove the
lockfile/installed-state split, env composition, single-flight, adoption,
and gc behavior. Real downloads, real archives, real locks — no mocked
stores."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

import pm.paths as paths
import pm.registry as registry
from pm.lock import Facts, Lockfile
from pm.package import InstallError, compose_env
from pm.packages import BinaryPackage, Venv
from pm.store import Store, current_target
from tests.pm._fixtures import make_tar, served as served


class FakeTool(BinaryPackage):
    name = "faketool"
    probe_version = False
    binary_rel = {"win32": "bin/faketool", "posix": "bin/faketool"}

    def fetch_url(self, version, target):
        return f"{FakeTool.base_url}/{self.name}-{version}.tar.gz"


class DepTool(FakeTool):
    name = "deptool"

    def env(self, entry, target):
        diff = super().env(entry, target)
        diff["DEPTOOL_SEEN"] = "1"
        return diff


class TopTool(FakeTool):
    name = "toptool"
    deps = ("deptool",)


class MultiTool(BinaryPackage):
    """A runtime split across two archives that must land in one entry."""
    name = "multitool"
    probe_version = False
    flatten = False
    binary_rel = {"win32": "bin/multitool", "posix": "bin/multitool"}

    def fetch_urls(self, version, target):
        return [f"{MultiTool.base_url}/{self.name}-{version}-a.tar.gz",
                f"{MultiTool.base_url}/{self.name}-{version}-b.tar.gz"]

    def fetch_url(self, version, target):
        return self.fetch_urls(version, target)[0]


@pytest.fixture
def pm_env(tmp_path, served, monkeypatch):
    docroot, base_url = served
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(runtime))
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    # Policy is pinned open here; the disabled-path tests pin it closed.
    import importlib

    ensure_mod = importlib.import_module("pm.install")
    monkeypatch.setattr(ensure_mod, "lazy_installs_allowed", lambda: True)

    # Restore the real registry wholesale on teardown. Tests must not
    # monkeypatch.setitem into the cleared dict: that undo runs after this
    # restore and deletes the built-in entry from the live registry.
    saved = dict(registry._packages)
    registry._packages.clear()
    for cls in (FakeTool, DepTool, TopTool, MultiTool, Venv):
        registry._packages[cls.name] = cls()
    FakeTool.base_url = base_url
    MultiTool.base_url = base_url

    lock_dir = tmp_path / "repo-lock"
    lock_dir.mkdir()
    lockfile_path = lock_dir / "lock.json"
    monkeypatch.setattr(paths, "lockfile_path", lambda: lockfile_path)

    name, digest = make_tar(docroot, "faketool-1.0.tar.gz", {"bin/faketool": "#!x"})
    lockfile = Lockfile(lockfile_path)
    lockfile.set_pin(
        "faketool", "1.0", {"any": {"url": f"{base_url}/faketool-1.0.tar.gz", "sha256": digest}}
    )
    lockfile.save()

    yield lockfile_path, runtime, docroot, base_url

    registry._packages.clear()
    registry._packages.update(saved)


def _pin(lockfile_path: Path, name: str, version: str, digest: str) -> None:
    lockfile = Lockfile(lockfile_path)
    url = f"{FakeTool.base_url}/{name}-{version}.tar.gz"
    lockfile.set_pin(name, version, {"any": {"url": url, "sha256": digest}})
    lockfile.save()




def test_bad_hash_rejected(pm_env):
    from pm.install import ensure

    lockfile_path, *_ = pm_env
    _pin(lockfile_path, "faketool", "1.0", "0" * 64)
    with pytest.raises(InstallError, match="install failed"):
        ensure("faketool", base_env={})



def test_multi_archive_merges_into_one_entry(pm_env, capsys):
    """A package split across two archives lands in ONE store entry: a
    second entry would put the DLLs where a loading executable can never
    find them. No per-archive entries, no leftover scratch."""
    from pm.install import ensure, is_installed

    lockfile_path, runtime, docroot, _ = pm_env
    _, digest_a = make_tar(docroot, "multitool-1.0-a.tar.gz",
                           {"bin/multitool": "#!a"})
    _, digest_b = make_tar(docroot, "multitool-1.0-b.tar.gz",
                           {"lib/extra.so": "y"})
    lockfile = Lockfile(lockfile_path)
    lockfile.set_pin("multitool", "1.0", {"any": [
        {"url": f"{FakeTool.base_url}/multitool-1.0-a.tar.gz", "sha256": digest_a},
        {"url": f"{FakeTool.base_url}/multitool-1.0-b.tar.gz", "sha256": digest_b},
    ]})
    lockfile.save()

    from pm import cli
    assert cli._install_names(["multitool"]) == 0
    out = capsys.readouterr().out
    assert "✓ multitool" in out and "multitool: 100.0%" in out
    assert "multitool: unpacking 1/2" in out and "multitool: unpacking 2/2" in out
    assert is_installed("multitool")

    entry_dirs = [p for p in runtime.iterdir() if p.name.startswith("multitool-")]
    assert len(entry_dirs) == 1, f"expected one merged entry, got {entry_dirs}"
    entry = entry_dirs[0]
    assert (entry / "bin" / "multitool").is_file()
    assert (entry / "lib" / "extra.so").is_file()
    # Both archives verified before publish: a digest mismatch in either
    # must fail loudly, not be silently dropped.
    old_facts = (runtime / "facts.json").read_bytes()
    _, good = make_tar(docroot, "multitool-2.0-a.tar.gz", {"bin/multitool": "#!new"})
    make_tar(docroot, "multitool-2.0-b.tar.gz", {"lib/extra.so": "z"})
    lockfile = Lockfile(lockfile_path)
    lockfile.set_pin("multitool", "2.0", {"any": [
        {"url": f"{FakeTool.base_url}/multitool-2.0-a.tar.gz", "sha256": good},
        {"url": f"{FakeTool.base_url}/multitool-2.0-b.tar.gz", "sha256": "0" * 64},
    ]})
    lockfile.save()
    with pytest.raises(InstallError, match="sha256 mismatch.*multitool-2.0-b"):
        ensure("multitool", base_env={})
    assert (runtime / "facts.json").read_bytes() == old_facts
    assert (entry / "bin/multitool").read_bytes() == b"#!a"
    assert (entry / "lib/extra.so").read_bytes() == b"y"
    assert not list(runtime.glob("multitool-2.0*"))





def test_deps_compose_dependents_win(pm_env):
    from pm.install import ensure

    lockfile_path, _, docroot, _ = pm_env
    name, digest = make_tar(docroot, "deptool-1.0.tar.gz", {"bin/faketool": "y"})
    _pin(lockfile_path, "deptool", "1.0", digest)
    name, digest = make_tar(docroot, "toptool-1.0.tar.gz", {"bin/faketool": "z"})
    _pin(lockfile_path, "toptool", "1.0", digest)

    runner = ensure("toptool", base_env={})
    assert runner.env["DEPTOOL_SEEN"] == "1"
    path = runner.env["PATH"]
    assert path.index("toptool-1.0") < path.index("deptool-1.0")

def test_cli_env_reports_only_package_exports(pm_env, monkeypatch, capsys):
    import json
    from argparse import Namespace
    from pm.cli import cmd_env
    from pm.install import ensure

    lockfile_path, _, docroot, _ = pm_env
    _, digest = make_tar(docroot, "deptool-1.0.tar.gz", {"bin/faketool": "y"})
    _pin(lockfile_path, "deptool", "1.0", digest)
    ensure("deptool", explicit=True)
    monkeypatch.setenv("FAKE_API_KEY", "never-print-this-secret")
    monkeypatch.setenv("PATH", "inherited-path-is-not-a-pm-export")
    assert cmd_env(Namespace(names=["deptool"])) == 0
    output = capsys.readouterr().out
    assert "never-print-this-secret" not in output
    assert "inherited-path-is-not-a-pm-export" not in output
    assert json.loads(output)["DEPTOOL_SEEN"] == "1"
    assert "deptool-1.0" in json.loads(output)["PATH"]


def test_activation_trusts_a_recorded_entry_a_deliberate_install_repairs(pm_env, monkeypatch):
    """Shell activation skips the byte re-hash; a deliberate install keeps it.

    Hashing every published tree costs seconds per shell, so activation trusts
    the digest the install recorded. That trust must not leak: the same
    corruption, installed without the flag, is still detected and rewritten.
    """
    import importlib
    from pm.cli import _install_names

    ensure = importlib.import_module("pm.install")
    lockfile_path, runtime, docroot, _ = pm_env
    _, digest = make_tar(docroot, "faketool-1.0.tar.gz", {"bin/faketool": "good"})
    _pin(lockfile_path, "faketool", "1.0", digest)
    assert _install_names(["faketool"]) == 0

    fact = Facts(runtime / "facts.json").get("faketool")
    assert fact is not None
    binary = runtime / fact["entry"] / "bin/faketool"
    binary.write_text("corrupt", encoding="utf-8")

    checked = []
    original = ensure._entry_verified

    def verify(package, fact, store, target):
        checked.append(package.name)
        return original(package, fact, store, target)

    monkeypatch.setattr(ensure, "_entry_verified", verify)
    assert _install_names(["faketool"], verify=False) == 0
    assert checked == []
    assert binary.read_text(encoding="utf-8") == "corrupt"

    assert _install_names(["faketool"]) == 0
    assert "faketool" in checked
    assert binary.read_text(encoding="utf-8") == "good"


def test_warm_install_verifies_shared_dependencies_once_under_lock(pm_env, monkeypatch):
    import importlib
    import os
    from collections import Counter
    from pm.filesystem import lock_fd
    from pm.cli import _install_names

    ensure = importlib.import_module("pm.install")
    lockfile_path, runtime, docroot, _ = pm_env
    for name in ("deptool", "toptool"):
        _, digest = make_tar(docroot, f"{name}-1.0.tar.gz", {"bin/faketool": name})
        _pin(lockfile_path, name, "1.0", digest)
    assert _install_names(["deptool", "toptool"]) == 0
    checked = Counter()
    locked = []
    original = ensure._entry_verified

    def verify(package, fact, store, target):
        fd = os.open(store.root / ".install.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            locked.append(not lock_fd(fd, wait=False))
        finally:
            os.close(fd)
        checked[package.name] += 1
        return original(package, fact, store, target)

    monkeypatch.setattr(ensure, "_entry_verified", verify)
    assert _install_names(["deptool", "toptool"]) == 0
    assert checked == {"deptool": 1, "toptool": 1}
    assert all(locked), "validation must share the publication lock"

    # The next operation must not reuse validity across a writer's mutation.
    fact = Facts(runtime / "facts.json").get("deptool")
    assert fact is not None
    binary = runtime / fact["entry"] / "bin/faketool"
    with Store(runtime).install_lock():
        binary.write_text("corrupt", encoding="utf-8")
    assert _install_names(["toptool"]) == 0
    assert binary.read_text(encoding="utf-8") == "deptool"


def test_standalone_warm_ensure_does_not_wait_for_unrelated_writer(pm_env):
    from concurrent.futures import ThreadPoolExecutor
    from pm.install import ensure

    _, runtime, _, _ = pm_env
    ensure("faketool", explicit=True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with Store(runtime).install_lock():
            # A downloader can hold this lock for minutes. A healthy unrelated
            # tool must remain usable without waiting for that writer to finish.
            future = pool.submit(ensure, "faketool", explicit=True, base_env={})
            runner = future.result(timeout=3)
            assert "faketool-1.0" in runner.env["PATH"]


def test_install_forgets_verification_when_state_operation_releases_lock(pm_env, monkeypatch):
    import importlib
    import os
    from pm.filesystem import lock_fd
    from pm.cli import _install_names

    ensure = importlib.import_module("pm.install")
    lockfile_path, runtime, docroot, _ = pm_env
    for name in ("deptool", "toptool"):
        _, digest = make_tar(docroot, f"{name}-1.0.tar.gz", {"bin/faketool": name})
        _pin(lockfile_path, name, "1.0", digest)
    assert _install_names(["toptool"]) == 0
    fact = Facts(runtime / "facts.json").get("deptool")
    assert fact is not None
    binary = runtime / fact["entry"] / "bin/faketool"

    def sync(**kwargs):
        # State operations provision their own tools. They must be able to
        # acquire the lock independently, and invalidate prior observations.
        fd = os.open(runtime / ".install.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            assert lock_fd(fd, wait=False), "tool lock leaked into the state operation"
            binary.write_text("corrupt", encoding="utf-8")
        finally:
            os.close(fd)

    monkeypatch.setattr(ensure, "sync_venv", sync)
    assert _install_names(["deptool", "venv", "toptool"]) == 0
    assert binary.read_text(encoding="utf-8") == "deptool"


def test_version_bump_selects_the_new_tool(pm_env):
    from pm.install import ensure

    lockfile_path, _, docroot, _ = pm_env

    ensure("faketool", base_env={})
    name, digest = make_tar(docroot, "faketool-2.0.tar.gz", {"bin/faketool": "#!2"})
    _pin(lockfile_path, "faketool", "2.0", digest)
    runner = ensure("faketool", base_env={})
    assert "faketool-2.0" in runner.env["PATH"]


def test_lazy_installs_disabled(pm_env, monkeypatch):
    import importlib

    ensure_mod = importlib.import_module("pm.install")
    monkeypatch.setattr(ensure_mod, "lazy_installs_allowed", lambda: False)
    with pytest.raises(InstallError, match="lazy installs are disabled"):
        ensure_mod.ensure("faketool", base_env={})


def test_missing_platform_is_declared(pm_env):
    from pm.install import ensure

    lockfile_path, *_ = pm_env
    pkg = registry._packages["faketool"]
    pkg.gaps = {current_target(): "no artifact for this platform"}
    try:
        with pytest.raises(InstallError, match="no artifact"):
            ensure("faketool", base_env={})
    finally:
        pkg.gaps = {}


def test_corrupt_facts_degrades_to_empty(pm_env):
    from pm.install import ensure, is_installed

    _, runtime, *_ = pm_env
    ensure("faketool", base_env={})
    (runtime / "facts.json").write_text("{ not json", encoding="utf-8")
    assert not is_installed("faketool")
    # the unparsable bytes were kept for post-mortem, not discarded
    assert (runtime / "facts.corrupt").is_file()



def test_concurrent_installs_do_not_clobber(pm_env):
    from pm.install import ensure, is_installed

    lockfile_path, _, docroot, _ = pm_env
    name, digest = make_tar(docroot, "deptool-1.0.tar.gz", {"bin/faketool": "y"})
    _pin(lockfile_path, "deptool", "1.0", digest)

    errors = []

    def run(target_name):
        try:
            ensure(target_name, base_env={})
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=run, args=("faketool",)),
        threading.Thread(target=run, args=("deptool",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert is_installed("faketool") and is_installed("deptool")


def test_single_flight_one_store_entry(pm_env, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager
    from http.server import SimpleHTTPRequestHandler
    from pm.install import ensure

    _, runtime, *_ = pm_env
    contenders = threading.Barrier(6)
    body_started, release_body = threading.Event(), threading.Event()
    bodies, stages = [], []
    install_lock, do_get, stage = Store.install_lock, SimpleHTTPRequestHandler.do_GET, FakeTool.stage

    @contextmanager
    def contending(self):
        contenders.wait(timeout=15)
        with install_lock(self):
            yield

    def body(self):
        if self.headers.get("Range") == "bytes=0-0":
            return do_get(self)
        bodies.append(self.path)
        body_started.set()
        assert release_body.wait(15), "server was never released"
        return do_get(self)

    def staged(self, *args):
        stages.append(self.name)
        return stage(self, *args)

    monkeypatch.setattr(Store, "install_lock", contending)
    monkeypatch.setattr(SimpleHTTPRequestHandler, "do_GET", body)
    monkeypatch.setattr(FakeTool, "stage", staged)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(ensure, "faketool", base_env={}) for _ in range(6)]
        try:
            assert body_started.wait(15)
        finally:
            release_body.set()
        runners = [future.result(timeout=30) for future in futures]
    assert bodies == ["/faketool-1.0.tar.gz"]
    assert stages == ["faketool"]
    assert all(runner.env == runners[0].env for runner in runners)
    fact = Facts(runtime / "facts.json").get("faketool")
    assert fact["entry"] in runners[0].env["PATH"]
    assert (runtime / fact["entry"] / "bin/faketool").read_bytes() == b"#!x"


def test_gc_keeps_used_removes_orphans(pm_env):
    from pm.cli import cmd_gc
    from pm.install import ensure

    _, runtime, *_ = pm_env
    ensure("faketool", base_env={})
    orphan = runtime / "orphan-9.9-nowhere"
    orphan.mkdir()
    # A killed installer's scratch dir; its restore point must survive gc.
    (runtime / ".staging-abandoned" / "tree").mkdir(parents=True)
    (runtime / ".previous-faketool-1.0").mkdir()
    cmd_gc(None)
    assert not orphan.exists()
    assert not (runtime / ".staging-abandoned").exists()
    assert (runtime / ".previous-faketool-1.0").is_dir()
    assert any(p.name.startswith("faketool-1.0") for p in runtime.iterdir())


def test_gc_removes_fetch_cache_archives(pm_env):
    """The fetch-<sha> download-cache dirs are install-time only — gc must
    drop them so a staged payload (and the CI cache that stores it) doesn't
    carry the raw archives. The live package entry survives."""
    from pm.cli import cmd_gc
    from pm.install import ensure

    lock_path, runtime, *_ = pm_env
    ensure("faketool", base_env={})
    # Downloads not consumed by a successful install remain eligible for GC.
    artifact = Lockfile(lock_path).artifacts("faketool", current_target())[0]
    store = Store(runtime)
    with store.scratch() as scratch:
        store.fetch(artifact["url"], artifact["sha256"], scratch)
    fetches = [p for p in runtime.iterdir() if p.name.startswith("fetch-")]
    assert fetches

    cmd_gc(None)
    assert not [p for p in runtime.iterdir() if p.name.startswith("fetch-")], \
        "gc must remove the fetch-<sha> archive cache"
    assert any(p.name.startswith("faketool-1.0") for p in runtime.iterdir()), \
        "the live package entry survives gc"


def test_env_for_never_installs(pm_env):
    from pm import env_for

    _, runtime, *_ = pm_env
    env = env_for("faketool", base_env={})
    assert "faketool-1.0" not in env.get("PATH", "")
    installed = [p for p in runtime.iterdir() if p.is_dir()] if runtime.is_dir() else []
    assert not any(p.name.startswith("faketool-") for p in installed)



def test_compose_env_dependents_win_non_path_too():
    env = compose_env([{"X": "dep"}, {"X": "dependent"}], base={})
    assert env["X"] == "dependent"



# ── bundle payload pieces ─────────────────────────────────────────────


def test_python_package_url_carries_release_tag():
    from pm.package import InstallError as PmInstallError
    from pm.registry import get_package

    python = get_package("python")
    url = python.fetch_url("3.14.7+20260901", "win32-arm64")
    assert "download/20260901/" in url
    assert "cpython-3.14.7+20260901-aarch64-pc-windows-msvc-install_only" in url

    try:
        python.fetch_url("3.14.7", "win32-arm64")
        raise AssertionError("bare version must be rejected")
    except PmInstallError:
        pass


@pytest.mark.platforms("macos")
def test_python_package_stably_signs_macos_runtime(tmp_path):
    import shutil
    import subprocess
    import sys
    from pm.registry import get_package

    python = get_package("python")
    staged = tmp_path / "staged"
    binary = staged / "python" / "bin" / "python3"
    binary.parent.mkdir(parents=True)
    (staged / "python" / "lib").mkdir()
    shutil.copy2(Path(sys._base_executable).resolve(), binary)
    subprocess.run(
        ["codesign", "--force", "--sign", "-", "--timestamp=none",
         "--identifier", "test.hermes.downloaded", "--requirements",
         '=designated => identifier "test.hermes.downloaded"', str(binary)],
        check=True, capture_output=True, timeout=30,
    )
    python.stage(Store(tmp_path / "store"), staged, "fixture", current_target())
    binary = python.binary(staged, current_target())
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(binary)],
                   check=True, capture_output=True, timeout=30)
    identity = subprocess.run(["codesign", "-d", "-r-", str(binary)],
                              check=True, capture_output=True, text=True, timeout=30)
    assert 'designated => identifier "com.nousresearch.hermes.managed-python"' in identity.stdout + identity.stderr


@pytest.mark.platforms("not macos")
def test_python_package_does_not_sign_non_macos_runtime(monkeypatch, tmp_path):
    import hermes_cli.macos_signing as signing

    monkeypatch.setattr(
        signing.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("codesign must not run outside macOS"),
    )

    assert signing.sign_managed_python(tmp_path / "python") is False


def test_machine_matches_binary_pe_headers(tmp_path):
    from pm.package import machine_matches_binary

    def pe(machine: int) -> bytes:
        head = bytearray(b"MZ" + b"\0" * 62)
        head[60:64] = (64).to_bytes(4, "little")
        return bytes(head) + b"PE\0\0" + machine.to_bytes(2, "little")

    amd = tmp_path / "amd.exe"
    amd.write_bytes(pe(0x8664))
    arm = tmp_path / "arm.exe"
    arm.write_bytes(pe(0xAA64))
    script = tmp_path / "tool"
    script.write_text("#!/bin/sh\n")

    assert machine_matches_binary(amd, "win32-x64") is True
    assert machine_matches_binary(amd, "win32-arm64") is False
    assert machine_matches_binary(arm, "win32-arm64") is True
    assert machine_matches_binary(script, "win32-x64") is None  # not a mismatch


def test_machine_matches_binary_elf(tmp_path):
    from pm.package import machine_matches_binary

    elf = bytearray(b"\x7fELF" + b"\0" * 60)
    elf[18:20] = (0xB7).to_bytes(2, "little")
    b = tmp_path / "tool"
    b.write_bytes(bytes(elf))
    assert machine_matches_binary(b, "linux-arm64") is True
    assert machine_matches_binary(b, "linux-x64") is False


def test_bundle_closure_uv_stays_internal_node_npm_ship(monkeypatch, tmp_path):
    """Locks the semantics split: uv is pm's install machinery and never
    ships by closure, node/npm are runtime tools and always do."""
    from scripts.bundles.native import _bundle_package_names
    from pm.lock import Lockfile
    from pm.registry import get_package

    lock = Lockfile(tmp_path / "lock.json")
    for name in ("uv", "node", "npm"):
        lock.set_pin(name, "1", {"any": {"url": "x", "sha256": "0" * 64}})
    lock.save()
    monkeypatch.setattr("scripts.bundles.native._lockfile", lambda: lock)
    names = _bundle_package_names()
    # uv stays internal (off PATH, off the default install) but ships in
    # the bundle via the explicit whitelist — install machinery rides along.
    assert get_package("uv").internal is True
    assert "uv" in names  # whitelisted cargo, not closure by right
    assert get_package("node").internal is False
    assert get_package("npm").internal is False
    assert "node" in names and "npm" in names


def test_arch_guard_allows_emulated_x64_on_win32_arm64(monkeypatch, tmp_path):
    """agent-browser on win32-arm64 ships the x64 PE (emulated). The guard
    must not reject it when the package declares the target emulated."""
    from scripts.bundles import native as cli
    from pm.lock import Facts, Lockfile
    from pm.registry import get_package

    store = tmp_path / "store"
    entry = store / "agent-browser-0.35.1"
    bin_dir = entry / "bin"
    bin_dir.mkdir(parents=True)
    # A real x64 PE header (MZ + PE sig + machine 0x8664).
    x64_pe = (
        b"MZ" + b"\0" * 58 + (0x80).to_bytes(4, "little") + b"\0" * 64
        + b"PE\0\0" + (0x8664).to_bytes(2, "little") + b"\0" * 54
    )
    binary = bin_dir / "agent-browser-win32-x64.exe"
    binary.write_bytes(x64_pe)
    from pm.package import machine_matches_binary

    assert machine_matches_binary(binary, "win32-arm64") is False

    lock = Lockfile(tmp_path / "lock.json")
    lock.set_pin("agent-browser", "0.35.1", {"any": {"url": "x", "sha256": "0" * 64}})
    lock.save()
    monkeypatch.setattr("scripts.bundles.native._lockfile", lambda: lock)
    monkeypatch.setattr("scripts.bundles.native.current_target", lambda: "win32-arm64")
    monkeypatch.setattr("scripts.bundles.native.get_package", lambda name: get_package(name))

    facts = Facts(store / "facts.json")
    facts.record("agent-browser", "0.35.1", entry.name, {}, store)

    problems = cli._arch_guard(store)
    assert problems == []


def test_python_stage_drops_unloadable_x64_vc_runtime_on_arm64(monkeypatch, tmp_path):
    from pm.registry import get_package

    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "vcruntime140_1.dll").write_bytes(b"x64")
    (staged / "vcruntime140.dll").write_bytes(b"arm64")

    monkeypatch.setattr("hermes_cli.macos_signing.sign_managed_python", lambda p: False)
    get_package("python").stage(None, staged, "3.14.7", "win32-arm64")

    assert not (staged / "vcruntime140_1.dll").exists()
    assert (staged / "vcruntime140.dll").is_file()


def test_python_stage_keeps_vc_runtimes_on_other_targets(monkeypatch, tmp_path):
    from pm.registry import get_package

    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "vcruntime140_1.dll").write_bytes(b"x64")

    monkeypatch.setattr("hermes_cli.macos_signing.sign_managed_python", lambda p: False)
    get_package("python").stage(None, staged, "3.14.7", "win32-x64")

    assert (staged / "vcruntime140_1.dll").is_file()


def test_verify_missing_binary_reports_path_and_listing(tmp_path):
    """A missing binary must say which path is missing and what the
    entry actually contains — the diagnosis that exposes a bad pin."""
    entry = tmp_path / "entry"
    (entry / "bin").mkdir(parents=True)
    (entry / "doc").write_text("x")
    reason = FakeTool().verify(entry, current_target())
    assert "bin/faketool" in reason
    assert "missing" in reason
    assert "doc" in reason


def test_verify_probe_failure_reports_reason(tmp_path):
    """A probe that cannot run must say so, not just fail."""

    class ProbingTool(FakeTool):
        probe_version = True

    entry = tmp_path / "entry"
    (entry / "bin").mkdir(parents=True)
    (entry / "bin" / "faketool").write_bytes(b"\x00\x01 not an executable")
    reason = ProbingTool().verify(entry, current_target())
    assert "--version" in reason


def test_probe_args_override_used_by_verify(tmp_path):
    """A binary that rejects `--version` (ffmpeg's BtbN autobuild does) can
    override the probe argv; verify() must honor the override everywhere
    the probe is described, not just in the subprocess argv."""

    class DashesTool(FakeTool):
        probe_version = True
        probe_args = ["-version"]

    entry = tmp_path / "entry"
    (entry / "bin").mkdir(parents=True)
    (entry / "bin" / "faketool").write_bytes(b"\x00\x01 not an executable")
    reason = DashesTool().verify(entry, current_target())
    assert "-version" in reason
    assert "--version" not in reason


def test_ffmpeg_binary_rel_follows_the_build_source(tmp_path):
    """BtbN (Windows `.zip`, Linux `.tar.xz`) ships bin/ffmpeg under one
    top-level dir; martin-riedl's macOS zip is a single `ffmpeg` at the root."""
    from pm.registry import get_package

    ffmpeg = get_package("ffmpeg")
    entry = tmp_path / "entry"
    entry.mkdir()
    (entry / "ffmpeg").write_bytes(b"x")
    (entry / "bin").mkdir()
    (entry / "bin" / "ffmpeg").write_bytes(b"x")
    assert ffmpeg.binary(entry, "darwin-x64") == entry / "ffmpeg"
    assert ffmpeg.binary(entry, "darwin-arm64") == entry / "ffmpeg"
    assert ffmpeg.binary(entry, "linux-x64") == entry / "bin" / "ffmpeg"
    assert ffmpeg.binary(entry, "linux-arm64") == entry / "bin" / "ffmpeg"
    assert ffmpeg.probe_args == ["-version"]


def test_verify_arch_mismatch_reports_target(tmp_path):
    """A wrong-arch binary is diagnosed before any probe is attempted."""
    target = current_target()
    arch = target.rsplit("-", 1)[-1]
    wrong = 0xAA64 if arch == "x64" else 0x8664
    entry = tmp_path / "entry"
    (entry / "bin").mkdir(parents=True)
    buf = bytearray(b"MZ" + b"\x00" * 0x3E)
    buf[0x3C:0x40] = (0x40).to_bytes(4, "little")
    buf += b"PE\x00\x00" + wrong.to_bytes(2, "little") + b"\x00" * 64
    (entry / "bin" / "faketool").write_bytes(bytes(buf))
    reason = FakeTool().verify(entry, target)
    assert reason and "not a" in reason and target in reason


def test_install_verify_failure_reports_reason(pm_env):
    """The CI failure: an entry that installs but fails verification must
    surface WHY in the InstallError, not a bare status."""
    lockfile_path, runtime, docroot, base_url = pm_env
    # Archive whose layout does not match binary_rel (bin/faketool).
    name, digest = make_tar(docroot, "faketool-2.0.tar.gz", {"nope/x": "y"})
    _pin(lockfile_path, "faketool", "2.0", digest)

    from pm.install import ensure

    with pytest.raises(InstallError) as exc:
        ensure("faketool", explicit=True)
    msg = str(exc.value)
    assert "failed verification" in msg
    assert "bin/faketool" in msg
    assert "missing" in msg



def test_store_path_dirs_include_node_npm_when_installed(tmp_path, monkeypatch):
    """The settled tools-on-path claim, made true: once node/npm are
    installed store packages (non-internal), their dirs enter the
    provisioned PATH. Regression test for the flag flip."""
    from pm import paths
    from pm.install import _store_path_dirs
    from pm.lock import Facts, Lockfile

    runtime = tmp_path / "runtime"
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(runtime))
    lock_dir = tmp_path / "repo-lock"
    lock_dir.mkdir()
    lockfile_path = lock_dir / "lock.json"
    monkeypatch.setattr(paths, "lockfile_path", lambda: lockfile_path)

    lock = Lockfile(lockfile_path)
    for name in ("node", "npm"):
        lock.set_pin(name, "1", {"any": {"url": "x", "sha256": "0" * 64}})
    lock.save()

    # Fabricate installed node/npm entries in the store the way pm would.
    store = paths.store_root()
    facts_path = paths.facts_path()
    facts_path.parent.mkdir(parents=True, exist_ok=True)
    facts = Facts(facts_path) if facts_path.is_file() else None
    import json as _json

    if facts is None:
        facts_path.write_text(_json.dumps({"schema": 1, "packages": {}}), encoding="utf-8")
        facts = Facts(facts_path)
    for name in ("node", "npm"):
        entry = f"{name}-1"
        entry_dir = store / entry
        entry_dir.mkdir(parents=True, exist_ok=True)
        binary = registry.get_package(name).binary(entry_dir, current_target())
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(b"installed tool")
        facts.record(
            name, "1", entry,
            {"PATH": ["{store}/" + entry]}, store_root=store,
            target=current_target(),
            artifacts=["0" * 64],
        )
    monkeypatch.setattr(paths, "lockfile_path", lambda: lockfile_path)

    dirs = _store_path_dirs()
    assert any(d.endswith("node-1") for d in dirs), dirs
    assert any(d.endswith("npm-1") for d in dirs), dirs

    # And the split holds: uv, still internal, contributes nothing even
    # if a (fabricated) fact exists for it.
    uv_entry = "uv-1"
    (store / uv_entry).mkdir(parents=True, exist_ok=True)
    facts.record("uv", "1", uv_entry, {"PATH": ["{store}/" + uv_entry]}, store_root=store)
    dirs = _store_path_dirs()
    assert not any(d.endswith("uv-1") for d in dirs), dirs

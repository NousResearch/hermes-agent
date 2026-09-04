"""pm end-to-end: install a fake tool from a loopback server, prove the
lockfile/installed-state split, env composition, single-flight, adoption,
and gc behavior. Real downloads, real archives, real locks — no mocked
stores."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest

import pm.paths as paths
import pm.registry as registry
from pm.lock import Facts, Lockfile
from pm.package import InstallError, Package, StatePackage, compose_env
from pm.packages import BinaryPackage
from pm.store import Store, current_target, flatten_single_dir


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
    # Policy is pinned open here; the disabled-path tests pin it closed.
    import importlib

    ensure_mod = importlib.import_module("pm.ensure")
    monkeypatch.setattr(ensure_mod, "lazy_installs_allowed", lambda: True)
    paths._stamp.cache_clear()

    saved = dict(registry._packages)
    registry._packages.clear()
    for cls in (FakeTool, DepTool, TopTool, MultiTool):
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


def test_install_and_env(pm_env):
    from pm.ensure import ensure, is_installed

    runner = ensure("faketool", base_env={})
    assert is_installed("faketool")
    assert "faketool-1.0" in runner.env["PATH"]


def test_idempotent_and_offline_after_install(pm_env):
    from pm.ensure import ensure

    _, _, docroot, _ = pm_env
    ensure("faketool", base_env={})
    (docroot / "faketool-1.0.tar.gz").unlink()
    runner = ensure("faketool", base_env={})
    assert "faketool-1.0" in runner.env["PATH"]


def test_bad_hash_rejected(pm_env):
    from pm.ensure import ensure

    lockfile_path, *_ = pm_env
    _pin(lockfile_path, "faketool", "1.0", "0" * 64)
    with pytest.raises(InstallError, match="install failed"):
        ensure("faketool", base_env={})


def test_fetch_is_a_store_entry(pm_env):
    from pm.ensure import ensure

    _, runtime, docroot, _ = pm_env
    ensure("faketool", base_env={})
    fetches = [p for p in runtime.iterdir() if p.name.startswith("fetch-")]
    assert len(fetches) == 1


def test_multi_archive_merges_into_one_entry(pm_env):
    """A package split across two archives lands in ONE store entry: a
    second entry would put the DLLs where a loading executable can never
    find them. No per-archive entries, no leftover scratch."""
    from pm.ensure import ensure, is_installed

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

    ensure("multitool", base_env={})
    assert is_installed("multitool")

    entry_dirs = [p for p in runtime.iterdir() if p.name.startswith("multitool-")]
    assert len(entry_dirs) == 1, f"expected one merged entry, got {entry_dirs}"
    entry = entry_dirs[0]
    assert (entry / "bin" / "multitool").is_file()
    assert (entry / "lib" / "extra.so").is_file()
    # Both archives verified before publish: a digest mismatch in either
    # must fail loudly, not be silently dropped.
    _, bad = make_tar(docroot, "multitool-2.0-b.tar.gz", {"lib/extra.so": "z"})
    lockfile = Lockfile(lockfile_path)
    lockfile.set_pin("multitool", "2.0", {"any": [
        {"url": f"{FakeTool.base_url}/multitool-2.0-a.tar.gz", "sha256": "0" * 64},
        {"url": f"{FakeTool.base_url}/multitool-2.0-b.tar.gz", "sha256": bad},
    ]})
    lockfile.save()
    with pytest.raises(InstallError):
        ensure("multitool", base_env={})


def test_install_emits_staged_progress(pm_env):
    """ensure() streams download -> unpack per artifact, labelled when a
    package has several. A slow download must not look frozen."""
    from pm.ensure import ensure

    lockfile_path, _, docroot, _ = pm_env
    _, digest_a = make_tar(docroot, "multitool-1.0-a.tar.gz", {"bin/multitool": "#!a"})
    _, digest_b = make_tar(docroot, "multitool-1.0-b.tar.gz", {"lib/extra.so": "y"})
    lockfile = Lockfile(lockfile_path)
    lockfile.set_pin("multitool", "1.0", {"any": [
        {"url": f"{FakeTool.base_url}/multitool-1.0-a.tar.gz", "sha256": digest_a},
        {"url": f"{FakeTool.base_url}/multitool-1.0-b.tar.gz", "sha256": digest_b},
    ]})
    lockfile.save()

    events: list[tuple[str, str]] = []
    ensure("multitool", base_env={},
           progress=lambda stage, d, t, label: events.append((stage, label)))

    assert ("download", "1/2") in events
    assert ("download", "2/2") in events
    assert ("unpack", "1/2") in events
    assert ("unpack", "2/2") in events
    # download before unpack within each artifact.
    stages = [s for s, _ in events]
    assert stages.index("download") < stages.index("unpack")


def test_install_names_streams_progress(pm_env, capsys):
    """The CLI install loop renders live download/unpack progress + a ✓
    line per package, so a slow bundle run is never silent in a piped
    log."""
    from pm import cli

    assert cli._install_names(["faketool"]) == 0
    out = capsys.readouterr().out
    assert "✓ faketool" in out
    assert "faketool: 100.0%" in out
    assert "faketool: unpacking" in out


def test_install_names_labels_multi_archive(pm_env, capsys):
    """A package split across archives gets a label on its progress lines,
    so the log says which archive is moving."""
    from pm import cli

    lockfile_path, _, docroot, _ = pm_env
    _, digest_a = make_tar(docroot, "multitool-1.0-a.tar.gz", {"bin/multitool": "#!a"})
    _, digest_b = make_tar(docroot, "multitool-1.0-b.tar.gz", {"lib/extra.so": "y"})
    lockfile = Lockfile(lockfile_path)
    lockfile.set_pin("multitool", "1.0", {"any": [
        {"url": f"{FakeTool.base_url}/multitool-1.0-a.tar.gz", "sha256": digest_a},
        {"url": f"{FakeTool.base_url}/multitool-1.0-b.tar.gz", "sha256": digest_b},
    ]})
    lockfile.save()

    assert cli._install_names(["multitool"]) == 0
    out = capsys.readouterr().out
    assert "multitool: unpacking 1/2" in out
    assert "multitool: unpacking 2/2" in out
    assert "✓ multitool" in out


def test_download_ticks_per_chunk(pm_env, monkeypatch):
    """The raw download stream reports byte progress on every chunk so a
    slow line proves liveness."""
    from pm.store import Store

    _, runtime, docroot, _ = pm_env
    ticks: list[tuple[int, int]] = []
    store = Store(runtime / "scratch")
    url = f"{FakeTool.base_url}/faketool-1.0.tar.gz"
    _, digest = make_tar(docroot, "faketool-1.0.tar.gz", {"bin/faketool": "#!x"})
    archive = store.fetch(url, digest, runtime / "scratch",
                          progress=lambda d, t: ticks.append((d, t)))
    assert archive.is_file()
    assert ticks and ticks[-1][0] == ticks[-1][1] == archive.stat().st_size


def test_deps_compose_dependents_win(pm_env):
    from pm.ensure import ensure

    lockfile_path, _, docroot, _ = pm_env
    name, digest = make_tar(docroot, "deptool-1.0.tar.gz", {"bin/faketool": "y"})
    _pin(lockfile_path, "deptool", "1.0", digest)
    name, digest = make_tar(docroot, "toptool-1.0.tar.gz", {"bin/faketool": "z"})
    _pin(lockfile_path, "toptool", "1.0", digest)

    runner = ensure("toptool", base_env={})
    assert runner.env["DEPTOOL_SEEN"] == "1"
    path = runner.env["PATH"]
    assert path.index("toptool-1.0") < path.index("deptool-1.0")


def test_version_bump_reinstalls_and_migrates(pm_env):
    from pm.ensure import ensure

    lockfile_path, _, docroot, _ = pm_env
    migrations = []
    registry._packages["faketool"].migrate = lambda prev, new: migrations.append((prev, new))

    ensure("faketool", base_env={})
    name, digest = make_tar(docroot, "faketool-2.0.tar.gz", {"bin/faketool": "#!2"})
    _pin(lockfile_path, "faketool", "2.0", digest)
    runner = ensure("faketool", base_env={})
    assert "faketool-2.0" in runner.env["PATH"]
    assert migrations == [("1.0", "2.0")]


def test_lazy_installs_disabled(pm_env, monkeypatch):
    import importlib

    ensure_mod = importlib.import_module("pm.ensure")
    monkeypatch.setattr(ensure_mod, "lazy_installs_allowed", lambda: False)
    with pytest.raises(InstallError, match="lazy installs are disabled"):
        ensure_mod.ensure("faketool", base_env={})


def test_missing_platform_is_declared(pm_env):
    from pm.ensure import ensure

    lockfile_path, *_ = pm_env
    pkg = registry._packages["faketool"]
    pkg.gaps = {current_target(): "no artifact for this platform"}
    try:
        with pytest.raises(InstallError, match="no artifact"):
            ensure("faketool", base_env={})
    finally:
        pkg.gaps = {}


def test_corrupt_facts_degrades_to_empty(pm_env):
    from pm.ensure import ensure, is_installed

    _, runtime, *_ = pm_env
    ensure("faketool", base_env={})
    (runtime / "facts.json").write_text("{ not json", encoding="utf-8")
    assert not is_installed("faketool")
    # the unparsable bytes were kept for post-mortem, not discarded
    assert (runtime / "facts.corrupt").is_file()


def test_sealed_install_refuses_even_explicit(pm_env, monkeypatch):
    from pm.ensure import ensure

    _, runtime, *_ = pm_env
    (runtime.parent / "manifest.json").write_text(
        '{"schema": 1}', encoding="utf-8"
    )
    with pytest.raises(InstallError, match="sealed"):
        ensure("faketool", base_env={}, explicit=True)


def test_concurrent_installs_do_not_clobber(pm_env):
    from pm.ensure import ensure, is_installed

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


def test_single_flight_one_store_entry(pm_env):
    from pm.ensure import ensure

    _, runtime, *_ = pm_env
    errors = []

    def go():
        try:
            ensure("faketool", base_env={})
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=go) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    entries = [p for p in runtime.iterdir() if p.name.startswith("faketool-")]
    assert len(entries) == 1


def test_gc_keeps_used_removes_orphans(pm_env):
    from pm.cli import cmd_gc
    from pm.ensure import ensure

    _, runtime, *_ = pm_env
    ensure("faketool", base_env={})
    orphan = runtime / "orphan-9.9-nowhere"
    orphan.mkdir()
    cmd_gc(None)
    assert not orphan.exists()
    assert any(p.name.startswith("faketool-1.0") for p in runtime.iterdir())


def test_gc_removes_fetch_cache_archives(pm_env):
    """The fetch-<sha> download-cache dirs are install-time only — gc must
    drop them so a staged payload (and the CI cache that stores it) doesn't
    carry the raw archives. The live package entry survives."""
    from pm.cli import cmd_gc
    from pm.ensure import ensure

    _, runtime, *_ = pm_env
    ensure("faketool", base_env={})

    # install() fetched via store.fetch → a fetch-<sha> archive cache dir.
    fetches = [p for p in runtime.iterdir() if p.name.startswith("fetch-")]
    assert fetches, "install should have left fetch-<sha> download-cache dirs"

    cmd_gc(None)
    assert not [p for p in runtime.iterdir() if p.name.startswith("fetch-")], \
        "gc must remove the fetch-<sha> archive cache"
    assert any(p.name.startswith("faketool-1.0") for p in runtime.iterdir()), \
        "the live package entry survives gc"


def test_env_for_never_installs(pm_env):
    from pm.ensure import env_for

    _, runtime, *_ = pm_env
    env = env_for("faketool", base_env={})
    assert "faketool-1.0" not in env.get("PATH", "")
    installed = [p for p in runtime.iterdir() if p.is_dir()] if runtime.is_dir() else []
    assert not any(p.name.startswith("faketool-") for p in installed)


def test_facts_adopt_by_path_substitution(tmp_path):
    store_a = tmp_path / "bundle-store"
    facts_a = Facts(store_a / "facts.json")
    store_a.mkdir()
    facts_a.record("tool", "1.0", "tool-1.0-any", {"PATH": [str(store_a / "tool-1.0-any" / "bin")]}, store_a)

    raw = (store_a / "facts.json").read_text(encoding="utf-8")
    assert "{{store}}" in raw and str(store_a) not in raw

    store_b = tmp_path / "user-store"
    store_b.mkdir()
    (store_a / "facts.json").rename(store_b / "facts.json")
    facts_b = Facts(store_b / "facts.json")
    env = facts_b.env_for("tool", store_b)
    assert env["PATH"] == [str(store_b / "tool-1.0-any" / "bin")]


def test_compose_env_dependents_win_non_path_too():
    env = compose_env([{"X": "dep"}, {"X": "dependent"}], base={})
    assert env["X"] == "dependent"


def test_flatten_refuses_layout_dirs(tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "tool").write_text("x")
    flatten_single_dir(tmp_path)
    assert (tmp_path / "bin" / "tool").is_file()


# ── bundle payload pieces ─────────────────────────────────────────────


def test_python_package_url_carries_release_tag():
    from pm.package import InstallError as PmInstallError
    from pm.registry import get_package

    python = get_package("python")
    url = python.fetch_url("3.11.16+20260814", "win32-arm64")
    assert "download/20260814/" in url
    assert "cpython-3.11.16+20260814-aarch64-pc-windows-msvc-install_only" in url

    try:
        python.fetch_url("3.11.16", "win32-arm64")
        raise AssertionError("bare version must be rejected")
    except PmInstallError:
        pass


def test_python_package_stably_signs_macos_runtime(monkeypatch, tmp_path):
    import pm.packages as packages
    from pm.registry import get_package

    python = get_package("python")
    binary = tmp_path / "bin" / "python3"
    binary.parent.mkdir()
    binary.touch()
    calls = []

    monkeypatch.setattr(packages.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(packages.shutil, "which", lambda name: "/usr/bin/codesign")
    monkeypatch.setattr(
        packages.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append((cmd, kwargs)) or type("Result", (), {"returncode": 0})(),
    )
    monkeypatch.setattr(python, "binary", lambda entry, target: binary)

    python.stage(Store(tmp_path / "store"), tmp_path, "3.11", "darwin-arm64")

    assert calls[0][0] == [
        "/usr/bin/codesign",
        "--force",
        "--deep",
        "--sign",
        "-",
        "--timestamp=none",
        "--identifier",
        "com.nousresearch.hermes.managed-python",
        "--requirements",
        '=designated => identifier "com.nousresearch.hermes.managed-python"',
        str(binary),
    ]
    assert calls[1][0] == ["/usr/bin/codesign", "--verify", "--deep", "--strict", str(binary)]


def test_python_package_does_not_sign_non_macos_runtime(monkeypatch, tmp_path):
    import pm.packages as packages

    monkeypatch.setattr(packages.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        packages.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("codesign must not run outside macOS"),
    )

    assert packages._macos_sign_managed_python(tmp_path / "python") is False


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


def test_adopt_noop_without_facts(pm_env, monkeypatch):
    import pm
    from pm import paths

    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(paths.store_root() / "nowhere"))
    paths._stamp.cache_clear()
    assert pm.adopt() is False


class FakeVenv(StatePackage):
    name = "venv"

    def __init__(self):
        self.applied: list[list[str]] = []
        self.lock_content = b"lock-v1"

    def expected_stamp(self, extras):
        import hashlib

        h = hashlib.sha256(self.lock_content)
        h.update(",".join(sorted(extras)).encode())
        return h.hexdigest()

    def apply(self, extras):
        self.applied.append(list(extras))


@pytest.fixture
def venv_env(pm_env):
    fake = FakeVenv()
    registry._packages["venv"] = fake
    return pm_env, fake


def test_sync_venv_applies_once_then_stamps(venv_env):
    from pm.ensure import sync_venv

    _, fake = venv_env
    sync_venv()
    sync_venv()
    assert fake.applied == [[]]


def test_sync_venv_unions_extras(venv_env):
    from pm.ensure import sync_venv

    _, fake = venv_env
    sync_venv(["telegram"])
    sync_venv(["anthropic"])
    sync_venv(["telegram"])
    assert fake.applied == [["telegram"], ["anthropic", "telegram"]]


def test_check_reports_venv_drift_and_missing_tools(venv_env):
    from pm.ensure import check, ensure, sync_venv

    (pm_env_tuple, fake) = venv_env
    lockfile_path, runtime, *_ = pm_env_tuple

    assert check() == []  # pm never touched this install: silent

    ensure("faketool", base_env={})
    sync_venv()
    assert check() == []

    fake.lock_content = b"lock-v2"  # uv.lock changed underneath
    problems = check()
    assert problems == ["venv: out of sync with uv.lock"]

    _pin(lockfile_path, "faketool", "9.9", "0" * 64)  # tool outdated now
    problems = check()
    assert "faketool: not installed or outdated" in problems


def test_bundle_package_names_include_browsers(monkeypatch, tmp_path):
    from pm.cli import _bundle_package_names
    from pm.lock import Lockfile

    lock = Lockfile(tmp_path / "lock.json")
    for name in ("uv", "python", "ripgrep", "chromium", "chromium-headless-shell", "node", "npm"):
        lock.set_pin(name, "1", {"any": {"url": "x", "sha256": "0" * 64}})
    lock.save()
    monkeypatch.setattr("pm.cli._lockfile", lambda: lock)
    names = _bundle_package_names()
    # Browsers now ship in every payload (win32-arm64 runs the x64 build
    # under emulation); nothing is excluded from the bundle.
    assert "chromium" in names
    assert "chromium-headless-shell" in names
    assert "python" in names
    assert "ripgrep" in names
    # node/npm are shipped runtime tools (TUI, plugins), not install
    # machinery; only uv remains internal.
    assert "node" in names
    assert "npm" in names


def test_bundle_closure_uv_stays_internal_node_npm_ship(monkeypatch, tmp_path):
    """Locks the semantics split: uv is pm's install machinery and never
    ships by closure, node/npm are runtime tools and always do."""
    from pm.cli import _bundle_package_names
    from pm.lock import Lockfile
    from pm.registry import get_package

    lock = Lockfile(tmp_path / "lock.json")
    for name in ("uv", "node", "npm"):
        lock.set_pin(name, "1", {"any": {"url": "x", "sha256": "0" * 64}})
    lock.save()
    monkeypatch.setattr("pm.cli._lockfile", lambda: lock)
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
    import pm.cli as cli
    from pm.lock import Facts, Lockfile
    from pm.registry import get_package

    store = tmp_path / "store"
    entry = store / "agent-browser-0.35.1"
    bin_dir = entry / "bin"
    bin_dir.mkdir(parents=True)
    # A real x64 PE header (MZ + PE sig + machine 0x8664).
    x64_pe = (
        b"MZ" + b"\0" * 58 + (0x80).to_bytes(4, "little")
        + b"PE\0\0" + (0x8664).to_bytes(2, "little") + b"\0" * 54
    )
    (bin_dir / "agent-browser-win32-x64.exe").write_bytes(x64_pe)

    lock = Lockfile(tmp_path / "lock.json")
    lock.set_pin("agent-browser", "0.35.1", {"any": {"url": "x", "sha256": "0" * 64}})
    lock.save()
    monkeypatch.setattr("pm.cli._lockfile", lambda: lock)
    monkeypatch.setattr("pm.cli.current_target", lambda: "win32-arm64")
    monkeypatch.setattr("pm.cli.get_package", lambda name: get_package(name))

    facts = Facts(store / "facts.json")
    facts.record("agent-browser", "0.35.1", entry.name, {}, store)

    problems = cli._arch_guard(store)
    assert problems == []


def test_python_stage_drops_unloadable_x64_vc_runtime_on_arm64(monkeypatch, tmp_path):
    import pm.packages as packages
    from pm.registry import get_package

    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "vcruntime140_1.dll").write_bytes(b"x64")
    (staged / "vcruntime140.dll").write_bytes(b"arm64")

    monkeypatch.setattr(packages, "_macos_sign_managed_python", lambda p: False)
    get_package("python").stage(None, staged, "3.11.16", "win32-arm64")

    assert not (staged / "vcruntime140_1.dll").exists()
    assert (staged / "vcruntime140.dll").is_file()


def test_python_stage_keeps_vc_runtimes_on_other_targets(monkeypatch, tmp_path):
    import pm.packages as packages
    from pm.registry import get_package

    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "vcruntime140_1.dll").write_bytes(b"x64")

    monkeypatch.setattr(packages, "_macos_sign_managed_python", lambda p: False)
    get_package("python").stage(None, staged, "3.11.16", "win32-x64")

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


def test_ffmpeg_posix_layout_resolves_at_entry_root(tmp_path):
    """martin-riedl zips are a single `ffmpeg` file at the zip root — the
    package must resolve it there (bin/ffmpeg is the BtbN win32 layout)."""
    from pm.registry import get_package

    ffmpeg = get_package("ffmpeg")
    entry = tmp_path / "entry"
    entry.mkdir()
    (entry / "ffmpeg").write_bytes(b"x")
    assert ffmpeg.binary(entry, "linux-arm64") == entry / "ffmpeg"
    assert ffmpeg.binary(entry, "darwin-x64") == entry / "ffmpeg"
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

    from pm.ensure import ensure

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
    from pm.ensure import _store_path_dirs
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

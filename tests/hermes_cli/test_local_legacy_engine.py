"""A pre-PM engine under runtimes/llamacpp/ moves into PM's store instead of forcing first-run setup."""

from __future__ import annotations

import json
import os
import threading

import pytest

import pm
from pm import paths
from pm.lock import Facts, Lockfile

PINNED = ("a" * 64, "c" * 64)  # CUDA pins the engine archive plus cudart; CPU pins one archive


@pytest.fixture(params=["cpu", "cuda"])
def legacy_env(request, tmp_path, monkeypatch):
    backend = request.param
    home = tmp_path / "home"
    store = home / "tools"
    lock_path = tmp_path / "lock.json"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(paths, "store_root", lambda: store)
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock_path)
    target = pm.current_target()
    package = pm.get_package(f"llamacpp-{backend}")
    if package.missing_reason(target):
        pytest.skip(f"{package.name} has no build for {target}")
    urls = package.fetch_urls("10964", target)
    lock = Lockfile(lock_path)
    lock.set_pin(package.name, "10964", {target: [{"url": u, "sha256": s} for u, s in zip(urls, PINNED)]})
    lock.save()
    # verify() runs `llama-server --version`; the fixture binary is not executable.
    monkeypatch.setattr(type(package), "verify", lambda self, entry, target: "")
    from hermes_cli.local_runtime import binaries

    monkeypatch.setattr(binaries, "_LEFT_IN_PLACE", set())
    return home, store, package, target, backend, list(PINNED[:len(urls)])


def legacy_install(home, package, target, backend, tag, digests, *, verified=True):
    install = home / "runtimes" / "llamacpp" / tag / backend
    install.mkdir(parents=True)
    package.binary(install, target).write_bytes(b"engine " + tag.encode())
    (install / "ggml-base.dll").write_bytes(b"dll")
    names = [url.rsplit("/", 1)[-1] for url in package.fetch_urls(tag.removeprefix("b"), target)]
    manifest = {"tag": tag, "backend": backend, "assets": dict(zip(names, digests))}
    if verified:
        manifest["verified_version"] = f"version: 0.4.1-dev (build {tag[1:]}, commit 0)"
    (install / "manifest.json").write_bytes(json.dumps(manifest).encode("utf-8"))
    return install


def test_matching_legacy_engine_moves_into_the_store_as_current(legacy_env):
    from hermes_cli.local_runtime import binaries

    home, store, package, target, backend, pinned = legacy_env
    install = legacy_install(home, package, target, backend, "b10964", pinned)

    engine = binaries.installed_engine(backend, allow_outdated=False)

    entry = store / package.store_entry("10964", target)
    assert engine == binaries.Engine(backend, "b10964", package.binary(entry, target))
    assert engine.binary.read_bytes() == b"engine b10964"
    assert not install.exists() and not install.parent.exists()
    fact = Facts(store / "facts.json").get(package.name)
    assert fact["version"] == "10964" and fact["artifacts"] == pinned


def test_older_legacy_engine_counts_as_installed_but_outdated(legacy_env):
    from hermes_cli.local_runtime import binaries

    home, _, package, target, backend, pinned = legacy_env
    legacy_install(home, package, target, backend, "b10679", ["d" * 64, *pinned[1:]])

    assert binaries.installed_engine(backend).tag == "b10679"
    assert pm.installed_package(package.name) is None  # the pane offers the update
    assert binaries.installed_engine(backend, allow_outdated=False) is None


def test_newest_legacy_engine_wins_and_older_stays(legacy_env):
    from hermes_cli.local_runtime import binaries

    home, _, package, target, backend, pinned = legacy_env
    old = legacy_install(home, package, target, backend, "b10679", ["d" * 64, *pinned[1:]])
    legacy_install(home, package, target, backend, "b10964", pinned)

    assert binaries.installed_engine(backend).tag == "b10964"
    assert old.is_dir()


@pytest.mark.parametrize("damage", ["unverified", "missing_digest", "wrong_backend", "tag_mismatch"])
def test_unverified_or_damaged_legacy_engine_is_left_alone(legacy_env, damage):
    from hermes_cli.local_runtime import binaries

    home, store, package, target, backend, pinned = legacy_env
    install = legacy_install(home, package, target, backend, "b10964",
                             ["not-a-digest", *pinned[1:]] if damage == "missing_digest" else pinned,
                             verified=damage != "unverified")
    manifest = json.loads((install / "manifest.json").read_bytes())
    if damage == "wrong_backend":
        manifest["backend"] = "vulkan"
    if damage == "tag_mismatch":
        manifest["tag"] = "b10679"
    (install / "manifest.json").write_bytes(json.dumps(manifest).encode("utf-8"))

    assert binaries.installed_engine(backend) is None
    assert install.is_dir()
    assert Facts(store / "facts.json").get(package.name) is None


def test_engine_that_fails_verification_stays_and_is_not_retried(legacy_env, monkeypatch):
    from hermes_cli.local_runtime import binaries

    home, _, package, target, backend, pinned = legacy_env
    install = legacy_install(home, package, target, backend, "b10964", pinned)
    probes = []

    def failing(self, entry, target):
        probes.append(entry)
        return "llama-server --version exited 1"

    monkeypatch.setattr(type(package), "verify", failing)
    assert binaries.installed_engine(backend) is None
    assert binaries.installed_engine(backend) is None
    assert install.is_dir() and probes == [install]


def test_existing_pm_engine_is_never_replaced(legacy_env):
    from hermes_cli.local_runtime import binaries
    from pm.store import tree_digest

    home, store, package, target, backend, pinned = legacy_env
    entry = store / package.store_entry("10964", target)
    entry.mkdir(parents=True)
    package.binary(entry, target).write_bytes(b"pm engine")
    Facts(store / "facts.json").record(package.name, "10964", entry.name, {}, store, target=target,
                                       artifacts=pinned, digest=tree_digest(entry))
    install = legacy_install(home, package, target, backend, "b10964", pinned)

    assert binaries.installed_engine(backend).binary.read_bytes() == b"pm engine"
    assert install.is_dir()


def test_busy_store_leaves_the_engine_for_a_later_call(legacy_env):
    from hermes_cli.local_runtime import binaries
    from pm.filesystem import lock_fd

    home, store, package, target, backend, pinned = legacy_env
    install = legacy_install(home, package, target, backend, "b10964", pinned)
    store.mkdir(parents=True, exist_ok=True)
    held, release = threading.Event(), threading.Event()

    def hold():
        # A second open file: msvcrt byte locks and flock both exclude it within one process.
        fd = os.open(store / ".install.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            assert lock_fd(fd, wait=True)
            held.set()
            release.wait(30)
        finally:
            os.close(fd)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert held.wait(10)
        assert binaries.installed_engine(backend) is None
        assert install.is_dir()
    finally:
        release.set()
        holder.join()
    assert binaries.installed_engine(backend).tag == "b10964"


def test_status_route_reports_the_moved_engine(legacy_env, monkeypatch):
    from hermes_cli.web_routers import local_models as lm

    home, _, package, target, backend, pinned = legacy_env
    legacy_install(home, package, target, backend, "b10964", pinned)
    monkeypatch.setattr(lm, "_runtime_section", lambda: {"enabled": True, "backend": backend})
    monkeypatch.setattr(lm, "_state_endpoint", lambda: None)

    status = lm.local_models_status()

    assert status["runtime_installed"] and status["runtime_backend"] == backend
    assert status["tag"] == "b10964" and not status["update_available"]

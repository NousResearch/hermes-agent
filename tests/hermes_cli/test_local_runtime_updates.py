"""Engine updates read PM pins; boot retains installed bytes without downloading."""

from __future__ import annotations

import pytest

import pm
from pm import paths
from pm.lock import Facts, Lockfile
from pm.store import tree_digest


@pytest.fixture
def runtime_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    store = home / "tools"
    lock_path = tmp_path / "lock.json"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(paths, "store_root", lambda: store)
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock_path)
    lock = Lockfile(lock_path)
    lock.set_pin("llamacpp-cpu", "10412", {pm.current_target(): {
        "url": "https://example.invalid/cpu.zip", "sha256": "a" * 64,
    }})
    lock.save()
    return store, lock


def record_engine(store, version, digest="b" * 64):
    package = pm.get_package("llamacpp-cpu")
    target = pm.current_target()
    entry = store / package.store_entry(version, target)
    entry.mkdir(parents=True, exist_ok=True)
    binary = package.binary(entry, target)
    binary.write_bytes(b"installed engine fixture")
    Facts(store / "facts.json").record(package.name, version, entry.name, {}, store,
                                       target=target, artifacts=[digest], digest=tree_digest(entry))
    return binary


def test_status_uses_pm_pin_and_ignores_legacy_tag(runtime_env, monkeypatch):
    from hermes_cli.web_routers import local_models as lm

    store, lock = runtime_env
    section = {"enabled": True, "backend": "cpu", "tag": "b99999"}
    monkeypatch.setattr(lm, "_runtime_section", lambda: section)
    monkeypatch.setattr(lm, "_state_endpoint", lambda: None)
    record_engine(store, "10362")
    status = lm.local_models_status()
    assert status["tag"] == "b10362"
    assert status["configured_tag"] == "b" + lock.version("llamacpp-cpu")
    assert status["runtime_installed"] and status["update_available"]
    section["enabled"] = False
    assert not lm.local_models_status()["update_available"]
    section["enabled"] = True
    record_engine(store, lock.version("llamacpp-cpu"), "a" * 64)
    assert not lm.local_models_status()["update_available"]


def test_boot_uses_previous_pm_engine_without_installing(runtime_env, monkeypatch):
    from hermes_cli.local_runtime import bootstrap, endpoint, supervisor

    store, _ = runtime_env
    binary = record_engine(store, "10362")
    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(endpoint, "_state_endpoint", lambda: None)
    monkeypatch.setattr(bootstrap, "_generate_presets", lambda *args: None)
    monkeypatch.setattr(bootstrap, "_start_idle_sweeper", lambda *args: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("boot attempted a PM install")

    monkeypatch.setattr(pm, "ensure", forbidden)
    spawned = []

    def start(self):
        spawned.append(self.binary)

    monkeypatch.setattr(supervisor.LlamaServerSupervisor, "start", start)
    config = {"local_runtime": {"enabled": True, "backend": "cpu", "tag": "b99999"}}
    assert bootstrap.ensure_local_runtime(config, force=True) is not None
    assert spawned == [binary]
    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    binary.unlink()
    assert bootstrap.ensure_local_runtime(config, force=True) is None
    assert spawned == [binary]

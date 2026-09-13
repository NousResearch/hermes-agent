"""A bundle reuses shipped bytes and adds missing pinned tools outside its seal."""
import json
from pathlib import Path

import pytest

import pm.paths as paths
from pm.lock import Facts, Lockfile
from tests.pm.test_pm_authority import pm_env, served  # noqa: F401 — fixtures


@pytest.mark.parametrize("sealed_install", [True, False])
def test_missing_bundle_tool_is_installed_in_writable_store(pm_env, tmp_path, monkeypatch, sealed_install):
    from pm.ensure import ensure, env_for, is_installed
    import importlib

    fixture = pm_env
    shipped = tmp_path / "payload" / "tools"
    shipped.mkdir(parents=True)
    (shipped.parent / "manifest.json").write_text('{"repo":"core","store":"tools","venv":"venv"}')
    monkeypatch.setattr(paths, "store_root", lambda: shipped)
    monkeypatch.setattr(paths, "facts_path", lambda: shipped / "facts.json")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    ensure_module = importlib.import_module("pm.ensure")
    monkeypatch.setattr(ensure_module, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr(ensure_module, "sealed", lambda: sealed_install)
    before = sorted(p.relative_to(shipped).as_posix() for p in shipped.rglob("*"))
    runner = ensure("faketool", explicit=True, base_env={})
    assert is_installed("faketool")
    assert sorted(p.relative_to(shipped).as_posix() for p in shipped.rglob("*")) == before
    writable = paths.writable_store_root()
    fact = Facts(writable / "facts.json").get("faketool")
    assert fact is not None and (writable / fact["entry"] / "bin" / "faketool").is_file()
    assert str(writable) in runner.env["PATH"]
    assert env_for("faketool", base_env={}) == runner.env


def test_adoption_records_verification_outside_shipped_payload(pm_env, monkeypatch):
    from pm.ensure import adopt
    from tests.pm.test_pm_authority import _bundle_payload

    _bundle_payload(pm_env)
    shipped_roots = (paths.store_root(), paths.repo_root())
    def snapshot():
        return {p: p.read_bytes() for root in shipped_roots for p in root.rglob("*") if p.is_file()}
    before = snapshot()
    assert adopt() is True
    assert snapshot() == before
    assert not (paths.store_root().parent / ".adopted").exists()
    assert adopt() is False


def test_corrupt_shipped_facts_are_only_read(tmp_path, monkeypatch):
    store = tmp_path / "payload" / "tools"
    store.mkdir(parents=True)
    (store.parent / "manifest.json").write_text("{}")
    facts = store / "facts.json"
    facts.write_text("{broken")
    monkeypatch.setattr(paths, "store_root", lambda: store)
    assert Facts(facts).get("anything") is None
    assert facts.read_text() == "{broken"
    assert not facts.with_suffix(".corrupt").exists()

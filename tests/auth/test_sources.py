"""Behavior of the explicit application boundary and source removal."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from auth.context import CredentialScope
from auth import store
from auth.source_removal import find_removal_step
from auth.sources import (
    CredentialEnvironment,
    is_source_suppressed,
    remove_provider_env_credential,
    save_provider_env_credential,
)


def environment(home, values, events):
    def save(key, value):
        values[key] = value
        events.append(("save", key))

    def remove(key):
        events.append(("remove", key))
        return values.pop(key, None) is not None

    def reconcile(old, new):
        events.append(("mirrors", old, new))
        return ["model.api_key"]

    return CredentialEnvironment(
        scope=CredentialScope(home),
        read_env=lambda: dict(values),
        save_env=save,
        remove_env=remove,
        reconcile_mirrors=reconcile,
        providers_for_env=lambda key: ["one", "two"],
        clear_models_cache=lambda provider: events.append(("cache", provider)),
        seed_pool=lambda provider: events.append(("seed", provider)),
    )


def test_remove_prunes_all_shared_sources_and_explicit_readd_lifts_suppression(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    key = "SHARED_TEST_KEY"
    env = environment(tmp_path, {key: "old-material"}, events := [])
    store._save_auth_store({
        "providers": {},
        "credential_pool": {
            "one": [
                {"id": "env-one", "source": f"env:{key}"},
                {"id": "manual", "source": "manual", "access_token": "retained"},
            ],
            "two": [{"id": "env-two", "source": f"env:{key}"}],
            "third": [{"id": "env-third", "source": f"env:{key}"}],
        },
    })
    result = remove_provider_env_credential(key, environment=env)
    assert result["found"]
    assert result["pool_pruned"] == ["one", "two", "third"]
    assert result["providers"] == ["one", "third", "two"]
    pool = store._load_auth_store()["credential_pool"]
    assert pool == {
        "one": [{"id": "manual", "source": "manual", "access_token": "retained"}]
    }
    assert all(is_source_suppressed(p, f"env:{key}") for p in result["providers"])
    assert ("mirrors", "old-material", None) in events
    assert {e[1] for e in events if e[0] == "cache"} == {"one", "two", "third"}

    save_provider_env_credential(key, "new-material", environment=env)
    assert not is_source_suppressed("one", f"env:{key}")
    assert not is_source_suppressed("two", f"env:{key}")
    assert is_source_suppressed("third", f"env:{key}")
    assert {e[1] for e in events if e[0] == "seed"} == {"one", "two"}


def test_boundary_rejects_foreign_profile_before_any_operation_and_allows_return(
    monkeypatch, tmp_path
):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    env = environment(first, {}, events := [])
    monkeypatch.setenv("HERMES_HOME", str(second))
    for operation in (
        lambda: save_provider_env_credential("KEY", "material", environment=env),
        lambda: remove_provider_env_credential("KEY", environment=env),
    ):
        with pytest.raises(ValueError, match="another profile"):
            operation()
    assert not events
    assert not (second / "auth.json").exists()
    monkeypatch.setenv("HERMES_HOME", str(first))
    save_provider_env_credential("KEY", "material", environment=env)
    assert ("save", "KEY") in events
    assert not (second / "auth.json").exists()


def test_failed_environment_write_does_not_report_success_or_reenable_source(
    monkeypatch, tmp_path
):
    from auth.sources import suppress_credential_source

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    suppress_credential_source("one", "env:KEY")
    env = environment(tmp_path, {}, events := [])

    def fail(*args):
        raise PermissionError("write refused")

    with pytest.raises(PermissionError):
        save_provider_env_credential(
            "KEY", "material", environment=replace(env, save_env=fail)
        )
    assert is_source_suppressed("one", "env:KEY")
    assert not events


def test_env_removal_uses_supplied_config_owner_and_preserves_other_sources(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("\ufeffKEY=material\n", encoding="utf-8")
    env = environment(tmp_path, {"KEY": "material"}, events := [])
    step = find_removal_step("one", "env:KEY", environment=env)
    result = step.remove_fn("one", SimpleNamespace(source="env:KEY"))
    assert result.cleaned == ["Cleared KEY from .env"]
    assert result.suppress
    assert ("remove", "KEY") in events
    assert find_removal_step("one", "manual") is None
    with pytest.raises(ValueError, match="application credential boundary"):
        find_removal_step("one", "env:KEY")


def test_copilot_removal_suppresses_all_duplicate_sources_without_removing_environment(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    step = find_removal_step("copilot", "env:GH_TOKEN")
    result = step.remove_fn("copilot", SimpleNamespace(source="env:GH_TOKEN"))
    assert result.suppress and not result.cleaned
    assert all(
        is_source_suppressed("copilot", source)
        for source in (
            "gh_cli",
            "env:COPILOT_GITHUB_TOKEN",
            "env:GH_TOKEN",
            "env:GITHUB_TOKEN",
        )
    )


def test_storage_imports_do_not_load_cli_or_discover_providers():
    import subprocess
    import sys
    from pathlib import Path

    code = """
import importlib.abc
import sys

class ForbidPresentationAndDiscovery(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"hermes_cli", "nous_cli", "providers"}:
            raise AssertionError("unexpected eager import: " + fullname)

sys.meta_path.insert(0, ForbidPresentationAndDiscovery())
from auth import store, provider_state, pool_persistence, sources, source_removal
assert store.AUTH_STORE_VERSION == 1
assert source_removal.find_removal_step("example", "manual") is None
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )

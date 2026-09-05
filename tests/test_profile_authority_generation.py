"""Authority must bind the inputs actually read, including startup provenance."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent import secret_scope
from agent.secret_sources import registry
from agent.secret_sources.base import FetchResult, SecretSource
from hermes_cli import env_loader
from tools.environments.local import build_subprocess_env


@pytest.fixture(autouse=True)
def isolated_sources(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(secret_scope, "_PROFILE_OWNED_NAME_HISTORY", {})
    token = secret_scope.set_secret_scope(None)
    env_loader.reset_secret_source_cache()
    registry._reset_registry_for_tests()
    yield
    secret_scope.reset_secret_scope(token)
    registry._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()


@pytest.mark.parametrize("filename", [".env", ".op.env"])
def test_startup_dotenv_revocation_before_first_child_keeps_ownership(tmp_path, monkeypatch, filename):
    name = "AUDIT_REVOKED_LOGIN"
    monkeypatch.delenv(name, raising=False)
    (tmp_path / filename).write_text(f"{name}=synthetic-old\n", encoding="utf-8")
    env_loader.load_hermes_dotenv(hermes_home=tmp_path, load_external_secrets=False)
    assert os.environ[name] == "synthetic-old"
    (tmp_path / filename).write_text("", encoding="utf-8")
    target = tmp_path / "target"
    target.mkdir()
    child = build_subprocess_env(base=dict(os.environ), profile_home=target,
                                 source_profile_home=tmp_path, enforce_profile_boundary=True)
    assert name not in child
    subprocess.run(
        [sys.executable, "-c", f"import os; assert {name!r} not in os.environ"],
        env=child, check=True, capture_output=True, text=True, encoding="utf-8", timeout=10,
    )


@pytest.mark.parametrize("startup", [False, True])
def test_source_change_during_fetch_cannot_publish_old_values_as_new_generation(tmp_path, startup):
    class ChangingSource(SecretSource):
        name = "audit_changing"
        shape = "bulk"

        def is_enabled(self, cfg):
            return cfg.get("enabled") is True

        def fetch(self, cfg, home_path):
            (home_path / "config.yaml").write_text("secrets: {}\n", encoding="utf-8")
            return FetchResult(secrets={"AUDIT_OLD_TOKEN": "synthetic-old"})

    assert registry.register_source(ChangingSource())
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.yaml").write_text(
        "secrets:\n  audit_changing:\n    enabled: true\n", encoding="utf-8")
    if startup:
        env_loader._apply_external_secret_sources(target)
        assert env_loader.get_external_secret_snapshot(target).status == "stale"
        assert str(target.resolve()) not in env_loader._APPLIED_HOMES
        # Startup is deliberately fail-open, but the resulting ambient name
        # must remain tainted when this profile later becomes a source.
        child = build_subprocess_env(base=dict(os.environ), profile_home=tmp_path,
                                     source_profile_home=target, enforce_profile_boundary=True)
        assert "AUDIT_OLD_TOKEN" not in child
    else:
        with pytest.raises(RuntimeError, match="external secret snapshot is stale"):
            build_subprocess_env(base={}, profile_home=target,
                                 source_profile_home=tmp_path, enforce_profile_boundary=True)
    assert not env_loader.get_external_secret_snapshot(target).data
    assert "AUDIT_OLD_TOKEN" not in secret_scope.build_profile_secret_scope(
        target, fail_closed_external=True)


def test_startup_disabled_source_race_is_not_empty_authority(tmp_path, monkeypatch):
    """A no-fetch startup decision still belongs to the config it actually read."""
    config = tmp_path / "config.yaml"
    config.write_text("secrets:\n  audit_pending:\n    enabled: false\n", encoding="utf-8")
    original = env_loader._load_secrets_config

    def change_after_read(home, **kwargs):
        cfg = original(home, **kwargs)
        config.write_text("secrets:\n  audit_pending:\n    enabled: true\n", encoding="utf-8")
        return cfg

    monkeypatch.setattr(env_loader, "_load_secrets_config", change_after_read)
    env_loader._apply_external_secret_sources(tmp_path)
    snapshot = env_loader.get_external_secret_snapshot(tmp_path)
    assert snapshot.status == "stale"
    assert snapshot.error_kind == "source_changed_during_fetch"
    assert not snapshot.data

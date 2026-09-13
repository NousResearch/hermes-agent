"""External-source failures must not authorize a child environment."""
from pathlib import Path

import pytest

from agent import secret_scope
from agent.secret_sources import registry
from agent.secret_sources.base import ErrorKind, FetchResult, SecretSource
from hermes_cli import env_loader
from tools.environments.local import build_subprocess_env


@pytest.fixture(autouse=True)
def isolated_sources(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    token = secret_scope.set_secret_scope(None)
    env_loader.reset_secret_source_cache()
    registry._reset_registry_for_tests()
    yield
    secret_scope.reset_secret_scope(token)
    registry._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("startup", [False, True])
def test_failed_source_refuses_child_and_recovers_without_cache_reset(tmp_path, partial, startup):
    """Real registry results must retain failure state through cold hydration."""
    class UnreliableSource(SecretSource):
        name = "audit_unreliable"
        shape = "bulk"
        recovered = False

        def is_enabled(self, cfg):
            return cfg.get("enabled") is True

        def fetch(self, cfg, home_path):
            if not self.recovered:
                return FetchResult().fail("synthetic outage", ErrorKind.INTERNAL)
            return FetchResult(secrets={"RECOVERED_TOKEN": "synthetic-recovered"})

    class HealthySource(UnreliableSource):
        name = "audit_healthy"

        def fetch(self, cfg, home_path):
            return FetchResult(secrets={"HEALTHY_TOKEN": "synthetic-healthy"})

    source = UnreliableSource()
    assert registry.register_source(source)
    assert registry.register_source(HealthySource())
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.yaml").write_text(
        "secrets:\n  audit_unreliable:\n    enabled: true\n"
        f"  audit_healthy:\n    enabled: {'true' if partial else 'false'}\n"
    )
    if startup:
        env_loader._apply_external_secret_sources(target)
        initial = env_loader.get_external_secret_snapshot(target)
        assert initial.status == ("degraded" if partial else "failed")
    with pytest.raises(RuntimeError, match="external secret snapshot is (failed|degraded)"):
        build_subprocess_env(base={}, profile_home=target,
                             source_profile_home=tmp_path, enforce_profile_boundary=True)
    snapshot = env_loader.get_external_secret_snapshot(target)
    assert snapshot.status == ("degraded" if partial else "failed")
    assert str(target.resolve()) not in env_loader._APPLIED_HOMES
    source.recovered = True
    child = build_subprocess_env(base={}, profile_home=target,
                                 source_profile_home=tmp_path, enforce_profile_boundary=True)
    assert child["HERMES_HOME"] == str(target)
    recovered = env_loader.get_external_secret_snapshot(target)
    assert recovered.status == "ready"
    assert recovered.data["RECOVERED_TOKEN"] == "synthetic-recovered"
    assert recovered.generation > snapshot.generation


@pytest.mark.parametrize("invalid_source", ["malformed", "unreadable"])
def test_invalid_profile_config_is_not_absent_authority(tmp_path, invalid_source):
    target = tmp_path / "target"
    target.mkdir()
    config = target / "config.yaml"
    if invalid_source == "malformed":
        config.write_text("secrets: [unterminated\n", encoding="utf-8")
    else:
        config.mkdir()
    with pytest.raises(RuntimeError, match="external secret snapshot is failed"):
        build_subprocess_env(base={}, profile_home=target,
                             source_profile_home=tmp_path, enforce_profile_boundary=True)

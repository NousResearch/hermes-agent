"""A routed cron fire's own per-run reload must not invalidate the scope it installed.

``_run_one_job_body`` installs the fire's profile scope, then ``run_job`` re-pulls external
sources (``_reload_dotenv_and_publish_delivery_target``). The re-pull starts a new source epoch;
the installed scope must follow it, or every child boundary built later in the fire is refused
as stale. Exercised through the production install -> reload -> child-env seams (A -> B -> A).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent import secret_scope
from agent.secret_sources import registry
from agent.secret_sources.base import FetchResult, SecretSource
from hermes_cli import env_loader


@pytest.fixture
def routed_homes(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    routed = launch / "profiles" / "ops"
    for home in (launch, routed):
        (home / "cron").mkdir(parents=True)
    (launch / ".env").write_text("FIRE_LOGIN=launch-value\n", encoding="utf-8")
    (routed / ".env").write_text("FIRE_LOGIN=routed-value\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("FIRE_LOGIN", "launch-value")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(secret_scope, "_PROFILE_OWNED_NAME_HISTORY", {})
    token = secret_scope.set_secret_scope(None)
    env_loader.reset_secret_source_cache()
    registry._reset_registry_for_tests()
    try:
        yield launch, routed
    finally:
        secret_scope.reset_secret_scope(token)
        registry._reset_registry_for_tests()
        env_loader.reset_secret_source_cache()


def _fire_child_env():
    from tools.environments.local import hermes_subprocess_env

    boundary = secret_scope.build_profile_env_boundary()
    return boundary, hermes_subprocess_env(base_env={"PATH": "/usr/bin"}, profile_boundary=boundary)


@pytest.mark.parametrize("source_config", [None, "secrets: {}\n"], ids=["absent", "empty"])
def test_unchanged_absent_or_empty_source_keeps_child_admission(routed_homes, source_config):
    from cron import scheduler
    from cron.scheduler_provider import _profile_cron_scope

    _launch, routed = routed_homes
    if source_config is not None:
        (routed / "config.yaml").write_text(source_config, encoding="utf-8")
    job = {"id": "routed", "deliver": "local"}

    with _profile_cron_scope(routed):
        tokens = scheduler._install_fire_secret_scope()
        try:
            installed = secret_scope.current_secret_scope()
            boundary, _ = _fire_child_env()
            assert boundary.target_values["FIRE_LOGIN"] == "routed-value"

            scheduler._reload_dotenv_and_publish_delivery_target(job)

            boundary, _ = _fire_child_env()  # raised "stale for target profile generation" before
            assert boundary.target_home == routed.resolve()
            assert boundary.target_values["FIRE_LOGIN"] == "routed-value"
            assert secret_scope.current_secret_scope_home() == str(routed.resolve())
            assert dict(secret_scope.current_secret_scope()) == dict(installed)
        finally:
            scheduler._reset_fire_secret_scope(tokens)

    # Back on the launch profile (A -> B -> A): nothing routed survives the fire.
    assert secret_scope.current_secret_scope() is None
    assert secret_scope.is_multiplex_active() is False


def test_rotated_source_binds_fresh_authority_after_reload(routed_homes):
    from cron import scheduler
    from cron.scheduler_provider import _profile_cron_scope

    _launch, routed = routed_homes
    values = iter(["vault-v1", "vault-v2"])

    class RotatingSource(SecretSource):
        name = "fire_rotation"
        shape = "bulk"

        def is_enabled(self, cfg):
            return cfg.get("enabled") is True

        def fetch(self, cfg, home_path):
            return FetchResult(secrets={"FIRE_VAULT_LOGIN": next(values)})

    assert registry.register_source(RotatingSource())
    (routed / "config.yaml").write_text(
        "secrets:\n  fire_rotation:\n    enabled: true\n", encoding="utf-8")

    with _profile_cron_scope(routed):
        tokens = scheduler._install_fire_secret_scope()
        try:
            before = secret_scope.current_secret_scope()
            assert before["FIRE_VAULT_LOGIN"] == "vault-v1"

            scheduler._reload_dotenv_and_publish_delivery_target({"id": "routed", "deliver": "local"})

            after = secret_scope.current_secret_scope()
            assert after["FIRE_VAULT_LOGIN"] == "vault-v2"
            assert after.generation != before.generation
            assert before["FIRE_VAULT_LOGIN"] == "vault-v1"  # earlier readers keep their snapshot
            boundary, child = _fire_child_env()
            assert boundary.target_values["FIRE_VAULT_LOGIN"] == "vault-v2"
            assert boundary.target_generation == after.generation
            assert child.get("FIRE_LOGIN") != "launch-value"
        finally:
            scheduler._reset_fire_secret_scope(tokens)

"""Cold cron fires must bind external secrets without sharing profile state (#126492)."""

import os


def test_cold_cron_fire_hydrates_sources_before_binding_scope(tmp_path, monkeypatch):
    import cron.scheduler as scheduler
    import hermes_constants
    from agent import secret_scope
    from agent.secret_sources import registry
    from agent.secret_sources.base import FetchResult, SecretSource, get_source_environment
    from cron.scheduler_provider import _profile_cron_scope
    from hermes_cli import env_loader

    launch = tmp_path / "launch"
    launch.mkdir()
    homes = [launch / "profiles" / name for name in ("a", "b")]
    for home in homes:
        (home / "cron").mkdir(parents=True)
        (home / ".env").write_text(
            f"CRON_VAULT_BOOTSTRAP_TOKEN={home.name}-bootstrap\n", encoding="utf-8"
        )
        (home / "config.yaml").write_text(
            "secrets:\n  cron_test_vault:\n    enabled: true\n", encoding="utf-8"
        )

    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("CRON_VAULT_TOKEN", "launch-token")
    monkeypatch.setenv("CRON_VAULT_BOOTSTRAP_TOKEN", "launch-bootstrap")
    monkeypatch.setattr(hermes_constants, "_PINNED_PROCESS_HERMES_HOME", None)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    # Keep both profiles cold, and restore registry/cache state after the test.
    for name in ("_APPLIED_HOMES", "_SOURCE_SUPPLIED_NAMES"):
        monkeypatch.setattr(env_loader, name, set())
    for name in (
        "_SECRET_SOURCES", "_SECRET_SOURCE_VALUES_BY_HOME", "_SECRET_SOURCE_RESTORE_BY_HOME"
    ):
        monkeypatch.setattr(env_loader, name, {})
    registry.list_sources()  # Initialize builtins before saving registry state.
    monkeypatch.setattr(registry, "_SOURCES", dict(registry._SOURCES))
    monkeypatch.setattr(registry, "_SOURCE_ORIGINS", dict(registry._SOURCE_ORIGINS))

    fetched_homes = []

    class VaultSource(SecretSource):
        name = "cron_test_vault"
        shape = "bulk"

        def fetch(self, cfg, home_path):
            fetched_homes.append(home_path.resolve())
            bootstrap = get_source_environment()["CRON_VAULT_BOOTSTRAP_TOKEN"]
            return FetchResult(secrets={"CRON_VAULT_TOKEN": f"resolved-{bootstrap}"})

    assert registry.register_source(VaultSource())
    parent_scope = {"CRON_VAULT_TOKEN": "launch-token"}
    parent_token = secret_scope.set_secret_scope(parent_scope, profile_home=str(launch))
    context_token = secret_scope.set_multiplex_context(False)
    environ_before = dict(os.environ)
    try:
        # Do not pre-hydrate: the first fire must see its source's value immediately.
        # A -> B -> A also checks that one profile's cached secrets cannot replace another's.
        for home in (homes[0], homes[1], homes[0]):
            with _profile_cron_scope(home):
                tokens = scheduler._install_fire_secret_scope()
                try:
                    assert secret_scope.is_multiplex_active()
                    assert secret_scope.get_secret("CRON_VAULT_TOKEN") == (
                        f"resolved-{home.name}-bootstrap"
                    )
                    assert dict(os.environ) == environ_before
                finally:
                    scheduler._reset_fire_secret_scope(tokens)
                assert secret_scope.current_secret_scope() is parent_scope
                assert not secret_scope.is_multiplex_active()
            assert secret_scope.get_secret("CRON_VAULT_TOKEN") == "launch-token"
            assert hermes_constants.get_hermes_home() == launch
            assert dict(os.environ) == environ_before
    finally:
        secret_scope.reset_multiplex_context(context_token)
        secret_scope.reset_secret_scope(parent_token)

    assert fetched_homes == [home.resolve() for home in homes]

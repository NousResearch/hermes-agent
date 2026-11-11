"""External cron workers must see plugin-registered secret sources (#121929).

``python -m cron.scheduler --external-worker-file`` starts with the builtin secret-source
registry alone; a plugin source (``ctx.register_secret_source()``) only exists after plugin
discovery, so ``hydrate_profile_secret_sources`` hydrated nothing and agent-mode jobs died at
credential resolution. Discovery must run under the payload's home override so a multiplexed
worker loads the OWNING profile's plugins, not the launch profile's.
"""
from __future__ import annotations

import json

import pytest

STUB_PLUGIN_INIT = '''
from agent.secret_sources.base import SECRET_SOURCE_API_VERSION, FetchResult, SecretSource


class TestVaultSource(SecretSource):
    api_version = SECRET_SOURCE_API_VERSION
    name = "testvault"
    label = "Test Vault"
    shape = "bulk"

    def fetch(self, cfg, home_path):
        return FetchResult(secrets={"TESTVAULT_API_KEY": "stub-vault-key"})


def register(ctx):
    ctx.register_secret_source(TestVaultSource())
'''


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """A launch home with no plugins and a profile home shipping the vault plugin."""
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    profile = tmp_path / "profile"
    plugin_dir = profile / "plugins" / "test-vault"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: test-vault\ndescription: stub\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(STUB_PLUGIN_INIT, encoding="utf-8")
    (profile / "config.yaml").write_text(
        "secrets:\n  sources: [testvault]\n  testvault:\n    enabled: true\n"
        "plugins:\n  enabled: [test-vault]\n",
        encoding="utf-8",
    )
    yield launch, profile
    from agent.secret_sources import registry
    from hermes_cli.env_loader import reset_secret_source_cache
    from hermes_cli.plugins import _reset_plugin_managers_for_tests

    registry._reset_registry_for_tests()
    reset_secret_source_cache()
    _reset_plugin_managers_for_tests()


def test_worker_hydrates_owning_profile_plugin_secret_source(homes, tmp_path, monkeypatch):
    _launch, profile = homes
    from cron import scheduler

    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps({"job": {"id": "job-1", "execution_id": "exec-1"},
                    "profile_home": str(profile), "multiplex_active": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "cron.executions.adopt_claimed_execution",
        lambda execution_id: {"id": execution_id, "status": "running"},
    )
    scopes = []

    def capture(*_args, **_kwargs):
        from agent.secret_scope import current_secret_scope
        scopes.append(dict(current_secret_scope() or {}))
        return True

    monkeypatch.setattr(scheduler, "run_one_job", capture)

    assert scheduler._run_external_worker_payload(payload, tmp_path / "exec-1.ready") is True
    assert scopes and scopes[0].get("TESTVAULT_API_KEY") == "stub-vault-key"

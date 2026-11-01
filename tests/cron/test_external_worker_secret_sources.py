"""External cron workers must see plugin-registered secret sources (#121929).

``python -m cron.scheduler --external-worker-file`` starts with the builtin secret-source
registry alone; a plugin source (``ctx.register_secret_source()``) only exists after plugin
discovery, so ``hydrate_profile_secret_sources`` hydrated nothing and agent-mode jobs died at
credential resolution. Discovery must run under the payload's home override so a multiplexed
worker loads the OWNING profile's plugins, not the launch profile's.
"""
from __future__ import annotations

import json
import os
import stat

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

BOOTSTRAP_PLUGIN_INIT = '''
from agent.secret_scope import get_secret


def register(ctx):
    if not get_secret("WORKER_BOOTSTRAP_CREDENTIAL"):
        raise RuntimeError("worker bootstrap credential unavailable")
    ctx.register_tool(
        name="worker_bootstrap_probe",
        toolset="worker_bootstrap",
        handler=lambda args, **kwargs: "ok",
        schema={
            "name": "worker_bootstrap_probe",
            "description": "External worker bootstrap regression probe",
            "parameters": {"type": "object", "properties": {}},
        },
    )
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


def test_worker_bootstraps_plugin_setup_from_command_secret_source(
        tmp_path, monkeypatch, caplog, capsys):
    """A scrubbed worker hydrates builtins before plugin setup, without cross-profile leakage."""
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.delenv("WORKER_BOOTSTRAP_CREDENTIAL", raising=False)
    profile = tmp_path / "profile"
    plugin_dir = profile / "plugins" / "bootstrap-probe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: bootstrap-probe\ndescription: regression probe\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(BOOTSTRAP_PLUGIN_INIT, encoding="utf-8")
    helper = tmp_path / "secret-helper.sh"
    helper.write_text(
        "#!/bin/sh\nprintf 'WORKER_BOOTSTRAP_CREDENTIAL=opaque-test-value\\n'\n",
        encoding="utf-8",
    )
    helper.chmod(helper.stat().st_mode | stat.S_IXUSR)
    (profile / "config.yaml").write_text(
        "secrets:\n"
        "  sources: [command]\n"
        "  command:\n"
        "    enabled: true\n"
        f"    command: {helper}\n"
        "plugins:\n"
        "  enabled: [bootstrap-probe]\n",
        encoding="utf-8",
    )
    payload = tmp_path / "bootstrap-payload.json"
    payload.write_text(
        json.dumps({"job": {"id": "job-bootstrap", "execution_id": "exec-bootstrap"},
                    "profile_home": str(profile), "multiplex_active": True}),
        encoding="utf-8",
    )

    from cron import scheduler
    from agent.secret_scope import current_secret_scope, is_multiplex_active
    from hermes_constants import hermes_home_key
    from tools.registry import registry as tool_registry

    monkeypatch.setattr(
        "cron.executions.adopt_claimed_execution",
        lambda execution_id: {"id": execution_id, "status": "running"},
    )
    observed = []

    def capture(*_args, **_kwargs):
        from toolsets import resolve_toolset

        resolved = resolve_toolset("worker_bootstrap")
        definitions = tool_registry.get_definitions(set(resolved), quiet=True)
        observed.append({
            "credential_present": bool((current_secret_scope() or {}).get(
                "WORKER_BOOTSTRAP_CREDENTIAL")),
            "resolved": resolved,
            "definition_names": [item["function"]["name"] for item in definitions],
        })
        return True

    monkeypatch.setattr(scheduler, "run_one_job", capture)
    try:
        assert scheduler._run_external_worker_payload(
            payload, tmp_path / "exec-bootstrap.ready") is True
        assert observed == [{
            "credential_present": True,
            "resolved": ["worker_bootstrap_probe"],
            "definition_names": ["worker_bootstrap_probe"],
        }]
        assert tool_registry.get_entry(
            "worker_bootstrap_probe", scope=str(launch)) is None
        assert "WORKER_BOOTSTRAP_CREDENTIAL" not in os.environ
        assert "opaque-test-value" not in caplog.text

        # The durable claim can disappear after setup. Even this early return must restore every
        # process context and must not publish the profile credential into the worker environment.
        failed_payload = tmp_path / "failed-payload.json"
        failed_payload.write_text(
            json.dumps({"job": {"id": "job-failed", "execution_id": "exec-failed"},
                        "profile_home": str(profile), "multiplex_active": True}),
            encoding="utf-8",
        )
        monkeypatch.setattr("cron.executions.adopt_claimed_execution", lambda _execution_id: None)
        assert scheduler._run_external_worker_payload(
            failed_payload, tmp_path / "exec-failed.ready") is False
        assert current_secret_scope() is None
        assert is_multiplex_active() is False
        assert "WORKER_BOOTSTRAP_CREDENTIAL" not in os.environ
        captured = capsys.readouterr()
        assert "opaque-test-value" not in captured.out + captured.err
    finally:
        from agent.secret_sources import registry as secret_registry
        from hermes_cli.env_loader import reset_secret_source_cache
        from hermes_cli.plugins import _reset_plugin_managers_for_tests

        _reset_plugin_managers_for_tests()
        secret_registry._reset_registry_for_tests()
        reset_secret_source_cache()

    assert current_secret_scope() is None
    assert is_multiplex_active() is False
    assert hermes_home_key() == hermes_home_key(launch)

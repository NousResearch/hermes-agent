"""External cron workers must see plugin-registered secret sources (#121929).

The worker process starts with the builtin secret-source registry alone; plugin
sources (the documented path for third-party vaults —
developer-guide/secret-source-plugin) only exist after plugin discovery runs.
These tests pin the real chain: a temp profile home with a real directory
plugin, discovered through ``discover_plugins()``, hydrating into the worker's
secret scope.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


STUB_PLUGIN_INIT = '''
from pathlib import Path

from agent.secret_sources.base import (
    SECRET_SOURCE_API_VERSION,
    FetchResult,
    SecretSource,
)


class TestVaultSource(SecretSource):
    """Bulk stub: one value, no backend."""

    api_version = SECRET_SOURCE_API_VERSION
    name = "testvault"
    label = "Test Vault"
    shape = "bulk"

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        return FetchResult(secrets={"TESTVAULT_API_KEY": "stub-vault-key"})


def register(ctx):
    ctx.register_secret_source(TestVaultSource())
'''


def _write_stub_plugin(profile_home: Path) -> None:
    plugin_dir = profile_home / "plugins" / "test-vault"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        "name: test-vault\ndescription: Stub secret-source plugin for tests\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(STUB_PLUGIN_INIT, encoding="utf-8")


def _write_profile_config(profile_home: Path) -> None:
    profile_home.mkdir(parents=True, exist_ok=True)
    (profile_home / "config.yaml").write_text(
        "secrets:\n"
        "  sources:\n"
        "    - testvault\n"
        "  testvault:\n"
        "    enabled: true\n"
        "plugins:\n"
        "  enabled:\n"
        "    - test-vault\n",
        encoding="utf-8",
    )


@pytest.fixture
def stub_profile_home(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    _write_profile_config(home)
    _write_stub_plugin(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home
    # Restore the pristine builtin-only registry for other tests in the run.
    import agent.secret_sources.registry as reg

    reg._reset_registry_for_tests()


def _run_worker_payload(payload_path: Path, ack_path: Path) -> bool:
    import cron.scheduler as scheduler

    return scheduler._run_external_worker_payload(payload_path, ack_path)


def _make_payload(tmp_path: Path, profile_home: Path) -> Path:
    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps({
            "job": {"id": "job-1", "execution_id": "exec-1"},
            "profile_home": str(profile_home),
        }),
        encoding="utf-8",
    )
    return payload


def test_worker_payload_hydrates_plugin_secret_sources(
    stub_profile_home, tmp_path, monkeypatch
):
    """The worker adopts the payload and builds a scope that resolves the
    plugin source's value — the exact step that failed for secondary profiles
    before the fix (#121929)."""

    payload = _make_payload(tmp_path, stub_profile_home)

    from cron.executions import create_execution, mark_execution_handoff_pending

    record = create_execution("job-1", source="builtin")
    payload.write_text(
        json.dumps({
            "job": {"id": "job-1", "execution_id": record["id"]},
            "profile_home": str(stub_profile_home),
        }),
        encoding="utf-8",
    )
    assert mark_execution_handoff_pending(record["id"]) is not None

    captured_scopes = []

    import cron.scheduler as scheduler

    def capture_scope(*args, **kwargs):
        from agent.secret_scope import current_secret_scope

        captured_scopes.append(dict(current_secret_scope() or {}))
        return True

    monkeypatch.setattr(scheduler, "run_one_job", capture_scope)

    assert _run_worker_payload(payload, tmp_path / "exec.ready") is True
    assert captured_scopes, "run_one_job never ran — scope never captured"
    assert (
        captured_scopes[0].get("TESTVAULT_API_KEY") == "stub-vault-key"
    ), "plugin secret source was not hydrated into the worker scope"


def test_worker_payload_without_plugins_still_hydrates_builtin_path(
    tmp_path, monkeypatch
):
    """The discovery call must not break workers whose profile has no plugins
    (the common case): the payload still runs and hydrates without error."""

    profile_home = tmp_path / "bare-profile"
    profile_home.mkdir(parents=True, exist_ok=True)
    # Create the execution row in the SAME store the worker adopts from —
    # the payload home's cron store, not the test process's launch home.
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    from cron.executions import create_execution, mark_execution_handoff_pending

    record = create_execution("job-bare", source="builtin")
    payload = tmp_path / "payload-bare.json"
    payload.write_text(
        json.dumps({
            "job": {"id": "job-bare", "execution_id": record["id"]},
            "profile_home": str(profile_home),
        }),
        encoding="utf-8",
    )
    assert mark_execution_handoff_pending(record["id"]) is not None

    import cron.scheduler as scheduler

    ran = {"ok": False}

    def mark_ran(*args, **kwargs):
        ran["ok"] = True
        return True

    monkeypatch.setattr(scheduler, "run_one_job", mark_ran)

    assert _run_worker_payload(payload, tmp_path / "exec-bare.ready") is True
    assert ran["ok"]
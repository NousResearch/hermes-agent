"""External provider setup uses the same profile-owned contract in both loaders."""

import json
import os
import textwrap
import time
from pathlib import Path

import pytest

PROVIDER = """
import os
from pathlib import Path
from agent.memory_provider import MemoryProvider, MemoryProviderConfigConflictError
from agent.secret_scope import get_secret
from hermes_constants import get_hermes_home
from plugins.memory.desktop_setup import report_progress
class Provider(MemoryProvider):
    name = "setup_probe"
    def is_available(self): return True
    def initialize(self, session_id, **kwargs): pass
    def get_tool_schemas(self): return []
    def get_desktop_config(self, *, hermes_home):
        return {"values": {"endpoint": "saved", "key": "must-not-leak"}, "is_set": {"key": True}}
    def handle_desktop_config_action(self, action, payload, *, hermes_home):
        if action == "health": return {"state": "ready"}
        if payload.get("crash"): raise SystemExit(1)
        if payload.get("confirm"):
            raise MemoryProviderConfigConflictError("Replace settings?", confirmation="overwrite")
        report_progress("working", "Waiting for the test to release the action.")
        gate = Path(hermes_home) / "release"
        import time
        deadline = time.monotonic() + 20
        while not gate.exists():
            if time.monotonic() > deadline: raise ValueError("Release timed out")
            time.sleep(.02)
        return {"home": str(get_hermes_home()), "secret": get_secret("SETUP_TEST_KEY"), "pid": os.getpid()}
def register(ctx): ctx.register_memory_provider(Provider())
"""
SCHEMA = """
from plugins.memory.config_schema import *
CONFIG_SCHEMA = ProviderConfigSchema(
    name="setup_probe", label="Probe", storage=STORAGE_PROVIDER_MANAGED, submit_action="save", status_action="health",
    fields=(
        ProviderField("endpoint", "Endpoint", visible_when=(ProviderFieldCondition("mode", values=("custom",)),)),
        ProviderField("key", "Key", kind=KIND_SECRET),
    ),
    actions=(ProviderConfigAction("validate", "Validate"),),
)
"""


def wait_job(client, profile, job):
    deadline = time.monotonic() + 20
    while job["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.03)
        response = client.get(
            f"/api/memory/providers/setup_probe/operations/{job['id']}",
            params={"profile": profile},
        )
        assert response.status_code == 200, response.text
        job = response.json()
    assert job["status"] != "running"
    return job


@pytest.mark.parametrize("isolation", ["in_process", "host"])
def test_external_actions_keep_owner_progress_and_single_flight(
    tmp_path, monkeypatch, isolation
):
    from fastapi.testclient import TestClient
    import plugins.memory as memory
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    root = tmp_path / ".hermes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(memory, "_MEMORY_PLUGINS_DIR", tmp_path / "no-bundled")
    homes = {"default": root, "work": root / "profiles/work"}
    for name, home in homes.items():
        plugin = home / "plugins/setup_probe"
        plugin.mkdir(parents=True)
        (plugin / "__init__.py").write_text(textwrap.dedent(PROVIDER))
        (plugin / "config_schema.py").write_text(textwrap.dedent(SCHEMA))
        (plugin / "plugin.yaml").write_text(
            "name: setup_probe\nversion: 1.0.0\nkind: memory\n"
        )
        (home / "config.yaml").write_text(
            f"plugins:\n  isolation: {isolation}\n  enabled: [setup_probe]\nmemory:\n  provider: setup_probe\n"
        )
        (home / ".env").write_text(f"SETUP_TEST_KEY={name}-secret\n")
    client = TestClient(app, headers={_SESSION_HEADER_NAME: _SESSION_TOKEN})
    jobs = {}
    for name in homes:
        url = "/api/memory/providers/setup_probe"
        legacy = client.get(
            url + "/config", params={"surface": "declared", "profile": name}
        )
        assert legacy.status_code == 200 and legacy.json()["fields"] == []
        response = client.get(
            url + "/config",
            params={"surface": "declared", "setup_api": 1, "profile": name},
        )
        assert response.status_code == 200, response.text
        assert "must-not-leak" not in response.text
        assert response.json()["fields"][0]["visible_when"][0]["values"] == ["custom"]
        for surface in ("declared", ""):
            denied = client.put(
                url + "/config",
                params={"surface": surface, "profile": name},
                json={"values": {"endpoint": "bad"}},
            )
            assert denied.status_code == 405, denied.text
        denied = client.post(
            url + "/setup",
            params={"profile": name},
            json={"values": {"endpoint": "bad"}},
        )
        assert denied.status_code == 405, denied.text
        first = client.post(
            url + "/actions/save", params={"profile": name}, json={"payload": {}}
        )
        assert first.status_code == 200, first.text
        jobs[name] = first.json()
        health = client.post(
            url + "/actions/health", params={"profile": name}, json={"payload": {}}
        )
        assert health.json() == {"status": "completed", "result": {"state": "ready"}}
        duplicate = client.post(
            url + "/actions/save", params={"profile": name}, json={"payload": {}}
        )
        assert duplicate.json()["id"] == jobs[name]["id"]
        busy = client.post(
            url + "/actions/save",
            params={"profile": name},
            json={"payload": {"different": True}},
        )
        assert busy.json()["status"] == "busy"
    try:
        # A request under B cannot read A's progress/results.
        denied = client.get(
            f"/api/memory/providers/setup_probe/operations/{jobs['default']['id']}",
            params={"profile": "work"},
        )
        assert denied.json()["status"] == "unavailable"
        for name in ("work", "default"):
            (homes[name] / "release").touch()
            done = wait_job(client, name, jobs[name])
            assert done["status"] == "completed", done
            assert done["result"]["home"] == str(homes[name])
            assert done["result"]["secret"] == f"{name}-secret"
            assert (done["result"]["pid"] != os.getpid()) == (isolation == "host")
        confirmation = client.post(
            "/api/memory/providers/setup_probe/actions/save",
            params={"profile": "work"},
            json={"payload": {"confirm": True}},
        ).json()
        assert (
            wait_job(client, "work", confirmation)["status"] == "confirmation_required"
        )
        unknown = client.post(
            "/api/memory/providers/setup_probe/actions/not-declared",
            params={"profile": "work"},
            json={"payload": {}},
        )
        assert unknown.status_code == 404
        crash = client.post(
            "/api/memory/providers/setup_probe/actions/save",
            params={"profile": "work"},
            json={"payload": {"crash": True}},
        ).json()
        assert wait_job(client, "work", crash)["status"] == "failed"
    finally:
        for home in homes.values():
            (home / "release").touch()
        if isolation == "host":
            from hermes_cli.plugins import get_plugin_manager
            from hermes_cli.web_server_profiles import _config_profile_scope

            for name in homes:
                with _config_profile_scope(name):
                    get_plugin_manager()._plugin_host().shutdown()

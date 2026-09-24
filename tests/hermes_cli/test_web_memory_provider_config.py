"""PUT /api/memory/providers/{name}/config saves without selecting unless asked; host-owned storage takes partial saves."""

import json
import textwrap

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli.web_routers import memory_providers as mp

INERT_RUNTIME = '# MemoryProvider\nraise AssertionError("no runtime initialization")\n'

NATIVE_RUNTIME = '''
    import json
    from pathlib import Path
    from agent.memory_provider import MemoryProvider

    class Probe(MemoryProvider):
        name = "probe"
        def initialize(self, session_id, **kwargs): raise AssertionError("must not initialize")
        def is_available(self):
            path = Path(__file__).parents[2] / "probe/config.json"
            return bool(json.loads(path.read_text()).get("endpoint", True)) if path.exists() else False
        def get_tool_schemas(self): return []
        def get_config_schema(self): return SCHEMA
        def save_config(self, values, hermes_home):
            path = Path(hermes_home, "probe/config.json")
            path.parent.mkdir(exist_ok=True)
            Path(hermes_home, "native-calls.json").write_text(json.dumps(values))
            path.write_text(json.dumps({**(json.loads(path.read_text()) if path.exists() else {}), **values}))
'''

DECLARED = '''
    from plugins.memory.config_schema import ProviderConfigSchema, ProviderField
    CONFIG_SCHEMA = ProviderConfigSchema(name="probe", label="Probe", fields=(
        ProviderField(key="endpoint", label="Endpoint"), ProviderField(key="count", label="Count", kind="number", default="3"),
        ProviderField(key="api_key", label="API key", kind="secret", env_key="PROBE_KEY"),
    ))
'''

URL = "/api/memory/providers/probe/config"


def _install(home, *, runtime=INERT_RUNTIME, raw_schema=None, declared=None, config=None, env="", terminal="local"):
    plugin = home / "plugins/probe"
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.yaml").write_text("name: probe\n")
    (plugin / "__init__.py").write_text(f"SCHEMA = {raw_schema!r}\n" + textwrap.dedent(runtime))
    if declared:
        (plugin / "config_schema.py").write_text(textwrap.dedent(declared))
    if config is not None:
        (home / "probe").mkdir(exist_ok=True)
        (home / "probe/config.json").write_text(json.dumps(config))
    (home / "config.yaml").write_text(yaml.safe_dump({"memory": {"provider": "original"}, "terminal": {"backend": terminal}}))
    (home / ".env").write_text(env)



def _selected(home):
    return yaml.safe_load((home / "config.yaml").read_text())["memory"]["provider"]


@pytest.fixture
def config_api(memory_homes, monkeypatch):
    from agent import secret_scope
    from tui_gateway import launch_profile_policy

    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    launch_profile_policy.activate_multi_profile_hosting()
    app = FastAPI()
    app.include_router(mp.router)
    try:
        with TestClient(app) as client:
            yield client, memory_homes
    finally:
        secret_scope.set_multiplex_active(False)


@pytest.mark.parametrize("activate", [False, True, None], ids=["save-only", "explicit", "legacy-default"])
def test_declared_schema_wins_and_selection_happens_only_when_asked(config_api, activate):
    client, homes = config_api
    home = homes["default"]
    _install(home, runtime=NATIVE_RUNTIME, raw_schema=[{"key": "endpoint", "required": True}], declared=DECLARED,
             config={"count": 3, "retained": "untouched"}, env="PROBE_KEY=disposable-test-secret\n")
    url = URL + "?surface=declared"
    prior_env = (home / ".env").read_bytes()
    response = client.put(url, json={"values": {"count": 5, "api_key": ""}, "activate": False})
    assert response.status_code == 200, response.text
    assert json.loads((home / "probe/config.json").read_text()) == {"count": 5, "retained": "untouched"}
    assert not (home / "native-calls.json").exists()  # host-owned flat storage, not the native writer
    assert _selected(home) == "original"
    assert client.put(url, json={"values": {"count": "not-a-number"}, "activate": False}).status_code == 400

    body = {"values": {"endpoint": "local-test-endpoint", "api_key": ""}, **({} if activate is None else {"activate": activate})}
    assert client.put(url, json=body).status_code == 200
    assert _selected(home) == ("original" if activate is False else "probe")
    assert (home / ".env").read_bytes() == prior_env
    fields = {f["key"]: f for f in client.get(url).json()["fields"]}
    assert fields["endpoint"]["value"] == "local-test-endpoint"
    assert fields["api_key"]["value"] == "" and fields["api_key"]["is_set"] is True


def test_inherited_save_config_uses_host_owned_generic_persistence(config_api):
    client, homes = config_api
    home = homes["default"]
    runtime = NATIVE_RUNTIME[:NATIVE_RUNTIME.index("        def save_config")]
    _install(home, runtime=runtime, raw_schema=[{"key": "count", "type": "integer", "default": 1}])
    assert client.get(URL).json()["capabilities"] == {
        "save_without_activation": True, "supports_partial_updates": True, "requires_full_form": False}
    response = client.put(URL, json={"values": {"count": 7}, "activate": False})
    assert response.status_code == 200, response.text
    config = yaml.safe_load((home / "config.yaml").read_text())["memory"]
    assert config["provider"] == "original" and config["probe"]["count"] == 7


def test_malformed_submission_is_rejected_before_any_secret_is_written(config_api):
    client, homes = config_api
    home = homes["default"]
    _install(home, declared=DECLARED, env="PROBE_KEY=old\n")
    response = client.put(URL + "?surface=declared", json={"values": {"count": "many", "api_key": "new"}, "activate": False})
    assert response.status_code == 400, response.text
    assert (home / ".env").read_text() == "PROBE_KEY=old\n" and not (home / "probe/config.json").exists()

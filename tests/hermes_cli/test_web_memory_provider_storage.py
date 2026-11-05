"""Dashboard config panel for ``storage=STORAGE_PROVIDER``: the provider's read_config_fields /
write_config_fields own the file layout (here a host-block file like a catalog provider keeps);
core coerces values, keeps secrets write-only, and never opens the file itself."""

import json

import pytest

_INIT = """
import json
from pathlib import Path

from agent.memory_provider import MemoryProvider

_ROOT_KEYS = {"baseUrl"}


def _read(home):
    path = Path(home) / "ownprov.json"
    return path, (json.loads(path.read_text(encoding="utf-8")) if path.exists() else {})


class OwnProv(MemoryProvider):
    name = "ownprov"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []

    def read_config_fields(self, hermes_home):
        _path, cfg = _read(hermes_home)
        block = (cfg.get("hosts") or {}).get("hermes") or {}
        values = {**{k: v for k, v in cfg.items() if k in _ROOT_KEYS}, **block}
        return {"values": values, "placeholders": {"workspace": "hermes", "aiPeer": "hermes"}}

    def write_config_fields(self, values, hermes_home):
        path, cfg = _read(hermes_home)  # a file that does not parse raises JSONDecodeError: no write
        block = cfg.setdefault("hosts", {}).setdefault("hermes", {})
        for key, value in values.items():
            target = cfg if key in _ROOT_KEYS else block
            if value is None:
                target.pop(key, None)
            else:
                target[key] = value
        path.write_text(json.dumps(cfg), encoding="utf-8")


def register(ctx):
    ctx.register_memory_provider(OwnProv())
"""

_SCHEMA = """
from plugins.memory.config_schema import (
    KIND_BOOL, KIND_SECRET, KIND_TEXT, STORAGE_PROVIDER, ProviderConfigSchema, ProviderField,
)

CONFIG_SCHEMA = ProviderConfigSchema(
    name="ownprov",
    label="Provider-owned storage",
    storage=STORAGE_PROVIDER,
    fields=(
        ProviderField(key="apiKey", label="API key", kind=KIND_SECRET, description="", env_key="OWNPROV_API_KEY"),
        ProviderField(key="baseUrl", label="Base URL", kind=KIND_TEXT, description=""),
        ProviderField(key="workspace", label="Workspace", kind=KIND_TEXT, description=""),
        ProviderField(key="saveMessages", label="Save messages", kind=KIND_BOOL, description="", default="true"),
    ),
)
"""

_URL = "/api/memory/providers/ownprov/config?surface=declared"


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    plugin_dir = get_hermes_home() / "plugins" / "ownprov"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(_INIT, encoding="utf-8")
    (plugin_dir / "config_schema.py").write_text(_SCHEMA, encoding="utf-8")
    test_client = TestClient(app)
    test_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return test_client


def _config_path():
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "ownprov.json"


def test_put_writes_through_the_provider_with_natives_and_saves_the_secret(client):
    from hermes_cli.config import load_config, load_env

    resp = client.put(_URL, json={"values": {
        "apiKey": "own-test-key", "baseUrl": "https://own.example.dev", "workspace": "myws", "saveMessages": "false",
    }})

    assert resp.status_code == 200
    assert load_config()["memory"]["provider"] == "ownprov"
    assert load_env()["OWNPROV_API_KEY"] == "own-test-key"
    cfg = json.loads(_config_path().read_text(encoding="utf-8"))
    assert cfg["baseUrl"] == "https://own.example.dev"
    assert cfg["hosts"]["hermes"] == {"apiKey": "own-test-key", "workspace": "myws", "saveMessages": False}


def test_get_keeps_secrets_write_only_and_shows_placeholders_and_presence(client):
    client.put(_URL, json={"values": {"apiKey": "secret-value", "saveMessages": "false"}})

    resp = client.get(_URL)

    assert resp.status_code == 200
    assert "secret-value" not in json.dumps(resp.json())
    fields = {field["key"]: field for field in resp.json()["fields"]}
    assert fields["apiKey"]["is_set"] is True and fields["apiKey"]["value"] == ""
    assert fields["workspace"]["placeholder"] == "hermes" and fields["workspace"]["is_set"] is False
    assert fields["saveMessages"]["is_set"] is True  # a stored False is still set
    assert fields["saveMessages"]["value"] == "false"


def test_a_blank_field_clears_the_key(client):
    client.put(_URL, json={"values": {"workspace": "myws"}})
    client.put(_URL, json={"values": {"workspace": ""}})
    assert "workspace" not in json.loads(_config_path().read_text(encoding="utf-8"))["hosts"]["hermes"]


def test_a_refused_write_is_a_400_with_no_secret_saved_and_no_switch(client):
    from hermes_cli.config import load_config, load_env

    _config_path().write_text("{not json", encoding="utf-8")

    resp = client.put(_URL, json={"values": {"workspace": "myws", "apiKey": "own-new-key"}})

    assert resp.status_code == 400
    assert _config_path().read_text(encoding="utf-8") == "{not json"
    assert (load_config().get("memory") or {}).get("provider") != "ownprov"
    assert "OWNPROV_API_KEY" not in load_env()

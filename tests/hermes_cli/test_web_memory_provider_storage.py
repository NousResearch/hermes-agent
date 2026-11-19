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


def test_a_config_the_provider_cannot_read_is_a_400(client):
    _config_path().write_text("{not json", encoding="utf-8")

    resp = client.get(_URL)

    assert resp.status_code == 400
    assert "Expecting property name" in resp.json()["detail"]


def test_a_refused_write_is_a_400_with_no_secret_saved_and_no_switch(client):
    from hermes_cli.config import load_config, load_env

    _config_path().write_text("{not json", encoding="utf-8")

    resp = client.put(_URL, json={"values": {"workspace": "myws", "apiKey": "own-new-key"}})

    assert resp.status_code == 400
    assert _config_path().read_text(encoding="utf-8") == "{not json"
    assert (load_config().get("memory") or {}).get("provider") != "ownprov"
    assert "OWNPROV_API_KEY" not in load_env()


def _install(name, init, schema):
    from hermes_constants import get_hermes_home
    plugin_dir = get_hermes_home() / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(init, encoding="utf-8")
    (plugin_dir / "config_schema.py").write_text(schema, encoding="utf-8")


def test_storage_provider_without_a_writer_is_a_501_with_no_secret_saved(client):
    from hermes_cli.config import load_env

    init = _INIT.replace("class OwnProv", "class _Unused").replace("def write_config_fields", "def _unused_write")
    init = init.replace("ctx.register_memory_provider(OwnProv())", "ctx.register_memory_provider(_Unused())")
    init = init.replace('name = "ownprov"', 'name = "nowriter"')
    _install("nowriter", init, _SCHEMA.replace('name="ownprov"', 'name="nowriter"'))

    resp = client.put("/api/memory/providers/nowriter/config?surface=declared",
                      json={"values": {"workspace": "x", "apiKey": "nw-key"}})

    assert resp.status_code == 501
    assert "OWNPROV_API_KEY" not in load_env()


def test_a_provider_that_fails_to_load_is_a_503_not_unknown(client):
    _install("broken", "from agent.memory_provider import MemoryProvider\nraise RuntimeError('import-time failure')\n",
             _SCHEMA.replace('name="ownprov"', 'name="broken"'))
    assert client.get("/api/memory/providers/broken/config?surface=declared").status_code == 503


def test_an_unknown_storage_kind_shows_no_panel_and_writes_no_stray_file(client):
    from hermes_constants import get_hermes_home

    _install("oddstore", _INIT.replace('name = "ownprov"', 'name = "oddstore"'),
             _SCHEMA.replace('name="ownprov"', 'name="oddstore"').replace("storage=STORAGE_PROVIDER", 'storage="retired_kind"'))
    url = "/api/memory/providers/oddstore/config?surface=declared"

    assert client.get(url).json()["fields"] == []
    assert client.put(url, json={"values": {"workspace": "x"}}).status_code == 400
    assert not (get_hermes_home() / "oddstore").exists()


def test_a_schema_that_fails_to_load_logs_its_traceback_once(client, caplog):
    _install("oldschema", _INIT.replace('name = "ownprov"', 'name = "oldschema"'),
             "from plugins.memory.config_schema import NO_SUCH_NAME\n")
    with caplog.at_level("ERROR", logger="plugins.memory.config_schema"):
        for _ in range(3):
            assert client.get("/api/memory/providers/oldschema/config?surface=declared").status_code == 200
    assert sum("failed to load config schema" in r.getMessage() for r in caplog.records if r.levelname == "ERROR") == 1


def _park(home, spelling, provider=None):
    import hermes_yaml as yaml

    path = home / "config.yaml"
    config = (yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None) or {}
    config.setdefault("plugins", {})["disabled"] = [spelling]
    if provider:
        config.setdefault("memory", {})["provider"] = provider
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_a_parked_provider_says_plugins_disabled_turned_it_off(client):
    from hermes_constants import get_hermes_home

    _park(get_hermes_home(), "ownprov", provider="ownprov")

    resp = client.get(_URL)

    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert "plugins.disabled" in detail and "`hermes plugins enable ownprov`" in detail, detail
    assert "external memory is off" in detail, detail


def test_a_parked_provider_while_another_is_selected_does_not_say_external_memory_is_off(client):
    from hermes_constants import get_hermes_home

    _park(get_hermes_home(), "ownprov", provider="otherprov")

    detail = client.get(_URL).json()["detail"]

    assert "plugins.disabled lists it" in detail and "external memory is off" not in detail, detail


def test_a_secondary_profile_panel_names_that_profile_and_its_refused_save_changes_nothing(client, monkeypatch):
    from hermes_cli import profiles
    from hermes_constants import get_hermes_home

    default_home = get_hermes_home()
    work_home = default_home / "profiles" / "work"
    plugin_dir = work_home / "plugins" / "ownprov"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(_INIT, encoding="utf-8")
    (plugin_dir / "config_schema.py").write_text(_SCHEMA, encoding="utf-8")
    (plugin_dir / "plugin.yaml").write_text("name: ownprov-manifest\n", encoding="utf-8")
    _park(work_home, "ownprov-manifest", provider="otherprov")  # the manifest spelling parks it too
    (work_home / "ownprov.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: default_home / "profiles")
    default_config = (default_home / "config.yaml").read_bytes() if (default_home / "config.yaml").exists() else None
    work_config = (work_home / "config.yaml").read_bytes()

    url = f"{_URL}&profile=work"
    got = client.get(url)
    put = client.put(url, json={"values": {"workspace": "w", "apiKey": "work-key"}})

    assert got.status_code == put.status_code == 503, (got.text, put.text)
    for detail in (got.json()["detail"], put.json()["detail"]):
        assert "`hermes -p work plugins enable ownprov`" in detail and "external memory is off" not in detail, detail
    assert (work_home / "config.yaml").read_bytes() == work_config  # no switch to ownprov
    assert (work_home / "ownprov.json").read_text(encoding="utf-8") == "{}"
    for home in (work_home, default_home):
        assert "work-key" not in ((home / ".env").read_text(encoding="utf-8") if (home / ".env").exists() else "")
    assert ((default_home / "config.yaml").read_bytes()
            if (default_home / "config.yaml").exists() else None) == default_config

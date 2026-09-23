"""A memory provider installed under ``$HERMES_HOME/plugins/`` (catalog install) keeps the
Desktop surfaces a bundled copy has: host-block config storage and the OAuth connect routes
resolve the provider's own ``client`` / ``oauth_flow`` modules through ``find_provider_dir``,
not a hard-coded ``plugins.memory.<name>`` import that only the bundled copy satisfies."""

import json
import sys
import textwrap

import pytest

_CLIENT = '''
import json, os
from pathlib import Path

def resolve_active_host():
    return "hermes"

def resolve_config_path():
    return Path(os.environ["HERMES_HOME"]) / "honcho.json"

def _host_block(cfg, host):
    return (cfg.get("hosts") or {}).get(host) or {}
'''

_OAUTH_FLOW = '''
from . import client  # relative import: only a real package load resolves it

def start_loopback_flow_background():
    return {"state": "pending", "config": str(client.resolve_config_path())}

def get_flow_status():
    return {"state": "idle"}
'''

_CONFIG_SCHEMA = '''
from plugins.memory.config_schema import STORAGE_HONCHO_HOST_BLOCK, ProviderConfigSchema, ProviderField

CONFIG_SCHEMA = ProviderConfigSchema(
    name="honcho", label="Honcho (user dir)", storage=STORAGE_HONCHO_HOST_BLOCK,
    fields=(ProviderField(key="workspace", label="Workspace", inline=True),),
)
'''


@pytest.fixture
def user_dir_honcho(monkeypatch, tmp_path, _isolate_hermes_home):
    """A user-dir ``honcho`` with the bundled copy gone: empty bundled root and the bundled
    module path blocked, the way a post-removal core + a catalog install look."""
    import plugins.memory as memory_pkg
    from hermes_constants import get_hermes_home

    plugin_dir = get_hermes_home() / "plugins" / "honcho"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(
        '"""fake provider: register_memory_provider"""\nfrom .client import resolve_active_host\n', encoding="utf-8"
    )
    for stem, source in (("client", _CLIENT), ("oauth_flow", _OAUTH_FLOW), ("config_schema", _CONFIG_SCHEMA)):
        (plugin_dir / f"{stem}.py").write_text(textwrap.dedent(source), encoding="utf-8")
    (get_hermes_home() / "honcho.json").write_text(
        json.dumps({"hosts": {"hermes": {"workspace": "from-user-dir"}}}), encoding="utf-8"
    )

    monkeypatch.setattr(memory_pkg, "_MEMORY_PLUGINS_DIR", tmp_path / "no-bundled")
    for name in ("plugins.memory.honcho", "plugins.memory.honcho.client", "plugins.memory.honcho.oauth_flow"):
        monkeypatch.setitem(sys.modules, name, None)
    return plugin_dir


def test_user_dir_host_block_provider_serves_its_declared_config(user_dir_honcho):
    from starlette.testclient import TestClient

    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    client = TestClient(app, headers={_SESSION_HEADER_NAME: _SESSION_TOKEN})
    resp = client.get("/api/memory/providers/honcho/config", params={"surface": "declared"})

    assert resp.status_code == 200, resp.text
    (field,) = resp.json()["fields"]
    assert (field["key"], field["value"]) == ("workspace", "from-user-dir")


def test_user_dir_provider_oauth_flow_resolves_from_its_directory(user_dir_honcho):
    from hermes_cli.memory_oauth import _resolve_flow

    flow = _resolve_flow("honcho")

    assert flow.__file__ == str(user_dir_honcho / "oauth_flow.py")
    assert flow.start_loopback_flow_background()["state"] == "pending"

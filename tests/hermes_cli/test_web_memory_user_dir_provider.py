"""A memory provider installed under ``$HERMES_HOME/plugins/`` (catalog install) keeps the
Desktop surfaces a bundled copy has: its provider-owned config panel and the OAuth connect routes
resolve through ``find_provider_dir``, not a hard-coded ``hermes_agent.plugins.memory.<name>`` import that
only the bundled copy satisfies."""

import json
import sys
import textwrap
from pathlib import Path

import pytest

_CLIENT = '''
import json
from pathlib import Path

def resolve_config_path(home):
    return Path(home) / "honcho.json"

def read_fields(home):
    raw = json.loads(resolve_config_path(home).read_text(encoding="utf-8"))
    return (raw.get("hosts") or {}).get("hermes") or {}
'''

_OAUTH_FLOW = '''
from . import client  # relative import: only a real package load resolves it

def start_loopback_flow_background():
    import os
    return {"state": "pending", "config": str(client.resolve_config_path(os.environ["HERMES_HOME"]))}

def get_flow_status():
    return {"state": "idle"}
'''

_CONFIG_SCHEMA = '''
from hermes_agent.plugins.memory.config_schema import STORAGE_PROVIDER, ProviderConfigSchema, ProviderField

CONFIG_SCHEMA = ProviderConfigSchema(
    name="honcho", label="Honcho (user dir)", storage=STORAGE_PROVIDER,
    fields=(ProviderField(key="workspace", label="Workspace", inline=True),),
)
'''


@pytest.fixture
def user_dir_honcho(monkeypatch, tmp_path, _isolate_hermes_home):
    """A user-dir ``honcho`` with the bundled copy gone: empty bundled root and the bundled
    module path blocked, the way a post-removal core + a catalog install look."""
    import hermes_agent.plugins.memory as memory_pkg
    from hermes_agent.hermes_constants import get_hermes_home

    plugin_dir = get_hermes_home() / "plugins" / "honcho"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(textwrap.dedent('''
        """fake provider: register_memory_provider"""
        from hermes_agent.agent.memory_provider import MemoryProvider
        from .client import read_fields  # relative import: only a real package load resolves it

        class UserDirProvider(MemoryProvider):
            name = "honcho"
            def is_available(self): return True
            def initialize(self, session_id, **kwargs): pass
            def get_tool_schemas(self): return []
            def read_config_fields(self, hermes_home): return {"values": read_fields(hermes_home)}

        def register(ctx):
            ctx.register_memory_provider(UserDirProvider())
        '''), encoding="utf-8")
    for stem, source in (("client", _CLIENT), ("oauth_flow", _OAUTH_FLOW), ("config_schema", _CONFIG_SCHEMA)):
        (plugin_dir / f"{stem}.py").write_text(textwrap.dedent(source), encoding="utf-8")
    (get_hermes_home() / "honcho.json").write_text(
        json.dumps({"hosts": {"hermes": {"workspace": "from-user-dir"}}}), encoding="utf-8"
    )

    monkeypatch.setattr(memory_pkg, "_MEMORY_PLUGINS_DIR", tmp_path / "no-bundled")
    for name in ("hermes_agent.plugins.memory.honcho", "hermes_agent.plugins.memory.honcho.client", "hermes_agent.plugins.memory.honcho.oauth_flow"):
        monkeypatch.setitem(sys.modules, name, None)
    return plugin_dir


def test_user_dir_host_block_provider_serves_its_declared_config(user_dir_honcho):
    from starlette.testclient import TestClient

    from hermes_agent.hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    client = TestClient(app, headers={_SESSION_HEADER_NAME: _SESSION_TOKEN})
    resp = client.get("/api/memory/providers/honcho/config", params={"surface": "declared"})

    assert resp.status_code == 200, resp.text
    (field,) = resp.json()["fields"]
    assert (field["key"], field["value"]) == ("workspace", "from-user-dir")


def test_user_dir_provider_oauth_flow_resolves_from_its_directory(user_dir_honcho):
    from hermes_agent.hermes_cli.memory_oauth import _resolve_flow

    flow = _resolve_flow("honcho")

    assert flow.__file__ == str(user_dir_honcho / "oauth_flow.py")
    assert flow.start_loopback_flow_background()["state"] == "pending"


def test_oauth_routes_load_the_provider_from_the_requested_profile(tmp_path, monkeypatch):
    """A desktop serving two homes must not import the launch home's OAuth flow for both."""
    import hermes_agent.plugins.memory as memory_pkg
    from hermes_agent.hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    from starlette.testclient import TestClient

    default_home = tmp_path / ".hermes"
    work_home = default_home / "profiles" / "work"
    work_home.mkdir(parents=True)
    (work_home / "profile.yaml").write_text("name: work\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr(memory_pkg, "_MEMORY_PLUGINS_DIR", tmp_path / "no-bundled")

    for home, label in ((default_home, "default"), (work_home, "work")):
        plugin_dir = home / "plugins" / "honcho"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "__init__.py").write_text('"""register_memory_provider"""\n', encoding="utf-8")
        (plugin_dir / "oauth_flow.py").write_text(
            "def get_flow_status():\n"
            f"    return {{'state': 'idle', 'profile': {label!r}}}\n"
            "def start_loopback_flow_background():\n"
            f"    return {{'state': 'pending', 'profile': {label!r}}}\n",
            encoding="utf-8",
        )

    client = TestClient(app, headers={_SESSION_HEADER_NAME: _SESSION_TOKEN})
    for profile in ("default", "work", "default"):
        status = client.get("/api/memory/providers/honcho/oauth/status", params={"profile": profile})
        assert status.status_code == 200, status.text
        assert status.json()["profile"] == profile
        start = client.post("/api/memory/providers/honcho/oauth/start", params={"profile": profile})
        assert start.status_code == 200, start.text
        assert start.json()["profile"] == profile

"""Standalone ``hermes mcp test`` must load enabled plugin secret sources (#133616).

Non-agent ``mcp`` subcommands skip startup plugin discovery, so a plugin-provided
credential used to fail the fail-closed ``${VAR}`` render even though runtime
startup resolved it fine. Each case resets the secret-source registry, the
dotenv cache, the inherited credential and the plugin manager first, so an
already-discovered plugin or a leftover environment value cannot mask the bug.
"""

import json
import os
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import pytest

ENV_KEY = "MCP_PROBE_FIXTURE_API_KEY"

PLUGIN = """from pathlib import Path
from agent.secret_sources.base import FetchResult, SecretSource

class FixtureVault(SecretSource):
    name = "fixture_vault"
    label = "Fixture vault"
    shape = "mapped"

    def fetch(self, cfg, home_path):
        result = FetchResult()
        if self.is_enabled(cfg):
            home = Path(home_path)
            result.secrets = {"MCP_PROBE_FIXTURE_API_KEY": (home / "canary.txt").read_text()}
            (home / "fetch.marker").write_text("fetched")
        return result

def register(ctx):
    ctx.register_secret_source(FixtureVault())
"""


@pytest.fixture
def mcp_endpoint():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            method = payload.get("method")
            calls.append((self.path, self.headers.get("Authorization"), method))
            expected = "Bearer fixture-" + self.path.split("/")[1]
            if self.headers.get("Authorization") != expected:
                self.send_response(401)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if "id" not in payload:
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if method == "initialize":
                result = {
                    "protocolVersion": payload["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fixture-mcp", "version": "1.0"},
                }
            elif method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": "fixture_read",
                            "description": "Read-only fixture",
                            "inputSchema": {"type": "object"},
                        }
                    ]
                }
            else:
                raise AssertionError("Unexpected MCP method: " + str(method))
            body = json.dumps({
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": result,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send_response(405)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    """Redirect config I/O under the env-selected home so A/B/A home switches track HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.config.get_hermes_home", lambda: Path(os.environ["HERMES_HOME"])
    )
    monkeypatch.setattr(
        "hermes_cli.config.get_config_path",
        lambda: Path(os.environ["HERMES_HOME"]) / "config.yaml",
    )
    monkeypatch.setattr(
        "hermes_cli.config.get_env_path",
        lambda: Path(os.environ["HERMES_HOME"]) / ".env",
    )
    yield
    os.environ.pop(ENV_KEY, None)  # apply_all writes straight into os.environ


def _seed_home(home, port, label, *, plugin_enabled=True, source_enabled=True):
    home.mkdir()
    plugin = home / "plugins" / "fixture-vault"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: fixture-vault\nversion: 1.0.0\ndescription: Synthetic local test source\n"
    )
    (plugin / "__init__.py").write_text(PLUGIN)
    (home / "canary.txt").write_text("fixture-" + label)
    # JSON is valid YAML; no actual profile/config writer is touched.
    (home / "config.yaml").write_text(
        json.dumps({
            "plugins": {"enabled": ["fixture-vault"] if plugin_enabled else []},
            "secrets": {"fixture_vault": {"enabled": source_enabled}},
            "mcp_servers": {
                "fixture": {
                    "url": f"http://127.0.0.1:{port}/{label}/mcp",
                    "headers": {"Authorization": "Bearer ${MCP_PROBE_FIXTURE_API_KEY}"},
                    "connect_timeout": 5,
                },
                # A diagnostic must never launch unrelated configured servers.
                "unrelated": {
                    "command": "hermes-unrelated-fixture-marker-cmd",
                    "args": ["never"],
                },
            },
        })
    )


def _purge_fixture_state(monkeypatch, home):
    """Enter each probe with the plugin undiscovered, its source unregistered, the dotenv
    cache empty and no inherited credential — the state a fresh CLI process would have."""
    monkeypatch.delenv(ENV_KEY, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from agent.secret_sources import registry

    with registry._REGISTRY_LOCK:
        registry._SOURCES.pop("fixture_vault", None)
        registry._SOURCE_ORIGINS.pop("fixture_vault", None)
        for scoped in registry._SCOPED_SOURCES.values():
            scoped.pop("fixture_vault", None)
    from hermes_cli.env_loader import reset_secret_source_cache

    reset_secret_source_cache(home)
    import hermes_cli.plugins as plugins_mod

    monkeypatch.setattr(plugins_mod, "_plugin_manager", None)
    # get_plugin_manager() also caches per resolved home; drop this home's entry so a revisit
    # (A/B/A) rediscovers instead of reusing an already-discovered manager that registered
    # into a registry we just purged.
    plugins_mod._plugin_managers_by_home.pop(plugins_mod._plugin_home_key(), None)


def test_mcp_test_hydrates_plugin_credential_per_home(
    tmp_path, mcp_endpoint, monkeypatch, capsys
):
    port, calls = mcp_endpoint
    from hermes_cli.mcp_config import cmd_mcp_test

    homes = []
    for label in ("a", "b"):
        home = tmp_path / label
        _seed_home(home, port, label)
        homes.append(home)

    expected = []
    for home, label in ((homes[0], "a"), (homes[1], "b"), (homes[0], "a")):
        _purge_fixture_state(monkeypatch, home)
        assert cmd_mcp_test(Namespace(name="fixture")) == 0, capsys.readouterr().out
        out = capsys.readouterr().out
        assert "fixture_read" in out
        assert (home / "fetch.marker").read_text() == "fetched"
        assert (
            home / "canary.txt"
        ).read_text() not in out  # credential value never echoed
        expected.append((f"/{label}/mcp", f"Bearer fixture-{label}"))

    # A/B/A: each probe carried its own home's credential, and the unrelated
    # configured server was never launched.
    assert [
        (path, auth) for path, auth, method in calls if method == "tools/list"
    ] == expected


@pytest.mark.parametrize("disabled", ["plugin", "source"])
def test_mcp_test_does_not_activate_disabled_secret_source(
    tmp_path, mcp_endpoint, monkeypatch, capsys, disabled
):
    port, calls = mcp_endpoint
    from hermes_cli.mcp_config import cmd_mcp_test

    home = tmp_path / "disabled"
    _seed_home(
        home,
        port,
        "disabled",
        plugin_enabled=disabled != "plugin",
        source_enabled=disabled != "source",
    )
    _purge_fixture_state(monkeypatch, home)

    assert cmd_mcp_test(Namespace(name="fixture")) == 1, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "not set in this profile" in out
    assert not (home / "fetch.marker").exists()  # the disabled source was never fetched
    assert calls == []  # and no HTTP traffic happened


def test_mcp_list_does_not_fetch_plugin_credentials(
    tmp_path, mcp_endpoint, monkeypatch, capsys
):
    port, calls = mcp_endpoint
    from hermes_cli.mcp_config import cmd_mcp_list

    home = tmp_path / "list"
    _seed_home(home, port, "list")
    _purge_fixture_state(monkeypatch, home)

    cmd_mcp_list()
    out = capsys.readouterr().out
    assert "fixture" in out
    assert not (home / "fetch.marker").exists()  # listing is non-fetching
    assert calls == []

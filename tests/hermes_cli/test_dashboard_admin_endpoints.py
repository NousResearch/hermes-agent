"""Tests for the dashboard admin API endpoints (MCP, pairing, webhooks,
credential pool, memory, gateway lifecycle, ops, skills hub).

These endpoints turn the web dashboard into an administration panel for
operators without CLI access to the host. The tests assert the request
contract and the CLI-config parity (servers/keys written via the API are
visible to the CLI data layer), not specific catalog values.
"""

import pytest


def _client():
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    # Keep the state DB under the isolated HERMES_HOME for any handler that
    # touches it.
    hermes_state.DEFAULT_DB_PATH = get_hermes_home() / "state.db"
    return client, _SESSION_HEADER_NAME


class TestMcpEndpoints:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, self.header = _client()

    def test_list_add_remove_roundtrip(self):
        assert self.client.get("/api/mcp/servers").json()["servers"] == []

        r = self.client.post(
            "/api/mcp/servers", json={"name": "srv1", "url": "https://x/mcp"}
        )
        assert r.status_code == 200
        assert r.json()["transport"] == "http"

        servers = self.client.get("/api/mcp/servers").json()["servers"]
        assert [s["name"] for s in servers] == ["srv1"]

        # CLI parity: the server is in config.yaml under mcp_servers.
        from hermes_cli.mcp_config import _get_mcp_servers

        assert "srv1" in _get_mcp_servers()

        assert self.client.delete("/api/mcp/servers/srv1").status_code == 200
        assert self.client.get("/api/mcp/servers").json()["servers"] == []

    def test_stdio_env_is_redacted_on_read(self):
        self.client.post(
            "/api/mcp/servers",
            json={
                "name": "srv2",
                "command": "npx",
                "args": ["-y", "pkg"],
                "env": {"API_KEY": "sk-secret-1234567890"},
            },
        )
        srv = self.client.get("/api/mcp/servers").json()["servers"][0]
        assert srv["env"]["API_KEY"] != "sk-secret-1234567890"

    def test_duplicate_rejected(self):
        self.client.post("/api/mcp/servers", json={"name": "dup", "url": "u"})
        r = self.client.post("/api/mcp/servers", json={"name": "dup", "url": "u"})
        assert r.status_code == 409

    def test_missing_transport_rejected(self):
        r = self.client.post("/api/mcp/servers", json={"name": "bad"})
        assert r.status_code == 400


class TestCredentialPoolEndpoints:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()

    def test_add_list_remove_and_cli_parity(self):
        assert self.client.get("/api/credentials/pool").json()["providers"] == []

        r = self.client.post(
            "/api/credentials/pool",
            json={"provider": "openrouter", "api_key": "sk-or-abcdef1234", "label": "p"},
        )
        assert r.status_code == 200 and r.json()["count"] == 1

        providers = self.client.get("/api/credentials/pool").json()["providers"]
        entry = providers[0]["entries"][0]
        # API redacts the key but exposes a preview + 1-based index.
        assert entry["index"] == 1
        assert entry["token_preview"] != "sk-or-abcdef1234"

        # CLI parity: the raw, usable key is retrievable via the pool API.
        from agent.credential_pool import load_pool

        raw = load_pool("openrouter").entries()
        assert raw[0].access_token == "sk-or-abcdef1234"

        assert self.client.delete("/api/credentials/pool/openrouter/1").status_code == 200
        assert self.client.delete("/api/credentials/pool/openrouter/99").status_code == 404

    def test_empty_body_rejected(self):
        r = self.client.post(
            "/api/credentials/pool", json={"provider": "", "api_key": ""}
        )
        assert r.status_code == 400


class TestMemoryEndpoints:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()
        from hermes_constants import get_hermes_home

        (get_hermes_home() / "memories").mkdir(parents=True, exist_ok=True)

    def test_status_and_select(self):
        data = self.client.get("/api/memory").json()
        assert "active" in data and "providers" in data and "builtin_files" in data

        r = self.client.put("/api/memory/provider", json={"provider": "built-in"})
        assert r.status_code == 200 and r.json()["active"] == ""

        r = self.client.put(
            "/api/memory/provider", json={"provider": "no-such-provider-xyz"}
        )
        assert r.status_code == 400

    def test_reset_targets(self):
        from hermes_constants import get_hermes_home

        mem = get_hermes_home() / "memories"
        (mem / "MEMORY.md").write_text("notes")
        (mem / "USER.md").write_text("user")

        r = self.client.post("/api/memory/reset", json={"target": "user"})
        assert r.status_code == 200 and "USER.md" in r.json()["deleted"]
        assert (mem / "MEMORY.md").exists()

        assert self.client.post(
            "/api/memory/reset", json={"target": "bogus"}
        ).status_code == 400


class TestPairingEndpoints:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()

    def test_list_and_bad_approve(self):
        data = self.client.get("/api/pairing").json()
        assert data == {"pending": [], "approved": []}
        r = self.client.post(
            "/api/pairing/approve", json={"platform": "telegram", "code": "NOPE99"}
        )
        assert r.status_code == 404


class TestWebhookEndpoints:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()

    def test_list_disabled_and_create_blocked(self):
        data = self.client.get("/api/webhooks").json()
        assert data["enabled"] is False
        r = self.client.post("/api/webhooks", json={"name": "gh", "deliver": "log"})
        assert r.status_code == 400


class TestOpsEndpoints:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()

    def test_hooks_list_reads_config(self):
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg["hooks"] = {
            "pre_tool_call": [
                {"matcher": "terminal", "command": "/bin/echo hi", "timeout": 5}
            ]
        }
        save_config(cfg)
        data = self.client.get("/api/ops/hooks").json()
        assert data["hooks"][0]["command"] == "/bin/echo hi"

    def test_checkpoints_list_empty(self):
        data = self.client.get("/api/ops/checkpoints").json()
        assert data == {"sessions": [], "total_bytes": 0}

    def test_import_missing_archive_404(self):
        r = self.client.post("/api/ops/import", json={"archive": "/no/such.zip"})
        assert r.status_code == 404


class TestAdminEndpointsAuthGate:
    """Every admin endpoint must sit behind the dashboard session-token gate."""

    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        from starlette.testclient import TestClient
        from hermes_cli.web_server import app

        # No session header → must be rejected.
        self.client = TestClient(app)

    @pytest.mark.parametrize(
        "path",
        [
            "/api/mcp/servers",
            "/api/pairing",
            "/api/webhooks",
            "/api/credentials/pool",
            "/api/memory",
            "/api/ops/hooks",
            "/api/ops/checkpoints",
        ],
    )
    def test_gated(self, path):
        resp = self.client.get(path)
        assert resp.status_code in (401, 403)

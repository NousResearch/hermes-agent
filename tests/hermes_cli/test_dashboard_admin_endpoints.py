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

    def test_enable_disable_toggle(self):
        self.client.post("/api/mcp/servers", json={"name": "tog", "url": "u"})
        r = self.client.put("/api/mcp/servers/tog/enabled", json={"enabled": False})
        assert r.status_code == 200 and r.json()["enabled"] is False
        srv = [
            s for s in self.client.get("/api/mcp/servers").json()["servers"]
            if s["name"] == "tog"
        ][0]
        assert srv["enabled"] is False
        # Toggling a missing server is a 404.
        assert self.client.put(
            "/api/mcp/servers/nope/enabled", json={"enabled": True}
        ).status_code == 404

    def test_catalog_lists_entries(self):
        r = self.client.get("/api/mcp/catalog")
        assert r.status_code == 200
        body = r.json()
        assert "entries" in body and "diagnostics" in body
        # The shipped optional-mcps/ catalog has at least one entry; each must
        # carry the install/enabled status fields the UI relies on.
        for e in body["entries"]:
            assert {"name", "transport", "installed", "enabled", "needs_install"} <= set(e)

    def test_catalog_install_unknown_404(self):
        r = self.client.post("/api/mcp/catalog/install", json={"name": "no-such-mcp-xyz"})
        assert r.status_code == 404



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
        assert "valid_events" in data and len(data["valid_events"]) >= 1

    def test_hook_create_and_delete(self):
        # Create with consent approval.
        r = self.client.post(
            "/api/ops/hooks",
            json={
                "event": "pre_tool_call",
                "command": "/bin/echo created",
                "matcher": "terminal",
                "timeout": 7,
                "approve": True,
            },
        )
        assert r.status_code == 200 and r.json()["approved"] is True

        hooks = self.client.get("/api/ops/hooks").json()["hooks"]
        created = [h for h in hooks if h["command"] == "/bin/echo created"]
        assert created and created[0]["allowed"] is True

        # Unknown event rejected.
        assert self.client.post(
            "/api/ops/hooks", json={"event": "no_such_event", "command": "/x"}
        ).status_code == 400

        # Delete it.
        r = self.client.request(
            "DELETE",
            "/api/ops/hooks",
            json={"event": "pre_tool_call", "command": "/bin/echo created"},
        )
        assert r.status_code == 200
        hooks2 = self.client.get("/api/ops/hooks").json()["hooks"]
        assert not [h for h in hooks2 if h["command"] == "/bin/echo created"]

    def test_checkpoints_list_empty(self):
        data = self.client.get("/api/ops/checkpoints").json()
        assert data == {"sessions": [], "total_bytes": 0}

    def test_import_missing_archive_404(self):
        r = self.client.post("/api/ops/import", json={"archive": "/no/such.zip"})
        assert r.status_code == 404


class TestSystemStatsEndpoint:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()

    def test_stats_shape(self):
        r = self.client.get("/api/system/stats")
        assert r.status_code == 200
        s = r.json()
        # Identity fields always present (stdlib-sourced).
        for key in ("os", "arch", "hostname", "python_version", "hermes_version"):
            assert key in s and s[key]
        # psutil flag tells the UI whether the richer metrics are populated.
        assert "psutil" in s


class TestCuratorEndpoints:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()

    def test_status_and_pause_toggle(self):
        r = self.client.get("/api/curator")
        assert r.status_code == 200
        body = r.json()
        assert {"enabled", "paused", "interval_hours"} <= set(body)
        # Pause then resume; the read reflects the write.
        r = self.client.put("/api/curator/paused", json={"paused": True})
        assert r.status_code == 200 and r.json()["paused"] is True
        assert self.client.get("/api/curator").json()["paused"] is True
        r = self.client.put("/api/curator/paused", json={"paused": False})
        assert r.status_code == 200 and r.json()["paused"] is False


class TestPortalEndpoint:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()

    def test_status_shape(self):
        r = self.client.get("/api/portal")
        assert r.status_code == 200
        body = r.json()
        assert {"logged_in", "features", "subscription_url", "provider"} <= set(body)
        assert isinstance(body["features"], list)


class TestSessionManagementEndpoints:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()
        from hermes_state import SessionDB

        db = SessionDB()
        db.create_session(session_id="sess-x", source="cli")
        db.close()

    def test_stats_not_shadowed_by_session_id_route(self):
        # /api/sessions/stats must resolve to the stats handler, not be captured
        # as {session_id}="stats" by the parameterized route registered after it.
        r = self.client.get("/api/sessions/stats")
        assert r.status_code == 200
        body = r.json()
        assert {"total", "active_store", "archived", "messages", "by_source"} <= set(body)
        assert body["total"] >= 1

    def test_rename(self):
        r = self.client.patch("/api/sessions/sess-x", json={"title": "Renamed"})
        assert r.status_code == 200 and r.json()["title"] == "Renamed"

    def test_export(self):
        r = self.client.get("/api/sessions/sess-x/export")
        assert r.status_code == 200 and "messages" in r.json()
        assert self.client.get("/api/sessions/nope/export").status_code == 404

    def test_prune_validation(self):
        r = self.client.post("/api/sessions/prune", json={"older_than_days": 9999})
        assert r.status_code == 200 and "removed" in r.json()
        assert self.client.post(
            "/api/sessions/prune", json={"older_than_days": 0}
        ).status_code == 400


class TestSkillsHubSearchEndpoint:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()

    def test_empty_query_returns_empty(self):
        # Empty query short-circuits (no network) and returns no results.
        r = self.client.get("/api/skills/hub/search?q=")
        assert r.status_code == 200 and r.json() == {"results": []}




class TestWebhookToggleEndpoint:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()
        # Enable the webhook platform so a subscription can be created.
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg.setdefault("platforms", {})["webhook"] = {
            "enabled": True,
            "extra": {"host": "0.0.0.0", "port": 8644},
        }
        save_config(cfg)

    def test_create_toggle_disable(self):
        r = self.client.post(
            "/api/webhooks", json={"name": "hook1", "deliver": "log", "events": ["push"]}
        )
        assert r.status_code == 200 and r.json()["enabled"] is True
        r = self.client.put("/api/webhooks/hook1/enabled", json={"enabled": False})
        assert r.status_code == 200 and r.json()["enabled"] is False
        subs = self.client.get("/api/webhooks").json()["subscriptions"]
        assert subs[0]["enabled"] is False
        assert self.client.put(
            "/api/webhooks/nope/enabled", json={"enabled": True}
        ).status_code == 404



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
            "/api/curator",
            "/api/portal",
            "/api/system/stats",
            "/api/hermes/update/check",
        ],
    )
    def test_gated(self, path):
        resp = self.client.get(path)
        assert resp.status_code in (401, 403)


class TestUpdateCheckEndpoint:
    """``GET /api/hermes/update/check`` reports availability without applying.

    Powers the dashboard's check-before-you-update flow: the System page
    shows the commit-behind count and asks the user to confirm before
    ``POST /api/hermes/update`` runs ``hermes update``.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()

    def test_git_install_reports_behind_count(self, monkeypatch):
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "detect_install_method", lambda *a, **k: "git")
        # Stub the shared checker so the contract is deterministic (no network).
        import hermes_cli.banner as banner

        monkeypatch.setattr(banner, "check_for_updates", lambda: 5)

        r = self.client.get("/api/hermes/update/check")
        assert r.status_code == 200
        body = r.json()
        assert {
            "install_method",
            "current_version",
            "behind",
            "update_available",
            "can_apply",
            "update_command",
            "message",
        } <= set(body)
        assert body["install_method"] == "git"
        assert body["behind"] == 5
        assert body["update_available"] is True
        # git/pip installs can apply the update in place from the dashboard.
        assert body["can_apply"] is True

    def test_up_to_date(self, monkeypatch):
        import hermes_cli.web_server as ws
        import hermes_cli.banner as banner

        monkeypatch.setattr(ws, "detect_install_method", lambda *a, **k: "git")
        monkeypatch.setattr(banner, "check_for_updates", lambda: 0)

        body = self.client.get("/api/hermes/update/check").json()
        assert body["behind"] == 0
        assert body["update_available"] is False

    def test_docker_is_not_applyable(self, monkeypatch):
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "detect_install_method", lambda *a, **k: "docker")
        body = self.client.get("/api/hermes/update/check").json()
        # Docker images are immutable — the dashboard can't apply an update.
        assert body["can_apply"] is False
        assert body["message"]
        assert body["behind"] is None

    def test_check_failure_is_soft(self, monkeypatch):
        import hermes_cli.web_server as ws
        import hermes_cli.banner as banner

        monkeypatch.setattr(ws, "detect_install_method", lambda *a, **k: "git")

        def _boom():
            raise RuntimeError("offline")

        monkeypatch.setattr(banner, "check_for_updates", _boom)
        # A failed check must not 500 — it returns behind=null with guidance.
        r = self.client.get("/api/hermes/update/check")
        assert r.status_code == 200
        body = r.json()
        assert body["behind"] is None
        assert body["update_available"] is False
        assert body["message"]


class TestDebugShareEndpoint:
    """POST /api/ops/debug-share returns the paste URLs synchronously so the
    dashboard can render them as copyable links (not a backgrounded log tail)."""

    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, self.header = _client()
        from hermes_constants import get_hermes_home

        logs = get_hermes_home() / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "agent.log").write_text("agent line\n")
        (logs / "errors.log").write_text("err line\n")
        (logs / "gateway.log").write_text("gw line\n")

    def test_returns_structured_urls(self, monkeypatch):
        import hermes_cli.debug as dbg

        count = [0]

        def _upload(content, expiry_days=7):
            count[0] += 1
            return f"https://paste.rs/p{count[0]}"

        monkeypatch.setattr(dbg, "upload_to_pastebin", _upload)
        monkeypatch.setattr(dbg, "_schedule_auto_delete", lambda *a, **k: None)
        monkeypatch.setattr(dbg, "_best_effort_sweep_expired_pastes", lambda: None)
        monkeypatch.setattr("hermes_cli.dump.run_dump", lambda a: None)

        r = self.client.post("/api/ops/debug-share", json={"redact": True})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert "Report" in body["urls"]
        assert body["redacted"] is True
        assert body["auto_delete_seconds"] == 21600
        assert isinstance(body["failures"], list)

    def test_redact_false_is_honored(self, monkeypatch):
        import hermes_cli.debug as dbg

        monkeypatch.setattr(
            dbg, "upload_to_pastebin", lambda c, expiry_days=7: "https://paste.rs/x"
        )
        monkeypatch.setattr(dbg, "_schedule_auto_delete", lambda *a, **k: None)
        monkeypatch.setattr(dbg, "_best_effort_sweep_expired_pastes", lambda: None)
        monkeypatch.setattr("hermes_cli.dump.run_dump", lambda a: None)

        r = self.client.post("/api/ops/debug-share", json={"redact": False})
        assert r.status_code == 200
        assert r.json()["redacted"] is False

    def test_default_body_redacts(self, monkeypatch):
        import hermes_cli.debug as dbg

        monkeypatch.setattr(
            dbg, "upload_to_pastebin", lambda c, expiry_days=7: "https://paste.rs/x"
        )
        monkeypatch.setattr(dbg, "_schedule_auto_delete", lambda *a, **k: None)
        monkeypatch.setattr(dbg, "_best_effort_sweep_expired_pastes", lambda: None)
        monkeypatch.setattr("hermes_cli.dump.run_dump", lambda a: None)

        # No JSON body at all — should default redact=True.
        r = self.client.post("/api/ops/debug-share")
        assert r.status_code == 200
        assert r.json()["redacted"] is True

    def test_upload_failure_returns_502(self, monkeypatch):
        import hermes_cli.debug as dbg

        monkeypatch.setattr(
            dbg,
            "upload_to_pastebin",
            lambda c, expiry_days=7: (_ for _ in ()).throw(RuntimeError("down")),
        )
        monkeypatch.setattr(dbg, "_schedule_auto_delete", lambda *a, **k: None)
        monkeypatch.setattr(dbg, "_best_effort_sweep_expired_pastes", lambda: None)
        monkeypatch.setattr("hermes_cli.dump.run_dump", lambda a: None)

        r = self.client.post("/api/ops/debug-share", json={"redact": True})
        assert r.status_code == 502

    def test_requires_session_token(self):
        # Drop the token header and confirm the global auth gate rejects it.
        bare = self.client
        r = bare.post(
            "/api/ops/debug-share",
            json={"redact": True},
            headers={self.header: "wrong-token"},
        )
        assert r.status_code == 401


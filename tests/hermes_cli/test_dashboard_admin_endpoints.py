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
        # carry the install/enabled status fields plus the inspection detail
        # the dashboard renders (transport target, install source, guidance) so
        # users can vet an entry before installing.
        for e in body["entries"]:
            assert {
                "name",
                "transport",
                "auth_type",
                "installed",
                "enabled",
                "needs_install",
                "command",
                "args",
                "url",
                "install_url",
                "install_ref",
                "bootstrap",
                "default_enabled",
                "post_install",
            } <= set(e)
            # http entries expose a url; stdio entries expose a command.
            if e["transport"] == "http":
                assert e["url"]
            elif e["transport"] == "stdio":
                assert e["command"]

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

    def test_enable_platform_starts_gateway_restart(self, monkeypatch):
        import hermes_cli.web_server as ws
        from hermes_cli.config import load_config

        ws._ACTION_PROCS.pop("gateway-restart", None)
        restart_calls = []

        class FakeRestartProc:
            pid = 4242

        def fake_spawn_action(subcommand, name):
            restart_calls.append((subcommand, name))
            return FakeRestartProc()

        monkeypatch.setattr(ws, "_spawn_hermes_action", fake_spawn_action)

        r = self.client.post("/api/webhooks/enable")

        assert r.status_code == 200
        assert r.json() == {
            "ok": True,
            "platform": "webhook",
            "enabled": True,
            "needs_restart": False,
            "restart_started": True,
            "restart_action": "gateway-restart",
            "restart_pid": 4242,
        }
        assert restart_calls == [(["gateway", "restart"], "gateway-restart")]
        assert load_config()["platforms"]["webhook"]["enabled"] is True
        assert self.client.get("/api/webhooks").json()["enabled"] is True

    def test_enable_platform_reports_restart_failure_after_save(self, monkeypatch):
        import hermes_cli.web_server as ws
        from hermes_cli.config import load_config

        ws._ACTION_PROCS.pop("gateway-restart", None)

        def fail_spawn_action(subcommand, name):
            assert subcommand == ["gateway", "restart"]
            assert name == "gateway-restart"
            raise RuntimeError("supervisor unavailable")

        monkeypatch.setattr(ws, "_spawn_hermes_action", fail_spawn_action)

        r = self.client.post("/api/webhooks/enable")

        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["platform"] == "webhook"
        assert data["enabled"] is True
        assert data["needs_restart"] is True
        assert data["restart_started"] is False
        assert "supervisor unavailable" in data["restart_error"]
        assert load_config()["platforms"]["webhook"]["enabled"] is True

    def test_enable_platform_reuses_inflight_gateway_restart(self, monkeypatch):
        import hermes_cli.web_server as ws
        from hermes_cli.config import load_config

        ws._ACTION_PROCS.pop("gateway-restart", None)

        class FakeRunningProc:
            pid = 5151

            def poll(self):
                return None

        monkeypatch.setitem(ws._ACTION_PROCS, "gateway-restart", FakeRunningProc())

        def fail_spawn_action(subcommand, name):
            raise AssertionError("must not spawn a second concurrent restart")

        monkeypatch.setattr(ws, "_spawn_hermes_action", fail_spawn_action)

        r = self.client.post("/api/webhooks/enable")

        assert r.status_code == 200
        data = r.json()
        assert data["needs_restart"] is False
        assert data["restart_started"] is True
        assert data["restart_pid"] == 5151
        assert load_config()["platforms"]["webhook"]["enabled"] is True


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
        # Empty query short-circuits (no network) and returns the enriched
        # empty shape (results + per-source counts + timeouts + installed map).
        r = self.client.get("/api/skills/hub/search?q=")
        assert r.status_code == 200
        body = r.json()
        assert body["results"] == []
        assert body["source_counts"] == {}
        assert body["timed_out"] == []
        assert body["installed"] == {}


class _FakeMeta:
    """Minimal SkillMeta stand-in for monkeypatched source search."""

    def __init__(self, identifier, trust_level="community", source="github"):
        self.name = identifier.rsplit("/", 1)[-1]
        self.description = "desc"
        self.source = source
        self.identifier = identifier
        self.trust_level = trust_level
        self.repo = "owner/repo"
        self.tags = ["a", "b"]
        # Used by the preview endpoint's getattr() fallbacks.
        self.files = {}


class _FakeBundle:
    def __init__(self, identifier, source="github", trust_level="community"):
        self.name = identifier.rsplit("/", 1)[-1]
        self.identifier = identifier
        self.source = source
        self.trust_level = trust_level
        self.description = "desc"
        self.repo = "owner/repo"
        self.tags = ["a", "b"]
        # Mix str + bytes to exercise the decode-or-placeholder branch.
        self.files = {
            "SKILL.md": b"---\nname: x\n---\nbody text",
            "icon.png": b"\xff\xd8\xff\xe0binary",
            "notes.txt": "plain string content",
        }
        self.metadata = {}


class TestSkillsHubSourcesEndpoint:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()

    def test_sources_lists_configured_hubs(self, monkeypatch):
        # The endpoint should enumerate the configured hub sources without
        # requiring any live network — monkeypatch the router.
        class _Src:
            is_available = False

            def __init__(self, sid):
                self._sid = sid

            def source_id(self):
                return self._sid

            def search(self, q, limit=10):
                return [_FakeMeta("hermes-index/featured-skill", "trusted")]

        def _fake_router():
            srcs = [_Src("official"), _Src("github")]
            # hermes-index source advertises availability + featured search.
            idx = _Src("hermes-index")
            idx.is_available = True
            srcs.insert(1, idx)
            return srcs

        monkeypatch.setattr(
            "tools.skills_hub.create_source_router", _fake_router
        )
        r = self.client.get("/api/skills/hub/sources")
        assert r.status_code == 200
        body = r.json()
        ids = {s["id"] for s in body["sources"]}
        assert {"official", "github", "hermes-index"} <= ids
        # Every source carries a human label.
        assert all(s.get("label") for s in body["sources"])
        assert body["index_available"] is True
        # Featured pulled from the index (zero extra API calls).
        assert len(body["featured"]) == 1
        assert body["featured"][0]["trust_level"] == "trusted"
        assert isinstance(body["installed"], dict)


class TestSkillsHubPreviewEndpoint:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()

    def test_preview_requires_identifier(self):
        r = self.client.get("/api/skills/hub/preview?identifier=")
        assert r.status_code == 400

    def test_preview_returns_skill_md_text(self, monkeypatch):
        monkeypatch.setattr(
            "tools.skills_hub.create_source_router", lambda: []
        )
        bundle = _FakeBundle("github/owner/repo/x")
        meta = _FakeMeta("github/owner/repo/x")
        monkeypatch.setattr(
            "hermes_cli.skills_hub._resolve_source_meta_and_bundle",
            lambda ident, sources: (meta, bundle, None),
        )
        r = self.client.get(
            "/api/skills/hub/preview?identifier=github/owner/repo/x"
        )
        assert r.status_code == 200
        body = r.json()
        # Bytes-stored SKILL.md decodes to text.
        assert "body text" in body["skill_md"]
        # Binary file is masked, text files decode.
        assert "icon.png" in body["files"]
        assert sorted(body["files"]) == ["SKILL.md", "icon.png", "notes.txt"]

    def test_preview_404_when_unresolved(self, monkeypatch):
        monkeypatch.setattr(
            "tools.skills_hub.create_source_router", lambda: []
        )
        monkeypatch.setattr(
            "hermes_cli.skills_hub._resolve_source_meta_and_bundle",
            lambda ident, sources: (None, None, None),
        )
        r = self.client.get("/api/skills/hub/preview?identifier=nope/x")
        assert r.status_code == 404


class TestSkillsHubScanEndpoint:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()

    def test_scan_requires_identifier(self):
        r = self.client.get("/api/skills/hub/scan?identifier=")
        assert r.status_code == 400

    def test_scan_returns_verdict_and_policy(self, monkeypatch):
        from tools.skills_guard import ScanResult, Finding

        monkeypatch.setattr(
            "tools.skills_hub.create_source_router", lambda: []
        )
        bundle = _FakeBundle("github/owner/repo/x", trust_level="community")
        monkeypatch.setattr(
            "hermes_cli.skills_hub._resolve_source_meta_and_bundle",
            lambda ident, sources: (None, bundle, None),
        )

        from pathlib import Path

        monkeypatch.setattr(
            "tools.skills_hub.quarantine_bundle", lambda b: Path("/tmp/_fake_q")
        )

        fake_result = ScanResult(
            skill_name="x",
            source="github/owner/repo/x",
            trust_level="community",
            verdict="caution",
            findings=[
                Finding(
                    pattern_id="p",
                    severity="high",
                    category="exfiltration",
                    file="SKILL.md",
                    line=10,
                    match="m",
                    description="leaks data",
                )
            ],
            summary="s",
        )
        monkeypatch.setattr(
            "tools.skills_guard.scan_skill",
            lambda path, source="community": fake_result,
        )
        # Avoid touching the filesystem during cleanup.
        monkeypatch.setattr("shutil.rmtree", lambda *a, **k: None)

        r = self.client.get(
            "/api/skills/hub/scan?identifier=github/owner/repo/x"
        )
        assert r.status_code == 200
        body = r.json()
        assert body["verdict"] == "caution"
        assert body["trust_level"] == "community"
        # community + caution => blocked by install policy.
        assert body["policy"] == "block"
        assert body["severity_counts"]["high"] == 1
        assert body["findings"][0]["category"] == "exfiltration"
        assert body["findings"][0]["file"] == "SKILL.md"

    def test_scan_404_when_no_bundle(self, monkeypatch):
        monkeypatch.setattr(
            "tools.skills_hub.create_source_router", lambda: []
        )
        monkeypatch.setattr(
            "hermes_cli.skills_hub._resolve_source_meta_and_bundle",
            lambda ident, sources: (None, None, None),
        )
        r = self.client.get("/api/skills/hub/scan?identifier=nope/x")
        assert r.status_code == 404




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
            "/api/appearance",
        ],
    )
    def test_gated(self, path):
        resp = self.client.get(path)
        assert resp.status_code in (401, 403)

    def test_webhooks_enable_post_gated(self):
        resp = self.client.post("/api/webhooks/enable")
        assert resp.status_code in (401, 403)

    def test_appearance_put_gated(self):
        resp = self.client.put("/api/appearance", json={"dashboard_theme": "mono"})
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

    def test_managed_runtime_dashboard_is_not_applyable(self, monkeypatch):
        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws, "_dashboard_local_update_managed_externally", lambda: True)
        monkeypatch.setattr(
            ws,
            "detect_install_method",
            lambda *a, **k: pytest.fail(
                "managed runtime update check should not probe install method"
            ),
        )

        body = self.client.get("/api/hermes/update/check").json()
        assert body["install_method"] == "managed-runtime"
        assert body["can_apply"] is False
        assert body["update_available"] is False
        assert body["behind"] is None
        assert "managed outside this dashboard" in body["message"]

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

    def test_git_behind_includes_commits(self, monkeypatch):
        import hermes_cli.web_server as ws
        import hermes_cli.banner as banner

        monkeypatch.setattr(ws, "detect_install_method", lambda *a, **k: "git")
        monkeypatch.setattr(banner, "check_for_updates", lambda: 3)
        monkeypatch.setattr(
            ws,
            "_recent_upstream_commits",
            lambda n=20: [
                {"sha": "abc1234", "summary": "feat: x", "author": "a", "at": 1},
            ],
        )

        body = self.client.get("/api/hermes/update/check").json()
        # The desktop overlay renders this as the "what's changed" list.
        assert isinstance(body["commits"], list)
        assert body["commits"][0]["sha"] == "abc1234"
        assert body["commits"][0]["summary"] == "feat: x"

    def test_up_to_date_omits_commits(self, monkeypatch):
        import hermes_cli.web_server as ws
        import hermes_cli.banner as banner

        monkeypatch.setattr(ws, "detect_install_method", lambda *a, **k: "git")
        monkeypatch.setattr(banner, "check_for_updates", lambda: 0)

        body = self.client.get("/api/hermes/update/check").json()
        # No commits list when there's nothing to show (additive, non-breaking).
        assert body.get("commits", []) == []


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


class TestToolsConfigEndpoints:
    """Provider selection, API-key save, and post-setup spawn for toolsets —
    the dashboard surface that replicates the `hermes tools` configurator."""

    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, self.header = _client()

    def test_list_toolsets_shape(self):
        r = self.client.get("/api/tools/toolsets")
        assert r.status_code == 200
        rows = r.json()
        assert isinstance(rows, list) and rows
        row = rows[0]
        for k in ("name", "label", "enabled", "configured", "tools"):
            assert k in row

    def test_toolset_config_provider_matrix(self):
        # `web` has a TOOL_CATEGORIES entry → providers list populated.
        r = self.client.get("/api/tools/toolsets/web/config")
        assert r.status_code == 200
        body = r.json()
        assert body["has_category"] is True
        assert isinstance(body["providers"], list)

    def test_unknown_toolset_config_400(self):
        r = self.client.get("/api/tools/toolsets/not_a_toolset/config")
        assert r.status_code == 400

    def test_save_env_writes_key_and_validates_allowlist(self):
        from hermes_cli.config import get_env_value

        cfg = self.client.get("/api/tools/toolsets/web/config").json()
        # Find a real env-var key from the visible provider matrix.
        key = None
        for prov in cfg["providers"]:
            for e in prov.get("env_vars", []):
                key = e["key"]
                break
            if key:
                break
        if not key:
            pytest.skip("no env-var-bearing web provider in this build")

        r = self.client.put(
            "/api/tools/toolsets/web/env", json={"env": {key: "test-secret-123"}}
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert key in body["saved"]
        assert body["is_set"][key] is True
        # CLI-config parity: the key landed in the .env store the CLI reads.
        assert get_env_value(key) == "test-secret-123"

    def test_save_env_rejects_unknown_key(self):
        r = self.client.put(
            "/api/tools/toolsets/web/env",
            json={"env": {"TOTALLY_BOGUS_KEY": "x"}},
        )
        assert r.status_code == 400

    def test_save_env_blank_value_skipped(self):
        cfg = self.client.get("/api/tools/toolsets/web/config").json()
        key = None
        for prov in cfg["providers"]:
            for e in prov.get("env_vars", []):
                key = e["key"]
                break
            if key:
                break
        if not key:
            pytest.skip("no env-var-bearing web provider in this build")
        r = self.client.put(
            "/api/tools/toolsets/web/env", json={"env": {key: "   "}}
        )
        assert r.status_code == 200
        assert key in r.json()["skipped"]

    def test_post_setup_unknown_key_400(self):
        r = self.client.post(
            "/api/tools/toolsets/browser/post-setup", json={"key": "bogus"}
        )
        assert r.status_code == 400

    def test_post_setup_unknown_toolset_400(self):
        r = self.client.post(
            "/api/tools/toolsets/not_a_toolset/post-setup",
            json={"key": "agent_browser"},
        )
        assert r.status_code == 400

    def test_post_setup_spawns_action(self, monkeypatch):
        import hermes_cli.web_server as ws

        spawned = {}

        class _FakeProc:
            pid = 4321

        def _fake_spawn(subcommand, name):
            spawned["subcommand"] = subcommand
            spawned["name"] = name
            return _FakeProc()

        monkeypatch.setattr(ws, "_spawn_hermes_action", _fake_spawn)
        r = self.client.post(
            "/api/tools/toolsets/browser/post-setup",
            json={"key": "agent_browser"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["name"] == "tools-post-setup"
        assert body["pid"] == 4321
        assert spawned["subcommand"] == ["tools", "post-setup", "agent_browser"]

    def test_endpoints_require_session_token(self):
        for method, path, payload in [
            ("get", "/api/tools/toolsets/web/config", None),
            ("put", "/api/tools/toolsets/web/env", {"env": {}}),
            ("post", "/api/tools/toolsets/web/post-setup", {"key": "ddgs"}),
        ]:
            fn = getattr(self.client, method)
            kwargs = {"headers": {self.header: "wrong-token"}}
            if payload is not None:
                kwargs["json"] = payload
            r = fn(path, **kwargs)
            assert r.status_code == 401, f"{method} {path} not gated"


class TestAppearanceEndpoints:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, self.header = _client()

    def test_get_appearance_lists_presets_themes_skins_and_custom_entries(self):
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
        (home / "skins").mkdir(parents=True, exist_ok=True)
        (home / "skins" / "ocean-chat.yaml").write_text(
            "name: ocean-chat\ndescription: Custom ocean skin\n",
            encoding="utf-8",
        )
        (home / "dashboard-themes").mkdir(parents=True, exist_ok=True)
        (home / "dashboard-themes" / "ocean.yaml").write_text(
            "name: ocean\nlabel: Ocean\ndescription: Custom ocean dashboard\n"
            "palette:\n  background: '#001122'\n  midground: '#aaddff'\n",
            encoding="utf-8",
        )

        r = self.client.get("/api/appearance")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["dashboard_theme"] == "default"
        assert body["chat_skin"] == "default"
        assert "hermes-teal" in {preset["id"] for preset in body["presets"]}
        assert "ocean" in {theme["name"] for theme in body["themes"]}
        assert "ocean-chat" in {skin["name"] for skin in body["skins"]}

    def test_set_appearance_writes_theme_and_skin(self, monkeypatch):
        import hermes_cli.web_server as ws
        from hermes_cli.config import load_config

        monkeypatch.setattr(
            ws,
            "_broadcast_tui_skin_changed",
            lambda: {"connected": 1, "sent": 1},
        )

        r = self.client.put(
            "/api/appearance",
            json={"dashboard_theme": "mono", "chat_skin": "ares"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["dashboard_theme"] == "mono"
        assert body["chat_skin"] == "ares"
        assert body["skin_broadcast"] == {"connected": 1, "sent": 1}

        cfg = load_config()
        assert cfg["dashboard"]["theme"] == "mono"
        assert cfg["display"]["skin"] == "ares"

    def test_set_appearance_rejects_invalid_values_without_corrupting_config(self):
        from hermes_cli.config import load_config

        ok = self.client.put(
            "/api/appearance",
            json={"dashboard_theme": "mono", "chat_skin": "mono"},
        )
        assert ok.status_code == 200, ok.text

        r = self.client.put(
            "/api/appearance",
            json={"dashboard_theme": "missing-theme", "chat_skin": "ares"},
        )
        assert r.status_code == 400

        cfg = load_config()
        assert cfg["dashboard"]["theme"] == "mono"
        assert cfg["display"]["skin"] == "mono"

    def test_config_schema_uses_dynamic_theme_and_skin_options(self):
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
        (home / "skins").mkdir(parents=True, exist_ok=True)
        (home / "skins" / "custom-chat.yaml").write_text(
            "name: custom-chat\ndescription: Custom chat skin\n",
            encoding="utf-8",
        )
        (home / "dashboard-themes").mkdir(parents=True, exist_ok=True)
        (home / "dashboard-themes" / "custom-dashboard.yaml").write_text(
            "name: custom-dashboard\nlabel: Custom Dashboard\n"
            "palette:\n  background: '#101010'\n  midground: '#eeeeee'\n",
            encoding="utf-8",
        )

        r = self.client.get("/api/config/schema")
        assert r.status_code == 200, r.text
        fields = r.json()["fields"]
        assert "custom-chat" in fields["display.skin"]["options"]
        assert "custom-dashboard" in fields["dashboard.theme"]["options"]
        assert "default-large" in fields["dashboard.theme"]["options"]
        assert "charizard" in fields["display.skin"]["options"]

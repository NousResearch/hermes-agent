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

    def test_http_bearer_auth_separates_secret_from_config(
        self, _isolate_hermes_home
    ):
        from hermes_constants import get_hermes_home

        secret = "dashboard-secret-value"
        response = self.client.post(
            "/api/mcp/servers",
            json={
                "name": "Bearer Server",
                "url": "https://example.com/mcp",
                "auth": "header",
                "bearer_token": f"Bearer {secret}",
            },
        )

        assert response.status_code == 200
        assert response.json()["auth"] == "header"
        assert "bearer_token" not in response.json()

        hermes_home = get_hermes_home()
        config_text = (hermes_home / "config.yaml").read_text()
        env_text = (hermes_home / ".env").read_text()
        assert secret not in config_text
        assert "Bearer ${MCP_BEARER_SERVER_API_KEY}" in config_text
        assert f"MCP_BEARER_SERVER_API_KEY={secret}" in env_text

    def test_http_oauth_mode_is_persisted_for_existing_auth_flow(self):
        response = self.client.post(
            "/api/mcp/servers",
            json={
                "name": "oauth-server",
                "url": "https://example.com/mcp",
                "auth": "oauth",
            },
        )

        assert response.status_code == 200
        assert response.json()["auth"] == "oauth"

        from hermes_cli.mcp_config import _get_mcp_servers

        assert _get_mcp_servers()["oauth-server"]["auth"] == "oauth"

    @pytest.mark.parametrize(
        ("payload", "error"),
        [
            (
                {"name": "bad", "url": "https://x/mcp", "env": {"KEY": "value"}},
                "only supported for stdio",
            ),
            (
                {"name": "bad", "url": "https://x/mcp", "args": ["ignored"]},
                "only supported for stdio",
            ),
            (
                {"name": "bad", "command": "npx", "auth": "oauth"},
                "not supported for stdio",
            ),
            (
                {"name": "bad", "url": "https://x/mcp", "auth": "header"},
                "Bearer token is required",
            ),
            (
                {
                    "name": "bad",
                    "url": "https://x/mcp",
                    "auth": "header",
                    "bearer_token": "Bearer   ",
                },
                "Bearer token is required",
            ),
            (
                {"name": "bad", "url": "https://x/mcp", "bearer_token": "secret"},
                "requires header authentication",
            ),
            (
                {"name": "bad", "url": "https://x/mcp", "command": "npx"},
                "exactly one",
            ),
            (
                {"name": "bad", "url": "https://x/mcp", "auth": "unknown"},
                "Unsupported auth mode",
            ),
        ],
    )
    def test_transport_auth_contract_is_enforced(self, payload, error):
        response = self.client.post("/api/mcp/servers", json=payload)

        assert response.status_code == 400
        assert error in response.json()["detail"]





class TestCredentialPoolEndpoints:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()



    def test_env_seeded_delete_stays_deleted(self):
        """#55217: DELETE must suppress the source or load_pool() resurrects it.

        load_pool() re-seeds from ~/.hermes/.env on every call, so removing
        just the pool row silently reverts on the next dashboard refresh.
        The endpoint must mirror `hermes auth remove`: clean up the backing
        source and suppress (provider, source).
        """
        from agent.credential_pool import load_pool
        from hermes_cli.auth import is_source_suppressed
        from hermes_cli.config import save_env_value

        fake_key = "sk-or-" + "x" * 20  # constructed, never a real key shape
        save_env_value("OPENROUTER_API_KEY", fake_key)

        entries = load_pool("openrouter").entries()
        assert [e.source for e in entries] == ["env:OPENROUTER_API_KEY"]

        r = self.client.delete("/api/credentials/pool/openrouter/1")
        assert r.status_code == 200

        # Suppressed exactly like the CLI removal path.
        assert is_source_suppressed("openrouter", "env:OPENROUTER_API_KEY")

        # Even if the backing var comes back (shell export, another process
        # rewriting .env), the removal must stay sticky.
        save_env_value("OPENROUTER_API_KEY", fake_key)
        assert load_pool("openrouter").entries() == []
        assert self.client.get("/api/credentials/pool").json()["providers"] == []

    def test_post_readd_lifts_suppression(self):
        """Re-adding via POST is an explicit re-engagement — suppressions lift.

        Mirrors `hermes auth add`, which clears every suppression for the
        provider so a user who deleted a credential and re-adds one isn't
        silently blocked from env re-seeding.
        """
        from agent.credential_pool import load_pool
        from hermes_cli.auth import is_source_suppressed
        from hermes_cli.config import save_env_value

        fake_key = "sk-or-" + "y" * 20
        save_env_value("OPENROUTER_API_KEY", fake_key)
        load_pool("openrouter")
        assert self.client.delete("/api/credentials/pool/openrouter/1").status_code == 200
        assert is_source_suppressed("openrouter", "env:OPENROUTER_API_KEY")

        r = self.client.post(
            "/api/credentials/pool",
            json={"provider": "openrouter", "api_key": "sk-or-" + "z" * 20},
        )
        assert r.status_code == 200
        assert not is_source_suppressed("openrouter", "env:OPENROUTER_API_KEY")

        # Key back in .env + suppression lifted → env entry seeds alongside
        # the manual one.
        save_env_value("OPENROUTER_API_KEY", fake_key)
        sources = sorted(e.source for e in load_pool("openrouter").entries())
        assert sources == ["env:OPENROUTER_API_KEY", "manual"]




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

    def test_approve_pending_request_id(self):
        from gateway.pairing import PairingStore

        store = PairingStore()
        bot_code = store.generate_code("telegram", "user1", "Alice")
        data = self.client.get("/api/pairing").json()
        request_id = data["pending"][0]["request_id"]

        assert request_id
        assert request_id != bot_code

        r = self.client.post(
            "/api/pairing/approve",
            json={"platform": "telegram", "request_id": request_id},
        )

        assert r.status_code == 200
        assert r.json()["user"]["user_id"] == "user1"
        assert self.client.get("/api/pairing").json()["pending"] == []

    def test_pairing_is_isolated_per_profile(self):
        """A named profile's approvals must land in the store its own gateway reads.

        The gateway keeps one PairingStore per served profile, so an approval
        written to the global store grants access the running gateway never
        consults — the user stays locked out with the dashboard showing them
        as approved.
        """
        from gateway.pairing import PairingStore
        from hermes_constants import get_hermes_home

        (get_hermes_home() / "profiles" / "work").mkdir(parents=True, exist_ok=True)
        PairingStore().generate_code("telegram", "global-1", "GlobalGuy")
        PairingStore(profile="work").generate_code("telegram", "work-1", "WorkGal")

        listed = self.client.get("/api/pairing?profile=work").json()["pending"]
        assert [row["user_id"] for row in listed] == ["work-1"]

        r = self.client.post(
            "/api/pairing/approve",
            json={
                "platform": "telegram",
                "request_id": listed[0]["request_id"],
                "profile": "work",
            },
        )
        assert r.status_code == 200

        # The grant is visible to the profile's own store — what the gateway reads.
        assert PairingStore(profile="work").is_approved("telegram", "work-1") is True

        # ...and it never leaked into the global store, whose own pending row
        # is still waiting. (Asserted against this user rather than an empty
        # list: the module-level PAIRING_DIR is bound at import, so the global
        # store carries whatever earlier cases in this class approved.)
        global_view = self.client.get("/api/pairing").json()
        assert PairingStore().is_approved("telegram", "work-1") is False
        assert "work-1" not in [row["user_id"] for row in global_view["approved"]]
        assert "global-1" in [row["user_id"] for row in global_view["pending"]]

    def test_unknown_profile_is_rejected(self):
        assert self.client.get("/api/pairing?profile=ghost").status_code == 404


class TestWebhookEndpoints:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()


    def test_create_webhook_persists_script(self):
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        cfg.setdefault("platforms", {})["webhook"] = {
            "enabled": True,
            "extra": {"host": "0.0.0.0", "port": 8644},
        }
        save_config(cfg)

        r = self.client.post(
            "/api/webhooks",
            json={
                "name": "todoist",
                "deliver": "log",
                "script": "todoist_filter.py",
            },
        )
        assert r.status_code == 200
        assert r.json()["script"] == "todoist_filter.py"

        subs = self.client.get("/api/webhooks").json()["subscriptions"]
        assert subs[0]["script"] == "todoist_filter.py"

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


class TestPortalEndpoint:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()


class TestSessionManagementEndpoints:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()
        from hermes_state import SessionDB

        db = SessionDB()
        db.create_session(session_id="sess-x", source="cli")
        db.close()


    def test_stats_source_counts_use_direct_aggregate(self, monkeypatch):
        """Source badges must not materialise rich session rows.

        Large stores can have thousands of sessions. The stats endpoint only
        needs grouped counts, so it should call ``session_count_by_source``
        instead of ``list_sessions_rich`` and build preview/last-active rows
        just to count source labels.
        """
        from hermes_state import SessionDB

        def fail_list_sessions_rich(self, *args, **kwargs):
            raise AssertionError("stats should use grouped source counts, not list_sessions_rich")

        monkeypatch.setattr(SessionDB, "list_sessions_rich", fail_list_sessions_rich)

        r = self.client.get("/api/sessions/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["by_source"]["cli"] >= 1



    def test_prune_attr_filter_suppresses_default_cutoff(self):
        # An attribute filter without an explicit older_than_days matches all
        # ages (mirrors the CLI: any filter disables the implicit 90-day
        # default). dry_run so nothing is deleted; the seeded session is
        # recent + ended, so it would be invisible under a 90-day cutoff.
        from hermes_state import SessionDB

        db = SessionDB()
        db.create_session(session_id="sess-recent-ended", source="cli")
        db.end_session("sess-recent-ended", "completed")
        db.close()

        r = self.client.post(
            "/api/sessions/prune",
            json={"source": "cli", "dry_run": True},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["matched"] >= 1
        assert "skipped_open" in body
        assert "oldest_started_at" in body and "newest_started_at" in body
        assert "oldest_last_active" in body and "newest_last_active" in body
        assert all("last_active" in session for session in body["sessions"])

    def test_prune_reports_open_sessions_excluded_by_safety_guard(self):
        from hermes_state import SessionDB

        db = SessionDB()
        db.create_session(session_id="sess-old-open", source="skip-test")
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (1.0, "sess-old-open"),
        )
        db._conn.commit()
        db.close()

        r = self.client.post(
            "/api/sessions/prune",
            json={"older_than_days": 1, "source": "skip-test"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["removed"] == 0
        assert body["skipped_open"] == 1

        db = SessionDB()
        assert db.get_session("sess-old-open") is not None
        db.close()



class TestSkillsHubSearchEndpoint:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()


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


class TestOfficialSkillsCatalogEndpoint:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()

    def test_lists_full_catalog_with_installed_flags(self, monkeypatch):
        # Serve the catalog from OptionalSkillSource.list_local() (no network),
        # with the per-profile installed map marking already-installed rows.
        metas = [
            _FakeMeta("official/gifs/gif-search", "builtin", "official"),
            _FakeMeta("official/creative/ascii-art", "builtin", "official"),
        ]
        monkeypatch.setattr(
            "tools.skills_hub.OptionalSkillSource.list_local",
            lambda self: metas,
        )
        monkeypatch.setattr(
            "hermes_cli.web_server._installed_hub_identifiers",
            lambda profile=None: {"official/gifs/gif-search": {"name": "gif-search"}},
        )
        r = self.client.get("/api/skills/hub/official")
        assert r.status_code == 200
        skills = r.json()["skills"]
        by_ident = {s["identifier"]: s for s in skills}
        assert set(by_ident) == {
            "official/gifs/gif-search",
            "official/creative/ascii-art",
        }
        # Category derived from the identifier's directory segment.
        assert by_ident["official/gifs/gif-search"]["category"] == "gifs"
        assert by_ident["official/creative/ascii-art"]["category"] == "creative"
        # Installed flag comes from the profile's hub lock.
        assert by_ident["official/gifs/gif-search"]["installed"] is True
        assert by_ident["official/creative/ascii-art"]["installed"] is False


class TestSkillsHubPreviewEndpoint:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, _ = _client()


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


class TestAdminEndpointsAuthGate:
    """Every admin endpoint must sit behind the dashboard session-token gate."""

    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        from starlette.testclient import TestClient
        from hermes_cli.web_server import app

        # No session header → must be rejected.
        self.client = TestClient(app)


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



class TestToolsConfigEndpoints:
    """Provider selection, API-key save, and post-setup spawn for toolsets —
    the dashboard surface that replicates the `hermes tools` configurator."""

    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client, self.header = _client()




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



    def test_post_setup_unknown_toolset_400(self):
        r = self.client.post(
            "/api/tools/toolsets/not_a_toolset/post-setup",
            json={"key": "agent_browser"},
        )
        assert r.status_code == 400




# ---------------------------------------------------------------------------
# _spawn_hermes_action env scrubbing (#52470)
# ---------------------------------------------------------------------------

def test_spawn_hermes_action_scrubs_gateway_loop_guard_env(monkeypatch, tmp_path):
    """The dashboard runs inside the gateway, so os.environ has
    _HERMES_GATEWAY=1. Spawned actions (e.g. `gateway restart`) must NOT inherit
    it, or the in-process restart-loop guard rejects the restart and it silently
    fails (#52470).
    """
    import hermes_cli.web_server as ws

    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setattr(ws, "_ACTION_LOG_DIR", tmp_path)
    # Isolate the module-global proc registry: _spawn_hermes_action stores
    # _FakeProc (no poll()) in _ACTION_PROCS, and later tests' lifespan
    # shutdown (_terminate_desktop_managed_gateway) would trip over it.
    monkeypatch.setattr(ws, "_ACTION_PROCS", {})

    captured = {}

    class _FakeProc:
        pid = 1234

    def _fake_popen(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return _FakeProc()

    monkeypatch.setattr(ws.subprocess, "Popen", _fake_popen)

    ws._spawn_hermes_action(["gateway", "restart"], "gateway-restart")

    assert "_HERMES_GATEWAY" not in captured["env"]
    assert captured["env"]["HERMES_NONINTERACTIVE"] == "1"


# ---------------------------------------------------------------------------
# Desktop lifespan reaps orphan gateways at serve startup (#77276)
# ---------------------------------------------------------------------------

def test_desktop_lifespan_reaps_orphan_gateways_on_startup(
    monkeypatch, _isolate_hermes_home
):
    """Starting a Desktop serve backend should reap orphan gateways left by a
    previous serve session before forking a fresh one (#77276).

    Graceful shutdown reaps the managed child, but an abnormal exit reparents
    the old gateway to launchd (PPID=1) where it keeps holding the QQ
    WebSocket. The lifespan calls _reap_unsupervised_gateway_orphans() once at
    startup under HERMES_DESKTOP=1 so the stale orphan is cleared first.
    """
    import hermes_cli.web_server as ws

    called = []

    def _fake_reap():
        called.append(True)
        return True

    monkeypatch.setenv("HERMES_DESKTOP", "1")
    # Keep the lifespan cheap: don't re-import the gateway module or spin up the
    # real cron scheduler thread.
    monkeypatch.setattr(ws, "_warm_gateway_module", lambda: None)
    monkeypatch.setattr(ws, "_start_desktop_cron_ticker", lambda *_args: None)
    # web_server imports the reaper lazily from hermes_cli.gateway, so patch it
    # on that module.
    import hermes_cli.gateway as g

    monkeypatch.setattr(g, "_reap_unsupervised_gateway_orphans", _fake_reap)

    client, _header = _client()
    with client:
        pass

    assert called == [True]


def test_desktop_lifespan_terminates_managed_gateway_restart(monkeypatch):
    """A Desktop-owned gateway child must not survive its serve backend."""
    import hermes_cli.web_server as ws

    calls = []

    class _FakeRunningProc:
        def poll(self):
            return None

        def terminate(self):
            calls.append("terminate")

    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.setattr(ws, "_warm_gateway_module", lambda: None)
    monkeypatch.setattr(ws, "_start_desktop_cron_ticker", lambda *_args: None)
    monkeypatch.setitem(ws._ACTION_PROCS, "gateway-restart", _FakeRunningProc())

    client, _header = _client()
    with client:
        pass

    assert calls == ["terminate"]

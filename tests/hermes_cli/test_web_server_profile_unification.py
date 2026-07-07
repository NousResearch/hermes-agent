"""Regression tests for the machine-dashboard multi-profile unification.

The dashboard is ONE machine-level management surface: config, env, MCP,
model, and chat-PTY endpoints accept an optional ``profile`` so the global
profile switcher can target any profile's HERMES_HOME. These tests pin:
reads/writes land in the REQUESTED profile, the dashboard's own profile
stays untouched, and the chat PTY env is scoped via HERMES_HOME.
"""
import pytest
import yaml


@pytest.fixture
def isolated_profiles(tmp_path, monkeypatch, _isolate_hermes_home):
    """Isolated default home + one named profile, each with config + .env."""
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles

    default_home = get_hermes_home()
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "worker_beta"
    for home in (default_home, worker_home):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    (worker_home / ".env").write_text("", encoding="utf-8")

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    return {"default": default_home, "worker_beta": worker_home}


@pytest.fixture
def client(monkeypatch, isolated_profiles):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _cfg(home):
    return yaml.safe_load((home / "config.yaml").read_text()) or {}


class TestProfileScopedConfig:
    def test_config_put_lands_in_target_profile_only(self, client, isolated_profiles):
        resp = client.put(
            "/api/config",
            json={"config": {"timezone": "Mars/Olympus"}, "profile": "worker_beta"},
        )
        assert resp.status_code == 200
        assert _cfg(isolated_profiles["worker_beta"]).get("timezone") == "Mars/Olympus"
        assert _cfg(isolated_profiles["default"]).get("timezone") != "Mars/Olympus"

    def test_config_get_reads_target_profile(self, client, isolated_profiles):
        (isolated_profiles["worker_beta"] / "config.yaml").write_text(
            "timezone: Venus/Cloud\n", encoding="utf-8"
        )
        resp = client.get("/api/config", params={"profile": "worker_beta"})
        assert resp.status_code == 200
        assert resp.json().get("timezone") == "Venus/Cloud"
        # Unscoped read sees the dashboard's own config.
        resp = client.get("/api/config")
        assert resp.json().get("timezone") != "Venus/Cloud"

    def test_config_query_param_equivalent_to_body(self, client, isolated_profiles):
        """The SPA's fetchJSON injects ?profile= — must scope like body.profile."""
        resp = client.put(
            "/api/config?profile=worker_beta",
            json={"config": {"timezone": "Pluto/Far"}},
        )
        assert resp.status_code == 200
        assert _cfg(isolated_profiles["worker_beta"]).get("timezone") == "Pluto/Far"
        assert _cfg(isolated_profiles["default"]).get("timezone") != "Pluto/Far"

    def test_config_raw_round_trip_scoped(self, client, isolated_profiles):
        resp = client.put(
            "/api/config/raw",
            json={"yaml_text": "timezone: Io/Volcano\n", "profile": "worker_beta"},
        )
        assert resp.status_code == 200
        resp = client.get("/api/config/raw", params={"profile": "worker_beta"})
        assert "Io/Volcano" in resp.json()["yaml"]
        resp = client.get("/api/config/raw")
        assert "Io/Volcano" not in resp.json()["yaml"]

    def test_config_raw_path_reflects_requested_profile(self, client, isolated_profiles):
        """The Config page header shows /api/config/raw's ``path`` — it must
        point at the SWITCHED profile's config.yaml, not the dashboard's own
        (the stale-path bug reported after the profile unification launch)."""
        resp = client.get("/api/config/raw", params={"profile": "worker_beta"})
        assert resp.status_code == 200
        assert resp.json()["path"] == str(isolated_profiles["worker_beta"] / "config.yaml")
        resp = client.get("/api/config/raw")
        assert resp.json()["path"] == str(isolated_profiles["default"] / "config.yaml")

    def test_unknown_profile_404(self, client, isolated_profiles):
        resp = client.get("/api/config", params={"profile": "ghost"})
        assert resp.status_code == 404


class TestProfileScopedEnv:
    def test_env_set_lands_in_target_profile_only(self, client, isolated_profiles):
        resp = client.put(
            "/api/env",
            json={"key": "FAL_KEY", "value": "test-fal-123", "profile": "worker_beta"},
        )
        assert resp.status_code == 200
        worker_env = (isolated_profiles["worker_beta"] / ".env").read_text()
        assert "test-fal-123" in worker_env
        default_env_path = isolated_profiles["default"] / ".env"
        if default_env_path.exists():
            assert "test-fal-123" not in default_env_path.read_text()

    def test_env_list_reads_target_profile(self, client, isolated_profiles):
        (isolated_profiles["worker_beta"] / ".env").write_text(
            "FAL_KEY=worker-only-value\n", encoding="utf-8"
        )
        resp = client.get("/api/env", params={"profile": "worker_beta"})
        assert resp.status_code == 200
        assert resp.json()["FAL_KEY"]["is_set"] is True
        resp = client.get("/api/env")
        assert resp.json()["FAL_KEY"]["is_set"] is False

    def test_env_delete_scoped(self, client, isolated_profiles):
        (isolated_profiles["worker_beta"] / ".env").write_text(
            "FAL_KEY=doomed\n", encoding="utf-8"
        )
        resp = client.request(
            "DELETE",
            "/api/env",
            json={"key": "FAL_KEY", "profile": "worker_beta"},
        )
        assert resp.status_code == 200
        assert "doomed" not in (isolated_profiles["worker_beta"] / ".env").read_text()


class TestProfileScopedMcp:
    def test_mcp_add_and_list_scoped(self, client, isolated_profiles):
        resp = client.post(
            "/api/mcp/servers",
            json={"name": "scoped-srv", "url": "http://localhost:1234/sse",
                  "profile": "worker_beta"},
        )
        assert resp.status_code == 200

        worker_cfg = _cfg(isolated_profiles["worker_beta"])
        assert "scoped-srv" in worker_cfg.get("mcp_servers", {})
        assert "scoped-srv" not in _cfg(isolated_profiles["default"]).get("mcp_servers", {})

        listing = client.get("/api/mcp/servers", params={"profile": "worker_beta"}).json()
        assert any(s["name"] == "scoped-srv" for s in listing["servers"])
        listing = client.get("/api/mcp/servers").json()
        assert not any(s["name"] == "scoped-srv" for s in listing["servers"])

    def test_mcp_enabled_toggle_scoped(self, client, isolated_profiles):
        (isolated_profiles["worker_beta"] / "config.yaml").write_text(
            "mcp_servers:\n  srv1:\n    url: http://x/sse\n", encoding="utf-8"
        )
        resp = client.put(
            "/api/mcp/servers/srv1/enabled",
            json={"enabled": False, "profile": "worker_beta"},
        )
        assert resp.status_code == 200
        worker_cfg = _cfg(isolated_profiles["worker_beta"])
        assert worker_cfg["mcp_servers"]["srv1"]["enabled"] is False

    def test_mcp_probe_runs_inside_profile_scope(
        self, client, isolated_profiles, monkeypatch
    ):
        """The test-server probe must execute with the selected profile's
        scope active so env-placeholder expansion reads the profile's .env,
        matching the config the server was saved into."""
        import hermes_cli.mcp_config as mcp_config
        from hermes_constants import get_hermes_home

        (isolated_profiles["worker_beta"] / "config.yaml").write_text(
            "mcp_servers:\n  probe-srv:\n    url: http://x/sse\n",
            encoding="utf-8",
        )
        seen = {}

        def fake_probe(name, config, connect_timeout=30, details=None):
            seen["home"] = str(get_hermes_home())
            return [("tool-a", "desc")]

        monkeypatch.setattr(mcp_config, "_probe_single_server", fake_probe)
        resp = client.post(
            "/api/mcp/servers/probe-srv/test", params={"profile": "worker_beta"}
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert seen["home"] == str(isolated_profiles["worker_beta"])

    def test_mcp_test_oauth_server_without_token_is_not_ok(
        self, client, isolated_profiles, monkeypatch
    ):
        """An `auth: oauth` server that serves tools/list anonymously must not
        false-green: a successful probe with no token on disk reports needs-auth."""
        import hermes_cli.mcp_config as mcp_config

        (isolated_profiles["worker_beta"] / "config.yaml").write_text(
            "mcp_servers:\n  oauth-srv:\n    url: http://x/sse\n    auth: oauth\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            mcp_config,
            "_probe_single_server",
            lambda name, config, connect_timeout=30, details=None: [("tool-a", "desc")],
        )
        monkeypatch.setattr(mcp_config, "_oauth_tokens_present", lambda name: False)

        resp = client.post(
            "/api/mcp/servers/oauth-srv/test", params={"profile": "worker_beta"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is False
        assert "oauth" in body["error"].lower()

        # With a token present, the same probe is genuinely authenticated.
        monkeypatch.setattr(mcp_config, "_oauth_tokens_present", lambda name: True)
        resp = client.post(
            "/api/mcp/servers/oauth-srv/test", params={"profile": "worker_beta"}
        )
        assert resp.json()["ok"] is True

    def test_mcp_remove_scoped(self, client, isolated_profiles):
        (isolated_profiles["worker_beta"] / "config.yaml").write_text(
            "mcp_servers:\n  srv2:\n    url: http://x/sse\n", encoding="utf-8"
        )
        # Removing from the DASHBOARD's profile must 404 (srv2 lives in worker).
        resp = client.delete("/api/mcp/servers/srv2")
        assert resp.status_code == 404
        resp = client.delete("/api/mcp/servers/srv2", params={"profile": "worker_beta"})
        assert resp.status_code == 200
        assert "srv2" not in _cfg(isolated_profiles["worker_beta"]).get("mcp_servers", {})


class TestProfileScopedModel:
    def test_model_set_main_scoped(self, client, isolated_profiles):
        resp = client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "openrouter",
                "model": "test/model-1",
                "confirm_expensive_model": True,
                "profile": "worker_beta",
            },
        )
        assert resp.status_code == 200
        worker_cfg = _cfg(isolated_profiles["worker_beta"])
        model_cfg = worker_cfg.get("model", {})
        assert isinstance(model_cfg, dict)
        assert model_cfg.get("provider") == "openrouter"
        default_model = _cfg(isolated_profiles["default"]).get("model", {})
        if isinstance(default_model, dict):
            assert default_model.get("default") != "test/model-1"

    def test_auxiliary_read_scoped_matches_write_target(
        self, client, isolated_profiles
    ):
        """Reads and writes must scope symmetrically: an aux pin written to
        the worker profile must show up ONLY in the worker-scoped read.
        (Regression: /api/model/auxiliary used to read unscoped while
        /api/model/set wrote scoped — the Models page displayed the
        dashboard profile's pins while editing the selected profile's.)"""
        (isolated_profiles["worker_beta"] / "config.yaml").write_text(
            "auxiliary:\n  vision:\n    provider: openrouter\n"
            "    model: worker/vision-pin\n",
            encoding="utf-8",
        )
        resp = client.get("/api/model/auxiliary", params={"profile": "worker_beta"})
        assert resp.status_code == 200
        vision = next(t for t in resp.json()["tasks"] if t["task"] == "vision")
        assert vision["model"] == "worker/vision-pin"

        # Unscoped read = the dashboard's own profile, which has no pin.
        resp = client.get("/api/model/auxiliary")
        assert resp.status_code == 200
        vision = next(t for t in resp.json()["tasks"] if t["task"] == "vision")
        assert vision["model"] != "worker/vision-pin"

    def test_auxiliary_unknown_profile_404(self, client, isolated_profiles):
        resp = client.get("/api/model/auxiliary", params={"profile": "ghost"})
        assert resp.status_code == 404

    def test_model_options_scoped_to_profile(self, client, isolated_profiles):
        """The Models picker must read the SAME profile model/set writes —
        current model/provider in the payload come from the scoped config."""
        (isolated_profiles["worker_beta"] / "config.yaml").write_text(
            "model:\n  provider: openrouter\n  default: worker/current-pin\n",
            encoding="utf-8",
        )
        resp = client.get("/api/model/options", params={"profile": "worker_beta"})
        assert resp.status_code == 200
        body = resp.json()
        # The payload carries the current selection somewhere stable; assert
        # the worker pin appears in the scoped response and not the unscoped.
        assert "worker/current-pin" in resp.text
        resp = client.get("/api/model/options")
        assert resp.status_code == 200
        assert "worker/current-pin" not in resp.text
        assert isinstance(body, dict)

    def test_model_options_unknown_profile_404(self, client, isolated_profiles):
        resp = client.get("/api/model/options", params={"profile": "ghost"})
        assert resp.status_code == 404

    def test_model_info_unknown_profile_404(self, client, isolated_profiles):
        """Regression: the broad except used to convert the 404 into a 200
        with empty model info ("no model set" — silently wrong)."""
        resp = client.get("/api/model/info", params={"profile": "ghost"})
        assert resp.status_code == 404

    def test_mcp_catalog_unknown_profile_404(self, client, isolated_profiles):
        resp = client.get("/api/mcp/catalog", params={"profile": "ghost"})
        assert resp.status_code == 404


class TestProfileScopedPostSetup:
    def test_post_setup_spawns_with_profile_flag(
        self, client, isolated_profiles, monkeypatch
    ):
        """Post-setup runs in a -p scoped subprocess so hooks that read
        config / write per-profile state see the same HERMES_HOME the rest
        of the drawer's writes targeted."""
        import hermes_cli.web_server as web_server

        calls = []

        class _FakeProc:
            pid = 777

        monkeypatch.setattr(
            web_server,
            "_spawn_hermes_action",
            lambda subcommand, name: calls.append(list(subcommand)) or _FakeProc(),
        )
        monkeypatch.setattr(
            "hermes_cli.tools_config.valid_post_setup_keys",
            lambda: {"agent_browser"},
        )
        resp = client.post(
            "/api/tools/toolsets/browser/post-setup",
            json={"key": "agent_browser", "profile": "worker_beta"},
        )
        assert resp.status_code == 200
        assert calls == [
            ["-p", "worker_beta", "tools", "post-setup", "agent_browser"]
        ]

    def test_post_setup_without_profile_keeps_legacy_argv(
        self, client, isolated_profiles, monkeypatch
    ):
        import hermes_cli.web_server as web_server

        calls = []

        class _FakeProc:
            pid = 777

        monkeypatch.setattr(
            web_server,
            "_spawn_hermes_action",
            lambda subcommand, name: calls.append(list(subcommand)) or _FakeProc(),
        )
        monkeypatch.setattr(
            "hermes_cli.tools_config.valid_post_setup_keys",
            lambda: {"agent_browser"},
        )
        resp = client.post(
            "/api/tools/toolsets/browser/post-setup",
            json={"key": "agent_browser"},
        )
        assert resp.status_code == 200
        assert calls == [["tools", "post-setup", "agent_browser"]]


class TestProfileScopedGateway:
    def test_lifecycle_spawns_with_profile_flag(
        self, client, isolated_profiles, monkeypatch
    ):
        import hermes_cli.web_server as web_server

        calls = []

        class _FakeProc:
            pid = 888

        monkeypatch.setattr(
            web_server,
            "_spawn_hermes_action",
            lambda subcommand, name: calls.append((list(subcommand), name)) or _FakeProc(),
        )
        web_server._ACTION_PROCS.pop("gateway-restart", None)
        web_server._ACTION_COMMANDS.pop("gateway-restart", None)

        for verb in ("start", "stop", "restart"):
            resp = client.post(f"/api/gateway/{verb}", params={"profile": "worker_beta"})
            assert resp.status_code == 200

        assert calls == [
            (["-p", "worker_beta", "gateway", "start"], "gateway-start"),
            (["-p", "worker_beta", "gateway", "stop"], "gateway-stop"),
            (["-p", "worker_beta", "gateway", "restart"], "gateway-restart"),
        ]

    def test_status_reads_requested_profile_home(
        self, client, isolated_profiles, monkeypatch
    ):
        import hermes_cli.web_server as web_server
        from hermes_constants import get_hermes_home

        seen_homes = []

        def fake_get_running_pid():
            seen_homes.append(str(get_hermes_home()))
            return None

        monkeypatch.setattr(web_server, "check_config_version", lambda: (1, 1))
        monkeypatch.setattr(web_server, "get_running_pid", fake_get_running_pid)
        monkeypatch.setattr(
            web_server,
            "read_runtime_status",
            lambda: {"gateway_state": "startup_failed", "platforms": {}},
        )
        monkeypatch.setattr(web_server, "_GATEWAY_HEALTH_URL", None)

        resp = client.get("/api/status", params={"profile": "worker_beta"})

        assert resp.status_code == 200
        assert seen_homes[0] == str(isolated_profiles["worker_beta"])
        assert resp.json()["hermes_home"] == str(isolated_profiles["worker_beta"])

    def test_status_uses_runtime_pid_when_profile_pid_file_is_missing(
        self, client, isolated_profiles, monkeypatch
    ):
        import hermes_cli.web_server as web_server

        worker_home = isolated_profiles["worker_beta"]
        (worker_home / ".env").write_text(
            "TELEGRAM_BOT_TOKEN=worker-token\n", encoding="utf-8"
        )
        (worker_home / "config.yaml").write_text(
            yaml.safe_dump({"platforms": {"telegram": {"enabled": True}}}),
            encoding="utf-8",
        )
        runtime = {
            "pid": 4242,
            "gateway_state": "running",
            "platforms": {"telegram": {"state": "connected"}},
            "exit_reason": None,
            "updated_at": "2026-06-17T00:00:00+00:00",
        }
        monkeypatch.setattr(web_server, "check_config_version", lambda: (1, 1))
        monkeypatch.setattr(web_server, "get_running_pid", lambda: None)
        monkeypatch.setattr(web_server, "read_runtime_status", lambda: runtime)
        monkeypatch.setattr(
            web_server, "get_runtime_status_running_pid", lambda payload: 4242
        )
        monkeypatch.setattr(web_server, "_GATEWAY_HEALTH_URL", None)
        from gateway.config import Platform

        class _FakeGatewayConfig:
            def get_connected_platforms(self):
                return [Platform.TELEGRAM]

        monkeypatch.setattr(
            "gateway.config.load_gateway_config", lambda: _FakeGatewayConfig()
        )

        resp = client.get("/api/status", params={"profile": "worker_beta"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["gateway_running"] is True
        assert data["gateway_pid"] == 4242
        assert data["gateway_state"] == "running"
        assert data["gateway_platforms"] == {"telegram": {"state": "connected"}}


class TestProfileScopedTelegramOnboarding:
    def test_apply_writes_target_profile_and_restarts_target(
        self, client, isolated_profiles, monkeypatch
    ):
        import time
        import hermes_cli.web_server as web_server

        with web_server._telegram_onboarding_lock:
            web_server._telegram_onboarding_pairings.clear()
            web_server._telegram_onboarding_pairings["pair-worker"] = (
                web_server._TelegramOnboardingPairing(
                    poll_token="poll-secret",
                    expires_at="2027-05-18T00:00:00.000Z",
                    expires_at_ts=time.time() + 600,
                    bot_token="123456:SECRET",
                    bot_username="worker_bot",
                    owner_user_id="123456789",
                )
            )

        calls = []

        class _FakeProc:
            pid = 889

        monkeypatch.setattr(
            web_server,
            "_spawn_hermes_action",
            lambda subcommand, name: calls.append((list(subcommand), name)) or _FakeProc(),
        )
        web_server._ACTION_PROCS.pop("gateway-restart", None)
        web_server._ACTION_COMMANDS.pop("gateway-restart", None)

        resp = client.post(
            "/api/messaging/telegram/onboarding/pair-worker/apply",
            params={"profile": "worker_beta"},
            json={"allowed_user_ids": ["123456789"]},
        )

        assert resp.status_code == 200
        assert resp.json()["restart_started"] is True
        assert calls == [
            (["-p", "worker_beta", "gateway", "restart"], "gateway-restart")
        ]

        worker_env = (isolated_profiles["worker_beta"] / ".env").read_text()
        assert "TELEGRAM_BOT_TOKEN=123456:SECRET" in worker_env
        assert "TELEGRAM_ALLOWED_USERS=123456789" in worker_env
        default_env_path = isolated_profiles["default"] / ".env"
        if default_env_path.exists():
            assert "TELEGRAM_BOT_TOKEN" not in default_env_path.read_text()

        worker_cfg = _cfg(isolated_profiles["worker_beta"])
        default_cfg = _cfg(isolated_profiles["default"])
        assert worker_cfg["platforms"]["telegram"]["enabled"] is True
        assert default_cfg.get("platforms", {}).get("telegram", {}).get("enabled") is not True


class TestProfileScopedChatPty:
    def test_chat_argv_scopes_hermes_home(self, isolated_profiles, monkeypatch):
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(
            "hermes_cli.main._make_tui_argv",
            lambda root, tui_dev=False: (["cat"], None),
            raising=False,
        )
        argv, cwd, env = web_server._resolve_chat_argv(profile="worker_beta")
        assert env is not None
        assert env["HERMES_HOME"] == str(isolated_profiles["worker_beta"])
        # Scoped chat must NOT attach to the dashboard's in-memory gateway.
        assert "HERMES_TUI_GATEWAY_URL" not in env

    def test_chat_argv_unscoped_keeps_legacy_env(self, isolated_profiles, monkeypatch):
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(
            "hermes_cli.main._make_tui_argv",
            lambda root, tui_dev=False: (["cat"], None),
            raising=False,
        )
        argv, cwd, env = web_server._resolve_chat_argv()
        assert env is not None
        assert env.get("HERMES_HOME") != str(isolated_profiles["worker_beta"])

    def test_chat_argv_unknown_profile_raises(self, isolated_profiles, monkeypatch):
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(
            "hermes_cli.main._make_tui_argv",
            lambda root, tui_dev=False: (["cat"], None),
            raising=False,
        )
        # Reuse the HTTPException class web_server itself raises — avoids a
        # direct fastapi import (unresolvable in the ty lint environment).
        with pytest.raises(web_server.HTTPException) as exc:
            web_server._resolve_chat_argv(profile="ghost")
        assert exc.value.status_code == 404


class TestProfileScopedChatResume:
    """Resuming a chat must resolve the session in its OWNING profile, not the
    dashboard's sticky-active/management profile.

    Regression for cross-profile "session not found": the dashboard stores
    sessions in per-profile state.db files, but the resume path (latest
    descendant + chat PTY) used the process/active profile only, so reopening a
    chat that belongs to a different profile than the one currently selected —
    including a bare desktop-restored ``/chat?resume=<id>`` URL carrying no
    ``?profile=`` — 404'd. See ``_find_session_profile`` /
    ``_session_latest_descendant`` / ``_resolve_chat_argv``.
    """

    @staticmethod
    def _seed(home, sid, *, parent=None):
        import hermes_state

        db = hermes_state.SessionDB(db_path=home / "state.db")
        try:
            kw = {"model": "test"}
            if parent is not None:
                kw["parent_session_id"] = parent
            db.create_session(session_id=sid, source="tui", **kw)
        finally:
            db.close()

    @staticmethod
    def _patch_tui(monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.main._make_tui_argv",
            lambda root, tui_dev=False: (["cat"], None),
            raising=False,
        )

    def test_find_session_profile_locates_owner(self, isolated_profiles):
        import hermes_cli.web_server as web_server

        self._seed(isolated_profiles["worker_beta"], "wb-sess")
        self._seed(isolated_profiles["default"], "def-sess")
        assert web_server._find_session_profile("wb-sess") == "worker_beta"
        assert web_server._find_session_profile("def-sess") == "default"
        assert web_server._find_session_profile("nope-zzz") is None

    def test_latest_descendant_walks_owning_profile_chain(
        self, isolated_profiles, monkeypatch
    ):
        import hermes_state
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(
            hermes_state, "DEFAULT_DB_PATH", isolated_profiles["default"] / "state.db"
        )
        # parent -> child compression chain lives ONLY in worker_beta
        self._seed(isolated_profiles["worker_beta"], "p")
        self._seed(isolated_profiles["worker_beta"], "c", parent="p")

        # explicit owning profile walks to the tip
        assert web_server._session_latest_descendant("p", profile="worker_beta")[0] == "c"
        # no profile -> cross-profile fallback still resolves it (was None before)
        assert web_server._session_latest_descendant("p")[0] == "c"
        # a wrong/sticky profile -> fallback corrects to the real owner
        assert web_server._session_latest_descendant("p", profile="default")[0] == "c"
        # genuinely unknown id stays clean
        assert web_server._session_latest_descendant("ghost-zzz") == (None, [])

    def test_latest_descendant_endpoint_honors_and_falls_back(
        self, client, isolated_profiles
    ):
        self._seed(isolated_profiles["worker_beta"], "wb1")
        # explicit profile
        r = client.get(
            "/api/sessions/wb1/latest-descendant", params={"profile": "worker_beta"}
        )
        assert r.status_code == 200 and r.json()["session_id"] == "wb1"
        # no profile -> fallback resolves the owner (regression: used to 404)
        r = client.get("/api/sessions/wb1/latest-descendant")
        assert r.status_code == 200 and r.json()["session_id"] == "wb1"
        # genuinely missing -> 404
        assert client.get("/api/sessions/ghost-zzz/latest-descendant").status_code == 404

    def test_chat_argv_resume_binds_owning_profile(
        self, isolated_profiles, monkeypatch
    ):
        import hermes_state
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(
            hermes_state, "DEFAULT_DB_PATH", isolated_profiles["default"] / "state.db"
        )
        self._patch_tui(monkeypatch)
        # worker_beta-owned session whose compression tip is "wc"
        self._seed(isolated_profiles["worker_beta"], "wp")
        self._seed(isolated_profiles["worker_beta"], "wc", parent="wp")

        # The reported bug: resume a worker_beta session while the switcher is on
        # the default/empty scope. Must bind worker_beta and walk to the tip.
        _, _, env = web_server._resolve_chat_argv(resume="wp", profile="")
        assert env["HERMES_HOME"] == str(isolated_profiles["worker_beta"])
        assert env["HERMES_TUI_RESUME"] == "wc"
        # Named owner spawns its own gateway (no in-memory attach).
        assert "HERMES_TUI_GATEWAY_URL" not in env

        # Correct explicit hint also binds worker_beta.
        _, _, env = web_server._resolve_chat_argv(resume="wp", profile="worker_beta")
        assert env["HERMES_HOME"] == str(isolated_profiles["worker_beta"])
        assert env["HERMES_TUI_RESUME"] == "wc"

    def test_chat_argv_resume_default_owned_stays_default(
        self, isolated_profiles, monkeypatch
    ):
        import hermes_state
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(
            hermes_state, "DEFAULT_DB_PATH", isolated_profiles["default"] / "state.db"
        )
        self._patch_tui(monkeypatch)
        self._seed(isolated_profiles["default"], "dp")

        # Even when the switcher points at worker_beta, a default-owned resume
        # must bind the default home (the wrong hint is corrected to the owner).
        _, _, env = web_server._resolve_chat_argv(resume="dp", profile="worker_beta")
        assert env["HERMES_HOME"] == str(isolated_profiles["default"])
        assert env["HERMES_TUI_RESUME"] == "dp"

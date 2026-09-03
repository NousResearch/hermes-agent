"""Regression tests for the machine-dashboard multi-profile unification.

The dashboard is ONE machine-level management surface: config, env, MCP,
model, and chat-PTY endpoints accept an optional ``profile`` so the global
profile switcher can target any profile's HERMES_HOME. These tests pin:
reads/writes land in the REQUESTED profile, the dashboard's own profile
stays untouched, and the chat PTY env is scoped via HERMES_HOME.
"""
import json
from contextlib import contextmanager

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
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}


def _write_jobs(home, jobs):
    cron_dir = home / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    (cron_dir / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")


class TestProfileScopedConfig:


    def test_config_query_param_equivalent_to_body(self, client, isolated_profiles):
        """The SPA's fetchJSON injects ?profile= — must scope like body.profile."""
        resp = client.put(
            "/api/config?profile=worker_beta",
            json={"config": {"timezone": "Pluto/Far"}},
        )
        assert resp.status_code == 200
        assert _cfg(isolated_profiles["worker_beta"]).get("timezone") == "Pluto/Far"
        assert _cfg(isolated_profiles["default"]).get("timezone") != "Pluto/Far"



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
        worker_env = (isolated_profiles["worker_beta"] / ".env").read_text(encoding="utf-8")
        assert "test-fal-123" in worker_env
        default_env_path = isolated_profiles["default"] / ".env"
        if default_env_path.exists():
            assert "test-fal-123" not in default_env_path.read_text(encoding="utf-8")


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
        assert "doomed" not in (isolated_profiles["worker_beta"] / ".env").read_text(
            encoding="utf-8"
        )


class TestProfileScopedWebhooks:
    def test_webhook_actions_stay_in_requested_profile(
        self, client, isolated_profiles, monkeypatch
    ):
        import hermes_cli.web_server as ws

        restart_profiles = []
        monkeypatch.setattr(
            ws,
            "_restart_gateway_after_webhook_enable",
            lambda profile: restart_profiles.append(profile) or {"restart_started": True},
        )

        params = {"profile": "worker_beta"}
        assert client.post("/api/webhooks/enable", params=params).status_code == 200
        assert restart_profiles == ["worker_beta"]
        assert _cfg(isolated_profiles["worker_beta"])["platforms"]["webhook"]["enabled"] is True
        assert "platforms" not in _cfg(isolated_profiles["default"])

        assert client.post(
            "/api/webhooks", params=params, json={"name": "worker-hook", "deliver": "log"}
        ).status_code == 200
        assert client.get("/api/webhooks", params=params).json()["subscriptions"][0]["name"] == "worker-hook"
        assert not (isolated_profiles["default"] / "webhook_subscriptions.json").exists()

        assert client.put(
            "/api/webhooks/worker-hook/enabled", params=params, json={"enabled": False}
        ).json()["enabled"] is False
        assert client.delete("/api/webhooks/worker-hook", params=params).status_code == 200
        assert client.get("/api/webhooks", params=params).json()["subscriptions"] == []


class TestProfileScopedMcp:

    def test_mcp_bearer_secret_is_profile_scoped(self, client, isolated_profiles):
        secret = "worker-only-secret"
        response = client.post(
            "/api/mcp/servers",
            params={"profile": "worker_beta"},
            json={
                "name": "profile-bearer",
                "url": "https://example.com/mcp",
                "auth": "header",
                "bearer_token": secret,
            },
        )

        assert response.status_code == 200
        worker_cfg = _cfg(isolated_profiles["worker_beta"])
        assert worker_cfg["mcp_servers"]["profile-bearer"]["headers"] == {
            "Authorization": "Bearer ${MCP_PROFILE_BEARER_API_KEY}",
        }
        assert secret in (isolated_profiles["worker_beta"] / ".env").read_text(encoding="utf-8")
        assert not (isolated_profiles["default"] / ".env").exists()
        assert "profile-bearer" not in _cfg(isolated_profiles["default"]).get(
            "mcp_servers", {}
        )



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

    def test_mcp_test_reports_optional_schema_chars(
        self, client, isolated_profiles, monkeypatch
    ):
        """The probe's per-tool `schema_chars` (details out-param) surfaces as an
        ADDITIVE per-tool field on the wire; tools without a size stay bare so
        older/partial probes degrade to 'no estimate' in the renderer."""
        import hermes_cli.mcp_config as mcp_config

        (isolated_profiles["worker_beta"] / "config.yaml").write_text(
            "mcp_servers:\n  sized-srv:\n    url: http://x/mcp\n",
            encoding="utf-8",
        )

        def fake_probe(name, config, connect_timeout=30, details=None):
            if details is not None:
                details["schema_chars"] = {"tool-a": 420}
            return [("tool-a", "desc-a"), ("tool-b", "desc-b")]

        monkeypatch.setattr(mcp_config, "_probe_single_server", fake_probe)

        resp = client.post(
            "/api/mcp/servers/sized-srv/test", params={"profile": "worker_beta"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        tools = {t["name"]: t for t in body["tools"]}
        assert tools["tool-a"]["schema_chars"] == 420
        # No size for tool-b → the key is simply absent (additive-optional).
        assert "schema_chars" not in tools["tool-b"]

    def test_mcp_test_without_schema_chars_keeps_old_wire_shape(
        self, client, isolated_profiles, monkeypatch
    ):
        """A probe that never fills schema_chars (older code path) produces the
        exact pre-overlay tool objects — nothing new for old renderers."""
        import hermes_cli.mcp_config as mcp_config

        (isolated_profiles["worker_beta"] / "config.yaml").write_text(
            "mcp_servers:\n  plain-srv:\n    url: http://x/mcp\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            mcp_config,
            "_probe_single_server",
            lambda name, config, connect_timeout=30, details=None: [("tool-a", "desc")],
        )

        resp = client.post(
            "/api/mcp/servers/plain-srv/test", params={"profile": "worker_beta"}
        )
        assert resp.status_code == 200
        assert resp.json()["tools"] == [{"name": "tool-a", "description": "desc"}]


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

    def test_main_assignment_reports_only_target_profile_cron_impact(
        self, client, isolated_profiles
    ):
        stale = {
            "name": "Worker summary",
            "enabled": True,
            "no_agent": False,
            "provider_snapshot": "openrouter",
            "model_snapshot": "old/model",
        }
        _write_jobs(
            isolated_profiles["worker_beta"], [{"id": "worker-job", **stale}]
        )
        _write_jobs(
            isolated_profiles["default"],
            [{"id": "default-job", **stale, "name": "Default summary"}],
        )

        resp = client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "nous",
                "model": "new/model",
                "confirm_expensive_model": True,
                "profile": "worker_beta",
            },
        )

        assert resp.status_code == 200
        assert resp.json()["cron_model_impact"] == {
            "available": True,
            "guard_enabled": True,
            "affected_count": 1,
            "truncated": False,
            "jobs": [
                {
                    "id": "worker-job",
                    "name": "Worker summary",
                    "drifted_axes": ["provider", "model"],
                }
            ],
        }

    def test_unavailable_impact_does_not_fail_persisted_assignment(
        self, client, isolated_profiles, monkeypatch
    ):
        import cron.jobs

        monkeypatch.setattr(cron.jobs, "load_jobs", lambda: {"malformed": True})

        resp = client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "nous",
                "model": "new/model",
                "confirm_expensive_model": True,
                "profile": "worker_beta",
            },
        )

        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert resp.json()["cron_model_impact"]["available"] is False
        assert _cfg(isolated_profiles["worker_beta"])["model"]["default"] == "new/model"

    def test_auxiliary_and_confirmation_responses_have_no_impact_summary(
        self, client, isolated_profiles
    ):
        _write_jobs(
            isolated_profiles["worker_beta"],
            [
                {
                    "id": "worker-job",
                    "enabled": True,
                    "provider_snapshot": "openrouter",
                    "model_snapshot": "old/model",
                }
            ],
        )

        auxiliary = client.post(
            "/api/model/set",
            json={
                "scope": "auxiliary",
                "provider": "nous",
                "model": "new/model",
                "profile": "worker_beta",
            },
        )
        confirmation = client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "openrouter",
                "model": "openai/gpt-5.5-pro",
                "profile": "worker_beta",
            },
        )

        assert auxiliary.status_code == 200
        assert "cron_model_impact" not in auxiliary.json()
        assert confirmation.status_code == 200
        assert confirmation.json()["confirm_required"] is True
        assert "cron_model_impact" not in confirmation.json()





    def test_model_options_uses_config_only_scope_for_selected_profile(
        self, client, monkeypatch
    ):
        """Regression (#58576): _profile_scope holds _SKILLS_PROFILE_LOCK
        across its body, and the payload build can block up to 15s on a
        models.dev cache miss — a cold request would starve concurrent
        /api/config on the same lock. The handler must scope the worker
        through _config_profile_scope (contextvar only, no lock) for the
        selected profile."""
        import hermes_cli.web_server as web_server

        scopes = []

        @contextmanager
        def _recording_config_scope(profile):
            scopes.append(("config", profile))
            yield object()

        @contextmanager
        def _recording_profile_scope(profile):
            scopes.append(("full", profile))
            yield object()

        monkeypatch.setattr(
            web_server, "_config_profile_scope", _recording_config_scope
        )
        monkeypatch.setattr(web_server, "_profile_scope", _recording_profile_scope)
        monkeypatch.setattr(
            "hermes_cli.inventory.load_picker_context", lambda: object()
        )
        monkeypatch.setattr(
            "hermes_cli.inventory.build_model_options_payload",
            lambda _ctx, **kwargs: {"providers": [], "model": "", "provider": ""},
        )

        resp = client.get("/api/model/options", params={"profile": "worker_beta"})
        assert resp.status_code == 200
        # Only the config-only scope may wrap the payload build; entering
        # _profile_scope would hold _SKILLS_PROFILE_LOCK across it (#58576).
        assert scopes == [("config", "worker_beta")]

    def test_model_info_unknown_profile_404(self, client, isolated_profiles):
        """Regression: the broad except used to convert the 404 into a 200
        with empty model info ("no model set" — silently wrong)."""
        resp = client.get("/api/model/info", params={"profile": "ghost"})
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

    def test_status_reads_requested_profile_home(
        self, client, isolated_profiles, monkeypatch
    ):
        import hermes_cli.web_server as web_server
        from hermes_constants import get_hermes_home

        seen_homes = []

        def fake_get_running_pid(*args, **kwargs):
            # /api/status?profile= now passes pid_path= explicitly (the TTL
            # cache would otherwise serve another profile's PID) — accept it.
            seen_homes.append(str(get_hermes_home()))
            return None

        monkeypatch.setattr(web_server, "check_config_version", lambda: (1, 1))
        # get_status probes via the TTL-cached wrapper (PR #53511 salvage);
        # patch the cached name so the fake still intercepts the probe.
        monkeypatch.setattr(web_server, "get_running_pid_cached", fake_get_running_pid)
        monkeypatch.setattr(
            web_server,
            "read_runtime_status",
            lambda *a, **k: {"gateway_state": "startup_failed", "platforms": {}},
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
        monkeypatch.setattr(
            web_server, "get_running_pid_cached", lambda *a, **k: None
        )
        monkeypatch.setattr(web_server, "read_runtime_status", lambda *a, **k: runtime)
        monkeypatch.setattr(
            web_server,
            "get_runtime_status_running_pid",
            lambda payload, **k: 4242,
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

    def test_status_keeps_fatal_platforms_on_startup_failed(
        self, client, isolated_profiles, monkeypatch
    ):
        """startup_failed keeps FATAL per-profile entries — they're the diagnosis.

        A multiplex gateway that dies at startup persists per-profile fatal
        entries (``alpha:telegram`` etc.). The dead-gateway platform clear must
        not erase them: exit_reason alone can't say which profile failed how.
        Non-fatal leftovers (e.g. a platform that connected before the crash)
        are still dropped — only fatals survive.
        """
        import hermes_cli.web_server as web_server

        runtime = {
            "pid": 4242,
            "gateway_state": "startup_failed",
            "platforms": {
                "telegram": {"state": "fatal", "error_code": "telegram_auth_error"},
                "alpha:telegram": {"state": "fatal", "error_code": "credential_collision"},
                "beta:discord": {"state": "connected"},
            },
            "exit_reason": "telegram: token rejected",
            "updated_at": "2026-06-17T00:00:00+00:00",
        }
        monkeypatch.setattr(web_server, "check_config_version", lambda: (1, 1))
        monkeypatch.setattr(
            web_server, "get_running_pid_cached", lambda *a, **k: None
        )
        monkeypatch.setattr(web_server, "read_runtime_status", lambda *a, **k: runtime)
        # Bare platform keys are checked against the configured set (fail
        # closed) — mirror a host that actually has telegram configured.
        monkeypatch.setattr(
            web_server, "_load_configured_gateway_platforms", lambda: {"telegram"}
        )
        monkeypatch.setattr(web_server, "_GATEWAY_HEALTH_URL", None)

        resp = client.get("/api/status", params={"profile": "worker_beta"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["gateway_running"] is False
        assert data["gateway_state"] == "startup_failed"
        assert data["gateway_exit_reason"] == "telegram: token rejected"
        # Fatal entries (root and namespaced) survive; the stale non-fatal is dropped.
        assert set(data["gateway_platforms"]) == {"telegram", "alpha:telegram"}
        assert data["gateway_platforms"]["alpha:telegram"]["error_code"] == "credential_collision"

    def test_status_clears_platforms_on_clean_stop(
        self, client, isolated_profiles, monkeypatch
    ):
        """A cleanly stopped gateway still reports no platforms (stale-noise rule)."""
        import hermes_cli.web_server as web_server

        runtime = {
            "pid": 4242,
            "gateway_state": "stopped",
            "platforms": {"telegram": {"state": "connected"}},
            "exit_reason": None,
            "updated_at": "2026-06-17T00:00:00+00:00",
        }
        monkeypatch.setattr(web_server, "check_config_version", lambda: (1, 1))
        monkeypatch.setattr(
            web_server, "get_running_pid_cached", lambda *a, **k: None
        )
        monkeypatch.setattr(web_server, "read_runtime_status", lambda *a, **k: runtime)
        monkeypatch.setattr(web_server, "_GATEWAY_HEALTH_URL", None)

        resp = client.get("/api/status", params={"profile": "worker_beta"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["gateway_state"] == "stopped"
        assert data["gateway_platforms"] == {}


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

        worker_env = (isolated_profiles["worker_beta"] / ".env").read_text(encoding="utf-8")
        assert "TELEGRAM_BOT_TOKEN=123456:SECRET" in worker_env
        assert "TELEGRAM_ALLOWED_USERS=123456789" in worker_env
        default_env_path = isolated_profiles["default"] / ".env"
        if default_env_path.exists():
            assert "TELEGRAM_BOT_TOKEN" not in default_env_path.read_text(encoding="utf-8")

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

    def test_chat_argv_bridges_selected_profile_terminal_config(
        self, isolated_profiles, monkeypatch
    ):
        import hermes_cli.web_server as web_server

        (isolated_profiles["default"] / "config.yaml").write_text(
            "terminal:\n"
            "  backend: docker\n"
            "  docker_image: launch-profile-image\n",
            encoding="utf-8",
        )
        (isolated_profiles["worker_beta"] / "config.yaml").write_text(
            "terminal:\n"
            "  backend: ssh\n"
            "  ssh_host: worker.example.test\n"
            "  cwd: '~'\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "launch-profile-image")
        monkeypatch.setenv("TERMINAL_SSH_USER", "operator-user")
        monkeypatch.setattr(
            "hermes_cli.main._make_tui_argv",
            lambda root, tui_dev=False: (["cat"], None),
            raising=False,
        )

        _argv, _cwd, env = web_server._resolve_chat_argv(profile="worker_beta")

        assert env is not None
        assert env["HERMES_HOME"] == str(isolated_profiles["worker_beta"])
        assert env["TERMINAL_ENV"] == "ssh"
        assert env["TERMINAL_SSH_HOST"] == "worker.example.test"
        assert env["TERMINAL_CWD"] == "~"
        assert env["TERMINAL_DOCKER_IMAGE"] != "launch-profile-image"
        assert env["TERMINAL_SSH_USER"] == "operator-user"

    def test_chat_argv_default_profile_preserves_exported_terminal_values(
        self, isolated_profiles, monkeypatch
    ):
        import hermes_cli.web_server as web_server

        (isolated_profiles["default"] / "config.yaml").write_text(
            "terminal:\n  backend: docker\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_SSH_USER", "operator-user")
        monkeypatch.setattr(
            "hermes_cli.main._make_tui_argv",
            lambda root, tui_dev=False: (["cat"], None),
            raising=False,
        )

        _argv, _cwd, env = web_server._resolve_chat_argv()

        assert env is not None
        assert env["TERMINAL_ENV"] == "docker"
        assert env["TERMINAL_SSH_USER"] == "operator-user"

    @pytest.mark.parametrize("placeholder", [".", "auto", "cwd"])
    def test_chat_argv_placeholder_cwd_preserves_exported_value(
        self, isolated_profiles, monkeypatch, placeholder
    ):
        import hermes_cli.web_server as web_server

        (isolated_profiles["default"] / "config.yaml").write_text(
            f"terminal:\n  backend: docker\n  cwd: {placeholder}\n",
            encoding="utf-8",
        )
        (isolated_profiles["worker_beta"] / "config.yaml").write_text(
            "terminal:\n  backend: ssh\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_CWD", "/operator/work")
        monkeypatch.setattr(
            "hermes_cli.main._make_tui_argv",
            lambda root, tui_dev=False: (["cat"], None),
            raising=False,
        )

        _argv, _cwd, env = web_server._resolve_chat_argv(profile="worker_beta")

        assert env is not None
        assert env["TERMINAL_ENV"] == "ssh"
        assert env["TERMINAL_CWD"] == "/operator/work"

    def test_chat_argv_warns_when_profile_terminal_bridge_fails(
        self, isolated_profiles, monkeypatch, caplog
    ):
        import logging

        import hermes_cli.config as config_mod
        import hermes_cli.web_server as web_server

        (isolated_profiles["default"] / "config.yaml").write_text(
            "terminal:\n  backend: docker\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setattr(
            "hermes_cli.main._make_tui_argv",
            lambda root, tui_dev=False: (["cat"], None),
            raising=False,
        )
        monkeypatch.setattr(
            config_mod,
            "apply_terminal_config_to_env",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bridge failed")),
        )

        with caplog.at_level(logging.WARNING, logger=web_server._log.name):
            _argv, _cwd, env = web_server._resolve_chat_argv(profile="worker_beta")

        assert env is not None
        assert env["HERMES_HOME"] == str(isolated_profiles["worker_beta"])
        assert "TERMINAL_ENV" not in env
        assert "Failed to apply terminal config bridge for dashboard chat" in caplog.text


class TestProfileScopedAudio:
    """Audio endpoints must honor ``profile`` like the rest of the dashboard.

    Historically /api/audio/transcribe|speak|elevenlabs/voices took no profile
    and always resolved the dashboard's own config/.env, so a non-default
    profile's TTS/STT settings were silently ignored (#53441 #45506 #66012
    #64057).
    """



    def test_transcribe_runs_inside_target_profile_home(
        self, client, isolated_profiles, monkeypatch
    ):
        import base64

        import tools.voice_mode as voice_mode

        seen = {}

        def _fake_transcribe(path):
            from hermes_constants import get_hermes_home

            seen["home"] = str(get_hermes_home())
            return {"success": True, "transcript": "hi", "provider": "fake"}

        monkeypatch.setattr(voice_mode, "transcribe_recording", _fake_transcribe)
        payload = base64.b64encode(b"\x00fakeaudio").decode("ascii")
        resp = client.post(
            "/api/audio/transcribe?profile=worker_beta",
            json={"data_url": f"data:audio/webm;base64,{payload}"},
        )
        assert resp.status_code == 200
        assert resp.json()["transcript"] == "hi"
        assert seen["home"] == str(isolated_profiles["worker_beta"])

    def test_audio_endpoints_unknown_profile_404(self, client, isolated_profiles):
        resp = client.get("/api/audio/elevenlabs/voices?profile=ghost")
        assert resp.status_code == 404
        resp = client.post("/api/audio/speak?profile=ghost", json={"text": "x"})
        assert resp.status_code == 404

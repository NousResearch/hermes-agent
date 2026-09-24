"""Regression tests for the machine-dashboard multi-profile unification.

The dashboard is ONE machine-level management surface: config, env, MCP,
model, and chat-PTY endpoints accept an optional ``profile`` so the global
profile switcher can target any profile's HERMES_HOME. These tests pin:
reads/writes land in the REQUESTED profile, the dashboard's own profile
stays untouched, and the chat PTY env is scoped via HERMES_HOME.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import hermes_yaml as yaml
import gateway.status as _gw_status
import hermes_cli.config as _cfg_mod
import hermes_cli.web_server_chat as _web_server_chat
import hermes_cli.web_server_gateway as _web_server_gateway
import hermes_cli.web_server_messaging as _web_server_messaging


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
        worker_env = (isolated_profiles["worker_beta"] / ".env").read_text()
        assert "test-fal-123" in worker_env
        default_env_path = isolated_profiles["default"] / ".env"
        if default_env_path.exists():
            assert "test-fal-123" not in default_env_path.read_text()


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
        assert secret in (isolated_profiles["worker_beta"] / ".env").read_text()
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


    def test_mcp_test_resolves_profile_secret_source_scope(
        self, client, isolated_profiles, monkeypatch
    ):
        """The probe's `${VAR}` interpolation must resolve from the REQUESTED
        profile's secret scope, not the dashboard process's os.environ: a
        secondary profile whose credential comes from an external secret source
        (Bitwarden/1Password) never has it in the shared process env, so the
        probe used to send the literal placeholder — or the default profile's
        value of the same name — and the server answered 400 (#109901)."""
        import hermes_cli.env_loader as env_loader
        import hermes_cli.mcp_config as mcp_config

        worker_home = isolated_profiles["worker_beta"]
        (worker_home / "config.yaml").write_text(
            "mcp_servers:\n  bw-srv:\n    url: http://x/mcp\n"
            "    headers:\n      Authorization: Bearer ${GITHUB_PERSONAL_ACCESS_TOKEN}\n",
            encoding="utf-8",
        )
        # The shared dashboard process carries the DEFAULT profile's value of the
        # same env name — the probe must not use it.
        monkeypatch.setenv("GITHUB_PERSONAL_ACCESS_TOKEN", "default-profile-token")

        def _worker_sources(hermes_home):
            if Path(hermes_home).resolve() == worker_home.resolve():
                return {"GITHUB_PERSONAL_ACCESS_TOKEN": "bw-worker-token"}
            return {}

        monkeypatch.setattr(env_loader, "get_secret_source_values", _worker_sources)

        resolved_headers = {}

        def fake_probe(name, config, connect_timeout=30, details=None):
            resolved = mcp_config._resolve_mcp_server_config(config)
            resolved_headers.update(resolved.get("headers", {}))
            return [("tool-a", "desc")]

        monkeypatch.setattr(mcp_config, "_probe_single_server", fake_probe)

        resp = client.post("/api/mcp/servers/bw-srv/test", params={"profile": "worker_beta"})

        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert resolved_headers["Authorization"] == "Bearer bw-worker-token"

    def test_mcp_list_expands_url_ref_from_profile_secret_scope(
        self, client, isolated_profiles, monkeypatch
    ):
        """Same class for the read endpoint: a ``${VAR}`` in a secondary profile's server
        ``url`` must expand from THAT profile's secret scope, never the dashboard process env."""
        import hermes_cli.env_loader as env_loader

        worker_home = isolated_profiles["worker_beta"]
        (worker_home / "config.yaml").write_text(
            "mcp_servers:\n  bw-srv:\n    url: ${MCP_GH_URL}\n", encoding="utf-8"
        )
        monkeypatch.setenv("MCP_GH_URL", "http://default-profile/mcp")
        monkeypatch.setattr(
            env_loader, "get_secret_source_values",
            lambda hermes_home: {"MCP_GH_URL": "http://worker/mcp"}
            if Path(hermes_home).resolve() == worker_home.resolve() else {},
        )

        resp = client.get("/api/mcp/servers", params={"profile": "worker_beta"})
        assert resp.status_code == 200
        assert [s["url"] for s in resp.json()["servers"]] == ["http://worker/mcp"]


class TestProfileScopedModel:
    @pytest.fixture(autouse=True)
    def _accept_any_model(self, monkeypatch):
        """These tests pin WHICH profile the write lands in, not catalog validation: the main
        slot now routes through ``switch_model`` (needs credentials + a listed model), so echo the
        request back as an accepted route."""
        from hermes_cli.model_switch import ModelSwitchResult

        def _switch(*, raw_input, explicit_provider, **_kw):
            return ModelSwitchResult(success=True, new_model=raw_input, target_provider=explicit_provider)

        monkeypatch.setattr("hermes_cli.model_switch.switch_model", _switch)

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

    def test_profile_create_validates_against_the_dashboard_home_and_writes_the_new_profile(
        self, client, isolated_profiles, monkeypatch
    ):
        """The create dialog's picker read THIS dashboard's catalog; the new profile is empty
        (no providers:, no .env), so validating there rejected every non-env provider and
        create silently returned model_set: false. Validation must see the dashboard home's
        config; the write must still land in the new profile only."""
        import hermes_cli.profiles as profiles_mod
        from hermes_constants import get_hermes_home

        monkeypatch.setattr(profiles_mod, "create_wrapper_script", lambda name: None)
        (isolated_profiles["default"] / "config.yaml").write_text(
            "providers:\n  mybox:\n    base_url: http://box:8000/v1\n    key_env: MYBOX_KEY\n", encoding="utf-8")
        seen: dict = {}

        def _switch(*, raw_input, explicit_provider, user_providers, **_kw):
            from hermes_cli.model_switch import ModelSwitchResult
            seen["user_providers"] = user_providers
            seen["home"] = get_hermes_home()
            if explicit_provider not in user_providers:
                return ModelSwitchResult(success=False, error_message=f"Unknown provider '{explicit_provider}'.")
            return ModelSwitchResult(success=True, new_model=raw_input, target_provider=explicit_provider)

        monkeypatch.setattr("hermes_cli.model_switch.switch_model", _switch)
        resp = client.post("/api/profiles", json={"name": "newbie", "provider": "mybox", "model": "qwen3"})
        assert resp.status_code == 200 and resp.json()["model_set"] is True
        assert "mybox" in seen["user_providers"] and seen["home"] == isolated_profiles["default"]
        new_home = isolated_profiles["default"] / "profiles" / "newbie"
        assert _cfg(new_home)["model"]["provider"] == "mybox" and _cfg(new_home)["model"]["default"] == "qwen3"
        assert "model" not in _cfg(isolated_profiles["default"])
        # A genuine rejection is reported with its reason, not a silent model_set: false.
        resp = client.post("/api/profiles", json={"name": "newbie2", "provider": "nobox", "model": "qwen3"})
        assert resp.status_code == 200 and resp.json()["model_set"] is False
        assert "Unknown provider 'nobox'" in resp.json()["model_error"]



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

        calls = []

        class _FakeProc:
            pid = 777

        monkeypatch.setattr(
            _web_server_gateway,
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

        monkeypatch.setattr(_cfg_mod, "check_config_version", lambda: (1, 1))
        # get_status probes via the TTL-cached wrapper (PR #53511 salvage);
        # patch the cached name so the fake still intercepts the probe.
        monkeypatch.setattr(_gw_status, "get_running_pid_cached", fake_get_running_pid)
        monkeypatch.setattr(
            _gw_status,
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
        monkeypatch.setattr(_cfg_mod, "check_config_version", lambda: (1, 1))
        monkeypatch.setattr(
            _gw_status, "get_running_pid_cached", lambda *a, **k: None
        )
        monkeypatch.setattr(_gw_status, "read_runtime_status", lambda *a, **k: runtime)
        monkeypatch.setattr(
            _gw_status,
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
            "desired_state": "running",
            "platforms": {
                "telegram": {"state": "fatal", "error_code": "telegram_auth_error"},
                "alpha:telegram": {"state": "fatal", "error_code": "credential_collision"},
                "beta:discord": {"state": "connected"},
            },
            "exit_reason": "telegram: token rejected",
            "updated_at": "2026-06-17T00:00:00+00:00",
        }
        monkeypatch.setattr(_cfg_mod, "check_config_version", lambda: (1, 1))
        monkeypatch.setattr(
            _gw_status, "get_running_pid_cached", lambda *a, **k: None
        )
        monkeypatch.setattr(_gw_status, "read_runtime_status", lambda *a, **k: runtime)
        # Bare platform keys are checked against the configured set (fail
        # closed) — mirror a host that actually has telegram configured.
        monkeypatch.setattr(
            _web_server_gateway, "_load_configured_gateway_platforms", lambda: {"telegram"}
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

    def test_status_hides_historical_startup_failure_after_operator_stop(
        self, client, isolated_profiles, monkeypatch
    ):
        """A durable stop intent takes precedence over an old startup failure."""
        import hermes_cli.web_server as web_server

        runtime = {
            "pid": 4242,
            "gateway_state": "startup_failed",
            "desired_state": "stopped",
            "platforms": {"telegram": {"state": "fatal"}},
            "exit_reason": "telegram: token rejected",
            "updated_at": "2026-06-17T00:00:00+00:00",
        }
        monkeypatch.setattr(_cfg_mod, "check_config_version", lambda: (1, 1))
        monkeypatch.setattr(
            _gw_status, "get_running_pid_cached", lambda *a, **k: None
        )
        monkeypatch.setattr(_gw_status, "read_runtime_status", lambda *a, **k: runtime)
        monkeypatch.setattr(web_server, "_GATEWAY_HEALTH_URL", None)

        resp = client.get("/api/status", params={"profile": "worker_beta"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["gateway_running"] is False
        assert data["gateway_state"] == "stopped"
        assert data["gateway_exit_reason"] is None
        assert data["gateway_platforms"] == {}

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
        monkeypatch.setattr(_cfg_mod, "check_config_version", lambda: (1, 1))
        monkeypatch.setattr(
            _gw_status, "get_running_pid_cached", lambda *a, **k: None
        )
        monkeypatch.setattr(_gw_status, "read_runtime_status", lambda *a, **k: runtime)
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

        with _web_server_messaging._telegram_onboarding_lock:
            _web_server_messaging._telegram_onboarding_pairings.clear()
            _web_server_messaging._telegram_onboarding_pairings["pair-worker"] = (
                _web_server_messaging._TelegramOnboardingPairing(
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
            _web_server_gateway,
            "_spawn_hermes_action",
            lambda subcommand, name: calls.append((list(subcommand), name)) or _FakeProc(),
        )
        _web_server_gateway._ACTION_PROCS.pop("gateway-restart", None)
        _web_server_gateway._ACTION_COMMANDS.pop("gateway-restart", None)

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

        monkeypatch.setattr(
            "hermes_cli.main_tui_launch._make_tui_argv",
            lambda root, tui_dev=False: (["cat"], None),
            raising=False,
        )
        argv, cwd, env = _web_server_chat._resolve_chat_argv(profile="worker_beta")
        assert env is not None
        assert env["HERMES_HOME"] == str(isolated_profiles["worker_beta"])
        # Scoped chat must NOT attach to the dashboard's in-memory gateway.
        assert "HERMES_TUI_GATEWAY_URL" not in env

    @pytest.mark.parametrize(
        ("target_env", "expected_overlap", "expected_target_only"),
        [
            (
                "FIRECRAWL_API_KEY=worker-overlap\n"
                "ANTHROPIC_API_KEY=worker-only\n"
                "AWS_ACCESS_KEY_ID=worker-aws-access\n"
                "AWS_SECRET_ACCESS_KEY=worker-aws-secret\n"
                "HERMES_CUSTOM_DEMO_API_KEY=worker-custom\n"
                "CUSTOM_BOX_API_KEY=worker-box\n"
                "HERMES_PROFILE=forged-profile\n",
                "worker-overlap",
                "worker-only",
            ),
            ("", None, None),
            (None, None, None),
        ],
        ids=("target-env", "empty-target-env", "missing-target-env"),
    )
    def test_chat_argv_child_gets_selected_profile_credentials_only(
        self,
        isolated_profiles,
        monkeypatch,
        target_env,
        expected_overlap,
        expected_target_only,
    ):
        launch_home = isolated_profiles["default"]
        worker_home = isolated_profiles["worker_beta"]
        (launch_home / ".env").write_text(
            "OPENAI_API_KEY=launch-only\n"
            "FIRECRAWL_API_KEY=launch-overlap\n",
            encoding="utf-8",
        )
        worker_env = worker_home / ".env"
        if target_env is None:
            worker_env.unlink()
        else:
            worker_env.write_text(target_env, encoding="utf-8")
        worker_config_path = worker_home / "config.yaml"
        worker_config = yaml.safe_load(worker_config_path.read_text(encoding="utf-8")) or {}
        worker_config["model"] = {
            "provider": "custom",
            "key_env": "CUSTOM_BOX_API_KEY",
        }
        worker_config_path.write_text(
            yaml.safe_dump(worker_config, sort_keys=False), encoding="utf-8"
        )
        launch_config_path = launch_home / "config.yaml"
        launch_config = yaml.safe_load(launch_config_path.read_text(encoding="utf-8")) or {}
        launch_config["model"] = {
            "provider": "custom",
            "key_env": "SOURCE_CUSTOM_TOKEN",
        }
        launch_config_path.write_text(
            yaml.safe_dump(launch_config, sort_keys=False), encoding="utf-8"
        )

        # Pin the explicit profile-HOME policy from #103651. The dashboard chat
        # must apply it after selecting the target profile, not before.
        (launch_home / "home").mkdir()
        (worker_home / "home").mkdir()
        monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
        monkeypatch.setenv("OPENAI_API_KEY", "launch-only")
        monkeypatch.setenv("FIRECRAWL_API_KEY", "launch-overlap")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "launch-aws-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "launch-aws-secret")
        monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/launch/.aws/credentials")
        monkeypatch.setenv("AWS_CONFIG_FILE", "/launch/.aws/config")
        monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "false")
        monkeypatch.setenv("AWS_CREDENTIAL_FILE", "/launch/.aws/legacy-credentials")
        monkeypatch.setenv("BOTO_CONFIG", "/launch/.boto")
        monkeypatch.setenv("AWS_SECURITY_TOKEN", "launch-legacy-token")
        monkeypatch.setenv("AWS_CREDENTIAL_EXPIRATION", "2099-01-01T00:00:00Z")
        monkeypatch.setenv("AWS_ACCOUNT_ID", "123456789012")
        monkeypatch.setenv("AWS_LOGIN_CACHE_DIRECTORY", "/launch/.aws/login/cache")
        monkeypatch.setenv("HERMES_CUSTOM_DEMO_API_KEY", "launch-custom")
        monkeypatch.setenv("CUSTOM_BOX_API_KEY", "launch-box")
        monkeypatch.setenv("SOURCE_CUSTOM_TOKEN", "launch-source-only")
        monkeypatch.setenv("HERMES_PROFILE", "default")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("UNRELATED_SETTING", "keep")
        monkeypatch.setattr(
            "hermes_cli.main_tui_launch._make_tui_argv",
            lambda root, tui_dev=False: (["cat"], None),
            raising=False,
        )

        _argv, _cwd, env = _web_server_chat._resolve_chat_argv(profile="worker_beta")
        probe = (
            "import json,os; print(json.dumps({k: os.environ.get(k) for k in "
            "('HERMES_HOME','HOME','OPENAI_API_KEY','FIRECRAWL_API_KEY',"
            "'ANTHROPIC_API_KEY','AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY',"
            "'AWS_SHARED_CREDENTIALS_FILE','AWS_CONFIG_FILE','AWS_EC2_METADATA_DISABLED',"
            "'AWS_CREDENTIAL_FILE','BOTO_CONFIG','AWS_SECURITY_TOKEN',"
            "'AWS_CREDENTIAL_EXPIRATION','AWS_ACCOUNT_ID',"
            "'AWS_LOGIN_CACHE_DIRECTORY',"
            "'HERMES_CUSTOM_DEMO_API_KEY','CUSTOM_BOX_API_KEY','HERMES_PROFILE',"
            "'SOURCE_CUSTOM_TOKEN',"
            "'UNRELATED_SETTING',"
            "'HERMES_TUI_GATEWAY_URL')}))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            env=env,
            capture_output=True,
            check=True,
            text=True,
            timeout=60,
        )
        seen = json.loads(result.stdout)

        assert seen == {
            "HERMES_HOME": str(worker_home),
            "HOME": str(worker_home / "home"),
            "OPENAI_API_KEY": None,
            "FIRECRAWL_API_KEY": expected_overlap,
            "ANTHROPIC_API_KEY": expected_target_only,
            "AWS_ACCESS_KEY_ID": "worker-aws-access" if target_env else None,
            "AWS_SECRET_ACCESS_KEY": "worker-aws-secret" if target_env else None,
            "AWS_SHARED_CREDENTIALS_FILE": str(worker_home / ".aws" / "credentials"),
            "AWS_CONFIG_FILE": str(worker_home / ".aws" / "config"),
            "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_CREDENTIAL_FILE": None,
            "BOTO_CONFIG": str(worker_home / ".boto"),
            "AWS_SECURITY_TOKEN": None,
            "AWS_CREDENTIAL_EXPIRATION": None,
            "AWS_ACCOUNT_ID": None,
            "AWS_LOGIN_CACHE_DIRECTORY": str(worker_home / ".aws" / "login" / "cache"),
            "HERMES_CUSTOM_DEMO_API_KEY": "worker-custom" if target_env else None,
            "CUSTOM_BOX_API_KEY": "worker-box" if target_env else None,
            "SOURCE_CUSTOM_TOKEN": None,
            "HERMES_PROFILE": "worker_beta",
            "UNRELATED_SETTING": "keep",
            "HERMES_TUI_GATEWAY_URL": None,
        }

    @pytest.mark.parametrize("target_dotenv_exists", [True, False])
    @pytest.mark.parametrize("explicit_launch_home", [True, False])
    def test_chat_argv_uses_frozen_launch_env_after_profile_activation(
        self, isolated_profiles, monkeypatch, target_dotenv_exists, explicit_launch_home
    ):
        from agent.secret_scope import set_multiplex_active
        from hermes_cli import profiles as profiles_mod
        from hermes_constants import get_default_hermes_root
        from tui_gateway import launch_profile_policy

        launch_home = isolated_profiles["default"]
        monkeypatch.setattr(
            profiles_mod, "_get_default_hermes_home", get_default_hermes_root
        )
        monkeypatch.setattr(
            profiles_mod,
            "_get_profiles_root",
            lambda: get_default_hermes_root() / "profiles",
        )
        worker_home = isolated_profiles["worker_beta"]
        (launch_home / "config.yaml").write_text(
            "terminal:\n  ssh_host: ${LAUNCH_HOST}\n", encoding="utf-8"
        )
        next_worker_home = launch_home / "profiles" / "worker_gamma"
        next_worker_home.mkdir(parents=True)
        (next_worker_home / "home").mkdir()
        (next_worker_home / "config.yaml").write_text(
            "terminal:\n  ssh_user: target-config-poison\n", encoding="utf-8"
        )
        late_root = launch_home / "late-root-poison"
        late_worker_home = late_root / "profiles" / "worker_gamma"
        late_worker_home.mkdir(parents=True)
        (late_worker_home / "config.yaml").write_text(
            "terminal:\n  ssh_user: late-root-poison\n", encoding="utf-8"
        )
        if target_dotenv_exists:
            (next_worker_home / ".env").write_text("", encoding="utf-8")
        real_home = launch_home / "real-home"
        real_home.mkdir()
        late_real_home = launch_home / "secondary-poison-home"
        late_real_home.mkdir()
        if not explicit_launch_home:
            monkeypatch.setattr(
                "hermes_constants._get_platform_default_hermes_home",
                lambda env=None: launch_home,
            )
            monkeypatch.delenv("HERMES_HOME")
        monkeypatch.setattr(launch_profile_policy, "_authority", None)
        monkeypatch.setattr("hermes_constants.is_container", lambda: False)
        set_multiplex_active(False)
        monkeypatch.setenv("HOME", str(real_home))
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "launch-only")
        monkeypatch.setenv("LAUNCH_HOST", "launch-config")
        monkeypatch.setenv("TERMINAL_SSH_USER", "launch-operator")
        monkeypatch.setenv("UNRELATED_SETTING", "launch-global")
        monkeypatch.setattr(
            "hermes_cli.main_tui_launch._make_tui_argv",
            lambda root, tui_dev=False: (["cat"], None),
            raising=False,
        )

        _argv, _cwd, first_env = _web_server_chat._resolve_chat_argv(
            profile="worker_beta"
        )
        # A secondary-profile task may mutate process-global HERMES_HOME. Profile-name
        # resolution must remain rooted in the same frozen launch authority as the
        # child environment, rather than accepting a different profiles tree.
        monkeypatch.setenv("HERMES_HOME", str(late_worker_home))
        monkeypatch.setenv("HERMES_REAL_HOME", str(late_real_home))
        monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
        monkeypatch.setenv("LAUNCH_HOST", "late-config-poison")
        monkeypatch.setenv("TERMINAL_SSH_USER", "secondary-poison")
        monkeypatch.setenv("UNRELATED_SETTING", "secondary-poison")
        _argv, _cwd, next_env = _web_server_chat._resolve_chat_argv(
            profile="worker_gamma"
        )
        _argv, _cwd, current_env = _web_server_chat._resolve_chat_argv(profile="current")

        probe = (
            "import json,os; print(json.dumps({k: os.environ.get(k) for k in "
            "('HERMES_HOME','HOME','HERMES_REAL_HOME','OPENAI_API_KEY',"
            "'TERMINAL_SSH_USER','TERMINAL_SSH_HOST','UNRELATED_SETTING')}))"
        )

        def observe(env):
            result = subprocess.run(
                [sys.executable, "-c", probe],
                env=env,
                capture_output=True,
                check=True,
                text=True,
                timeout=60,
            )
            return json.loads(result.stdout)

        assert observe(first_env) == {
            "HERMES_HOME": str(worker_home),
            "HOME": str(real_home),
            "HERMES_REAL_HOME": str(real_home),
            "OPENAI_API_KEY": None,
            "TERMINAL_SSH_USER": "launch-operator",
            "TERMINAL_SSH_HOST": None,
            "UNRELATED_SETTING": "launch-global",
        }
        assert observe(next_env) == {
            "HERMES_HOME": str(next_worker_home),
            "HOME": str(real_home),
            "HERMES_REAL_HOME": str(real_home),
            "OPENAI_API_KEY": None,
            "TERMINAL_SSH_USER": "target-config-poison",
            "TERMINAL_SSH_HOST": None,
            "UNRELATED_SETTING": "launch-global",
        }
        assert observe(current_env) == {
            "HERMES_HOME": str(launch_home) if explicit_launch_home else None,
            "HOME": str(real_home),
            "HERMES_REAL_HOME": str(real_home),
            "OPENAI_API_KEY": "launch-only",
            "TERMINAL_SSH_USER": "launch-operator",
            "TERMINAL_SSH_HOST": "launch-config",
            "UNRELATED_SETTING": "launch-global",
        }

    def test_chat_argv_does_not_restore_launch_external_source_terminal_secret(
        self, isolated_profiles, monkeypatch
    ):
        from hermes_cli import env_loader
        from tui_gateway import launch_profile_policy

        launch_home = isolated_profiles["default"]
        marker = '{"LAUNCH_TOKEN":"external-source-secret"}'
        monkeypatch.setattr(launch_profile_policy, "_authority", None)
        monkeypatch.setenv("TERMINAL_DOCKER_ENV", marker)
        monkeypatch.setenv("TERMINAL_SSH_USER", "operator-user")
        monkeypatch.setitem(
            env_loader._SECRET_SOURCE_VALUES_BY_HOME,
            str(launch_home.resolve()),
            {
                "TERMINAL_DOCKER_ENV": marker,
                # External sources snapshot skipped-existing values too. This
                # one remains a genuine operator export, not source-owned.
                "TERMINAL_SSH_USER": "operator-user",
            },
        )
        monkeypatch.setitem(
            env_loader._SECRET_SOURCE_OWNED_NAMES_BY_HOME,
            str(launch_home.resolve()),
            frozenset({"terminal_docker_env"}),
        )
        monkeypatch.setattr(
            "hermes_cli.main_tui_launch._make_tui_argv",
            lambda root, tui_dev=False: (["cat"], None),
            raising=False,
        )

        _argv, _cwd, env = _web_server_chat._resolve_chat_argv(profile="worker_beta")

        assert env is not None
        assert env["HERMES_HOME"] == str(isolated_profiles["worker_beta"])
        assert env.get("TERMINAL_DOCKER_ENV") != marker
        assert env["TERMINAL_SSH_USER"] == "operator-user"

    def test_chat_argv_scrubs_skipped_external_credentials_but_keeps_global_export(
        self, isolated_profiles, monkeypatch
    ):
        from hermes_cli import env_loader
        from tui_gateway import launch_profile_policy

        launch_home = isolated_profiles["default"]
        target_home = isolated_profiles["worker_beta"]
        target_env = target_home / ".env"
        target_env.write_text(
            target_env.read_text(encoding="utf-8")
            + "VAULT_OVERLAP=target-value\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(launch_profile_policy, "_authority", None)
        monkeypatch.setenv("VAULT_OVERLAP", "launch-value")
        monkeypatch.setenv("VAULT_LAUNCH_ONLY", "launch-only")
        monkeypatch.setenv("TERMINAL_SSH_USER", "operator-user")
        home_key = str(launch_home.resolve())
        monkeypatch.setitem(
            env_loader._SECRET_SOURCE_VALUES_BY_HOME,
            home_key,
            {
                "VAULT_OVERLAP": "launch-value",
                "VAULT_LAUNCH_ONLY": "launch-only",
                "TERMINAL_SSH_USER": "operator-user",
            },
        )
        monkeypatch.setitem(
            env_loader._SECRET_SOURCE_OWNED_NAMES_BY_HOME,
            home_key,
            frozenset(),
        )
        monkeypatch.setattr(
            "hermes_cli.main_tui_launch._make_tui_argv",
            lambda root, tui_dev=False: (["cat"], None),
            raising=False,
        )

        _argv, _cwd, env = _web_server_chat._resolve_chat_argv(profile="worker_beta")

        assert env is not None
        assert env["VAULT_OVERLAP"] == "target-value"
        assert "VAULT_LAUNCH_ONLY" not in env
        assert env["TERMINAL_SSH_USER"] == "operator-user"

    def test_chat_argv_resume_uses_frozen_profile_store(
        self, isolated_profiles, monkeypatch
    ):
        import os

        from hermes_cli import profiles
        from hermes_state import SessionDB
        from tui_gateway import launch_profile_policy

        launch_home = isolated_profiles["default"]
        launch_profile = isolated_profiles["worker_beta"]
        late_root = launch_home / "late-root"
        late_profile = late_root / "profiles" / "worker_beta"
        late_profile.mkdir(parents=True)
        (late_profile / "config.yaml").write_text("{}\n", encoding="utf-8")

        for home, child in (
            (launch_profile, "child-from-launch-root"),
            (late_profile, "child-from-poison-root"),
        ):
            db = SessionDB(db_path=home / "state.db")
            db.create_session("resume-parent", source="webui")
            db.create_session(child, source="webui", parent_session_id="resume-parent")
            db.close()

        monkeypatch.setattr(launch_profile_policy, "_authority", None)
        monkeypatch.setattr(
            profiles,
            "_get_profiles_root",
            lambda: Path(os.environ["HERMES_HOME"]) / "profiles",
        )
        monkeypatch.setattr(
            "hermes_cli.main_tui_launch._make_tui_argv",
            lambda root, tui_dev=False: (["cat"], None),
            raising=False,
        )

        _web_server_chat._resolve_chat_argv(profile="worker_beta")
        monkeypatch.setenv("HERMES_HOME", str(late_root))

        _argv, _cwd, env = _web_server_chat._resolve_chat_argv(
            profile="worker_beta", resume="resume-parent"
        )

        assert env is not None
        assert env["HERMES_HOME"] == str(launch_profile)
        assert env["HERMES_TUI_RESUME"] == "child-from-launch-root"

    def test_chat_argv_keeps_profile_authority_over_dotenv_overrides(
        self, isolated_profiles, monkeypatch
    ):
        launch_home = isolated_profiles["default"]
        worker_home = isolated_profiles["worker_beta"]
        (launch_home / ".env").write_text(
            'TERMINAL_DOCKER_ENV={"LAUNCH_TOKEN":"secret"}\n',
            encoding="utf-8",
        )
        (worker_home / ".env").write_text(
            "HERMES_HOME=/wrong/profile\n"
            "HOME=/wrong/home\n"
            "HERMES_TUI_GATEWAY_URL=ws://wrong.example.test/api/ws\n",
            encoding="utf-8",
        )
        (launch_home / "config.yaml").write_text(
            "terminal:\n  home_mode: real\n",
            encoding="utf-8",
        )
        (worker_home / "config.yaml").write_text(
            "terminal:\n  home_mode: profile\n",
            encoding="utf-8",
        )
        (launch_home / "home").mkdir()
        (worker_home / "home").mkdir()
        monkeypatch.setenv("TERMINAL_HOME_MODE", "real")
        monkeypatch.setenv("TERMINAL_DOCKER_ENV", '{"LAUNCH_TOKEN":"secret"}')
        monkeypatch.setenv("TERMINAL_SSH_USER", "operator-user")
        monkeypatch.setattr(
            "hermes_cli.main_tui_launch._make_tui_argv",
            lambda root, tui_dev=False: (["cat"], None),
            raising=False,
        )

        _argv, _cwd, env = _web_server_chat._resolve_chat_argv(profile="worker_beta")

        assert env is not None
        assert env["HERMES_HOME"] == str(worker_home)
        assert env["HOME"] == str(worker_home / "home")
        assert env["TERMINAL_HOME_MODE"] == "profile"
        assert env.get("TERMINAL_DOCKER_ENV") != '{"LAUNCH_TOKEN":"secret"}'
        assert env["TERMINAL_SSH_USER"] == "operator-user"
        assert "HERMES_TUI_GATEWAY_URL" not in env

    def test_chat_argv_bridges_selected_profile_terminal_config(
        self, isolated_profiles, monkeypatch
    ):

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
            "hermes_cli.main_tui_launch._make_tui_argv",
            lambda root, tui_dev=False: (["cat"], None),
            raising=False,
        )

        _argv, _cwd, env = _web_server_chat._resolve_chat_argv(profile="worker_beta")

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

        (isolated_profiles["default"] / "config.yaml").write_text(
            "terminal:\n  backend: docker\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_SSH_USER", "operator-user")
        monkeypatch.setattr(
            "hermes_cli.main_tui_launch._make_tui_argv",
            lambda root, tui_dev=False: (["cat"], None),
            raising=False,
        )

        _argv, _cwd, env = _web_server_chat._resolve_chat_argv()

        assert env is not None
        assert env["TERMINAL_ENV"] == "docker"
        assert env["TERMINAL_SSH_USER"] == "operator-user"

    @pytest.mark.parametrize("placeholder", [".", "auto", "cwd"])
    def test_chat_argv_placeholder_cwd_preserves_exported_value(
        self, isolated_profiles, monkeypatch, placeholder
    ):

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
            "hermes_cli.main_tui_launch._make_tui_argv",
            lambda root, tui_dev=False: (["cat"], None),
            raising=False,
        )

        _argv, _cwd, env = _web_server_chat._resolve_chat_argv(profile="worker_beta")

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
            "hermes_cli.main_tui_launch._make_tui_argv",
            lambda root, tui_dev=False: (["cat"], None),
            raising=False,
        )
        monkeypatch.setattr(
            config_mod,
            "apply_terminal_config_to_env",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bridge failed")),
        )

        with caplog.at_level(logging.WARNING, logger=web_server._log.name):
            _argv, _cwd, env = _web_server_chat._resolve_chat_argv(profile="worker_beta")

        assert env is not None
        assert env["HERMES_HOME"] == str(isolated_profiles["worker_beta"])
        assert "TERMINAL_ENV" not in env


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

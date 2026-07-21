"""Tests for dashboard Grok realtime voice dispatch support."""

import asyncio
from contextlib import contextmanager
import json
from unittest.mock import patch

import pytest

from hermes_cli.config import DEFAULT_CONFIG


class _FakeHTTPResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_voice_realtime_defaults_are_safe():
    """Behavioral invariants for safe-by-default realtime voice config.

    Avoid snapshotting mutable model/voice literals; assert the contracts
    that keep the feature off and the surface narrow until explicit setup.
    """
    import hermes_cli.web_server as web_server

    realtime = DEFAULT_CONFIG["voice"]["realtime"]
    assert realtime["enabled"] is False

    payload = web_server._voice_realtime_payload(realtime)
    serialized = json.dumps(payload).lower()
    assert payload["enabled"] is False
    assert "api_key" not in serialized
    assert "client_secret" not in serialized
    assert "authorization" not in serialized
    assert "xai_api_key" not in serialized

    tools_by_name = {tool["name"]: tool for tool in payload["tools"]}
    assert set(tools_by_name) == {
        "start_delegate",
        "get_delegate_status",
        "stop_delegate",
    }
    assert "start_hermes_run" not in tools_by_name
    assert "approve_delegate_action" not in tools_by_name

    assert web_server._voice_realtime_ttl_seconds({"ephemeral_token_ttl_seconds": 10}) == 30
    assert web_server._voice_realtime_ttl_seconds({"ephemeral_token_ttl_seconds": 900}) == 300
    assert web_server._voice_realtime_ttl_seconds({"ephemeral_token_ttl_seconds": "nope"}) == 300


class TestVoiceRealtimeEndpoints:
    @pytest.fixture(autouse=True)
    def _client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_cli.web_server as web_server

        self.web_server = web_server
        for name in (
            "_VOICE_RUN_STATUSES",
            "_VOICE_ACTIVE_RUN_AGENTS",
            "_VOICE_ACTIVE_RUN_TASKS",
            "_VOICE_RUN_APPROVAL_SESSIONS",
            "_VOICE_STOP_REQUESTED",
        ):
            store = getattr(web_server, name, None)
            if hasattr(store, "clear"):
                store.clear()
        self.client = TestClient(web_server.app)
        self.client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN

    def _enable_voice_config(self, monkeypatch, **overrides):
        import hermes_cli.web_server as web_server

        cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        cfg["voice"]["realtime"]["enabled"] = True
        cfg["voice"]["realtime"].update(overrides)
        monkeypatch.setattr(web_server, "load_config", lambda: cfg)
        return cfg

    def test_config_endpoint_requires_dashboard_token(self):
        client = self.client
        client.headers.pop(self.web_server._SESSION_HEADER_NAME, None)

        resp = client.get("/api/voice/realtime/config")

        assert resp.status_code == 401

    def test_client_secret_endpoint_requires_dashboard_token(self):
        client = self.client
        client.headers.pop(self.web_server._SESSION_HEADER_NAME, None)

        resp = client.post("/api/voice/realtime/xai-client-secret")

        assert resp.status_code == 401

    def test_config_endpoint_returns_sanitized_realtime_config(self, monkeypatch):
        self._enable_voice_config(monkeypatch, voice="rex")

        resp = self.client.get("/api/voice/realtime/config")

        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["provider"] == "xai"
        assert data["model"] == "grok-voice-latest"
        assert data["voice"] == "rex"
        assert "api_key" not in json.dumps(data).lower()
        tools_by_name = {tool["name"]: tool for tool in data["tools"]}
        assert set(tools_by_name) == {
            "start_delegate",
            "get_delegate_status",
            "stop_delegate",
        }
        serialized = json.dumps(data)
        assert "start_hermes_run" not in serialized
        assert "approve_delegate_action" not in serialized
        assert "approve_hermes_action" not in serialized
        for name in ("get_delegate_status", "stop_delegate"):
            assert "delegate_id" in tools_by_name[name]["parameters"]["properties"]

    def test_client_secret_rejects_when_disabled(self, monkeypatch):
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        monkeypatch.setattr(self.web_server, "load_config", lambda: cfg)

        resp = self.client.post("/api/voice/realtime/xai-client-secret")

        assert resp.status_code == 400
        assert "disabled" in resp.text.lower()

    def test_client_secret_requires_xai_api_key(self, monkeypatch):
        self._enable_voice_config(monkeypatch)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setattr(self.web_server, "get_env_value", lambda key: None)

        resp = self.client.post("/api/voice/realtime/xai-client-secret")

        assert resp.status_code == 400
        assert "XAI_API_KEY" in resp.text

    def test_client_secret_uses_hermes_env_value(self, monkeypatch):
        self._enable_voice_config(monkeypatch)
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        monkeypatch.setattr(
            self.web_server,
            "get_env_value",
            lambda key: "xai-dotenv-secret" if key == "XAI_API_KEY" else None,
        )
        captured = {}

        def fake_urlopen(req, timeout=0):
            captured["headers"] = dict(req.header_items())
            return _FakeHTTPResponse({"value": "ephemeral-secret", "expires_at": 1893456000})

        with patch("urllib.request.urlopen", fake_urlopen):
            resp = self.client.post("/api/voice/realtime/xai-client-secret")

        assert resp.status_code == 200
        assert captured["headers"]["Authorization"] == "Bearer xai-dotenv-secret"
        assert "xai-dotenv-secret" not in resp.text

    def test_client_secret_mints_ephemeral_token_without_leaking_api_key(self, monkeypatch):
        self._enable_voice_config(monkeypatch, ephemeral_token_ttl_seconds=9999)
        monkeypatch.setenv("XAI_API_KEY", "xai-real-secret")
        captured = {}

        def fake_urlopen(req, timeout=0):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeHTTPResponse({"value": "ephemeral-secret", "expires_at": 1893456000})

        with patch("urllib.request.urlopen", fake_urlopen):
            resp = self.client.post("/api/voice/realtime/xai-client-secret")

        assert resp.status_code == 200
        data = resp.json()
        assert data["client_secret"] == {"value": "ephemeral-secret", "expires_at": 1893456000}
        assert data["model"]
        assert data["voice"]
        assert data.get("profile") in (None, "")
        assert "api_key" not in json.dumps(data).lower()
        assert captured["url"] == "https://api.x.ai/v1/realtime/client_secrets"
        assert captured["body"] == {"expires_after": {"seconds": 300}}
        assert captured["headers"]["Authorization"] == "Bearer xai-real-secret"
        assert "xai-real-secret" not in resp.text

    def test_voice_run_rejects_when_realtime_disabled(self, monkeypatch):
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        monkeypatch.setattr(self.web_server, "load_config", lambda: cfg)

        resp = self.client.post("/api/voice/runs", json={"input": "Check the current time"})

        assert resp.status_code == 400
        assert "disabled" in resp.text.lower()

    def test_voice_run_starts_and_gets_completed_status(self, monkeypatch):
        self._enable_voice_config(monkeypatch)
        captured_coroutines = []

        class FakeTask:
            def __init__(self, coro):
                self.coro = coro

            def done(self):
                return False

            def cancel(self):
                return None

        class FakeAgent:
            def __init__(self):
                self.interrupted = False
                self.session_prompt_tokens = 1
                self.session_completion_tokens = 2
                self.session_total_tokens = 3

            def run_conversation(self, user_message, conversation_history=None, task_id=None):
                assert user_message == "Check the current time"
                assert task_id.startswith("voice_delegate:")
                return {"final_response": "It is handled.", "completed": True}

            def interrupt(self, reason):
                self.interrupted = True

        def fake_create_task(coro):
            captured_coroutines.append(coro)
            return FakeTask(coro)

        monkeypatch.setattr(self.web_server, "_create_voice_dispatch_agent", lambda **kwargs: FakeAgent())
        monkeypatch.setattr(self.web_server.asyncio, "create_task", fake_create_task)

        resp = self.client.post("/api/voice/runs", json={"input": "Check the current time"})

        assert resp.status_code == 202
        run_id = resp.json()["run_id"]
        assert run_id.startswith("voice_delegate_")
        assert captured_coroutines

        queued = self.client.get(f"/api/voice/runs/{run_id}")
        assert queued.status_code == 200
        assert queued.json()["status"] == "queued"

        asyncio.run(captured_coroutines[0])

        completed = self.client.get(f"/api/voice/runs/{run_id}")
        assert completed.status_code == 200
        data = completed.json()
        assert data["status"] == "completed"
        assert data["output"] == "It is handled."
        assert data["usage"] == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}


    def test_delegate_endpoint_starts_isolated_delegate_and_persists_to_sqlite(self, monkeypatch):
        self._enable_voice_config(monkeypatch)
        captured_coroutines = []
        created_agents = []

        class FakeTask:
            def __init__(self, coro):
                self.coro = coro

            def done(self):
                return False

        class FakeAgent:
            def __init__(self, **kwargs):
                created_agents.append(kwargs)
                self.session_prompt_tokens = 10
                self.session_completion_tokens = 20
                self.session_total_tokens = 30

            def run_conversation(self, user_message, conversation_history=None, task_id=None):
                assert user_message == "Audit the repo"
                assert task_id.startswith("voice_delegate:")
                return {"final_response": "Delegate complete.", "completed": True}

            def interrupt(self, reason):
                raise AssertionError("delegate should not be interrupted")

        def fake_create_task(coro):
            captured_coroutines.append(coro)
            return FakeTask(coro)

        monkeypatch.setattr(self.web_server, "_create_voice_dispatch_agent", lambda **kwargs: FakeAgent(**kwargs))
        monkeypatch.setattr(self.web_server.asyncio, "create_task", fake_create_task)

        resp = self.client.post("/api/voice/delegates", json={"input": "Audit the repo"})

        assert resp.status_code == 202
        started = resp.json()
        delegate_id = started["delegate_id"]
        assert delegate_id.startswith("voice_delegate_")
        assert started["run_id"] == delegate_id
        assert started["object"] == "hermes.voice.delegate"
        assert captured_coroutines

        asyncio.run(captured_coroutines[0])

        assert created_agents
        assert created_agents[0]["session_id"] == delegate_id
        assert created_agents[0]["gateway_session_key"] == f"voice:{delegate_id}"

        # Simulate a dashboard process restart: in-memory state is gone, but
        # the local SQLite delegate ledger still has the run and event trail.
        self.web_server._VOICE_RUN_STATUSES.clear()
        self.web_server._VOICE_ACTIVE_RUN_AGENTS.clear()
        self.web_server._VOICE_ACTIVE_RUN_TASKS.clear()
        self.web_server._VOICE_RUN_APPROVAL_SESSIONS.clear()

        completed = self.client.get(f"/api/voice/delegates/{delegate_id}")
        assert completed.status_code == 200
        data = completed.json()
        assert data["object"] == "hermes.voice.delegate"
        assert data["delegate_id"] == delegate_id
        assert data["status"] == "completed"
        assert data["output"] == "Delegate complete."
        assert data["usage"] == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        assert any(event["event"] == "run.completed" for event in data["events"])

        events = self.client.get(f"/api/voice/delegates/{delegate_id}/events")
        assert events.status_code == 200
        assert events.json()["events"][-1]["event"] == "run.completed"
        legacy_events = self.client.get(f"/api/voice/runs/{delegate_id}/events")
        assert legacy_events.status_code == 200
        assert legacy_events.json()["events"][-1]["event"] == "run.completed"

    def test_legacy_run_endpoints_alias_delegate_ledger(self, monkeypatch):
        self._enable_voice_config(monkeypatch)
        captured_coroutines = []

        class FakeTask:
            def __init__(self, coro):
                self.coro = coro

            def done(self):
                return False

        class FakeAgent:
            session_prompt_tokens = 0
            session_completion_tokens = 0
            session_total_tokens = 0

            def run_conversation(self, user_message, conversation_history=None, task_id=None):
                return {"final_response": "legacy ok", "completed": True}

            def interrupt(self, reason):
                pass

        monkeypatch.setattr(self.web_server, "_create_voice_dispatch_agent", lambda **kwargs: FakeAgent())
        monkeypatch.setattr(self.web_server.asyncio, "create_task", lambda coro: captured_coroutines.append(coro) or FakeTask(coro))

        resp = self.client.post("/api/voice/runs", json={"input": "Legacy call"})

        assert resp.status_code == 202
        run_id = resp.json()["run_id"]
        assert run_id.startswith("voice_delegate_")
        asyncio.run(captured_coroutines[0])
        assert self.client.get(f"/api/voice/runs/{run_id}").json()["delegate_id"] == run_id


    def test_delegate_ledger_redacts_secret_shaped_status_and_events(self):
        secret = "Authorization: Bearer sk-test-secret-value-1234567890"
        self.web_server._set_voice_run_status(
            "voice_delegate_secret",
            "completed",
            session_id="voice_delegate_secret",
            input_preview=f"use api_key={secret}",
            output=f"finished with {secret}",
            error=f"token={secret}",
            usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        )
        self.web_server._append_voice_run_event(
            "voice_delegate_secret",
            {"event": "tool.completed", "timestamp": 123.0, "preview": f"secret={secret}", "api_key": secret},
        )

        resp = self.client.get("/api/voice/delegates/voice_delegate_secret")

        assert resp.status_code == 200
        payload = json.dumps(resp.json())
        assert "sk-test-secret-value" not in payload
        assert "api_key=Authorization" not in payload
        assert "[redacted]" in payload

    def test_delegate_approval_sessions_are_unique_even_with_same_client_session_id(self, monkeypatch):
        self._enable_voice_config(monkeypatch, dispatch={"max_active_delegates": 2, "default_toolsets": []})
        captured_coroutines = []

        class FakeTask:
            def __init__(self, coro):
                self.coro = coro

            def done(self):
                return False

        def fake_create_task(coro):
            captured_coroutines.append(coro)
            return FakeTask(coro)

        monkeypatch.setattr(self.web_server.asyncio, "create_task", fake_create_task)

        first = self.client.post("/api/voice/delegates", json={"input": "first", "session_id": "shared"}).json()["delegate_id"]
        second = self.client.post("/api/voice/delegates", json={"input": "second", "session_id": "shared"}).json()["delegate_id"]

        assert first != second
        assert self.web_server._VOICE_RUN_APPROVAL_SESSIONS[first] == f"voice:{first}"
        assert self.web_server._VOICE_RUN_APPROVAL_SESSIONS[second] == f"voice:{second}"
        for coro in captured_coroutines:
            coro.close()

    def test_voice_approval_endpoint_rejects_always_choice_for_human_ui_path(self, monkeypatch):
        self.web_server._set_voice_run_status(
            "voice_delegate_approval",
            "waiting_for_approval",
            session_id="voice_delegate_approval",
            last_event="approval.request",
        )
        self.web_server._VOICE_RUN_APPROVAL_SESSIONS["voice_delegate_approval"] = "voice:voice_delegate_approval"

        from tools import approval as approval_module

        resolved_choices = []

        def fake_resolve_gateway_approval(session_key, choice, resolve_all=False):
            resolved_choices.append((session_key, choice, resolve_all))
            return 1

        monkeypatch.setattr(approval_module, "resolve_gateway_approval", fake_resolve_gateway_approval)

        resp = self.client.post("/api/voice/delegates/voice_delegate_approval/approval", json={"choice": "always"})

        assert resp.status_code == 400
        assert "always" in resp.text.lower()
        assert resolved_choices == []

    def test_waiting_delegate_status_exposes_approval_request_details_for_human_ui(self):
        delegate_id = "voice_delegate_approval_detail"
        self.web_server._set_voice_run_status(
            delegate_id,
            "waiting_for_approval",
            session_id=delegate_id,
            last_event="approval.request",
        )
        self.web_server._append_voice_run_event(
            delegate_id,
            {
                "event": "approval.request",
                "timestamp": 123.0,
                "command": "python deploy.py --prod",
                "description": "Run a production deployment command from the delegate.",
                "pattern_key": "shell_command",
                "pattern_keys": ["shell_command", "network_send"],
                "choices": ["once", "session", "deny"],
            },
        )

        resp = self.client.get(f"/api/voice/delegates/{delegate_id}")

        assert resp.status_code == 200
        approval = resp.json().get("approval_request")
        assert isinstance(approval, dict)
        assert approval["command"] == "python deploy.py --prod"
        assert approval["description"] == "Run a production deployment command from the delegate."
        assert approval["pattern_key"] == "shell_command"
        assert approval["pattern_keys"] == ["shell_command", "network_send"]
        assert approval["choices"] == ["once", "session", "deny"]
        assert "always" not in approval["choices"]
        assert approval["timestamp"] == 123.0

    def test_waiting_approval_loaded_from_ledger_is_stopped_after_restart(self):
        delegate_id = "voice_delegate_restart_waiting"
        self.web_server._set_voice_run_status(
            delegate_id,
            "waiting_for_approval",
            session_id=delegate_id,
            last_event="approval.request",
        )
        self.web_server._append_voice_run_event(
            delegate_id,
            {
                "event": "approval.request",
                "timestamp": 123.0,
                "command": "python deploy.py --prod",
                "choices": ["once", "session", "deny"],
            },
        )
        self.web_server._VOICE_RUN_STATUSES.clear()
        self.web_server._VOICE_ACTIVE_RUN_TASKS.clear()
        self.web_server._VOICE_ACTIVE_RUN_AGENTS.clear()
        self.web_server._VOICE_RUN_APPROVAL_SESSIONS.clear()

        resp = self.client.get(f"/api/voice/delegates/{delegate_id}")

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "stopped"
        assert "dashboard process restarted" in payload["error"]
        assert payload.get("approval_request") is None

    def test_voice_run_stop_interrupts_active_agent_without_cancelling_executor_task(self):
        class FakeAgent:
            def __init__(self):
                self.reason = None

            def interrupt(self, reason):
                self.reason = reason

        class FakeTask:
            def __init__(self):
                self.cancelled = False

            def done(self):
                return False

            def cancel(self):
                self.cancelled = True

        agent = FakeAgent()
        task = FakeTask()
        self.web_server._VOICE_RUN_STATUSES["voice_run_test"] = {"run_id": "voice_run_test", "status": "running"}
        self.web_server._VOICE_ACTIVE_RUN_AGENTS["voice_run_test"] = agent
        self.web_server._VOICE_ACTIVE_RUN_TASKS["voice_run_test"] = task

        resp = self.client.post("/api/voice/runs/voice_run_test/stop")

        assert resp.status_code == 200
        assert resp.json()["status"] == "stopping"
        assert agent.reason == "Stop requested via voice dispatch"
        assert task.cancelled is False

    def test_voice_run_stop_does_not_reopen_terminal_run(self):
        self.web_server._VOICE_RUN_STATUSES["voice_run_done"] = {
            "object": "hermes.voice.run",
            "run_id": "voice_run_done",
            "status": "completed",
            "output": "done",
        }

        resp = self.client.post("/api/voice/runs/voice_run_done/stop")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["output"] == "done"
        assert self.web_server._voice_active_run_count() == 0

    def test_voice_run_stop_before_agent_registration_prevents_execution(self, monkeypatch):
        self._enable_voice_config(monkeypatch)
        captured_coroutines = []
        created_agents = []

        class FakeTask:
            def done(self):
                return False

        def fake_create_task(coro):
            captured_coroutines.append(coro)
            return FakeTask()

        def fail_if_created(**kwargs):
            created_agents.append(kwargs)
            raise AssertionError("agent should not be created after queued stop")

        monkeypatch.setattr(self.web_server.asyncio, "create_task", fake_create_task)
        monkeypatch.setattr(self.web_server, "_create_voice_dispatch_agent", fail_if_created)

        started = self.client.post("/api/voice/runs", json={"input": "Do not run this"})
        run_id = started.json()["run_id"]
        stopped = self.client.post(f"/api/voice/runs/{run_id}/stop")

        assert stopped.status_code == 200
        assert stopped.json()["status"] == "stopping"

        asyncio.run(captured_coroutines[0])

        status = self.client.get(f"/api/voice/runs/{run_id}").json()
        assert status["status"] == "stopped"
        assert created_agents == []
        assert self.web_server._voice_active_run_count() == 0


class TestVoiceProfileAndAsyncSafety:
    @pytest.fixture(autouse=True)
    def _client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_cli.web_server as web_server

        self.web_server = web_server
        for name in (
            "_VOICE_RUN_STATUSES",
            "_VOICE_ACTIVE_RUN_AGENTS",
            "_VOICE_ACTIVE_RUN_TASKS",
            "_VOICE_RUN_APPROVAL_SESSIONS",
            "_VOICE_STOP_REQUESTED",
        ):
            store = getattr(web_server, name, None)
            if hasattr(store, "clear"):
                store.clear()
        self.client = TestClient(web_server.app)
        self.client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN

    def _enable_voice_config(self, monkeypatch, **overrides):
        import hermes_cli.web_server as web_server

        cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        cfg["voice"]["realtime"]["enabled"] = True
        cfg["voice"]["realtime"].update(overrides)
        monkeypatch.setattr(web_server, "load_config", lambda: cfg)
        return cfg

    def test_client_secret_offloads_blocking_http(self, monkeypatch):
        """xAI mint must not run urlopen on the event loop."""
        self._enable_voice_config(monkeypatch)
        monkeypatch.setattr(self.web_server, "get_env_value", lambda key: "xai-test-key" if key == "XAI_API_KEY" else None)

        calls = {"to_thread": 0}

        async def fake_to_thread(fn, *args, **kwargs):
            calls["to_thread"] += 1
            assert fn is self.web_server._mint_xai_realtime_client_secret
            return {"value": "ephemeral-secret", "expires_at": 123}

        monkeypatch.setattr(self.web_server.asyncio, "to_thread", fake_to_thread)

        resp = self.client.post("/api/voice/realtime/xai-client-secret")
        assert resp.status_code == 200
        assert calls["to_thread"] == 1
        data = resp.json()
        assert data["client_secret"]["value"] == "ephemeral-secret"
        assert "xai-test-key" not in json.dumps(data)

    def test_voice_operations_use_selected_profile_scope(self, monkeypatch, tmp_path):
        """Config/env/ledger/agent paths should honor ?profile=."""
        import hermes_cli.web_server as web_server

        self._enable_voice_config(monkeypatch, voice="rex")
        seen = {"profiles": []}
        homes = {"current": tmp_path / "default"}
        homes["current"].mkdir(parents=True, exist_ok=True)

        @contextmanager
        def fake_scope(profile):
            seen["profiles"].append(profile)
            home = tmp_path / str(profile or "default")
            home.mkdir(parents=True, exist_ok=True)
            previous = homes["current"]
            homes["current"] = home
            try:
                yield home
            finally:
                homes["current"] = previous

        monkeypatch.setattr(web_server, "_profile_scope", fake_scope)
        monkeypatch.setattr(web_server, "get_hermes_home", lambda: homes["current"])
        monkeypatch.setattr(web_server, "get_env_value", lambda key: "xai-test-key" if key == "XAI_API_KEY" else None)

        async def fake_to_thread(fn, *args, **kwargs):
            return {"value": "ephemeral-secret", "expires_at": 999}

        monkeypatch.setattr(web_server.asyncio, "to_thread", fake_to_thread)

        cfg_resp = self.client.get("/api/voice/realtime/config?profile=work")
        assert cfg_resp.status_code == 200
        assert "work" in seen["profiles"]

        secret_resp = self.client.post("/api/voice/realtime/xai-client-secret?profile=work")
        assert secret_resp.status_code == 200
        assert seen["profiles"].count("work") >= 2

        # Seed a delegate under profile A and ensure profile B cannot read it.
        run = web_server._set_voice_run_status(
            "voice_delegate_profile_a",
            "completed",
            profile="alpha",
            session_id="s1",
            input_preview="alpha work",
            output="done",
        )
        assert run["profile"] == "alpha"
        missing = self.client.get("/api/voice/delegates/voice_delegate_profile_a?profile=beta")
        assert missing.status_code == 404
        found = self.client.get("/api/voice/delegates/voice_delegate_profile_a?profile=alpha")
        assert found.status_code == 200
        assert found.json()["profile"] == "alpha"

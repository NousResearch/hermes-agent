import asyncio
import hmac
import hashlib
import json
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.runtime_governor import (
    RuntimeGovernorClient,
    RuntimeGovernorDecision,
    RuntimeGovernorHeartbeat,
    get_runtime_governor,
)
from gateway.session import SessionSource


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def getcode(self):
        return self.status

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_runtime_governor_admit_signs_sidecar_payload(monkeypatch):
    captured = {}

    def _fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = request.data
        captured["timestamp"] = request.get_header("X-hermes-timestamp")
        captured["signature"] = request.get_header("X-hermes-signature")
        return _FakeResponse(
            {
                "success": True,
                "data": {
                    "allowed": True,
                    "leaseId": "lease-1",
                    "deadlineAt": "2026-04-25T12:00:00Z",
                },
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    client = RuntimeGovernorClient(
        base_url="http://127.0.0.1:9090/api/runtime",
        instance_id="inst_1",
        api_key="secret",
        required=True,
        timeout_seconds=2,
        default_user_id="user-1",
        default_tier="free",
    )

    decision = client.admit(platform="telegram", session_key="s1", message_preview="hello")

    assert decision.allowed is True
    assert decision.lease_id == "lease-1"
    assert captured["url"] == "http://127.0.0.1:9090/api/runtime/admit"
    body = json.loads(captured["body"].decode("utf-8"))
    assert body == {
        "userId": "user-1",
        "tier": "free",
        "platform": "telegram",
        "sessionKey": "s1",
        "sourceMessageId": "",
        "messagePreview": "hello",
        "instanceId": "inst_1",
    }
    expected_signature = hmac.new(
        b"secret",
        captured["timestamp"].encode("utf-8") + b"." + captured["body"],
        hashlib.sha256,
    ).hexdigest()
    assert captured["signature"] == expected_signature
    assert captured["timeout"] == 2


def test_required_runtime_governor_missing_config_denies(monkeypatch):
    monkeypatch.setenv("HERMES_RUNTIME_GOVERNOR_REQUIRED", "1")
    monkeypatch.delenv("HERMES_RUNTIME_GOVERNOR_URL", raising=False)
    monkeypatch.delenv("HERMES_RUNTIME_GOVERNOR_INSTANCE_ID", raising=False)
    monkeypatch.delenv("API_SERVER_KEY", raising=False)

    decision = get_runtime_governor().admit(platform="telegram", session_key="s1")

    assert decision.allowed is False
    assert decision.reason == "policy_unavailable"


class _RuntimeStepAgent:
    last_instance = None

    def __init__(self, *args, **kwargs):
        type(self).last_instance = self
        self.tools = []
        self.step_callback = None
        self.interrupt_message = None

    def interrupt(self, message=None):
        self.interrupt_message = message

    def get_activity_summary(self):
        return {"api_call_count": 1}

    def run_conversation(self, user_message, conversation_history=None, task_id=None):
        if self.step_callback:
            self.step_callback(1, [])
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }


def _install_runtime_step_agent(monkeypatch):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _RuntimeStepAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._pending_model_notes = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False, emit=AsyncMock())
    runner.config = SimpleNamespace(streaming=None)
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    return runner


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
    )


@pytest.mark.asyncio
async def test_runtime_lease_gets_heartbeat_even_without_user_hooks(monkeypatch):
    _install_runtime_step_agent(monkeypatch)
    runner = _make_runner()
    heartbeats = []

    class _Governor:
        def heartbeat(self, lease_id):
            heartbeats.append(lease_id)
            return RuntimeGovernorHeartbeat(should_stop=False)

    monkeypatch.setattr(
        "gateway.runtime_governor.get_runtime_governor",
        lambda: _Governor(),
    )
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
        },
    )
    import hermes_cli.tools_config as tools_config
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    result = await runner._run_agent(
        message="hi",
        context_prompt="",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="agent:main:telegram:dm:12345",
        runtime_lease_id="lease-1",
    )

    assert result["final_response"] == "ok"
    assert heartbeats == ["lease-1"]


@pytest.mark.asyncio
async def test_runtime_heartbeat_cutoff_interrupts_agent(monkeypatch):
    _install_runtime_step_agent(monkeypatch)
    runner = _make_runner()

    class _Governor:
        def heartbeat(self, lease_id):
            return RuntimeGovernorHeartbeat(
                should_stop=True,
                reason="runtime_cap",
                user_message="Runtime cap reached.",
            )

    monkeypatch.setattr(
        "gateway.runtime_governor.get_runtime_governor",
        lambda: _Governor(),
    )
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
        },
    )
    import hermes_cli.tools_config as tools_config
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    result = await runner._run_agent(
        message="hi",
        context_prompt="",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="agent:main:telegram:dm:12345",
        runtime_lease_id="lease-1",
    )

    assert result["runtime_cutoff"] is True
    assert result["final_response"] == "Runtime cap reached."
    assert _RuntimeStepAgent.last_instance.interrupt_message == "Runtime cap reached."

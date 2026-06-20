"""Regression tests for gateway-owned Codex app-server thread continuity."""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from collections import OrderedDict
from types import SimpleNamespace

from gateway.config import Platform
from gateway.session import SessionSource
import gateway.run as gateway_run


class _CodexThreadAgent:
    last_instance = None

    def __init__(self, *args, **kwargs):
        type(self).last_instance = self
        self.tools = []
        self.session_id = kwargs.get("session_id")
        self.model = kwargs.get("model")
        self.provider = kwargs.get("provider")
        self.base_url = kwargs.get("base_url")
        self.api_key = kwargs.get("api_key")
        self.api_mode = kwargs.get("api_mode")
        self.context_compressor = SimpleNamespace(
            last_prompt_tokens=0,
            last_input_tokens=0,
            last_output_tokens=0,
            context_length=0,
        )
        self._codex_session = SimpleNamespace(_thread_id="thread-live-123")

    def run_conversation(self, user_message: str, **kwargs):
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
            "codex_thread_id": "thread-result-456",
        }


class _TimedOutCodexThreadAgent(_CodexThreadAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._codex_session = SimpleNamespace(_thread_id="thread-timeout-123")
        self._interrupted = threading.Event()
        self.released = False

    def run_conversation(self, user_message: str, **kwargs):
        self._interrupted.wait(timeout=2.0)
        return {
            "final_response": "late",
            "messages": [],
            "api_calls": 3,
            "codex_thread_id": "thread-late-result",
        }

    def get_activity_summary(self):
        return {
            "last_activity_desc": "waiting on codex app-server",
            "seconds_since_activity": 999.0,
            "current_tool": None,
            "api_call_count": 3,
            "max_iterations": 20,
        }

    def interrupt(self, reason=None):
        self._interrupted.set()

    def release_clients(self):
        self.released = True
        self._interrupted.set()


def _make_runner(persisted_thread_id: str):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._pending_messages = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._session_run_generation = {}
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = gateway_run.threading.Lock()
    runner._pending_skills_reload_notes = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._voice_mode = {}
    runner._show_reasoning = False
    runner._service_tier = None
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        streaming=SimpleNamespace(
            enabled=False,
            transport="off",
            cursor="",
            edit_interval=1.0,
            buffer_threshold=20,
            fresh_final_after_seconds=0.0,
        ),
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
    )
    runner.session_store = SimpleNamespace(
        _entries={
            "agent:main:local:dm": SimpleNamespace(
                codex_thread_id=persisted_thread_id,
            )
        }
    )
    return runner


def _patch_runtime(monkeypatch, tmp_path, agent_cls):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("agent:\n  model: test\n", encoding="utf-8")

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openai",
            "api_mode": "codex_app_server",
            "base_url": "https://stub.invalid",
            "api_key": "stub",
        },
    )

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


def test_run_agent_pins_persisted_codex_thread_and_returns_live_thread_id(
    monkeypatch, tmp_path
):
    _patch_runtime(monkeypatch, tmp_path, _CodexThreadAgent)

    _CodexThreadAgent.last_instance = None
    runner = _make_runner("thread-persisted-789")
    source = SessionSource(
        platform=Platform.LOCAL,
        chat_id="cli",
        chat_type="dm",
        user_id="user-1",
    )

    result = asyncio.run(
        runner._run_agent(
            message="ping",
            context_prompt="",
            history=[],
            source=source,
            session_id="session-1",
            session_key="agent:main:local:dm",
        )
    )

    assert result["final_response"] == "ok"
    assert result["codex_thread_id"] == "thread-live-123"
    assert _CodexThreadAgent.last_instance is not None
    assert (
        _CodexThreadAgent.last_instance._resume_codex_thread_id
        == "thread-persisted-789"
    )


def test_queued_followup_merge_preserves_codex_thread_id_when_followup_lacks_it():
    current = {"history_offset": 1, "codex_thread_id": "thread-current-123"}
    followup = {"history_offset": 9, "final_response": "next"}

    merged = gateway_run._preserve_queued_followup_history_offset(current, followup)

    assert merged["history_offset"] == 1
    assert merged["codex_thread_id"] == "thread-current-123"


def test_inactivity_timeout_returns_live_codex_thread_id(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0.001")
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT_WARNING", "0")
    _patch_runtime(monkeypatch, tmp_path, _TimedOutCodexThreadAgent)

    real_wait = asyncio.wait

    async def fast_wait(awaitables, *, timeout=None, return_when=asyncio.ALL_COMPLETED):
        effective_timeout = 0.01 if timeout == 5.0 else timeout
        return await real_wait(
            awaitables,
            timeout=effective_timeout,
            return_when=return_when,
        )

    monkeypatch.setattr(gateway_run.asyncio, "wait", fast_wait)

    _TimedOutCodexThreadAgent.last_instance = None
    runner = _make_runner("")
    source = SessionSource(
        platform=Platform.LOCAL,
        chat_id="cli",
        chat_type="dm",
        user_id="user-1",
    )

    result = asyncio.run(
        runner._run_agent(
            message="ping",
            context_prompt="",
            history=[],
            source=source,
            session_id="session-1",
            session_key="agent:main:local:dm",
        )
    )

    assert result["failed"] is True
    assert result["codex_thread_id"] == "thread-timeout-123"
    assert _TimedOutCodexThreadAgent.last_instance is not None

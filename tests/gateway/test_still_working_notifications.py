"""Tests for configurable gateway still-working notifications.

The gateway emits periodic "Still working..." heartbeats for long-running agent
turns. These tests cover both the config loader and the runtime heartbeat
messages that _run_agent emits.
"""

import sys
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class TestLoadStillWorkingInterval:
    def test_defaults_to_600_seconds(self, monkeypatch, tmp_path):
        import gateway.run as gw

        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        assert GatewayRunner._load_still_working_interval("telegram") == 600.0

    def test_reads_global_interval_from_config(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n  still_working_interval: 300\n",
            encoding="utf-8",
        )
        import gateway.run as gw

        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        assert GatewayRunner._load_still_working_interval("telegram") == 300.0

    def test_platform_override_wins(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n"
            "  still_working_interval: 600\n"
            "  still_working_overrides:\n"
            "    signal: off\n"
            "    telegram: 120\n",
            encoding="utf-8",
        )
        import gateway.run as gw

        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        assert GatewayRunner._load_still_working_interval("signal") is None
        assert GatewayRunner._load_still_working_interval("telegram") == 120.0

    @pytest.mark.parametrize(
        "raw_yaml",
        [
            "display:\n  still_working_interval: 0\n",
            "display:\n  still_working_interval: false\n",
            "display:\n  still_working_interval: off\n",
        ],
    )
    def test_zero_false_or_off_disables_globally(self, monkeypatch, tmp_path, raw_yaml):
        (tmp_path / "config.yaml").write_text(raw_yaml, encoding="utf-8")
        import gateway.run as gw

        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        assert GatewayRunner._load_still_working_interval("telegram") is None

    def test_invalid_value_defaults_to_600_seconds(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n  still_working_interval: banana\n",
            encoding="utf-8",
        )
        import gateway.run as gw

        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        assert GatewayRunner._load_still_working_interval("telegram") == 600.0

    def test_falls_back_to_legacy_env_var_when_new_display_key_missing(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text("display:\n  tool_progress: off\n", encoding="utf-8")
        import gateway.run as gw

        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "180")
        assert GatewayRunner._load_still_working_interval("telegram") == 180.0

    def test_invalid_platform_override_inherits_global_interval(self, monkeypatch, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "display:\n"
            "  still_working_interval: 300\n"
            "  still_working_overrides:\n"
            "    signal: banana\n",
            encoding="utf-8",
        )
        import gateway.run as gw

        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        assert GatewayRunner._load_still_working_interval("signal") == 300.0

    @pytest.mark.parametrize(
        "raw_yaml",
        [
            "display:\n  still_working_interval: true\n",
            "display:\n  still_working_interval: on\n",
        ],
    )
    def test_true_or_on_global_value_maps_to_default_interval(self, monkeypatch, tmp_path, raw_yaml):
        (tmp_path / "config.yaml").write_text(raw_yaml, encoding="utf-8")
        import gateway.run as gw

        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        assert GatewayRunner._load_still_working_interval("telegram") == 600.0

    @pytest.mark.parametrize(
        "raw_value",
        ["true", "on"],
    )
    def test_true_or_on_platform_override_inherits_global_interval(self, monkeypatch, tmp_path, raw_value):
        (tmp_path / "config.yaml").write_text(
            "display:\n"
            "  still_working_interval: 300\n"
            "  still_working_overrides:\n"
            f"    signal: {raw_value}\n",
            encoding="utf-8",
        )
        import gateway.run as gw

        monkeypatch.setattr(gw, "_hermes_home", tmp_path)
        assert GatewayRunner._load_still_working_interval("signal") == 300.0


class _HeartbeatCaptureAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
                "message_id": "heartbeat-1",
            }
        )
        return SendResult(success=True, message_id="heartbeat-1")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        for message in reversed(self.sent):
            if message.get("message_id") == message_id:
                message["content"] = content
                break
        else:
            self.sent.append(
                {
                    "chat_id": chat_id,
                    "content": content,
                    "reply_to": None,
                    "metadata": None,
                    "message_id": message_id,
                }
            )
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _make_runner(adapter):
    import gateway.run as gateway_run

    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_model_overrides = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    return runner


def _install_fake_run_agent(monkeypatch, *, activity_summary, run_duration=0.05):
    class _FakeStillWorkingAgent:
        def __init__(self, **kwargs):
            self._activity_summary = dict(activity_summary)

        def get_activity_summary(self):
            return dict(self._activity_summary)

        def run_conversation(self, message, conversation_history=None, task_id=None):
            time.sleep(run_duration)
            return {"final_response": "done", "messages": [], "api_calls": 1}

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _FakeStillWorkingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


async def _run_agent_with_heartbeat(monkeypatch, tmp_path, config_text, *, activity_summary, platform=Platform.TELEGRAM):
    (tmp_path / "config.yaml").write_text(config_text, encoding="utf-8")

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")
    _install_fake_run_agent(monkeypatch, activity_summary=activity_summary)

    adapter = _HeartbeatCaptureAdapter(platform=platform)
    runner = _make_runner(adapter)
    source = SessionSource(platform=platform, chat_id="123", chat_type="dm")
    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key=f"agent:main:{platform.value}:dm:123",
    )
    return adapter, result


class TestStillWorkingRuntime:
    @pytest.mark.asyncio
    async def test_runtime_heartbeat_preserves_last_activity_context(self, monkeypatch, tmp_path):
        adapter, result = await _run_agent_with_heartbeat(
            monkeypatch,
            tmp_path,
            "display:\n"
            "  still_working_interval: 0.01\n"
            "  platforms:\n"
            "    telegram:\n"
            "      busy_ack_detail: true\n",
            activity_summary={
                "current_tool": None,
                "last_activity_desc": "waiting on provider response",
                "api_call_count": 2,
                "max_iterations": 90,
                "seconds_since_activity": 0.0,
            },
        )

        assert result["final_response"] == "done"
        assert adapter.sent
        assert any(
            "iteration 2/90" in msg["content"] and "waiting on provider response" in msg["content"]
            for msg in adapter.sent
        )

    @pytest.mark.asyncio
    async def test_runtime_heartbeat_disabled_emits_no_messages(self, monkeypatch, tmp_path):
        adapter, result = await _run_agent_with_heartbeat(
            monkeypatch,
            tmp_path,
            "display:\n  still_working_interval: off\n",
            activity_summary={
                "current_tool": "terminal",
                "last_activity_desc": "tool_call",
                "api_call_count": 1,
                "max_iterations": 90,
                "seconds_since_activity": 0.0,
            },
        )

        assert result["final_response"] == "done"
        assert adapter.sent == []

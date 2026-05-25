"""Tests for the bundled observability/posthog plugin."""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "posthog"


class _FakePosthog:
    instances: list["_FakePosthog"] = []

    def __init__(self, project_api_key, **kwargs):
        self.project_api_key = project_api_key
        self.kwargs = kwargs
        self.events: list[dict] = []
        self.flushed = 0
        _FakePosthog.instances.append(self)

    def capture(self, *args, **kwargs):
        self.events.append({"args": args, "kwargs": kwargs})

    def flush(self):
        self.flushed += 1


class _Ctx:
    def __init__(self):
        self.hooks = []

    def register_hook(self, name, fn):
        self.hooks.append((name, fn))


def _fresh_plugin(monkeypatch=None):
    mod_name = "plugins.observability.posthog"
    sys.modules.pop(mod_name, None)
    mod = importlib.import_module(mod_name)
    if monkeypatch is not None:
        _FakePosthog.instances.clear()
        monkeypatch.setattr(mod, "Posthog", _FakePosthog, raising=False)
    return mod


def _clear_env(monkeypatch):
    for key in (
        "HERMES_POSTHOG_PROJECT_TOKEN",
        "HERMES_POSTHOG_HOST",
        "HERMES_POSTHOG_DISTINCT_ID",
        "HERMES_POSTHOG_ENV",
        "HERMES_POSTHOG_RELEASE",
        "HERMES_POSTHOG_SAMPLE_RATE",
        "HERMES_POSTHOG_MAX_CHARS",
        "HERMES_POSTHOG_PRIVACY_MODE",
        "HERMES_POSTHOG_SYNC_MODE",
        "HERMES_POSTHOG_DEBUG",
        "POSTHOG_PROJECT_API_KEY",
        "POSTHOG_HOST",
    ):
        monkeypatch.delenv(key, raising=False)


class TestManifest:
    def test_plugin_directory_exists(self):
        assert PLUGIN_DIR.is_dir()
        assert (PLUGIN_DIR / "plugin.yaml").exists()
        assert (PLUGIN_DIR / "__init__.py").exists()
        assert (PLUGIN_DIR / "README.md").exists()

    def test_manifest_fields(self):
        data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
        assert data["name"] == "posthog"
        assert data["version"]
        assert "posthog" in data["pip_dependencies"]
        assert set(data["hooks"]) == {
            "pre_api_request", "post_api_request",
            "pre_llm_call", "post_llm_call",
            "pre_tool_call", "post_tool_call",
        }
        assert "HERMES_POSTHOG_PROJECT_TOKEN" in data["requires_env"]


class TestDiscovery:
    def test_plugin_is_discovered_as_standalone_opt_in(self, tmp_path, monkeypatch):
        from hermes_cli import plugins as plugins_mod

        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        manager = plugins_mod.PluginManager()
        manager.discover_and_load()

        loaded = manager._plugins.get("observability/posthog")
        assert loaded is not None, "plugin not discovered"
        assert loaded.enabled is False
        assert "not enabled" in (loaded.error or "").lower()


class TestRuntimeGate:
    def test_get_posthog_returns_none_without_credentials(self, monkeypatch):
        _clear_env(monkeypatch)
        plugin = _fresh_plugin(monkeypatch)
        assert plugin._get_posthog() is None

    def test_get_posthog_caches_failure_no_env_rereads(self, monkeypatch):
        _clear_env(monkeypatch)
        plugin = _fresh_plugin(monkeypatch)
        assert plugin._get_posthog() is None

        import os
        real_get = os.environ.get
        called = {"n": 0}

        def tracking_get(key, default=None):
            if key.startswith(("HERMES_POSTHOG_", "POSTHOG_")):
                called["n"] += 1
            return real_get(key, default)

        monkeypatch.setattr(os.environ, "get", tracking_get)
        for _ in range(10):
            assert plugin._get_posthog() is None
        assert called["n"] == 0

    def test_get_posthog_does_not_import_hermes_config(self, monkeypatch):
        _clear_env(monkeypatch)
        sys.modules.pop("hermes_cli.config", None)
        plugin = _fresh_plugin(monkeypatch)
        for _ in range(5):
            plugin._get_posthog()
        assert "hermes_cli.config" not in sys.modules

    def test_valid_credentials_initialize_client(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_POSTHOG_PROJECT_TOKEN", "phc_real_project_token")
        monkeypatch.setenv("HERMES_POSTHOG_HOST", "https://eu.i.posthog.com")
        plugin = _fresh_plugin(monkeypatch)

        client = plugin._get_posthog()

        assert isinstance(client, _FakePosthog)
        assert client.project_api_key == "phc_real_project_token"
        assert client.kwargs["host"] == "https://eu.i.posthog.com"

    def test_invalid_max_chars_falls_back_instead_of_raising(self, monkeypatch, caplog):
        _clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_POSTHOG_MAX_CHARS", "not-an-int")
        plugin = _fresh_plugin(monkeypatch)

        with caplog.at_level(logging.WARNING, logger="plugins.observability.posthog"):
            assert plugin._safe_value("x" * 12005).endswith("[truncated 5 chars]")

        assert "Invalid HERMES_POSTHOG_MAX_CHARS" in caplog.text

    def test_trace_state_pruning_bounds_long_running_processes(self, monkeypatch):
        _clear_env(monkeypatch)
        plugin = _fresh_plugin(monkeypatch)
        plugin._TRACE_STATE.clear()
        now = 10_000.0
        plugin._TRACE_STATE["fresh"] = plugin.TraceState(trace_id="fresh", last_updated_at=now)
        plugin._TRACE_STATE["stale"] = plugin.TraceState(
            trace_id="stale",
            last_updated_at=now - plugin._TRACE_STATE_TTL_SECONDS - 1,
        )

        plugin._prune_trace_states(now)

        assert "fresh" in plugin._TRACE_STATE
        assert "stale" not in plugin._TRACE_STATE


class TestPlaceholderDetection:
    LOGGER_NAME = "plugins.observability.posthog"

    @pytest.mark.parametrize("placeholder", ["placeholder", "test-key", "your-posthog-token", "change-me", "xxx"])
    def test_common_placeholder_tokens_warn_and_skip(self, monkeypatch, caplog, placeholder):
        _clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_POSTHOG_PROJECT_TOKEN", placeholder)
        plugin = _fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            assert plugin._get_posthog() is None
        assert "HERMES_POSTHOG_PROJECT_TOKEN" in caplog.text
        assert _FakePosthog.instances == []

    def test_valid_phc_token_does_not_warn(self, monkeypatch, caplog):
        _clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_POSTHOG_PROJECT_TOKEN", "phc_real_project_token")
        plugin = _fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            assert isinstance(plugin._get_posthog(), _FakePosthog)
        assert "placeholder" not in caplog.text.lower()


class TestHooks:
    def test_hooks_noop_without_client(self, monkeypatch):
        _clear_env(monkeypatch)
        plugin = _fresh_plugin(monkeypatch)
        plugin.on_pre_llm_call(task_id="t", session_id="s", messages=[{"role": "user", "content": "hi"}])
        plugin.on_pre_llm_request(task_id="t", session_id="s", api_call_count=1, request_messages=[])
        plugin.on_post_llm_call(task_id="t", session_id="s", api_call_count=1)
        plugin.on_pre_tool_call(tool_name="read_file", args={}, task_id="t", session_id="s")
        plugin.on_post_tool_call(tool_name="read_file", args={}, result="ok", task_id="t", session_id="s")

    def test_registers_all_hooks(self, monkeypatch):
        plugin = _fresh_plugin(monkeypatch)
        ctx = _Ctx()
        plugin.register(ctx)
        assert [name for name, _fn in ctx.hooks] == [
            "pre_api_request", "post_api_request",
            "pre_llm_call", "post_llm_call",
            "pre_tool_call", "post_tool_call",
        ]

    def test_generation_capture_emits_posthog_ai_generation_event(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_POSTHOG_PROJECT_TOKEN", "phc_real_project_token")
        monkeypatch.setenv("HERMES_POSTHOG_DISTINCT_ID", "agent-test")
        plugin = _fresh_plugin(monkeypatch)

        plugin.on_pre_llm_request(
            task_id="task-1",
            session_id="session-1",
            platform="discord",
            model="gpt-5-mini",
            provider="openai",
            api_mode="responses",
            api_call_count=1,
            request_messages=[{"role": "user", "content": "hello"}],
        )
        plugin.on_post_llm_call(
            task_id="task-1",
            session_id="session-1",
            provider="openai",
            base_url="https://api.openai.com/v1",
            api_mode="responses",
            model="gpt-5-mini",
            api_call_count=1,
            assistant_response="hi there",
            api_duration=1.234,
            usage={"input_tokens": 3, "output_tokens": 4},
        )

        client = plugin._get_posthog()
        event = client.events[-1]["kwargs"]
        assert event["distinct_id"] == "agent-test"
        assert event["event"] == "$ai_generation"
        props = event["properties"]
        assert props["$ai_trace_id"]
        assert props["$ai_session_id"] == "session-1"
        assert props["$ai_model"] == "gpt-5-mini"
        assert props["$ai_provider"] == "openai"
        assert props["$ai_input"] == [{"role": "user", "content": "hello"}]
        assert props["$ai_output_choices"] == [{"role": "assistant", "content": "hi there"}]
        assert props["$ai_input_tokens"] == 3
        assert props["$ai_output_tokens"] == 4
        assert props["$ai_latency"] == 1.234
        assert client.flushed == 1

    def test_privacy_mode_omits_generation_input_and_output(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_POSTHOG_PROJECT_TOKEN", "phc_real_project_token")
        monkeypatch.setenv("HERMES_POSTHOG_PRIVACY_MODE", "true")
        plugin = _fresh_plugin(monkeypatch)

        plugin.on_pre_llm_request(
            task_id="task-privacy",
            session_id="session-privacy",
            api_call_count=1,
            request_messages=[{"role": "user", "content": "secret"}],
        )
        plugin.on_post_llm_call(
            task_id="task-privacy",
            session_id="session-privacy",
            api_call_count=1,
            assistant_response="secret answer",
        )

        props = plugin._get_posthog().events[-1]["kwargs"]["properties"]
        assert "$ai_input" not in props
        assert "$ai_output_choices" not in props

    def test_tool_call_emits_posthog_ai_span_event(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_POSTHOG_PROJECT_TOKEN", "phc_real_project_token")
        plugin = _fresh_plugin(monkeypatch)

        plugin.on_pre_llm_request(
            task_id="task-tools",
            session_id="session-tools",
            api_call_count=1,
            request_messages=[{"role": "user", "content": "read"}],
        )
        plugin.on_pre_tool_call(
            task_id="task-tools",
            session_id="session-tools",
            tool_name="read_file",
            tool_call_id="call-1",
            args={"path": "/tmp/a"},
        )
        plugin.on_post_tool_call(
            task_id="task-tools",
            session_id="session-tools",
            tool_name="read_file",
            tool_call_id="call-1",
            args={"path": "/tmp/a"},
            result='{"content": "1|hello", "total_lines": 1, "file_size": 7, "is_binary": false, "is_image": false}',
        )

        event = plugin._get_posthog().events[-1]["kwargs"]
        assert event["event"] == "$ai_span"
        props = event["properties"]
        assert props["$ai_trace_id"]
        assert props["$ai_session_id"] == "session-tools"
        assert props["$ai_span_name"] == "Tool: read_file"
        assert props["hermes.tool_name"] == "read_file"
        assert props["$ai_input_state"] == {"path": "/tmp/a"}
        assert props["$ai_output_state"]["returned_lines"] == {"start": 1, "end": 1, "count": 1}

    def test_serialize_tool_calls_extracts_tool_names_for_generation(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_POSTHOG_PROJECT_TOKEN", "phc_real_project_token")
        plugin = _fresh_plugin(monkeypatch)

        class _Fn:
            name = "terminal"
            arguments = '{"command":"date"}'

        class _ToolCall:
            id = "call-terminal"
            type = "function"
            function = _Fn()

        class _Assistant:
            content = None
            reasoning = None
            tool_calls = [_ToolCall()]

        plugin.on_pre_llm_request(task_id="task-tc", session_id="session-tc", api_call_count=1, request_messages=[])
        plugin.on_post_llm_call(task_id="task-tc", session_id="session-tc", api_call_count=1, assistant_message=_Assistant())

        props = plugin._get_posthog().events[-1]["kwargs"]["properties"]
        assert props["$ai_tools_called"] == ["terminal"]
        assert props["$ai_tool_call_count"] == 1

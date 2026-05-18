from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


PLUGIN_NAME = "progressive-context-e2e"


def _write_progressive_context_plugin(hermes_home):
    plugin_dir = hermes_home / "plugins" / PLUGIN_NAME
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {PLUGIN_NAME}\nversion: 0.1.0\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        """
from agent.context_engine import ContextCompressionCandidate, ContextEngine


class ProgressiveContextEngine(ContextEngine):
    def __init__(self):
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.threshold_tokens = 1_000_000
        self.context_length = 100_000
        self.compression_count = 0
        self.prepare_calls = 0
        self.applied_calls = 0

    @property
    def name(self):
        return "progressive-context-e2e"

    def update_from_response(self, usage):
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None):
        return messages

    def should_prepare_async_compression(self, prompt_tokens=None, messages=None):
        return bool(prompt_tokens and prompt_tokens >= 10_000)

    def prepare_async_compression(self, messages, current_tokens=None, focus_topic=None):
        self.prepare_calls += 1
        return ContextCompressionCandidate(
            messages=[
                {"role": "user", "content": "plugin compressed context"},
                {"role": "assistant", "content": "plugin compressed answer"},
            ],
        )

    def on_async_compression_applied(self, candidate, **kwargs):
        self.applied_calls += 1


_engine = ProgressiveContextEngine()


def _transform_tool_result(**kwargs):
    return "plugin-transform:" + kwargs["result"]


def register(ctx):
    ctx.register_context_engine(_engine)
    ctx.register_hook("transform_tool_result", _transform_tool_result)
""",
        encoding="utf-8",
    )


def _write_llm_progressive_context_plugin(hermes_home):
    plugin_dir = hermes_home / "plugins" / PLUGIN_NAME
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {PLUGIN_NAME}\nversion: 0.1.0\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        """
from agent.context_engine import ContextCompressionCandidate, ContextEngine


LLM_CALLS = []


def fake_llm_compress(messages, current_tokens=None, focus_topic=None):
    LLM_CALLS.append({
        "messages": list(messages),
        "current_tokens": current_tokens,
        "focus_topic": focus_topic,
    })
    user_text = next(
        (m.get("content", "") for m in messages if m.get("role") == "user"),
        "",
    )
    return f"fake llm summary: {user_text}"


class LlmProgressiveContextEngine(ContextEngine):
    def __init__(self):
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.threshold_tokens = 1_000_000
        self.context_length = 100_000
        self.compression_count = 0
        self.applied_calls = 0

    @property
    def name(self):
        return "progressive-context-e2e"

    def update_from_response(self, usage):
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None):
        return messages

    def should_prepare_async_compression(self, prompt_tokens=None, messages=None):
        return bool(prompt_tokens and prompt_tokens >= 10_000)

    def prepare_async_compression(self, messages, current_tokens=None, focus_topic=None):
        summary = fake_llm_compress(
            messages,
            current_tokens=current_tokens,
            focus_topic=focus_topic,
        )
        return ContextCompressionCandidate(
            messages=[
                {"role": "user", "content": "[llm compressed context]\\n" + summary},
                {"role": "assistant", "content": "llm summary ready"},
            ],
        )

    def on_async_compression_applied(self, candidate, **kwargs):
        self.applied_calls += 1


_engine = LlmProgressiveContextEngine()


def register(ctx):
    ctx.register_context_engine(_engine)
""",
        encoding="utf-8",
    )


def _write_slow_llm_progressive_context_plugin(hermes_home):
    plugin_dir = hermes_home / "plugins" / PLUGIN_NAME
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {PLUGIN_NAME}\nversion: 0.1.0\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        """
import threading

from agent.context_engine import ContextCompressionCandidate, ContextEngine


LLM_CALLS = []
LLM_STARTED = threading.Event()
LLM_RELEASE = threading.Event()


def fake_llm_compress(messages, current_tokens=None, focus_topic=None):
    LLM_STARTED.set()
    if not LLM_RELEASE.wait(timeout=30):
        raise RuntimeError("test did not release fake llm")
    LLM_CALLS.append({
        "messages": list(messages),
        "current_tokens": current_tokens,
        "focus_topic": focus_topic,
    })
    user_text = next(
        (m.get("content", "") for m in messages if m.get("role") == "user"),
        "",
    )
    return f"slow fake llm summary: {user_text}"


class SlowLlmProgressiveContextEngine(ContextEngine):
    def __init__(self):
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.threshold_tokens = 1_000_000
        self.context_length = 100_000
        self.compression_count = 0
        self.applied_calls = 0

    @property
    def name(self):
        return "progressive-context-e2e"

    def update_from_response(self, usage):
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None):
        return messages

    def should_prepare_async_compression(self, prompt_tokens=None, messages=None):
        return bool(prompt_tokens and prompt_tokens >= 10_000)

    def prepare_async_compression(self, messages, current_tokens=None, focus_topic=None):
        summary = fake_llm_compress(
            messages,
            current_tokens=current_tokens,
            focus_topic=focus_topic,
        )
        return ContextCompressionCandidate(
            messages=[
                {"role": "user", "content": "[slow llm compressed context]\\n" + summary},
                {"role": "assistant", "content": "slow llm summary ready"},
            ],
        )

    def on_async_compression_applied(self, candidate, **kwargs):
        self.applied_calls += 1


_engine = SlowLlmProgressiveContextEngine()


def register(ctx):
    ctx.register_context_engine(_engine)
""",
        encoding="utf-8",
    )


def _write_config(hermes_home):
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(
        f"""
context:
  engine: {PLUGIN_NAME}
model:
  context_length: 100000
plugins:
  enabled:
    - {PLUGIN_NAME}
""",
        encoding="utf-8",
    )


def _mock_response(content, usage=None):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="test/model",
        usage=SimpleNamespace(**usage) if usage else None,
    )


def _wait_for_pending(agent, timeout=2):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        candidate = getattr(agent, "_pending_async_context_candidate", None)
        if candidate is not None:
            return candidate
        time.sleep(0.01)
    return None


def _make_tool_defs(*names):
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def test_general_plugin_context_engine_async_compression_and_tool_transform_e2e(
    tmp_path,
    monkeypatch,
):
    hermes_home = tmp_path / "hermes_home"
    _write_config(hermes_home)
    _write_progressive_context_plugin(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import hermes_cli.plugins as plugins_mod

    monkeypatch.setattr(plugins_mod, "_plugin_manager", None)

    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("terminal")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent.context_compressor.name == PLUGIN_NAME

    from tools.registry import registry
    import model_tools

    monkeypatch.setattr(
        registry,
        "dispatch",
        lambda name, args, **kwargs: '{"raw": "tool-output"}',
    )
    monkeypatch.setattr(model_tools, "_READ_SEARCH_TOOLS", frozenset())

    transformed = model_tools.handle_function_call(
        "terminal",
        {"command": "cat large.log"},
        task_id="task",
        session_id="session",
        tool_call_id="call",
        skip_pre_tool_call_hook=True,
    )
    assert transformed == 'plugin-transform:{"raw": "tool-output"}'

    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            "first answer",
            usage={"prompt_tokens": 25_000, "completion_tokens": 10},
        ),
        _mock_response(
            "second answer",
            usage={"prompt_tokens": 2_000, "completion_tokens": 10},
        ),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        first = agent.run_conversation("first ask")
        assert _wait_for_pending(agent) is not None

        second = agent.run_conversation(
            "second ask",
            conversation_history=first["messages"],
        )

    assert first["completed"] is True
    assert second["completed"] is True
    assert second["messages"][0]["content"] == "plugin compressed context"
    assert second["messages"][1]["content"] == "plugin compressed answer"
    assert second["messages"][-2]["content"] == "second ask"
    assert second["messages"][-1]["content"] == "second answer"
    assert not any(msg.get("content") == "first ask" for msg in second["messages"])
    assert agent.context_compressor.prepare_calls == 1
    assert agent.context_compressor.applied_calls == 1


def test_general_plugin_async_compression_can_call_fake_llm(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_home"
    _write_config(hermes_home)
    _write_llm_progressive_context_plugin(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import hermes_cli.plugins as plugins_mod

    monkeypatch.setattr(plugins_mod, "_plugin_manager", None)

    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent.context_compressor.name == PLUGIN_NAME

    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            "first answer",
            usage={"prompt_tokens": 25_000, "completion_tokens": 10},
        ),
        _mock_response(
            "second answer",
            usage={"prompt_tokens": 2_000, "completion_tokens": 10},
        ),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        first = agent.run_conversation("first ask")
        assert _wait_for_pending(agent) is not None

        second = agent.run_conversation(
            "second ask",
            conversation_history=first["messages"],
        )

    assert first["completed"] is True
    assert second["completed"] is True
    assert agent.context_compressor.applied_calls == 1
    assert second["messages"][0]["content"].startswith("[llm compressed context]")
    assert "fake llm summary: first ask" in second["messages"][0]["content"]
    assert second["messages"][1]["content"] == "llm summary ready"
    assert second["messages"][-2]["content"] == "second ask"
    assert second["messages"][-1]["content"] == "second answer"

    plugin_manager = plugins_mod.get_plugin_manager()
    plugin_module = plugin_manager._plugins[PLUGIN_NAME].module
    assert len(plugin_module.LLM_CALLS) == 1
    call = plugin_module.LLM_CALLS[0]
    assert call["current_tokens"] == 25_000
    assert call["messages"][0]["content"] == "first ask"


def test_general_plugin_async_compression_can_apply_after_intervening_turn(
    tmp_path,
    monkeypatch,
):
    hermes_home = tmp_path / "hermes_home"
    _write_config(hermes_home)
    _write_slow_llm_progressive_context_plugin(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import hermes_cli.plugins as plugins_mod

    monkeypatch.setattr(plugins_mod, "_plugin_manager", None)

    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent.context_compressor.name == PLUGIN_NAME

    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            "first answer",
            usage={"prompt_tokens": 25_000, "completion_tokens": 10},
        ),
        _mock_response(
            "second answer",
            usage={"prompt_tokens": 25_000, "completion_tokens": 10},
        ),
        _mock_response(
            "third answer",
            usage={"prompt_tokens": 2_000, "completion_tokens": 10},
        ),
    ]

    plugin_manager = plugins_mod.get_plugin_manager()
    plugin_module = plugin_manager._plugins[PLUGIN_NAME].module

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        try:
            first = agent.run_conversation("first ask")
            assert plugin_module.LLM_STARTED.wait(timeout=2)
            assert _wait_for_pending(agent, timeout=0.1) is None

            second = agent.run_conversation(
                "second ask",
                conversation_history=first["messages"],
            )
            assert _wait_for_pending(agent, timeout=0.1) is None

            plugin_module.LLM_RELEASE.set()
            assert _wait_for_pending(agent) is not None

            third = agent.run_conversation(
                "third ask",
                conversation_history=second["messages"],
            )
        finally:
            plugin_module.LLM_RELEASE.set()

    assert first["completed"] is True
    assert second["completed"] is True
    assert third["completed"] is True
    assert agent.context_compressor.applied_calls == 1

    assert second["messages"][0]["content"] == "first ask"
    assert second["messages"][1]["content"] == "first answer"
    assert second["messages"][-2]["content"] == "second ask"
    assert second["messages"][-1]["content"] == "second answer"

    assert third["messages"][0]["content"].startswith("[slow llm compressed context]")
    assert "slow fake llm summary: first ask" in third["messages"][0]["content"]
    assert third["messages"][1]["content"] == "slow llm summary ready"
    assert any(msg.get("content") == "second ask" for msg in third["messages"])
    assert any(msg.get("content") == "second answer" for msg in third["messages"])
    assert third["messages"][-2]["content"] == "third ask"
    assert third["messages"][-1]["content"] == "third answer"
    assert not any(msg.get("content") == "first ask" for msg in third["messages"])
    assert not any(msg.get("content") == "first answer" for msg in third["messages"])

    assert len(plugin_module.LLM_CALLS) == 1
    call = plugin_module.LLM_CALLS[0]
    assert call["current_tokens"] == 25_000
    assert call["messages"][0]["content"] == "first ask"

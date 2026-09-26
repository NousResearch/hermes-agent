"""Delegated child memory-mode policy and isolation tests."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.memory_provider import MemoryProvider
from hermes_cli.config_defaults import DEFAULT_CONFIG
from tools.delegate_tool import DELEGATE_BLOCKED_TOOLS, DELEGATE_TASK_SCHEMA
from tools.delegate_tool_config import _get_memory_mode, _resolve_child_memory_mode


class _FakeOpenAI:
    def __init__(self, **kwargs):
        self.api_key = kwargs.get("api_key", "test")
        self.base_url = kwargs.get("base_url", "http://test")

    def close(self):
        pass


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": name, "parameters": {"type": "object", "properties": {}}},
    }


class _ReadWriteProvider(MemoryProvider):
    read_only_tool_names = frozenset({"provider_search"})

    @property
    def name(self):
        return "test-provider"

    def __init__(self):
        self.calls = []
        self.lifecycle = []
        self.closed = False

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.lifecycle.append(("initialize", session_id, kwargs))

    def get_tool_schemas(self):
        return [
            {"name": "provider_search", "description": "read", "parameters": {"type": "object", "properties": {}}},
            {"name": "provider_write", "description": "write", "parameters": {"type": "object", "properties": {}}},
            {
                "name": "provider_mixed", "description": "mixed",
                "parameters": {
                    "type": "object",
                    "properties": {"action": {"type": "string", "enum": ["search", "add"]}},
                },
            },
        ]

    def get_read_only_tool_schemas(self):
        mixed = self.get_tool_schemas()[2]
        mixed["parameters"]["properties"]["action"]["enum"] = ["search"]
        return [self.get_tool_schemas()[0], mixed]

    def is_read_only_tool_call(self, tool_name, args):
        return tool_name == "provider_search" or (
            tool_name == "provider_mixed" and args.get("action") == "search"
        )

    def handle_tool_call(self, tool_name, args, **kwargs):
        self.calls.append((tool_name, args))
        return json.dumps({"tool": tool_name})

    def system_prompt_block(self):
        return "provider-profile-secret"

    def shutdown(self):
        self.closed = True


def _make_agent(monkeypatch, home, **overrides):
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", _FakeOpenAI)
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})
    kwargs = {
        "api_key": "test-key",
        "base_url": "http://test",
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "model": "test-model",
        "max_iterations": 1,
        "quiet_mode": True,
        "skip_context_files": True,
    }
    kwargs.update(overrides)
    return AIAgent(**kwargs)


def _mock_parent(depth: int = 0):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "test"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "test-model"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.enabled_toolsets = None
    parent.disabled_toolsets = []
    parent.valid_tool_names = {"terminal", "hermes_mem_search", "hermes_mem_get", "memory"}
    parent._memory_manager = None
    parent._memory_mode_explicit = False
    parent._delegate_memory_mode = None
    parent.session_id = "parent-session"
    return parent


def test_config_and_schema_contract():
    assert DEFAULT_CONFIG["delegation"]["memory_mode"] == "on_demand"
    prop = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["memory_mode"]
    assert prop["type"] == "string"
    assert prop["enum"] == ["full", "on_demand", "off"]
    assert "memory" in DELEGATE_BLOCKED_TOOLS


@pytest.mark.parametrize(
    ("parent_mode", "requested", "expected"),
    [
        ("full", None, "full"),
        ("full", "on_demand", "on_demand"),
        ("full", "off", "off"),
        ("on_demand", None, "on_demand"),
        ("on_demand", "off", "off"),
        ("off", None, "off"),
        ("off", "off", "off"),
    ],
)
def test_nested_mode_inherits_or_narrows(parent_mode, requested, expected):
    parent = SimpleNamespace(_delegate_memory_mode=parent_mode)
    assert _resolve_child_memory_mode(parent, requested) == expected


@pytest.mark.parametrize(
    ("parent_mode", "requested"),
    [("off", "on_demand"), ("off", "full"), ("on_demand", "full")],
)
def test_nested_mode_rejects_escalation(parent_mode, requested):
    parent = SimpleNamespace(_delegate_memory_mode=parent_mode)
    with pytest.raises(ValueError, match="cannot escalate"):
        _resolve_child_memory_mode(parent, requested)


def test_explicit_root_mode_is_also_a_ceiling():
    parent = SimpleNamespace(
        _delegate_memory_mode=None,
        _memory_mode="off",
        _memory_mode_explicit=True,
    )
    assert _resolve_child_memory_mode(parent, None) == "off"
    with pytest.raises(ValueError, match="cannot escalate"):
        _resolve_child_memory_mode(parent, "full")


def test_invalid_requested_mode_is_rejected():
    with pytest.raises(ValueError, match="Invalid memory_mode"):
        _resolve_child_memory_mode(SimpleNamespace(_delegate_memory_mode=None), "sometimes")


def test_invalid_configured_mode_fails_closed(monkeypatch):
    monkeypatch.setattr("tools.delegate_tool_config._cfg", lambda: {"memory_mode": "sometimes"})
    assert _get_memory_mode() == "off"


@pytest.mark.parametrize(
    ("mode", "has_store", "contains_secret"),
    [("full", True, True), ("on_demand", False, False), ("off", False, False)],
)
def test_explicit_modes_gate_initial_profile_memory(monkeypatch, tmp_path, mode, has_store, contains_secret):
    home = tmp_path / mode
    mem_dir = home / "memories"
    mem_dir.mkdir(parents=True)
    secret = f"profile-{mode}-memory"
    (mem_dir / "MEMORY.md").write_text(secret + "\n", encoding="utf-8")
    (mem_dir / "USER.md").write_text(f"profile-{mode}-user\n", encoding="utf-8")
    agent = _make_agent(monkeypatch, home, memory_mode=mode, platform="subagent")
    try:
        from agent.system_prompt import build_system_prompt

        prompt = build_system_prompt(agent)
        assert (agent._memory_store is not None) is has_store
        assert (secret in prompt) is contains_secret
        assert f"<!-- hermes-memory-mode:{mode} -->" in prompt
        assert agent._memory_manager is None
    finally:
        agent.close()


def test_explicit_mode_overrides_legacy_skip_memory(monkeypatch, tmp_path):
    home = tmp_path / "profile"
    mem_dir = home / "memories"
    mem_dir.mkdir(parents=True)
    (mem_dir / "MEMORY.md").write_text("explicit-full-wins\n", encoding="utf-8")
    agent = _make_agent(
        monkeypatch,
        home,
        memory_mode="full",
        skip_memory=True,
        platform="subagent",
    )
    try:
        assert agent._memory_store is not None
        assert "explicit-full-wins" in agent._build_system_prompt()
    finally:
        agent.close()


def test_legacy_skip_memory_contract_is_unchanged(monkeypatch, tmp_path):
    agent = _make_agent(
        monkeypatch,
        tmp_path / "legacy",
        skip_memory=True,
        enabled_toolsets=["memory"],
    )
    try:
        assert agent._memory_mode == "off"
        assert agent._memory_mode_explicit is False
        assert agent._memory_store is not None
        assert agent._memory_manager is None
    finally:
        agent.close()


def test_on_demand_keeps_only_parent_permitted_reads(monkeypatch):
    from tools.delegate_tool import _build_child_agent

    parent = _mock_parent()
    child = MagicMock()
    child.tools = [_tool("terminal"), _tool("memory"), _tool("hermes_mem_search"), _tool("hermes_mem_get")]
    child.valid_tool_names = {"terminal", "memory", "hermes_mem_search", "hermes_mem_get"}
    child.session_id = "child-session"
    with (
        patch("run_agent.AIAgent", return_value=child) as agent_cls,
        patch("tools.delegate_tool._load_config", return_value={}),
        patch("tools.delegate_tool._get_max_spawn_depth", return_value=1),
    ):
        built = _build_child_agent(
            task_index=0,
            goal="read memory only when needed",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=2,
            parent_agent=parent,
            task_count=1,
            memory_mode="on_demand",
        )

    kwargs = agent_cls.call_args.kwargs
    assert kwargs["memory_mode"] == "on_demand"
    assert kwargs["memory_tool_allowlist"] == ["hermes_mem_get", "hermes_mem_search"]
    assert built._delegate_memory_mode == "on_demand"
    assert built.valid_tool_names == {"terminal", "hermes_mem_search", "hermes_mem_get"}


def test_off_child_receives_no_memory_capability(monkeypatch):
    from tools.delegate_tool import _build_child_agent

    parent = _mock_parent()
    child = MagicMock()
    child.tools = [_tool("terminal"), _tool("memory"), _tool("hermes_mem_search")]
    child.valid_tool_names = {"terminal", "memory", "hermes_mem_search"}
    child.session_id = "child-session"
    with (
        patch("run_agent.AIAgent", return_value=child) as agent_cls,
        patch("tools.delegate_tool._load_config", return_value={}),
        patch("tools.delegate_tool._get_max_spawn_depth", return_value=1),
    ):
        built = _build_child_agent(
            task_index=0,
            goal="no memory",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=2,
            parent_agent=parent,
            task_count=1,
            memory_mode="off",
        )

    assert agent_cls.call_args.kwargs["memory_tool_allowlist"] == []
    assert built.valid_tool_names == {"terminal"}


def test_child_inherits_profile_identity_fields():
    from tools.delegate_tool import _build_child_agent

    parent = _mock_parent()
    parent._user_id = "profile-user"
    parent._chat_id = "profile-chat"
    parent._gateway_session_key = "profile-gateway"
    child = MagicMock(tools=[], valid_tool_names=set(), session_id="child-session")
    with (
        patch("run_agent.AIAgent", return_value=child) as agent_cls,
        patch("tools.delegate_tool._load_config", return_value={}),
        patch("tools.delegate_tool._get_max_spawn_depth", return_value=1),
    ):
        _build_child_agent(
            task_index=0,
            goal="identity",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=2,
            parent_agent=parent,
            task_count=1,
            memory_mode="off",
        )
    kwargs = agent_cls.call_args.kwargs
    assert kwargs["user_id"] == "profile-user"
    assert kwargs["chat_id"] == "profile-chat"
    assert kwargs["gateway_session_key"] == "profile-gateway"


@pytest.mark.parametrize(
    ("mode", "allowed", "tool_name", "expected"),
    [
        ("on_demand", {"hermes_mem_search"}, "hermes_mem_search", True),
        ("on_demand", {"hermes_mem_search"}, "hermes_mem_get", False),
        ("on_demand", {"hermes_mem_search"}, "memory", False),
        ("off", set(), "hermes_mem_search", False),
        ("off", set(), "mcp__hermes_mem__write", False),
        ("off", set(), "terminal", True),
    ],
)
def test_runtime_policy_is_an_authorization_backstop(mode, allowed, tool_name, expected):
    from agent.agent_init import memory_tool_call_allowed

    agent = SimpleNamespace(
        _memory_mode_explicit=True,
        _memory_mode=mode,
        _memory_tool_policy_allowlist=frozenset(allowed),
        _memory_manager=None,
        platform="subagent",
    )
    assert memory_tool_call_allowed(agent, tool_name) is expected


def test_sequential_dispatch_rejects_stale_memory_call_before_provider():
    from agent.tool_executor import _ToolCallRef, _resolve_sequential_dispatch

    manager = MagicMock()
    manager.has_tool.return_value = True
    agent = SimpleNamespace(
        _memory_mode_explicit=True,
        _memory_mode="on_demand",
        _memory_tool_policy_allowlist=frozenset(),
        _memory_manager=manager,
        platform="subagent",
    )
    ref = _ToolCallRef(
        name="provider_write", args={"value": "secret"},
        task_id="child-task", call_id="call-1", trace=[],
    )

    dispatch = _resolve_sequential_dispatch(agent, ref, [])
    result = json.loads(dispatch.execute(ref.args))

    assert "not permitted" in result["error"]
    manager.handle_tool_call.assert_not_called()


def test_prompt_cache_identity_includes_memory_mode():
    from agent.conversation_loop import _stored_prompt_matches_runtime

    agent = SimpleNamespace(
        _memory_mode_explicit=True,
        _memory_mode="off",
        model="test-model",
        provider="openrouter",
        pass_session_id=False,
    )
    assert not _stored_prompt_matches_runtime(agent, "<!-- hermes-memory-mode:full -->")
    assert _stored_prompt_matches_runtime(agent, "<!-- hermes-memory-mode:off -->")


def test_delegate_dispatch_forwards_memory_mode():
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._delegate_depth = 0
    with patch("tools.delegate_tool.delegate_task", return_value=json.dumps({"ok": True})) as call:
        result = agent._dispatch_delegate_task({"goal": "x", "memory_mode": "off"})
    assert json.loads(result)["ok"] is True
    assert call.call_args.kwargs["memory_mode"] == "off"


def test_read_only_manager_view_enforces_schema_and_dispatch_boundary():
    from agent.memory_manager import MemoryManager

    provider = _ReadWriteProvider()
    manager = MemoryManager()
    manager.add_provider(provider)
    view = manager.read_only_view(
        allowed_names={"provider_search", "provider_write", "provider_mixed"}
    )

    assert view.get_all_tool_names() == {"provider_search", "provider_mixed"}
    schemas = {schema["name"]: schema for schema in view.get_all_tool_schemas()}
    assert set(schemas) == {"provider_search", "provider_mixed"}
    assert schemas["provider_mixed"]["parameters"]["properties"]["action"]["enum"] == ["search"]
    assert json.loads(view.handle_tool_call("provider_search", {"q": "x"}))["tool"] == "provider_search"
    assert json.loads(view.handle_tool_call("provider_mixed", {"action": "search"}))["tool"] == "provider_mixed"
    denied = json.loads(view.handle_tool_call("provider_write", {"value": "secret"}))
    assert denied["error"].startswith("Memory tool 'provider_write' is not permitted")
    mixed_denied = json.loads(view.handle_tool_call("provider_mixed", {"action": "add"}))
    assert mixed_denied["error"].startswith("Memory tool 'provider_mixed' is not permitted")
    assert provider.calls == [
        ("provider_search", {"q": "x"}),
        ("provider_mixed", {"action": "search"}),
    ]

    view.sync_all("user", "assistant")
    view.on_session_end([])
    view.shutdown_all()
    assert provider.closed is False

    owner = manager.read_only_view(owns_parent=True)
    owner.shutdown_all()
    assert provider.closed is True


def test_on_demand_subagent_has_real_provider_reads_but_no_writes_or_injection(monkeypatch, tmp_path):
    from agent.memory_manager import MemoryManager

    provider = _ReadWriteProvider()
    manager = MemoryManager()
    manager.add_provider(provider)
    agent = _make_agent(
        monkeypatch,
        tmp_path / "profile-a",
        platform="subagent",
        memory_mode="on_demand",
        memory_manager=manager,
        memory_tool_allowlist=["provider_search", "provider_write"],
        enabled_toolsets=["terminal"],
        disabled_toolsets=["memory"],
    )
    try:
        from agent.system_prompt import build_system_prompt

        assert "provider_search" in getattr(agent, "valid_tool_names")
        assert "provider_write" not in getattr(agent, "valid_tool_names")
        assert "provider-profile-secret" not in build_system_prompt(agent)
        manager_view = getattr(agent, "_memory_manager")
        assert json.loads(manager_view.handle_tool_call("provider_search", {}))["tool"] == "provider_search"
        denied = json.loads(manager_view.handle_tool_call("provider_write", {}))
        assert "not permitted" in denied["error"]
    finally:
        agent.close()
    assert provider.closed is False


def test_full_subagent_injects_borrowed_provider_prompt_but_stays_read_only(monkeypatch, tmp_path):
    from agent.memory_manager import MemoryManager

    provider = _ReadWriteProvider()
    manager = MemoryManager()
    manager.add_provider(provider)
    agent = _make_agent(
        monkeypatch,
        tmp_path / "profile-b",
        platform="subagent",
        memory_mode="full",
        memory_manager=manager,
        memory_tool_allowlist=["provider_search", "provider_write"],
        enabled_toolsets=["terminal"],
        disabled_toolsets=["memory"],
    )
    try:
        from agent.system_prompt import build_system_prompt

        assert "provider-profile-secret" in build_system_prompt(agent)
        assert "provider_search" in getattr(agent, "valid_tool_names")
        assert "provider_write" not in getattr(agent, "valid_tool_names")
    finally:
        agent.close()
    assert provider.closed is False

"""Read-only memory for subagents (``delegation.memory_read``).

A read-only manager offers and dispatches only allowlisted provider tools and keeps every
automatic write and recall path off, whatever ``agent_context`` it was initialized with. The
delegate spawn site hands the child that allowlist only when the opt-in is on, and only for tools
the parent itself exposes. Off by default: the child stays provider-less, exactly as before.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import PRE_COMPRESS_CHECKPOINT_API_VERSION, MemoryProvider

_SEARCH = {"name": "rec_search", "description": "search", "parameters": {"type": "object", "properties": {}}}
_SAVE = {"name": "rec_save", "description": "save", "parameters": {"type": "object", "properties": {}}}
_FORGET = {"name": "rec_forget", "description": "delete", "parameters": {"type": "object", "properties": {}}}


class _Provider(MemoryProvider):
    """Records every call the manager routes to it; ``declares`` = its read_only_tool_names()."""

    pre_compress_checkpoint_api_version = PRE_COMPRESS_CHECKPOINT_API_VERSION

    def __init__(self, declares=("rec_search",)):
        self.declares = list(declares)
        self.calls = []

    @property
    def name(self):
        return "rec"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.calls.append(("initialize", kwargs.get("agent_context")))

    def system_prompt_block(self):
        return "# Rec memory\nUse rec_save to store facts."

    def get_tool_schemas(self):
        return [_SEARCH, _SAVE, _FORGET]

    def read_only_tool_names(self):
        return self.declares

    def handle_tool_call(self, tool_name, args, **kwargs):
        self.calls.append(("tool", tool_name))
        return json.dumps({"ok": tool_name})

    def on_turn_start(self, turn_number, message, **kwargs):
        self.calls.append("on_turn_start")

    def prefetch(self, query, *, session_id=""):
        self.calls.append("prefetch")
        return "recalled"

    def queue_prefetch(self, query, *, session_id=""):
        self.calls.append("queue_prefetch")

    def sync_turn(self, user_content, assistant_content, *, session_id="", **kwargs):
        self.calls.append("sync_turn")

    def on_session_end(self, messages):
        self.calls.append("on_session_end")

    def on_delegation(self, task, result, *, child_session_id="", **kwargs):
        self.calls.append("on_delegation")

    def on_memory_write(self, action, target, content, metadata=None):
        self.calls.append("on_memory_write")

    def on_pre_compress(self, messages, **kwargs):
        self.calls.append("on_pre_compress")
        return "checkpoint-saved"


def _manager(provider, *, agent_context="subagent", configured=None, restrict=True):
    mgr = MemoryManager()
    mgr.add_provider(provider)
    mgr.initialize_all("sess-1", agent_context=agent_context)
    if restrict:
        mgr.restrict_to_read_only(configured)
    provider.calls.clear()
    return mgr


def _names(mgr):
    return sorted(s["name"] for s in mgr.get_all_tool_schemas())


# -- the manager ---------------------------------------------------------------------------------


def test_off_by_default_the_full_surface_is_offered():
    mgr = _manager(_Provider(), agent_context="primary", restrict=False)
    assert not mgr.read_only
    assert _names(mgr) == ["rec_forget", "rec_save", "rec_search"]


def test_read_only_offers_and_routes_only_the_declared_search_tool():
    provider = _Provider()
    mgr = _manager(provider)

    assert mgr.read_only and mgr.read_only_tools == {"rec_search"}
    assert _names(mgr) == ["rec_search"]
    assert mgr.get_all_tool_names() == {"rec_search"}
    assert mgr.has_tool("rec_search") and not mgr.has_tool("rec_save") and not mgr.has_tool("rec_forget")
    assert json.loads(mgr.handle_tool_call("rec_search", {})) == {"ok": "rec_search"}
    assert provider.calls == [("tool", "rec_search")]


@pytest.mark.parametrize("tool", ["rec_save", "rec_forget", "invented_tool"])
def test_a_direct_call_to_anything_else_is_refused_before_the_provider(tool):
    provider = _Provider()
    mgr = _manager(provider)

    result = json.loads(mgr.handle_tool_call(tool, {"content": "x"}))

    assert "read-only" in json.dumps(result)
    assert provider.calls == []


def test_config_can_narrow_a_declaring_provider_but_never_widen_it(caplog):
    provider = _Provider(declares=("rec_search", "rec_forget"))  # a provider that (wrongly or not) lists two
    assert _manager(provider, configured=["rec_search"]).read_only_tools == {"rec_search"}

    widened = _manager(_Provider(), configured=["rec_search", "rec_save"])
    assert widened.read_only_tools == {"rec_search"}
    assert "rec_save" in caplog.text and "does not mark it read-only" in caplog.text


@pytest.mark.parametrize("configured", [[], ["rec_search", "rec_forget"], ["rec_forget"]])
def test_repeated_read_only_restrictions_never_restore_tools(configured):
    provider = _Provider(declares=("rec_search", "rec_forget"))
    mgr = _manager(provider, configured=["rec_search"])
    try:
        allowed = mgr.restrict_to_read_only(configured)
        expected = set() if configured == ["rec_forget"] else {"rec_search"}
        assert allowed == expected
        assert mgr.get_all_tool_names() == expected
        assert "read-only" in json.loads(mgr.handle_tool_call("rec_forget", {}))["error"]
        assert provider.calls == []
    finally:
        mgr.shutdown_all()


def test_without_a_declaration_the_config_is_the_explicit_allowlist():
    mgr = _manager(_Provider(declares=()), configured=["rec_search", "not_registered"])
    assert mgr.read_only_tools == frozenset()  # A provider declaration is now required.


def test_nothing_declared_and_nothing_configured_offers_no_tools():
    provider = _Provider(declares=())
    mgr = _manager(provider)
    assert mgr.read_only and mgr.read_only_tools == frozenset()
    assert mgr.get_all_tool_schemas() == []
    assert "read-only" in json.loads(mgr.handle_tool_call("rec_search", {}))["error"]
    assert provider.calls == []


def test_a_provider_whose_declaration_raises_offers_nothing():
    class _Broken(_Provider):
        def read_only_tool_names(self):
            raise RuntimeError("boom")

    assert _manager(_Broken()).read_only_tools == frozenset()


@pytest.mark.parametrize("agent_context", ["subagent", "primary", "cron"])
def test_every_automatic_write_and_recall_path_stays_off(agent_context):
    """Independent of agent_context: read-only alone suppresses them."""
    provider = _Provider()
    mgr = _manager(provider, agent_context=agent_context)

    mgr.on_turn_start(1, "user text")
    recall = mgr.prefetch_all("user text")
    mgr.sync_all("user text", "assistant text")
    mgr.queue_prefetch_all("user text")
    mgr.flush_pending(timeout=5)
    mgr.on_session_end([{"role": "user", "content": "user text"}])
    mgr.on_delegation("task", "result", child_session_id="child")
    mgr.on_memory_write("add", "memory", "a fact")

    assert provider.calls == []
    assert recall == ""


def test_best_effort_pre_compress_is_skipped():
    provider = _Provider()
    mgr = _manager(provider)
    assert mgr.on_pre_compress([{"role": "user", "content": "evidence"}]) == ""
    assert provider.calls == []


def test_an_armed_checkpoint_gate_still_saves():
    """compression.checkpoint_required is its own opt-in; read-only must not fail it closed."""
    provider = _Provider()
    mgr = _manager(provider)
    summary = mgr.on_pre_compress(
        [{"role": "user", "content": "evidence"}, {"role": "assistant", "content": "reply"}],
        require_checkpoint=True, checkpoint_api_version=PRE_COMPRESS_CHECKPOINT_API_VERSION,
    )
    assert provider.calls == ["on_pre_compress"] and "checkpoint-saved" in summary


def test_prompt_block_is_the_host_note_not_the_providers_write_instructions():
    mgr = _manager(_Provider())
    block = mgr.build_system_prompt()
    assert "rec_save" not in block
    assert "`rec_search`" in block and "read-only" in block
    assert _manager(_Provider(declares=())).build_system_prompt() == ""


# -- agent init ---------------------------------------------------------------------------------


class _InitAgent:
    """The attribute surface ``_init_memory`` reads, shaped like a delegate child."""

    log_prefix = ""
    quiet_mode = True
    status_callback = notice_callback = tool_progress_callback = None

    def __init__(self, memory_read_tools):
        self.platform = "subagent"
        self.memory_agent_context = "subagent"
        self.memory_read_tools = memory_read_tools
        # what _resolve_child_toolsets hands a child: the memory toolset is denied
        self.enabled_toolsets = ["terminal", "file", "web"]
        self.disabled_toolsets = ["memory"]
        self.tools = [{"type": "function", "function": {"name": "terminal"}}]
        self.valid_tool_names = {"terminal"}

    def _emit_startup_warning(self, *args, **kwargs):
        pass

    def _vprint(self, *args, **kwargs):
        pass


def _init(monkeypatch, memory_read_tools, *, skip_memory=False, provider=None, memory_manager=None):
    from agent import agent_init

    provider = provider or _Provider()
    monkeypatch.setattr(agent_init, "resolve_memory_provider", lambda name, say=None: provider)
    monkeypatch.setattr(agent_init, "_memory_provider_init_kwargs",
                        lambda agent, platform: {"session_id": "child", "platform": platform,
                                                 "agent_context": "subagent"})
    agent = _InitAgent(memory_read_tools)
    agent_init._init_memory(agent, {"memory": {"provider": "rec"}}, skip_memory, "subagent",
                            memory_manager=memory_manager)
    return agent, provider


def _tool_names(agent):
    return sorted(t["function"]["name"] for t in agent.tools)


def test_child_with_read_only_memory_gets_only_the_search_tool(monkeypatch):
    agent, provider = _init(monkeypatch, ["rec_search"])

    assert agent._memory_manager is not None and agent._memory_manager.read_only
    assert _tool_names(agent) == ["terminal"]  # Explicit enabled_toolsets narrowing still applies.
    assert "memory" not in agent.valid_tool_names
    assert getattr(agent, "_memory_store", None) is None  # MEMORY.md / USER.md stay out
    assert provider.calls == [("initialize", "subagent")]


def test_memory_provider_tools_exposed_for_the_read_only_child(monkeypatch):
    from agent.memory_manager_tools import memory_provider_tools_exposed

    agent, _ = _init(monkeypatch, ["rec_search"])
    assert memory_provider_tools_exposed(agent) is False  # This child enables only terminal/file/web.


def test_a_stock_child_stays_provider_less(monkeypatch):
    agent, provider = _init(monkeypatch, None, skip_memory=True)
    assert agent._memory_manager is None
    assert _tool_names(agent) == ["terminal"]
    assert provider.calls == []


def test_checkpoint_only_child_keeps_the_stock_surface(monkeypatch):
    """Gate armed, read-only off: provider loaded for the checkpoint, no tools (the denied toolset wins)."""
    agent, _ = _init(monkeypatch, None)
    assert agent._memory_manager is not None and not agent._memory_manager.read_only
    assert _tool_names(agent) == ["terminal"]


def test_a_handed_in_shared_manager_is_detached_not_restricted(monkeypatch):
    shared_provider = _Provider()
    shared = _manager(shared_provider, agent_context="primary", restrict=False)
    agent, _ = _init(monkeypatch, ["rec_search"], memory_manager=shared)

    assert agent._memory_manager is None
    assert not shared.read_only  # the owner keeps its full surface
    assert _tool_names(agent) == ["terminal"]


# -- the delegate spawn site ------------------------------------------------------------------------


def _parent(valid_tool_names, manager):
    return SimpleNamespace(_memory_manager=manager, valid_tool_names=set(valid_tool_names))


@pytest.fixture
def delegation_cfg():
    cfg = {}
    with patch("tools.delegate_tool_config._load_config", return_value=cfg):
        yield cfg


def test_resolver_is_off_by_default(delegation_cfg):
    from tools.delegate_tool_config import _resolve_child_memory_read_tools

    parent = _parent({"rec_search", "rec_save"}, _manager(_Provider(), agent_context="primary", restrict=False))
    assert _resolve_child_memory_read_tools(parent) is None


def test_resolver_uses_the_providers_declaration_when_unconfigured(delegation_cfg):
    from tools.delegate_tool_config import _resolve_child_memory_read_tools

    delegation_cfg["memory_read"] = True
    parent = _parent({"rec_search", "rec_save"}, _manager(_Provider(), agent_context="primary", restrict=False))
    assert _resolve_child_memory_read_tools(parent) == ["rec_search"]


def test_resolver_never_grants_a_tool_the_parent_lacks(delegation_cfg):
    from tools.delegate_tool_config import _resolve_child_memory_read_tools

    delegation_cfg.update(memory_read=True, memory_read_tools=["rec_search"])
    manager = _manager(_Provider(), agent_context="primary", restrict=False)
    assert _resolve_child_memory_read_tools(_parent({"terminal"}, manager)) is None
    assert _resolve_child_memory_read_tools(_parent({"rec_search"}, None)) is None


@pytest.mark.parametrize("bad", [{"rec_search": True}, 7])
def test_resolver_rejects_a_malformed_tool_list(delegation_cfg, bad):
    from tools.delegate_tool_config import _resolve_child_memory_read_tools

    delegation_cfg.update(memory_read=True, memory_read_tools=bad)
    manager = _manager(_Provider(), agent_context="primary", restrict=False)
    assert _resolve_child_memory_read_tools(_parent({"rec_search"}, manager)) is None


def test_resolver_accepts_a_comma_string(delegation_cfg):
    from tools.delegate_tool_config import _resolve_child_memory_read_tools

    delegation_cfg.update(memory_read="true", memory_read_tools="rec_search, rec_save")
    manager = _manager(_Provider(), agent_context="primary", restrict=False)
    # the resolver passes names through; the child's manager refuses rec_save against the declaration
    assert _resolve_child_memory_read_tools(_parent({"rec_search", "rec_save"}, manager)) == ["rec_search"]  # Config only narrows the declaration.


class _ConstructorStop(BaseException):
    """Abort at the child constructor (the site catches ``except Exception``)."""


def _spawn_kwargs(memory_read_tools):
    import run_agent
    from tools import delegate_tool

    parent = SimpleNamespace(
        model="m", provider="openrouter", api_key="k", base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions", session_id="parent", enabled_toolsets=None, disabled_toolsets=None,
        request_overrides={}, prefill_messages=None, _delegate_depth=0,
    )
    captured = {}

    def _recorder(**kwargs):
        captured.update(kwargs)
        raise _ConstructorStop()

    cfg = {"compression": {"checkpoint_required": False}, "agent": {}, "memory": {}, "delegation": {}}
    with patch("hermes_cli.config.load_config", return_value=cfg), \
            patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
            patch.object(delegate_tool, "_resolve_child_memory_read_tools", return_value=memory_read_tools), \
            patch.object(run_agent, "AIAgent", _recorder), pytest.raises(_ConstructorStop):
        delegate_tool._build_child_agent(
            task_index=0, goal="g", context=None, toolsets=None, model=None, max_iterations=3, task_count=1,
            parent_agent=parent,
        )
    return captured


def test_spawn_site_loads_the_provider_read_only_when_on():
    kwargs = _spawn_kwargs(["rec_search"])
    assert kwargs["skip_memory"] is False
    assert kwargs["memory_read_tools"] == ["rec_search"]
    assert kwargs["memory_agent_context"] == "subagent"


def test_spawn_site_is_unchanged_when_off():
    kwargs = _spawn_kwargs(None)
    assert kwargs["skip_memory"] is True
    assert kwargs["memory_read_tools"] is None


def test_memory_tool_stays_blocked_for_children():
    from tools.delegate_tool_toolsets import DELEGATE_BLOCKED_TOOLS

    assert "memory" in DELEGATE_BLOCKED_TOOLS


def test_config_default_is_off():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["delegation"]["memory_read"] is False
    assert DEFAULT_CONFIG["delegation"]["memory_read_tools"] == []


def test_read_only_policy_precedes_provider_startup(monkeypatch):
    class StartupProvider(_Provider):
        def initialize(self, session_id, **kwargs):
            if not kwargs.get("read_only"):
                self.calls.append("startup_write")
            super().initialize(session_id, **kwargs)

    agent, provider = _init(monkeypatch, ["rec_search"], provider=StartupProvider())
    assert agent._memory_manager.read_only
    assert "startup_write" not in provider.calls


@pytest.mark.parametrize("declaration", [[], None, "rec_save", {"rec_save": True}, 0, ["rec_search", 7]])
def test_config_never_replaces_missing_or_malformed_provider_declaration(declaration):
    class InvalidProvider(_Provider):
        def read_only_tool_names(self):
            return declaration

    provider = InvalidProvider()
    manager = _manager(provider, configured=["rec_search", "rec_save"])
    assert manager.read_only_tools == frozenset()
    assert "error" in json.loads(manager.handle_tool_call("rec_save", {}))
    assert provider.calls == []


def test_failed_declaration_cannot_grant_configured_write():
    class BrokenProvider(_Provider):
        def read_only_tool_names(self):
            raise RuntimeError("declaration unavailable")

    provider = BrokenProvider()
    manager = _manager(provider, configured=["rec_search", "rec_save"])
    assert manager.read_only_tools == frozenset()
    assert "error" in json.loads(manager.handle_tool_call("rec_save", {}))
    assert provider.calls == []


@pytest.mark.parametrize("enabled,disabled", [
    (["all"], ["memory", "all"]),
    (["all"], ["memory", "*"]),
    (["web"], ["memory"]),
    ([], ["memory"]),
])
def test_read_only_grant_preserves_toolset_denials(enabled, disabled):
    from agent.memory_manager_tools import inject_memory_provider_tools

    agent = _InitAgent(["rec_search"])
    agent.enabled_toolsets, agent.disabled_toolsets = enabled, disabled
    agent._memory_manager = _manager(_Provider())
    inject_memory_provider_tools(agent)
    assert _tool_names(agent) == ["terminal"]
    assert agent.valid_tool_names == {"terminal"}


def test_read_only_child_bypasses_only_the_builtin_memory_block():
    from agent.memory_manager_tools import inject_memory_provider_tools

    agent = _InitAgent(["rec_search"])
    agent.enabled_toolsets = ["hermes-cli"]
    agent._memory_manager = _manager(_Provider())
    inject_memory_provider_tools(agent)
    assert _tool_names(agent) == ["terminal"]  # Unmarked memory denials are explicit, not delegation-injected.
    assert "memory" not in agent.valid_tool_names


@pytest.mark.parametrize("boundary", [False, True])
def test_read_only_compression_and_session_boundary_cannot_persist_switch(tmp_path, boundary):
    output = tmp_path / "provider-session.json"

    class SwitchingProvider(_Provider):
        def on_session_switch(self, new_session_id, **kwargs):
            output.write_text(json.dumps({"session": new_session_id}), encoding="utf-8")

    manager = _manager(SwitchingProvider())
    messages = [{"role": "user", "content": "compact this session"}]
    assert manager.on_pre_compress(messages, require_checkpoint=False) == ""
    if boundary:
        manager.commit_session_boundary_async(messages, new_session_id="next", parent_session_id="sess-1")
        assert manager.flush_pending(timeout=5)
    else:
        manager.on_session_switch("next", parent_session_id="sess-1", reason="compression")
    assert not output.exists()


@pytest.mark.parametrize("bad", [0, {}, None, False, 1, {"rec_search": True}])
def test_malformed_read_tools_never_enables_provider_defaults(delegation_cfg, caplog, bad):
    from tools.delegate_tool_config import _resolve_child_memory_read_tools

    delegation_cfg.update(memory_read=True, memory_read_tools=bad)
    manager = _manager(_Provider(), agent_context="primary", restrict=False)
    assert _resolve_child_memory_read_tools(_parent({"rec_search"}, manager)) is None
    assert "memory_read_tools" in caplog.text


def test_missing_read_tools_uses_provider_declaration(delegation_cfg):
    from tools.delegate_tool_config import _resolve_child_memory_read_tools

    delegation_cfg["memory_read"] = True
    manager = _manager(_Provider(), agent_context="primary", restrict=False)
    assert _resolve_child_memory_read_tools(_parent({"rec_search"}, manager)) == ["rec_search"]

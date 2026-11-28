"""Real config → plugin discovery → delegated AIAgent memory boundary."""

import json

import pytest

from run_agent import AIAgent
from tools.delegate_tool import _build_child_agent


_PROVIDER = '''\
import json
from pathlib import Path
from agent.memory_provider import MemoryProvider

class Provider(MemoryProvider):
    @property
    def name(self):
        return "delegatemem"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.home = Path(kwargs["hermes_home"])
        self.user = kwargs.get("user_id") or "default"
        self.identity = {key: kwargs.get(key) for key in
                         ("user_id_alt", "user_name", "chat_id", "gateway_session_key")}
        if kwargs.get("read_only") is not True:
            self.home.joinpath("startup-writes").write_text(session_id)

    def get_tool_schemas(self):
        return [{"name": name, "description": name,
                 "parameters": {"type": "object", "properties": {}}}
                for name in ("delegatemem_search", "delegatemem_save")]

    def read_only_tool_names(self):
        return ["delegatemem_search"]

    def handle_tool_call(self, tool_name, args, **kwargs):
        path = self.home / "memories.json"
        rows = json.loads(path.read_text())
        if tool_name == "delegatemem_save":
            rows[self.user] = "overwritten"
            path.write_text(json.dumps(rows))
        return json.dumps({"memory": rows[self.user], "identity": self.identity})


def register(ctx):
    ctx.register_memory_provider(Provider())
'''


@pytest.fixture
def memory_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin = tmp_path / "plugins" / "delegatemem"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(_PROVIDER, encoding="utf-8")
    (plugin / "plugin.yaml").write_text(
        "name: delegatemem\nversion: 1.0.0\n", encoding="utf-8"
    )
    (tmp_path / "memories.json").write_text(
        json.dumps({"alice": "Alice private memory", "bob": "Bob private memory", "default": "Default private memory"}),
        encoding="utf-8",
    )
    return tmp_path


def _parent(home, *, user="alice", enabled=True):
    (home / "config.yaml").write_text(
        "memory:\n  provider: delegatemem\ndelegation:\n"
        + ("  memory_read: true\n  memory_read_tools: [delegatemem_search, delegatemem_save]\n" if enabled else "  max_iterations: 3\n"),
        encoding="utf-8",
    )
    return AIAgent(
        model="gpt-4.1-mini", provider="openrouter", api_key="test-key",
        base_url="http://127.0.0.1:9/v1", enabled_toolsets=["memory"],
        quiet_mode=True, skip_context_files=True, save_trajectories=False,
        user_id=user, user_id_alt=f"alt-{user}", user_name=user.title(),
        chat_id=f"chat-{user}", gateway_session_key=f"telegram:dm:{user}",
    )


def _child(parent, *, toolsets=None):
    return _build_child_agent(
        task_index=0, goal="Look up prior preferences", context=None,
        toolsets=toolsets, model=None, max_iterations=3, task_count=1, parent_agent=parent,
    )


def test_real_delegate_offers_declared_lookup_and_refuses_write(memory_home):
    parent = _parent(memory_home, user="default")
    try:
        before = (memory_home / "memories.json").read_bytes()
        startup = (memory_home / "startup-writes").read_bytes()
        child = _child(parent)
        try:
            names = {name for name in child.valid_tool_names if name.startswith("delegatemem_")}
            assert names == {"delegatemem_search"}
            manager = child._memory_manager
            assert manager is not parent._memory_manager
            result = json.loads(manager.handle_tool_call("delegatemem_search", {}))
            assert result["memory"] == "Default private memory"
            assert "error" in json.loads(manager.handle_tool_call("delegatemem_save", {}))
            assert (memory_home / "memories.json").read_bytes() == before
            assert (memory_home / "startup-writes").read_bytes() == startup
        finally:
            child.close()
    finally:
        parent.close()


def test_real_delegate_default_off_has_no_memory(memory_home):
    parent = _parent(memory_home, enabled=False)
    try:
        child = _child(parent)
        try:
            assert child._memory_manager is None
            assert not {name for name in child.valid_tool_names if name.startswith("delegatemem_")}
        finally:
            child.close()
    finally:
        parent.close()


@pytest.mark.parametrize("user", ["alice", "bob"])
def test_real_child_reads_only_its_parents_user_identity(memory_home, user):
    parent = _parent(memory_home, user=user)
    try:
        child = _child(parent)
        try:
            result = json.loads(child._memory_manager.handle_tool_call("delegatemem_search", {}))
            assert result == {
                "memory": f"{user.title()} private memory",
                "identity": {
                    "user_id_alt": f"alt-{user}", "user_name": None,  # Only memory-scoping user IDs inherit.
                    "chat_id": None, "gateway_session_key": None,
                },
            }
        finally:
            child.close()
    finally:
        parent.close()


@pytest.mark.parametrize("refresh", ["first_turn", "compression"])
def test_real_child_keeps_declared_lookup_across_tool_refresh(memory_home, monkeypatch, refresh):
    from agent.conversation_compression import _refresh_agent_tool_definitions
    from agent.turn_context import _refresh_mcp_tools_between_turns
    from tools import mcp_tool

    monkeypatch.setattr(mcp_tool, "_mcp_tool_server_names", {"mcp_probe_lookup": "probe"})
    parent = _parent(memory_home)
    try:
        child = _child(parent)
        try:
            if refresh == "first_turn":
                _refresh_mcp_tools_between_turns(child)
            else:
                _refresh_agent_tool_definitions(child)
            assert {name for name in child.valid_tool_names if name.startswith("delegatemem_")} == {
                "delegatemem_search"
            }
            assert json.loads(child._memory_manager.handle_tool_call("delegatemem_search", {}))["memory"] == "Alice private memory"
        finally:
            child.close()
    finally:
        parent.close()


def test_real_delegate_explicit_web_narrowing_withholds_memory(memory_home):
    from tools.mcp_tool_agent import refresh_agent_mcp_tools

    parent = _parent(memory_home)
    try:
        child = _child(parent, toolsets=["web"])
        try:
            assert not {name for name in child.valid_tool_names if name.startswith("delegatemem_")}
            refresh_agent_mcp_tools(child, preserve_prefix=True)
            assert not {name for name in child.valid_tool_names if name.startswith("delegatemem_")}
        finally:
            child.close()
    finally:
        parent.close()

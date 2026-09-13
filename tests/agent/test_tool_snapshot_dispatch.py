"""Snapshot-bound direct dispatch retains route identity outside publication locks."""
from types import SimpleNamespace
import pytest
from model_tools import handle_function_call

class TestSnapshotDispatch:
    def test_snapshot_dispatch_uses_captured_entry_after_releasing_agent_lock(
        self,
        monkeypatch,
    ):
        """The final registry route is immutable and executes outside the lock."""
        import model_tools
        from tools import mcp_tool_agent as mcp_tool

        agent = SimpleNamespace(
            _tool_snapshot_epoch=7,
            _memory_provider_tool_names=set(),
            _context_engine_tool_names=set(),
        )
        entry = SimpleNamespace(
            schema={
                "name": "snapshot_tool",
                "parameters": {"type": "object", "properties": {}},
            },
        )
        agent._tool_registry_routes = {"snapshot_tool": entry}
        dispatch_calls = []

        monkeypatch.setattr(
            model_tools.registry,
            "entry_is_current",
            lambda name, candidate: name == "snapshot_tool" and candidate is entry,
        )

        def _dispatch_entry(name, captured_entry, args, **kwargs):
            acquired = mcp_tool._agent_tools_lock.acquire(timeout=1)
            assert acquired, "handler dispatch retained the agent snapshot lock"
            mcp_tool._agent_tools_lock.release()
            dispatch_calls.append((name, captured_entry, args, kwargs))
            return "captured-result"

        monkeypatch.setattr(
            model_tools.registry,
            "dispatch_entry",
            _dispatch_entry,
        )
        monkeypatch.setattr(
            model_tools.registry,
            "dispatch",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("live registry dispatch must not be used")
            ),
        )

        result = handle_function_call(
            "snapshot_tool",
            {"value": 1},
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
            tool_snapshot_agent=agent,
            expected_tool_snapshot_epoch=7,
        )

        assert result == "captured-result"
        assert dispatch_calls == [
            (
                "snapshot_tool",
                entry,
                {"value": 1},
                {
                    "task_id": None,
                    "session_id": None,
                    "user_task": None,
                },
            ),
        ]


    def test_snapshot_dispatch_change_reaches_bounded_retry(self, monkeypatch):
        """A stale registry route must not be converted to a generic tool error."""
        import model_tools
        from agent.tool_snapshot import ToolSnapshotChangedError

        agent = SimpleNamespace(
            _tool_snapshot_epoch=8,
            _memory_provider_tool_names=set(),
            _context_engine_tool_names=set(),
            _tool_registry_routes={},
        )
        monkeypatch.setattr(
            model_tools.registry,
            "dispatch",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("stale route must not use live dispatch")
            ),
        )

        with pytest.raises(
            ToolSnapshotChangedError,
            match="tool snapshot changed",
        ):
            handle_function_call(
                "snapshot_tool",
                {},
                skip_pre_tool_call_hook=True,
                skip_tool_request_middleware=True,
                tool_snapshot_agent=agent,
                expected_tool_snapshot_epoch=7,
            )




@pytest.mark.parametrize("owner", ["memory_provider", "context_engine", "registry"])
def test_captured_owner_survives_transfer_before_invoke(owner, monkeypatch):
    from agent.inline_tool_executors import InlineToolContext, resolve_invoke_tool_executor
    from agent.tool_snapshot import bind_tool_execution_route, bind_tool_execution_snapshot

    handler = lambda name, args, **kw: (name, args, kw)
    agent = SimpleNamespace(
        _tool_snapshot_epoch=1, _memory_provider_tool_names={"shared"} if owner == "memory_provider" else set(),
        _context_engine_tool_names={"shared"} if owner == "context_engine" else set(),
        _memory_manager=SimpleNamespace(handle_tool_call=handler),
        context_compressor=SimpleNamespace(handle_tool_call=handler), _tool_registry_routes={"shared": object()},
    )
    monkeypatch.setattr("tools.mcp_tool_agent.capture_agent_tool_execution_route",
                        lambda *_args: (owner, handler))
    message = SimpleNamespace(_hermes_tool_snapshot_epoch=1)
    with bind_tool_execution_snapshot(agent, message), bind_tool_execution_route(agent, "shared"):
        agent._memory_provider_tool_names = set()
        agent._context_engine_tool_names = set()
        agent._memory_manager = None
        agent.context_compressor = None
        executor = resolve_invoke_tool_executor(agent, "shared")
        if owner == "registry":
            assert executor is None
        else:
            assert executor is not None
            assert executor(agent, {"old": 1}, InlineToolContext("task", messages=[]))[:2] == ("shared", {"old": 1})


@pytest.mark.parametrize("owner", ["memory_provider", "context_engine"])
def test_missing_owned_handler_cannot_fall_back_to_registry(owner):
    from agent.tool_snapshot import ToolSnapshotChangedError, bind_tool_execution_route, bind_tool_execution_snapshot
    from tools.registry import registry
    from uuid import uuid4

    name = "snapshot_missing_owner_" + uuid4().hex
    registry.register(name=name, toolset="snapshot-test", schema={"name": name}, handler=lambda _args, **_kw: "wrong-owner")
    try:
        agent = SimpleNamespace(
            _tool_snapshot_epoch=1, _tool_registry_routes={}, valid_tool_names={name},
            _memory_provider_tool_names={name} if owner == "memory_provider" else set(),
            _context_engine_tool_names={name} if owner == "context_engine" else set(),
            _memory_manager=SimpleNamespace(handle_tool_call=None),
            context_compressor=SimpleNamespace(handle_tool_call=None),
        )
        with pytest.raises(ToolSnapshotChangedError), bind_tool_execution_snapshot(
            agent, SimpleNamespace(_hermes_tool_snapshot_epoch=1)
        ), bind_tool_execution_route(agent, name):
            pytest.fail("an unadvertised registry route replaced the missing owned handler")
    finally:
        registry.deregister(name)


def test_legacy_direct_registry_extension_keeps_optional_route_map():
    from agent.tool_snapshot import bind_tool_execution_route, bind_tool_execution_snapshot, captured_tool_route
    from tools.registry import registry
    from uuid import uuid4

    name = "snapshot_legacy_" + uuid4().hex
    registry.register(name=name, toolset="snapshot-test", schema={"name": name}, handler=lambda _args, **_kw: "legacy")
    try:
        agent = SimpleNamespace(_tool_snapshot_epoch=1, _tool_registry_routes={}, valid_tool_names={name},
                                _memory_provider_tool_names=set(), _context_engine_tool_names=set())
        with bind_tool_execution_snapshot(agent, SimpleNamespace(_hermes_tool_snapshot_epoch=1)), bind_tool_execution_route(agent, name):
            assert captured_tool_route(name) == ("registry", registry.get_entry(name))
    finally:
        registry.deregister(name)

"""A server-owned deadline must survive the outer executor without widening a sibling batch."""
import json
from types import SimpleNamespace

from tools.registry import registry


def test_registered_deadline_extends_only_its_own_call():
    name = "mcp__deadline_probe__slow"
    from tools.mcp_tool_registration import _register_candidates, _tool_candidates

    tool = SimpleNamespace(name="slow", description="analysis", inputSchema={"type": "object"})
    candidates = _tool_candidates("deadline_probe", [tool], lambda name: True, 1000)
    _register_candidates("deadline_probe", candidates, check_fn=lambda: True, scope=lambda: None, lazy=True,
                         execution_timeout=1000)
    try:
        from agent.tool_dispatch_helpers import _extend_tool_timeout

        assert _extend_tool_timeout(name, 420) == 1000
        assert _extend_tool_timeout("read_file", 420) == 420
        assert _extend_tool_timeout(name, 1200) == 1200
        assert _extend_tool_timeout(name, None) is None
        _register_candidates("deadline_probe", candidates, check_fn=lambda: True, scope=lambda: None, lazy=True)
        assert _extend_tool_timeout(name, 100) == 100
    finally:
        registry.deregister(name)


def test_extended_deadline_is_a_barrier_for_direct_and_bridged_batches():
    # Even a normally parallel-safe tool must not extend a batch containing unrelated work.
    name = "tool_describe"
    old = registry.get_entry(name)
    registry.register(name=name, toolset="deadline-probe", schema={"name": name},
                      handler=lambda args, **kw: "ok", execution_timeout=1000, override=True)
    try:
        from agent.tool_dispatch_helpers import _batch_admission

        for exposed, args in [(name, {}), ("tool_call", {"calls": [{"name": name, "arguments": {}}]})]:
            call = SimpleNamespace(function=SimpleNamespace(name=exposed, arguments=json.dumps(args)))
            assert _batch_admission(call, None) is None
    finally:
        current = registry.get_entry(name)
        assert current is not None
        registry.restore_registration(name, current, old)

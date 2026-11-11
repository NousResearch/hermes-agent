"""Registry-dispatch regression for #84245: scoped projection must follow the
scope-resolved entry's schema, not a second bare-name lookup.

Kept free of ``tools.arg_coercion`` imports so the file collects against a base
that predates the projection helper — the behavior difference is the proof.
"""

import json
from unittest.mock import MagicMock

from tools.registry import ToolEntry, ToolRegistry


def _entry(name, schema, handler):
    return ToolEntry(
        name=name, toolset="test", schema=schema,
        handler=handler, check_fn=None, requires_env=[],
        is_async=False, description="test", emoji="t",
    )


def _schema(additional):
    return {
        "name": "shared",
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "additionalProperties": additional,
        },
    }


def test_dispatch_projects_with_scope_resolved_schema():
    """Two scopes share a tool name with different strictness; dispatch under
    scope A must strip args per A's schema even though the ambient registration
    is looser."""
    reg = ToolRegistry()
    strict_handler = MagicMock(return_value='{"ok": true}')
    loose_handler = MagicMock(return_value='{"ok": true}')
    reg._tools["shared"] = _entry("shared", _schema(additional=True), loose_handler)
    reg._scoped_tools["profile_A"] = {
        "shared": _entry("shared", _schema(additional=False), strict_handler)
    }

    reg.dispatch("shared", {"cmd": "ls", "secret": True}, scope="profile_A")
    sent = strict_handler.call_args[0][0]
    assert "secret" not in sent, (
        "scope-resolved strict schema was not used for projection — "
        "the handler received an argument its schema forbids"
    )

    loose_handler.reset_mock()
    reg.dispatch("shared", {"cmd": "ls", "secret": True})
    sent = loose_handler.call_args[0][0]
    assert "secret" in sent, (
        "ambient loose schema was not used — projection used the wrong scope"
    )


def test_terminal_legacy_notifications_and_wrong_tool_guard_survive_projection(monkeypatch):
    import tools.terminal_tool as terminal
    from tools.registry import registry

    received = []
    monkeypatch.setattr(terminal, "terminal_tool", lambda **kw: received.append(kw) or '{}')

    for legacy in ({"notify_on_complete": True}, {"watch_patterns": ["ready"]}):
        result = json.loads(registry.dispatch(
            "terminal", {"command": "echo ready", "background": True, "force": True, **legacy}))
        assert "error" not in result
        assert all(received[-1][key] == value for key, value in legacy.items())
        assert "force" not in received[-1]

    wrong_tool = json.loads(registry.dispatch("terminal", {"code": "print(1)"}))
    assert "Use execute_code" in wrong_tool["error"]
    assert len(received) == 2


def test_scoped_compatibility_allowlist_does_not_authorize_another_scope():
    reg = ToolRegistry()
    handler = MagicMock(return_value='{}')
    ambient = _entry("shared", _schema(additional=False), handler)
    ambient.compat_args = frozenset({"force"})
    scoped = _entry("shared", _schema(additional=False), handler)
    scoped.compat_args = frozenset({"legacy"})
    reg._tools["shared"] = ambient
    reg._scoped_tools["profile_A"] = {"shared": scoped}

    reg.dispatch("shared", {"cmd": "ls", "legacy": True, "force": True}, scope="profile_A")

    assert handler.call_args[0][0] == {"cmd": "ls", "legacy": True}

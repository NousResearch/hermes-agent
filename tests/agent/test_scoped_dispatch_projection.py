"""Registry-dispatch regression for #84245: scoped projection must follow the
scope-resolved entry's schema, not a second bare-name lookup.

Kept free of ``tools.arg_coercion`` imports so the file collects against a base
that predates the projection helper — the behavior difference is the proof.
"""

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

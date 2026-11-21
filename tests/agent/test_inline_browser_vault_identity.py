"""The inline browser_vault executor hands an overriding handler the registry path's call identity."""

from types import SimpleNamespace

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext


def test_override_handler_sees_session_and_approval_identity(monkeypatch):
    import tools.browser_vault_tool  # registers browser_vault
    from tools import approval_context
    from tools.registry import registry

    seen = {}

    def plugin_handler(args, **kwargs):
        seen.update(kwargs)
        seen["bound_turn"] = approval_context._approval_turn_id.get()
        seen["bound_session"] = approval_context._approval_session_id.get()
        seen["bound_call"] = approval_context._approval_tool_call_id.get()
        return '{"ok": true}'

    monkeypatch.setattr(registry.get_entry("browser_vault"), "handler", plugin_handler)
    agent = SimpleNamespace(session_id="sess-1", _current_turn_id="turn-1", drive_preview_callback=None)
    ctx = InlineToolContext(effective_task_id="task-1", tool_call_id="call-1")

    INLINE_TOOL_EXECUTORS["browser_vault"](agent, {"action": "list"}, ctx)

    assert seen["task_id"] == "task-1"
    assert seen["session_id"] == "sess-1"
    assert seen["conversation_id"]
    assert (seen["bound_turn"], seen["bound_session"], seen["bound_call"]) == ("turn-1", "sess-1", "call-1")
    # The binding is scoped to the call.
    assert approval_context._approval_turn_id.get() == ""


def test_plugin_fork_never_holds_browser_vault():
    # Why the executor forwards no turn_origin: only a plugin fork's worker sets one, and its
    # allowlist drops every pre-registry (inline) tool, browser_vault included.
    from agent.plugin_session_fork import fork_tool_allowlist

    assert "browser_vault" not in fork_tool_allowlist(["browser_vault", "my_plugin_tool"])

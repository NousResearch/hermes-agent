# ABOUTME: Checks argument updates attached to pre-tool approval directives.
# ABOUTME: Runs real plugin callbacks and preserves veto precedence.

from hermes_cli import plugins


def _manager(monkeypatch, callbacks):
    manager = plugins.PluginManager()
    manager._discovered = True
    manager._hooks["pre_tool_call"] = callbacks
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    return manager


def test_approval_argument_updates_compose_without_mutating_input(monkeypatch):
    def require_approval(**kwargs):
        return {"action": "approve", "args": {"path": "reviewed.txt"},
                "message": "Review this write", "rule_key": "reviewed-write"}

    def update_content(**kwargs):
        return {"action": "modify", "args": {"content": "checked content"}}

    _manager(monkeypatch, [require_approval, update_content])
    original = {"path": "requested.txt", "content": "requested content"}
    details = plugins._get_pre_tool_call_directive_details("write_file", original)

    assert (details.action, details.message, details.rule_key) == (
        "approve", "Review this write", "reviewed-write")
    assert details.modified_args == {"path": "reviewed.txt", "content": "checked content"}
    assert original == {"path": "requested.txt", "content": "requested content"}


def test_approval_argument_updates_do_not_override_a_later_veto(monkeypatch):
    def require_approval(**kwargs):
        return {"action": "approve", "args": {"path": "reviewed.txt"}}

    def deny(**kwargs):
        return {"action": "block", "message": "The write is prohibited"}

    _manager(monkeypatch, [require_approval, deny])
    details = plugins._get_pre_tool_call_directive_details(
        "write_file", {"path": "requested.txt", "content": "content"})

    assert (details.action, details.message) == ("block", "The write is prohibited")
    assert details.modified_args == {"path": "reviewed.txt", "content": "content"}

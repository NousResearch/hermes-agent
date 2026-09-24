"""Plugin command invocation is additive and never guesses a conversation owner."""

import pytest


def test_invocation_passes_only_declared_context_without_retrying_handlers():
    from hermes_cli.plugins_command import invoke_plugin_command

    context = {"session_id": "conversation-a", "task_id": "task-a"}
    assert invoke_plugin_command(lambda raw: raw, "status", **context) == "status"

    def narrow(raw, *, session_id):
        return raw, session_id

    assert invoke_plugin_command(narrow, "on", **context) == ("on", "conversation-a")

    def wide(raw, **kwargs):
        return raw, kwargs

    assert invoke_plugin_command(wide, "off", **context) == ("off", context)

    def positional(session_id, /, **kwargs):
        return session_id, kwargs

    assert invoke_plugin_command(positional, "status", **context) == ("status", context)
    calls = []

    def broken(raw, *, task_id):
        calls.append(task_id)
        raise TypeError("inside the handler")

    with pytest.raises(TypeError, match="inside the handler"):
        invoke_plugin_command(broken, "on", **context)
    assert calls == ["task-a"]

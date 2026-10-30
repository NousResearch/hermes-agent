"""A plugin's mandatory consent cannot inherit remembered or unattended approval."""

import json

import pytest

from hermes_cli import plugins
from tools import approval


@pytest.fixture
def policy(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("approvals:\n  mode: 'off'\ncommand_allowlist: []\n")
    manager = plugins.PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    ctx = plugins.PluginContext(plugins.PluginManifest(name="owner-policy", source="test"), manager)
    ctx.register_hook("pre_tool_call", lambda **kw: {
        "action": "approve", "message": "Owner policy requires a fresh human answer",
        "approval_policy": "mandatory", "rule_key": "owner-policy:write",
    })
    # Both remembered consent and yolo would bypass an ordinary approve.
    monkeypatch.setattr(approval, "_yolo_active", lambda: True)
    monkeypatch.setattr(approval, "is_approved", lambda *args: True)
    monkeypatch.setattr(approval, "_present_with_selected_transport", lambda **kw: {"selected": False})
    yield ctx
    manager.unload()


@pytest.mark.parametrize("choice", ["once", "deny", "session", "always", "timeout", "cancelled", None])
def test_mandatory_policy_asks_each_time_and_dispatches_only_fresh_once(policy, monkeypatch, choice):
    import model_tools

    asked, executed = [], []
    original = {"path": "/workspace/draft.txt", "content": "owner-visible draft"}
    monkeypatch.setattr(approval, "_presence", lambda cb=None: (cb, True, False, False))

    def ask(command, description, **flags):
        asked.append((json.loads(command), flags))
        # Mutating the caller's input while awaiting the human cannot change
        # the separate argument snapshot the host will execute.
        original["content"] = "changed after display"
        return choice

    def dispatch(name, args, **kwargs):
        executed.append(dict(args))
        return '{"ok":true}'

    def no_persistence(*args, **kwargs):
        pytest.fail("mandatory consent attempted to persist a grant")

    monkeypatch.setattr(approval, "prompt_dangerous_approval", ask)
    monkeypatch.setattr(approval, "_persist_choice", no_persistence)
    monkeypatch.setattr(model_tools.registry, "dispatch", dispatch)
    for _ in range(2):
        original["content"] = "owner-visible draft"
        result = json.loads(model_tools.handle_function_call(
            "write_file", original, session_id="owner-session", task_id="owner-task"))
        assert ("error" not in result) is (choice == "once")
    assert len(asked) == 2
    assert all(not flags["allow_session"] and not flags["allow_permanent"] for _, flags in asked)
    assert all(item["content"] == "owner-visible draft" for item in executed)
    assert len(executed) == (2 if choice == "once" else 0)


@pytest.mark.parametrize("mode", ["no-human", "veto", "unknown-policy", "ordinary"])
def test_mandatory_failure_veto_and_ordinary_compatibility(policy, monkeypatch, mode):
    import model_tools

    asked, executed = [], []
    monkeypatch.setattr(approval, "_presence", lambda cb=None: (None, False, False, False))
    monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **kw: asked.append(a) or "once")
    monkeypatch.setattr(model_tools.registry, "dispatch", lambda *a, **kw: executed.append(a) or '{"ok":true}')
    if mode == "veto":
        policy.register_hook("pre_tool_call", lambda **kw: {"action": "block", "message": "explicit veto"})
    if mode in {"unknown-policy", "ordinary"}:
        policy._manager._hooks["pre_tool_call"].clear()
        directive = {"action": "approve", "message": "ordinary consent"}
        if mode == "unknown-policy":
            directive["approval_policy"] = "unsupported"
        policy.register_hook("pre_tool_call", lambda **kw: directive)
    result = json.loads(model_tools.handle_function_call(
        "write_file", {"path": "/workspace/draft.txt", "content": "example"},
        session_id="owner-session", task_id="owner-task"))
    assert ("error" not in result) is (mode == "ordinary")
    assert len(executed) == (1 if mode == "ordinary" else 0)
    assert not asked

"""Mandatory policy is independent of unattended modes and delegation callbacks."""
import json

import pytest

from hermes_cli import plugins
from tools import approval


@pytest.mark.parametrize("context", ["cron", "single_query", "delegation"])
def test_automatic_context_cannot_supply_mandatory_once(monkeypatch, context):
    from agent.delegation_context import delegated_child_context
    from contextlib import nullcontext
    from tools.delegate_tool_config import _subagent_auto_approve
    from tools.terminal_tool import set_approval_callback

    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "")
    monkeypatch.setenv("HERMES_EXEC_ASK", "")
    monkeypatch.setenv("HERMES_CRON_SESSION", "1" if context == "cron" else "")
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1" if context == "single_query" else "")
    monkeypatch.setattr(approval, "_yolo_active", lambda: True)
    monkeypatch.setattr(approval, "is_approved", lambda *a: True)
    scope = delegated_child_context() if context == "delegation" else nullcontext()
    set_approval_callback(_subagent_auto_approve)
    try:
        with scope:
            result = approval.request_tool_approval(
                "dummy", "Exact operation", approval_policy="mandatory", effective_args={"value": "A"})
        assert not result["approved"], result
    finally:
        set_approval_callback(None)


@pytest.mark.parametrize("value", [float("nan"), object()])
def test_mandatory_unserializable_arguments_never_reach_gate(monkeypatch, value):
    import model_tools
    manager = plugins.PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    ctx = plugins.PluginContext(plugins.PluginManifest(name="dummy-policy", source="test"), manager)
    ctx.register_hook("pre_tool_call", lambda **kw: {
        "action": "approve", "approval_policy": "mandatory", "message": "Exact operation",
    })
    calls = []
    monkeypatch.setattr(approval, "request_tool_approval", lambda *a, **kw: calls.append(kw))
    try:
        result = json.loads(model_tools.handle_function_call("write_file", {"value": value}))
        assert "not serializable" in result["error"]
        assert not calls
    finally:
        manager.unload()


def test_mandatory_feature_detection_import_is_public(tmp_path):
    from hermes_cli.plugin_validate import validate_plugin_dir
    from hermes_cli.plugins import VALID_TOOL_APPROVAL_POLICIES

    assert "mandatory" in VALID_TOOL_APPROVAL_POLICIES
    plugin_dir = tmp_path / "public_consent"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.yaml").write_text("name: public_consent\nversion: 1.0.0\ndescription: Consent test\n")
    (plugin_dir / "__init__.py").write_text(
        "from hermes_cli.plugins import VALID_TOOL_APPROVAL_POLICIES\n"
        "def register(ctx):\n    pass\n")
    report = validate_plugin_dir(plugin_dir)
    assert report.ok, report.failures
    assert not [w for w in report.warnings if w.startswith("private core imports")]

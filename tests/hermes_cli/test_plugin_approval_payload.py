"""Approval sees the final tool payload."""

import json
import types

import pytest


@pytest.mark.parametrize("sensitive_value", ["839512", "private medical narrative"])
def test_approval_does_not_display_otp_or_arbitrary_text(monkeypatch, sensitive_value):
    from hermes_cli.plugins import _dispatch_pre_tool_call_hooks

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *a, **kw: [
        {"action": "approve", "message": "confirm operation"},
    ])
    seen = {}

    def decide(tool_name, reason, **kwargs):
        seen["reason"] = reason
        return {"approved": False, "message": "declined"}

    monkeypatch.setattr("tools.approval.request_tool_approval", decide)
    _dispatch_pre_tool_call_hooks("write_file", {
        "path": "/safe/report.txt", "otp": "839512", "content": "private medical narrative",
    })
    assert sensitive_value not in seen["reason"]


def test_approval_does_not_display_pin_or_oauth_code(monkeypatch):
    from hermes_cli.plugins import _dispatch_pre_tool_call_hooks

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *a, **kw: [
        {"action": "approve", "message": "confirm login"},
    ])
    seen = {}

    def decide(tool_name, reason, **kwargs):
        seen["reason"] = reason
        return {"approved": False, "message": "declined"}

    monkeypatch.setattr("tools.approval.request_tool_approval", decide)
    _dispatch_pre_tool_call_hooks("http_request", {
        "pin": "836492",
        "url": "https://example.invalid/callback?code=oauth-code-example&state=abc",
    })
    assert "836492" not in seen["reason"]
    assert "oauth-code-example" not in seen["reason"]


def test_approve_gate_receives_final_modified_args(monkeypatch):
    """Approval must describe the post-modifier snapshot, not stale input."""
    from hermes_cli.plugins import _dispatch_pre_tool_call_hooks

    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda hook_name, **kwargs: [
            {"action": "approve", "message": "confirm write"},
            {"action": "modify", "args": {"path": "/safe/final", "client_secret": "secret-value"}},
        ],
    )
    seen = {}

    def _approve(tool_name, reason, **kwargs):
        seen["reason"] = reason
        return {"approved": False, "message": "declined"}

    monkeypatch.setattr("tools.approval.request_tool_approval", _approve)
    block_msg, modified = _dispatch_pre_tool_call_hooks(
        "write_file", {"path": "/unsafe", "content": "x"}
    )

    assert block_msg == "declined"
    assert modified == {"path": "/safe/final", "content": "x", "client_secret": "secret-value"}
    assert "/safe/final" in seen["reason"]
    assert "/unsafe" not in seen["reason"]
    assert "secret-value" not in seen["reason"]
    assert json.loads(seen["reason"].split("(secrets redacted): ", 1)[1])["client_secret"] != "secret-value"

def test_approve_gate_includes_args_even_without_modify(monkeypatch):
    """Approve-only hooks should still show the current payload to the human gate."""
    from hermes_cli.plugins import _dispatch_pre_tool_call_hooks

    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda hook_name, **kwargs: [{"action": "approve", "message": "confirm write"}],
    )
    seen = {}

    def _approve(tool_name, reason, **kwargs):
        seen["reason"] = reason
        return {"approved": False, "message": "declined"}

    monkeypatch.setattr("tools.approval.request_tool_approval", _approve)

    block_msg, modified = _dispatch_pre_tool_call_hooks(
        "write_file", {"path": "/safe/current", "bearer_token": "secret-value"}
    )

    assert block_msg == "declined"
    assert modified is None
    assert "/safe/current" in seen["reason"]
    assert "secret-value" not in seen["reason"]

def test_approve_gate_receives_tool_execution_middleware_args(monkeypatch):
    """Approval must describe the final payload after execution middleware rewrites it."""
    import model_tools

    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda hook_name, **kwargs: [{"action": "approve", "message": "confirm write"}],
    )

    def _middleware(**kwargs):
        return kwargs["next_call"]({
            "path": "/etc/shadow",
            "content": "x",
            "bearer_token": "secret-value",
        })

    manager = types.SimpleNamespace(
        _middleware={"tool_execution": [_middleware]},
        _report_hook_failure=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("hermes_cli.plugins._delivery_manager", lambda: manager)

    seen = {}

    def _approve(tool_name, reason, **kwargs):
        seen["reason"] = reason
        return {"approved": False, "message": "declined"}

    monkeypatch.setattr("tools.approval.request_tool_approval", _approve)
    monkeypatch.setattr(
        model_tools.registry,
        "dispatch",
        lambda *args, **kwargs: pytest.fail("tool executed before approval was resolved"),
    )

    result = model_tools.handle_function_call(
        "write_file", {"path": "/safe/final", "content": "x"}, task_id="task-1"
    )

    assert json.loads(result) == {"error": "declined"}
    assert "/etc/shadow" in seen["reason"]
    assert "/safe/final" not in seen["reason"]
    assert "secret-value" not in seen["reason"]


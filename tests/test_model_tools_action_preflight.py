"""Integration coverage for action_preflight at the model_tools dispatch boundary."""

import json

import model_tools
from tools.action_preflight import SemanticReceipt, TrustedActionDecision, classify_tool_action


def _noop_unrelated_hooks(monkeypatch):
    monkeypatch.setattr(
        "acp_adapter.edit_approval.maybe_require_edit_approval",
        lambda name, args: None,
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda hook_name, **kwargs: [],
    )


def _dispatch_spy(monkeypatch):
    from tools.registry import registry

    dispatched = []

    def _dispatch(name, args, **kwargs):
        dispatched.append((name, args, kwargs))
        return json.dumps({"ok": True, "tool": name})

    monkeypatch.setattr(registry, "dispatch", _dispatch)
    return dispatched


def _enable_action_preflight(monkeypatch):
    monkeypatch.setenv("HERMES_ACTION_PREFLIGHT_ENABLED", "1")


def test_default_action_preflight_is_off_and_documented_by_dispatch(monkeypatch):
    _noop_unrelated_hooks(monkeypatch)
    monkeypatch.delenv("HERMES_ACTION_PREFLIGHT_ENABLED", raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    dispatched = _dispatch_spy(monkeypatch)

    out = model_tools.handle_function_call(
        "send_message",
        {"action": "send", "target": "discord:#ops", "message": "hello"},
        skip_pre_tool_call_hook=True,
    )

    assert json.loads(out) == {"ok": True, "tool": "send_message"}
    assert [call[0] for call in dispatched] == ["send_message"]


def test_config_can_enable_action_preflight_without_env(monkeypatch):
    _noop_unrelated_hooks(monkeypatch)
    monkeypatch.delenv("HERMES_ACTION_PREFLIGHT_ENABLED", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"security": {"action_preflight": {"enabled": True}}},
    )
    dispatched = _dispatch_spy(monkeypatch)

    out = model_tools.handle_function_call(
        "send_message",
        {"action": "send", "target": "discord:#ops", "message": "hello"},
        skip_pre_tool_call_hook=True,
    )

    payload = json.loads(out)
    assert "error" in payload
    assert "approval receipt required" in payload["error"]
    assert dispatched == []


def test_read_only_tool_dispatches_with_active_preflight_without_receipt(monkeypatch):
    _enable_action_preflight(monkeypatch)
    _noop_unrelated_hooks(monkeypatch)
    dispatched = _dispatch_spy(monkeypatch)

    out = model_tools.handle_function_call(
        "read_file",
        {"path": "README.md"},
        skip_pre_tool_call_hook=True,
    )

    assert json.loads(out) == {"ok": True, "tool": "read_file"}
    assert [call[0] for call in dispatched] == ["read_file"]


def test_publish_call_with_active_preflight_fails_closed_before_dispatch(monkeypatch):
    _enable_action_preflight(monkeypatch)
    _noop_unrelated_hooks(monkeypatch)
    dispatched = _dispatch_spy(monkeypatch)

    out = model_tools.handle_function_call(
        "send_message",
        {"action": "send", "target": "discord:#ops", "message": "hello"},
        skip_pre_tool_call_hook=True,
    )

    payload = json.loads(out)
    assert "error" in payload
    assert "approval receipt required" in payload["error"]
    assert dispatched == []


def test_model_supplied_approval_status_does_not_count_as_trusted_decision(monkeypatch):
    _enable_action_preflight(monkeypatch)
    _noop_unrelated_hooks(monkeypatch)
    dispatched = _dispatch_spy(monkeypatch)

    out = model_tools.handle_function_call(
        "send_message",
        {
            "action": "send",
            "target": "discord:#ops",
            "message": "hello",
            "approval_status": "approved",
        },
        skip_pre_tool_call_hook=True,
    )

    payload = json.loads(out)
    assert "error" in payload
    assert "trusted" in payload["error"] or "receipt" in payload["error"]
    assert dispatched == []


def test_trusted_decision_with_matching_payload_hash_reaches_dispatch(monkeypatch):
    _enable_action_preflight(monkeypatch)
    _noop_unrelated_hooks(monkeypatch)
    dispatched = _dispatch_spy(monkeypatch)
    args = {"action": "send", "target": "discord:#ops", "message": "hello"}
    preflight = classify_tool_action("send_message", args)
    decision = TrustedActionDecision(
        receipt=SemanticReceipt.for_preflight(preflight, approved=True),
        source="test-trusted-store",
    )

    out = model_tools.handle_function_call(
        "send_message",
        args,
        trusted_action_decision=decision,
        skip_pre_tool_call_hook=True,
    )

    assert json.loads(out) == {"ok": True, "tool": "send_message"}
    assert [call[0] for call in dispatched] == ["send_message"]


def test_payload_hash_mismatch_after_approval_fails_before_dispatch(monkeypatch):
    _enable_action_preflight(monkeypatch)
    _noop_unrelated_hooks(monkeypatch)
    dispatched = _dispatch_spy(monkeypatch)
    approved_args = {"action": "send", "target": "discord:#ops", "message": "old"}
    mutated_args = {"action": "send", "target": "discord:#ops", "message": "mutated"}
    approved_preflight = classify_tool_action("send_message", approved_args)
    decision = TrustedActionDecision(
        receipt=SemanticReceipt.for_preflight(approved_preflight, approved=True),
        source="test-trusted-store",
    )

    out = model_tools.handle_function_call(
        "send_message",
        mutated_args,
        trusted_action_decision=decision,
        skip_pre_tool_call_hook=True,
    )

    payload = json.loads(out)
    assert "error" in payload
    assert "payload hash mismatch" in payload["error"]
    assert dispatched == []


def test_terminal_hardline_guard_remains_in_terminal_tool_not_replaced(monkeypatch):
    # Default action_preflight is off, so terminal dispatch still reaches the
    # existing tools.approval hardline guard, which must block independently.
    _noop_unrelated_hooks(monkeypatch)
    monkeypatch.delenv("HERMES_ACTION_PREFLIGHT_ENABLED", raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    import tools.terminal_tool  # noqa: F401 - ensure registry registration

    out = model_tools.handle_function_call(
        "terminal",
        {"command": "rm -rf /", "timeout": 1},
        skip_pre_tool_call_hook=True,
    )

    payload = json.loads(out)
    text = json.dumps(payload).lower()
    assert "hardline" in text
    assert "blocked" in text

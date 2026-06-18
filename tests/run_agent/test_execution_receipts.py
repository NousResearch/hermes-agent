"""Tests for Hermes execution receipt construction and emission."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch



def test_emit_tool_execution_receipt_noops_without_listener(monkeypatch):
    from agent.execution_receipts import emit_tool_execution_receipt

    calls = []
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    receipt = emit_tool_execution_receipt(
        session_id="session-1",
        task_id="task-1",
        tool_name="terminal",
        tool_call_id="call-1",
        status="ok",
        duration_ms=3,
        args={"command": "echo secret"},
        result="secret output",
    )

    assert receipt is None
    assert calls == []


def test_emit_tool_execution_receipt_builds_redacted_envelope(monkeypatch):
    from agent.execution_receipts import emit_tool_execution_receipt

    calls = []
    monkeypatch.setattr(
        "hermes_cli.plugins.has_hook",
        lambda name: name == "execution_receipt",
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda name, **kwargs: calls.append((name, kwargs)) or [],
    )

    receipt = emit_tool_execution_receipt(
        session_id="session-1",
        task_id="task-1",
        turn_id="turn-1",
        api_request_id="api-1",
        tool_name="terminal",
        tool_call_id="call-1",
        status="ok",
        duration_ms=3,
        sequence_number=7,
        trace_id="trace-1",
        args={"command": "echo secret", "workdir": "/tmp/private-project"},
        result="secret output",
    )

    assert len(calls) == 1
    assert calls[0][0] == "execution_receipt"
    assert calls[0][1]["receipt"] == receipt
    assert receipt["schema_version"] == "hermes.execution_receipt.v0"
    assert receipt["receipt_type"] == "tool_complete"
    assert receipt["receipt_id"]
    assert receipt["trace_id"] == "trace-1"
    assert receipt["span_id"]
    assert receipt["parent_span_id"] is None
    assert receipt["sequence_number"] == 7
    assert receipt["timestamp"]
    assert receipt["session_id"] == "session-1"
    assert receipt["task_id"] == "task-1"
    assert receipt["turn_id"] == "turn-1"
    assert receipt["api_request_id"] == "api-1"
    assert receipt["tool_call_id"] == "call-1"
    assert receipt["tool_name"] == "terminal"
    assert receipt["status"] == "ok"
    assert receipt["duration_ms"] == 3
    assert receipt["links"] == []
    assert receipt["evidence_gaps"] == []
    assert receipt["redaction_policy_version"] == "execution_receipts.v0"
    assert receipt["redaction_status"] == "ok"
    assert receipt["args"]["redacted"] is True
    assert receipt["result"]["redacted"] is True

    serialized = json.dumps(receipt)
    assert "echo secret" not in serialized
    assert "/tmp/private-project" not in serialized
    assert "secret output" not in serialized


def test_emit_tool_execution_receipt_is_fail_open(monkeypatch):
    from agent.execution_receipts import emit_tool_execution_receipt

    monkeypatch.setattr(
        "hermes_cli.plugins.has_hook",
        lambda name: name == "execution_receipt",
    )

    def boom(*args, **kwargs):
        raise RuntimeError("observer failed")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", boom)

    receipt = emit_tool_execution_receipt(
        session_id="session-1",
        task_id="task-1",
        tool_name="terminal",
        tool_call_id="call-1",
        status="error",
        duration_ms=5,
        args={"command": "false"},
        result="error output",
        error_type="tool_error",
        error_message="failed with token secret-token-123",
    )

    assert receipt is not None
    assert receipt["status"] == "error"
    assert receipt["error_type"] == "tool_error"
    assert receipt["error_message"] is None
    assert receipt["error_message_metadata"] == {
        "redacted": True,
        "kind": "str",
        "char_count": len("failed with token secret-token-123"),
        "size_bytes": len("failed with token secret-token-123"),
    }
    assert "secret-token-123" not in json.dumps(receipt)


def _make_tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _make_agent(*tool_names: str):
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    setattr(agent, "session_id", "session-1")
    setattr(agent, "_current_turn_id", "turn-1")
    setattr(agent, "_current_api_request_id", "api-1")
    return agent


def _mock_tool_call(name="web_search", args=None, call_id="call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(args or {}),
        ),
    )


def _mock_assistant_msg(*tool_calls):
    return SimpleNamespace(content="", tool_calls=list(tool_calls))


def _capture_receipts(monkeypatch, *, block_tool_name: str | None = None):
    receipts = []

    monkeypatch.setattr(
        "hermes_cli.plugins.has_hook",
        lambda name: name == "execution_receipt",
    )

    def fake_invoke_hook(name, **kwargs):
        if name == "pre_tool_call" and kwargs.get("tool_name") == block_tool_name:
            return [{"action": "block", "message": "blocked for test"}]
        if name == "execution_receipt":
            receipts.append(kwargs["receipt"])
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)
    return receipts


def test_sequential_tool_execution_noops_without_receipt_listener(monkeypatch):
    agent = _make_agent("web_search")
    messages = []
    assistant = _mock_assistant_msg(
        _mock_tool_call("web_search", {"q": "top secret query"}, call_id="call-noop")
    )

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)

    def fail_if_invoked(*args, **kwargs):
        raise AssertionError("execution_receipt hook should not be invoked")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fail_if_invoked)

    with patch("run_agent.handle_function_call", return_value='{"ok": true}'):
        agent._execute_tool_calls_sequential(assistant, messages, "task-1")

    assert not hasattr(agent, "_execution_receipt_sequence")
    assert not hasattr(agent, "_execution_receipt_trace_id")


def test_sequential_tool_execution_emits_redacted_receipt(monkeypatch):
    receipts = _capture_receipts(monkeypatch)
    agent = _make_agent("web_search")
    messages = []
    assistant = _mock_assistant_msg(
        _mock_tool_call(
            "web_search",
            {"q": "top secret query"},
            call_id="call-web-search",
        )
    )

    with patch("run_agent.handle_function_call", return_value='{"ok": true}'):
        agent._execute_tool_calls_sequential(assistant, messages, "task-1")

    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["schema_version"] == "hermes.execution_receipt.v0"
    assert receipt["receipt_type"] == "tool_complete"
    assert receipt["session_id"] == "session-1"
    assert receipt["task_id"] == "task-1"
    assert receipt["turn_id"] == "turn-1"
    assert receipt["api_request_id"] == "api-1"
    assert receipt["tool_call_id"] == "call-web-search"
    assert receipt["tool_name"] == "web_search"
    assert receipt["status"] == "ok"
    assert receipt["sequence_number"] == 1
    assert receipt["trace_id"].startswith("trace-")
    assert receipt["duration_ms"] >= 0
    assert "top secret query" not in json.dumps(receipt)


def test_sequential_blocked_tool_emits_blocked_receipt(monkeypatch):
    receipts = _capture_receipts(monkeypatch, block_tool_name="web_search")
    agent = _make_agent("web_search")
    messages = []
    assistant = _mock_assistant_msg(
        _mock_tool_call("web_search", {"q": "blocked secret"}, call_id="call-blocked")
    )

    with patch("run_agent.handle_function_call") as mock_handle:
        agent._execute_tool_calls_sequential(assistant, messages, "task-1")

    mock_handle.assert_not_called()
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["tool_call_id"] == "call-blocked"
    assert receipt["status"] == "blocked"
    assert receipt["error_type"] == "plugin_block"
    assert receipt["error_message"] is None
    assert receipt["error_message_metadata"]["redacted"] is True
    assert "blocked secret" not in json.dumps(receipt)
    assert "blocked for test" not in json.dumps(receipt)


def test_sequential_interrupt_skip_emits_cancelled_receipts(monkeypatch):
    receipts = _capture_receipts(monkeypatch)
    agent = _make_agent("web_search", "read_file")
    agent._interrupt_requested = True
    messages = []
    assistant = _mock_assistant_msg(
        _mock_tool_call("web_search", {"q": "skip secret"}, call_id="call-skip-1"),
        _mock_tool_call("read_file", {"path": "/tmp/skip-secret"}, call_id="call-skip-2"),
    )

    agent._execute_tool_calls_sequential(assistant, messages, "task-1")

    assert len(receipts) == 2
    assert {receipt["status"] for receipt in receipts} == {"cancelled"}
    assert {receipt["tool_call_id"] for receipt in receipts} == {"call-skip-1", "call-skip-2"}
    serialized = json.dumps(receipts)
    assert "skip secret" not in serialized
    assert "/tmp/skip-secret" not in serialized


def test_concurrent_interrupt_skip_emits_cancelled_receipts(monkeypatch):
    receipts = _capture_receipts(monkeypatch)
    agent = _make_agent("web_search", "read_file")
    agent._interrupt_requested = True
    messages = []
    assistant = _mock_assistant_msg(
        _mock_tool_call("web_search", {"q": "skip secret"}, call_id="call-skip-1"),
        _mock_tool_call("read_file", {"path": "/tmp/skip-secret"}, call_id="call-skip-2"),
    )

    agent._execute_tool_calls_concurrent(assistant, messages, "task-1")

    assert len(receipts) == 2
    assert {receipt["status"] for receipt in receipts} == {"cancelled"}
    assert {receipt["tool_call_id"] for receipt in receipts} == {"call-skip-1", "call-skip-2"}
    serialized = json.dumps(receipts)
    assert "skip secret" not in serialized
    assert "/tmp/skip-secret" not in serialized


def test_sequential_post_tool_interrupt_emits_cancelled_receipt_for_remaining(monkeypatch):
    receipts = _capture_receipts(monkeypatch)
    agent = _make_agent("web_search", "read_file")
    messages = []
    assistant = _mock_assistant_msg(
        _mock_tool_call("web_search", {"q": "first secret"}, call_id="call-1"),
        _mock_tool_call("read_file", {"path": "/tmp/remaining-secret"}, call_id="call-2"),
    )

    def interrupt_after_first(*args, **kwargs):
        agent._interrupt_requested = True
        return '{"ok": true}'

    with patch("run_agent.handle_function_call", side_effect=interrupt_after_first):
        agent._execute_tool_calls_sequential(assistant, messages, "task-1")

    assert len(receipts) == 2
    by_id = {receipt["tool_call_id"]: receipt for receipt in receipts}
    assert by_id["call-1"]["status"] == "ok"
    assert by_id["call-2"]["status"] == "cancelled"
    assert by_id["call-2"]["error_type"] == "user_interrupt"
    serialized = json.dumps(receipts)
    assert "first secret" not in serialized
    assert "/tmp/remaining-secret" not in serialized


def test_enabled_plugin_writes_receipt_from_agent_loop(monkeypatch, tmp_path):
    from hermes_cli import plugins as pmod

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - observability/execution_receipts\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(pmod, "_plugin_manager", None)
    pmod.discover_plugins(force=True)

    agent = _make_agent("web_search")
    messages = []
    assistant = _mock_assistant_msg(
        _mock_tool_call("web_search", {"q": "integration secret"}, call_id="call-integrated")
    )

    with patch("run_agent.handle_function_call", return_value='{"ok": true}'):
        agent._execute_tool_calls_sequential(assistant, messages, "task-1")

    path = hermes_home / "execution-receipts" / "receipts.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    receipt = json.loads(lines[0])
    assert receipt["tool_call_id"] == "call-integrated"
    assert receipt["tool_name"] == "web_search"
    assert receipt["status"] == "ok"
    assert "integration secret" not in lines[0]


def test_concurrent_tool_execution_emits_one_receipt_per_tool(monkeypatch):
    receipts = _capture_receipts(monkeypatch)
    agent = _make_agent("web_search", "read_file")
    messages = []
    assistant = _mock_assistant_msg(
        _mock_tool_call("web_search", {"q": "secret one"}, call_id="call-1"),
        _mock_tool_call("read_file", {"path": "/tmp/secret-two"}, call_id="call-2"),
    )

    with patch("run_agent.handle_function_call", return_value='{"ok": true}'):
        agent._execute_tool_calls_concurrent(assistant, messages, "task-1")

    assert len(receipts) == 2
    assert {receipt["tool_call_id"] for receipt in receipts} == {"call-1", "call-2"}
    assert {receipt["tool_name"] for receipt in receipts} == {"web_search", "read_file"}
    assert sorted(receipt["sequence_number"] for receipt in receipts) == [1, 2]
    assert len({receipt["trace_id"] for receipt in receipts}) == 1
    serialized = json.dumps(receipts)
    assert "secret one" not in serialized
    assert "/tmp/secret-two" not in serialized

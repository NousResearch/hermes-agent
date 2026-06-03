import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SERVER_PATH = Path(__file__).resolve().parents[2] / "scripts/cc-switch-codex-bridge/server.py"


def load_bridge():
    spec = importlib.util.spec_from_file_location("cc_switch_codex_bridge_server", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sse_payloads(raw: bytes):
    frames = raw.decode("utf-8").strip().split("\n\n")
    return [json.loads(frame.split("data: ", 1)[1]) for frame in frames if "data: " in frame]


def test_claude_alias_preserves_requested_model_and_resolves_gpt55():
    bridge = load_bridge()

    requested, resolved, effort = bridge._resolve_model_and_effort(
        {"model": "claude-sonnet-4-8-latest"}
    )
    assert requested == "claude-sonnet-4-8-latest"
    assert resolved == "gpt-5.5"
    assert effort == "xhigh"
    assert "claude-sonnet-4-8-latest" in bridge._supported_model_ids()




def test_claude_alias_context_window_matches_gpt55_backend():
    bridge = load_bridge()

    payload = {"model": "claude-opus-4-8-latest", "messages": [{"role": "user", "content": "hello"}]}

    assert bridge._context_window_for_payload(payload) == 272000
    assert bridge._context_window_error(payload) is None

    oversized = {
        "model": "claude-opus-4-8-latest",
        "messages": [{"role": "user", "content": "x" * 1_120_000}],
    }
    error = bridge._context_window_error(oversized)

    assert error is not None
    assert error["error"]["type"] == "context_length_exceeded"
    assert error["error"]["context_window"] == 272000
    assert error["error"]["requested_model"] == "claude-opus-4-8-latest"
    assert error["error"]["resolved_model"] == "gpt-5.5"

def test_tool_result_history_gets_continuation_instruction_and_flattened_fallback():
    bridge = load_bridge()
    bridge.STRUCTURED_TOOL_HISTORY = False

    converted = bridge._anthropic_messages_to_openai(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_pwd",
                        "name": "Bash",
                        "input": {"command": "pwd"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_pwd",
                        "content": [{"type": "text", "text": "/repo"}],
                    }
                ],
            },
        ],
        "You are Claude Code.",
    )

    assert converted[0]["role"] == "system"
    assert bridge.BRIDGE_TOOL_CONTINUATION_INSTRUCTION in converted[0]["content"]
    assert all(msg["role"] != "tool" for msg in converted)
    assert "<completed_tool_result" in converted[-1]["content"]


def test_codex_call_preserves_all_valid_tool_calls_and_logs_meta(monkeypatch):
    bridge = load_bridge()
    captured = {}

    tool_calls = [
        SimpleNamespace(
            id="call_pwd",
            function=SimpleNamespace(name="Bash", arguments='{"command":"pwd"}'),
        ),
        SimpleNamespace(
            id="call_status",
            function=SimpleNamespace(name="Bash", arguments='{"command":"git status --short"}'),
        ),
    ]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            kwargs["_on_reasoning_delta"]("plan")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        message=SimpleNamespace(content=None, tool_calls=tool_calls),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    import agent.auxiliary_client as aux

    monkeypatch.setattr(aux, "_build_codex_client", lambda model: (fake_client, model))

    result, label, meta = bridge._codex_call(
        {
            "model": "claude-sonnet-4-8-latest",
            "messages": [{"role": "user", "content": "run two commands"}],
            "tools": [
                {"name": "Bash", "input_schema": {"type": "object", "properties": {}}},
            ],
        }
    )

    assert label is None
    assert captured["model"] == "gpt-5.5"
    assert captured["extra_body"] == {"reasoning": {"effort": "xhigh"}}
    assert result["model"] == "claude-sonnet-4-8-latest"
    assert result["stop_reason"] == "tool_use"
    assert [b["id"] for b in result["content"] if b["type"] == "tool_use"] == [
        "call_pwd",
        "call_status",
    ]
    assert meta["tool_use_names"] == ["Bash", "Bash"]
    assert meta["malformed_tool_count"] == 0
    assert result["content"][0]["type"] == "thinking"


def test_codex_call_uses_final_structured_reasoning_without_live_delta(monkeypatch):
    bridge = load_bridge()

    class FakeCompletions:
        def create(self, **kwargs):
            assert callable(kwargs.get("_on_reasoning_delta"))
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            content="done",
                            tool_calls=[],
                            reasoning="final reasoning summary",
                        ),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3),
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    import agent.auxiliary_client as aux

    monkeypatch.setattr(aux, "_build_codex_client", lambda model: (fake_client, model))

    result, _label, meta = bridge._codex_call(
        {"model": "claude-sonnet-4-8-latest", "messages": [{"role": "user", "content": "think"}]}
    )

    assert [block["type"] for block in result["content"]] == ["thinking", "text"]
    assert result["content"][0]["thinking"] == "final reasoning summary"
    assert meta["content_block_types"] == ["thinking", "text"]


def test_codex_call_malformed_tool_arguments_returns_diagnostic_end_turn(monkeypatch):
    bridge = load_bridge()
    tool_calls = [
        SimpleNamespace(
            id="call_bad",
            function=SimpleNamespace(name="Bash", arguments="{not json"),
        )
    ]

    class FakeCompletions:
        def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        message=SimpleNamespace(content=None, tool_calls=tool_calls),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

    import agent.auxiliary_client as aux

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(aux, "_build_codex_client", lambda model: (fake_client, model))

    result, _label, meta = bridge._codex_call(
        {"messages": [{"role": "user", "content": "bad tool"}], "tools": [{"name": "Bash"}]}
    )

    assert result["stop_reason"] == "end_turn"
    assert [b["type"] for b in result["content"]] == ["text"]
    assert "bridge_tool_call_argument_error" in result["content"][0]["text"]
    assert meta["malformed_tool_count"] == 1


@pytest.mark.parametrize("protocol_log", [False, True])
def test_streaming_final_thinking_block_emitted_with_protocol_log_off_and_on(
    monkeypatch, tmp_path, protocol_log
):
    bridge = load_bridge()
    monkeypatch.setattr(bridge, "PROTOCOL_LOG", False)
    if protocol_log:
        monkeypatch.setenv("CODEX_ANTHROPIC_BRIDGE_PROTOCOL_LOG", "1")
        monkeypatch.setenv(
            "CODEX_ANTHROPIC_BRIDGE_PROTOCOL_LOG_FILE",
            str(tmp_path / "protocol.jsonl"),
        )
    else:
        monkeypatch.delenv("CODEX_ANTHROPIC_BRIDGE_PROTOCOL_LOG", raising=False)
        monkeypatch.delenv("CODEX_ANTHROPIC_BRIDGE_PROTOCOL_LOG_FILE", raising=False)

    def fake_codex_call(payload, message_id=None, on_reasoning_delta=None, **_kwargs):
        assert callable(on_reasoning_delta)
        return (
            {
                "id": message_id or "msg_test",
                "type": "message",
                "role": "assistant",
                "model": payload.get("model", "claude-sonnet-4-8-latest"),
                "content": [
                    {"type": "thinking", "thinking": "final-only reasoning"},
                    {"type": "text", "text": "done"},
                ],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
            None,
            {
                "requested_model": "claude-sonnet-4-8-latest",
                "resolved_model": "gpt-5.5",
                "effort": "high",
                "finish_reason": "stop",
                "stop_reason": "end_turn",
                "tool_use_ids": [],
                "tool_use_names": [],
                "malformed_tool_count": 0,
            },
        )

    class FakeHandler:
        def __init__(self):
            self.wfile = io.BytesIO()
            self.headers = {}

        def send_response(self, status):
            self.status = status

        def send_header(self, name, value):
            pass

        def end_headers(self):
            pass

    monkeypatch.setattr(bridge, "_codex_call", fake_codex_call)
    handler = FakeHandler()

    bridge.Handler._handle_streaming_messages(
        handler,
        {
            "model": "claude-sonnet-4-8-latest",
            "stream": True,
            "messages": [{"role": "user", "content": "think"}],
        },
        "req_test",
    )

    payloads = _sse_payloads(handler.wfile.getvalue())
    deltas = [p.get("delta", {}) for p in payloads if p.get("type") == "content_block_delta"]
    block_starts = [
        p.get("content_block", {}).get("type")
        for p in payloads
        if p.get("type") == "content_block_start"
    ]

    assert handler.status == 200
    assert block_starts[:2] == ["thinking", "text"]
    assert [d.get("type") for d in deltas].count("thinking_delta") == 1
    assert any(d.get("thinking") == "final-only reasoning" for d in deltas)
    assert payloads[-1]["type"] == "message_stop"
    if protocol_log:
        rows = (tmp_path / "protocol.jsonl").read_text().splitlines()
        assert rows
        logged = json.loads(rows[-1])
        assert logged["outgoing_content_block_types"] == ["thinking", "text"]


def test_streaming_sse_order_includes_thinking_signature_and_stop_reason():
    bridge = load_bridge()

    class FakeHandler:
        def __init__(self):
            self.wfile = io.BytesIO()
            self._bridge_sse_events = []

    handler = FakeHandler()
    result = {
        "content": [
            {"type": "thinking", "thinking": "reasoning"},
            {"type": "text", "text": "done"},
            {"type": "tool_use", "id": "call_next", "name": "Bash", "input": {"command": "pwd"}},
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }

    bridge._sse_write(handler, "message_start", {"type": "message_start", "message": {}})
    bridge._sse_emit_result_content(handler, result)
    bridge._sse_write(
        handler,
        "message_delta",
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}},
    )
    bridge._sse_write(handler, "message_stop", {"type": "message_stop"})

    payloads = _sse_payloads(handler.wfile.getvalue())
    deltas = [p.get("delta", {}) for p in payloads if p.get("type") == "content_block_delta"]

    assert handler._bridge_sse_events[0] == "message_start"
    assert handler._bridge_sse_events[-2:] == ["message_delta", "message_stop"]
    assert any(delta.get("type") == "thinking_delta" for delta in deltas)
    assert any(delta.get("type") == "signature_delta" for delta in deltas)
    assert payloads[-2]["delta"]["stop_reason"] == "tool_use"
    assert bridge._event_order_checksum(handler._bridge_sse_events)

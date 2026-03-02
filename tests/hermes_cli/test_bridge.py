"""Tests for Hermes stdio bridge command."""

import argparse
import io
import json
import time
from typing import Any, Callable, Optional, cast
from unittest.mock import patch

from hermes_cli.bridge import bridge_command
from hermes_cli.main import cmd_bridge

JsonDict = dict[str, Any]
EventCallback = Callable[[JsonDict], None]


def _bridge_args(**overrides: Any) -> argparse.Namespace:
    base: JsonDict = {
        "model": None,
        "base_url": None,
        "api_key": None,
        "max_iterations": None,
        "toolsets": None,
        "disabled_toolsets": None,
        "cwd": None,
        "once": False,
        "assistant_delta": False,
        "assistant_delta_chunk_size": 120,
        "native_assistant_delta": True,
        "native_stream": False,
        "stream_first_chunk_timeout_ms": None,
        "stream_idle_timeout_ms": None,
        "verbose": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cmd_bridge_delegates_to_bridge_module() -> None:
    args = _bridge_args()
    with patch("hermes_cli.bridge.bridge_command", return_value=7) as mock_bridge:
        result = cmd_bridge(args)

    assert result == 7
    mock_bridge.assert_called_once_with(args)


def test_bridge_command_ping_roundtrip() -> None:
    args = _bridge_args()
    stdin = io.StringIO('{"type":"ping","request_id":"req-1"}\n')
    stdout = io.StringIO()

    rc = bridge_command(args, stdin=stdin, stdout=stdout, stderr=io.StringIO())

    assert rc == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    assert lines == [{"type": "pong", "request_id": "req-1"}]


def test_bridge_command_run_missing_message_emits_validation_error() -> None:
    args = _bridge_args(once=True)
    stdin = io.StringIO('{"type":"run","request_id":"req-2"}\n')
    stdout = io.StringIO()

    rc = bridge_command(args, stdin=stdin, stdout=stdout, stderr=io.StringIO())

    assert rc == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["type"] == "error"
    assert lines[0]["request_id"] == "req-2"
    assert lines[0]["stage"] == "request_validation"


def test_bridge_command_interrupt_without_active_session_errors() -> None:
    args = _bridge_args(once=True)
    stdin = io.StringIO('{"type":"interrupt","request_id":"req-int","session_id":"s1"}\n')
    stdout = io.StringIO()

    rc = bridge_command(args, stdin=stdin, stdout=stdout, stderr=io.StringIO())

    assert rc == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["type"] == "error"
    assert lines[0]["request_id"] == "req-int"
    assert lines[0]["stage"] == "request_validation"


def test_bridge_command_assistant_delta_conversion() -> None:
    args = _bridge_args(once=True)
    stdin = io.StringIO(
        '{"type":"run","request_id":"req-delta","session_id":"sess-1","message":"hello","assistant_delta":true,"assistant_delta_chunk_size":5}\n'
    )
    stdout = io.StringIO()

    class FakeAIAgent:
        def __init__(self, **kwargs: Any) -> None:
            callback = kwargs.get("event_callback")
            assert callable(callback)
            self._event_callback: EventCallback = cast(EventCallback, callback)

        def run_conversation(self, **kwargs: Any) -> JsonDict:
            self._event_callback(
                {
                    "type": "assistant_message",
                    "request_id": kwargs.get("request_id"),
                    "session_id": kwargs.get("task_id") or "sess-1",
                    "timestamp": "2026-01-01T00:00:00.000Z",
                    "content": "hello world",
                    "has_tool_calls": False,
                    "finish_reason": "stop",
                    "reasoning_present": False,
                }
            )
            return {
                "messages": [],
                "final_response": "done",
                "api_calls": 1,
                "completed": True,
                "partial": False,
                "interrupted": False,
            }

    with patch("hermes_cli.bridge.AIAgent", FakeAIAgent):
        rc = bridge_command(args, stdin=stdin, stdout=stdout, stderr=io.StringIO())

    assert rc == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    types = [entry["type"] for entry in lines]

    assert "session" in types
    assert "assistant_delta" in types
    assert "assistant_message_end" in types
    assert "final" in types
    assert "assistant_message" not in types


def test_bridge_command_interrupt_ack_for_active_session() -> None:
    args = _bridge_args()
    stdin = io.StringIO(
        '{"type":"run","request_id":"req-run","session_id":"sess-int","message":"start"}\n'
        '{"type":"interrupt","request_id":"req-stop","session_id":"sess-int","message":"stop"}\n'
        '{"type":"shutdown","request_id":"req-shutdown"}\n'
    )
    stdout = io.StringIO()

    class FakeAIAgent:
        def __init__(self, **kwargs: Any) -> None:
            self._interrupted = False

        def interrupt(self, message: Optional[str] = None) -> None:
            self._interrupted = True

        def run_conversation(self, **kwargs: Any) -> JsonDict:
            for _ in range(50):
                if self._interrupted:
                    break
                time.sleep(0.01)
            return {
                "messages": [],
                "final_response": "stopped" if self._interrupted else "done",
                "api_calls": 1,
                "completed": not self._interrupted,
                "partial": False,
                "interrupted": self._interrupted,
            }

    with patch("hermes_cli.bridge.AIAgent", FakeAIAgent):
        rc = bridge_command(args, stdin=stdin, stdout=stdout, stderr=io.StringIO())

    assert rc == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    by_type: dict[str, list[JsonDict]] = {}
    for line in lines:
        by_type.setdefault(line["type"], []).append(line)

    assert "interrupt_ack" in by_type
    assert by_type["interrupt_ack"][0]["request_id"] == "req-stop"
    assert by_type["interrupt_ack"][0]["accepted"] is True
    assert "final" in by_type
    assert by_type["final"][0]["interrupted"] is True


def test_bridge_native_assistant_delta_passthrough_without_synthetic_duplication() -> None:
    args = _bridge_args(once=True, assistant_delta=True, native_assistant_delta=True)
    stdin = io.StringIO(
        '{"type":"run","request_id":"req-native","session_id":"sess-native","message":"hello"}\n'
    )
    stdout = io.StringIO()

    class FakeAIAgent:
        def __init__(self, **kwargs: Any) -> None:
            callback = kwargs.get("event_callback")
            assert callable(callback)
            self._event_callback: EventCallback = cast(EventCallback, callback)

        def run_conversation(self, **kwargs: Any) -> JsonDict:
            self._event_callback(
                {
                    "type": "assistant_delta",
                    "delta": "hello ",
                    "delta_index": 1,
                    "native": True,
                }
            )
            self._event_callback(
                {
                    "type": "assistant_delta",
                    "delta": "world",
                    "delta_index": 2,
                    "native": True,
                }
            )
            self._event_callback(
                {
                    "type": "assistant_message_end",
                    "delta_chunks": 2,
                    "native": True,
                }
            )
            self._event_callback(
                {
                    "type": "assistant_message",
                    "content": "hello world",
                    "has_tool_calls": False,
                    "finish_reason": "stop",
                    "reasoning_present": False,
                }
            )
            return {
                "messages": [],
                "final_response": "done",
                "api_calls": 1,
                "completed": True,
                "partial": False,
                "interrupted": False,
            }

    with patch("hermes_cli.bridge.AIAgent", FakeAIAgent):
        rc = bridge_command(args, stdin=stdin, stdout=stdout, stderr=io.StringIO())

    assert rc == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    delta_events = [evt for evt in lines if evt.get("type") == "assistant_delta"]
    end_events = [evt for evt in lines if evt.get("type") == "assistant_message_end"]
    assert len(delta_events) == 2
    assert len(end_events) == 1
    assert [evt.get("delta") for evt in delta_events] == ["hello ", "world"]


def test_bridge_assistant_delta_fallback_when_no_native_deltas() -> None:
    args = _bridge_args(once=True, assistant_delta=True, native_assistant_delta=True)
    stdin = io.StringIO(
        '{"type":"run","request_id":"req-fallback","session_id":"sess-fallback","message":"hello"}\n'
    )
    stdout = io.StringIO()

    class FakeAIAgent:
        def __init__(self, **kwargs: Any) -> None:
            callback = kwargs.get("event_callback")
            assert callable(callback)
            self._event_callback: EventCallback = cast(EventCallback, callback)

        def run_conversation(self, **kwargs: Any) -> JsonDict:
            self._event_callback(
                {
                    "type": "assistant_message",
                    "timestamp": "2026-01-01T00:00:00.000Z",
                    "content": "fallback text",
                    "has_tool_calls": False,
                    "finish_reason": "stop",
                    "reasoning_present": False,
                }
            )
            return {
                "messages": [],
                "final_response": "done",
                "api_calls": 1,
                "completed": True,
                "partial": False,
                "interrupted": False,
            }

    with patch("hermes_cli.bridge.AIAgent", FakeAIAgent):
        rc = bridge_command(args, stdin=stdin, stdout=stdout, stderr=io.StringIO())

    assert rc == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    assert any(evt.get("type") == "assistant_delta" for evt in lines)
    assert any(evt.get("type") == "assistant_message_end" for evt in lines)
    assert not any(evt.get("type") == "assistant_message" for evt in lines)


def test_bridge_event_order_native_deltas_before_final() -> None:
    args = _bridge_args(once=True, assistant_delta=True, native_assistant_delta=True)
    stdin = io.StringIO(
        '{"type":"run","request_id":"req-order","session_id":"sess-order","message":"hello"}\n'
    )
    stdout = io.StringIO()

    class FakeAIAgent:
        def __init__(self, **kwargs: Any) -> None:
            callback = kwargs.get("event_callback")
            assert callable(callback)
            self._event_callback: EventCallback = cast(EventCallback, callback)

        def run_conversation(self, **kwargs: Any) -> JsonDict:
            self._event_callback({"type": "assistant_delta", "delta": "a", "delta_index": 1, "native": True})
            self._event_callback({"type": "assistant_message_end", "delta_chunks": 1, "native": True})
            self._event_callback({"type": "assistant_message", "content": "a", "finish_reason": "stop"})
            return {
                "messages": [],
                "final_response": "done",
                "api_calls": 1,
                "completed": True,
                "partial": False,
                "interrupted": False,
            }

    with patch("hermes_cli.bridge.AIAgent", FakeAIAgent):
        rc = bridge_command(args, stdin=stdin, stdout=stdout, stderr=io.StringIO())

    assert rc == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    types = [evt.get("type") for evt in lines]
    assert types.index("assistant_delta") < types.index("assistant_message_end")
    assert types.index("assistant_message_end") < types.index("assistant_message")
    assert types.index("assistant_message") < types.index("final")

from __future__ import annotations

import json
import os
import socket
import struct
import threading

from hermes_cli.observability import stack_monitor


def test_build_envelope_excludes_content_and_secrets():
    event = stack_monitor.build_envelope(
        "post_tool_call",
        {
            "session_id": "session-1",
            "turn_id": "turn-1",
            "tool_call_id": "call-1",
            "tool_name": "write_file",
            "args": {"path": "/private/file", "content": "secret-body"},
            "result": "secret-result",
            "prompt": "private prompt",
            "api_key": "secret-key",
            "status": "ok",
        },
        sequence=1,
    )
    encoded = json.dumps(event, sort_keys=True)
    assert event["kind"] == "tool"
    assert event["privacy"] == {
        "tier": "MetadataOnly",
        "redaction": "ContentDisabled",
        "content_fields": 0,
    }
    assert "secret-body" not in encoded
    assert "secret-result" not in encoded
    assert "private prompt" not in encoded
    assert "secret-key" not in encoded
    assert "args" not in encoded
    assert "result" not in encoded


def test_terminal_gap_is_an_explicit_negative_witness():
    event = stack_monitor.build_envelope(
        "terminal_observation_gap",
        {
            "session_id": "session-1",
            "request_id": "request-1",
            "gap_kind": "llm_call",
            "missing_terminal_hook": "post_api_request",
        },
        sequence=2,
    )
    assert event["kind"] == "llm_call"
    assert event["status"] == "cancelled"
    assert event["payload"]["reason"] == "session_end_without_terminal_hook"
    assert event["payload"]["missing_terminal_hook"] == "post_api_request"


def test_session_end_exposes_started_vs_terminal_coverage():
    event = stack_monitor.build_envelope(
        "on_session_end",
        {
            "session_id": "session-1",
            "coverage": {
                "started_llm": 2,
                "terminal_llm": 0,
                "started_tool": 1,
                "terminal_tool": 0,
            },
        },
        sequence=3,
    )
    assert event["payload"]["coverage"] == {
        "started_llm": 2,
        "terminal_llm": 0,
        "started_tool": 1,
        "terminal_tool": 0,
    }


def test_observe_lifecycle_sends_bounded_frame(monkeypatch):
    socket_path = "/tmp/ares-observability-test.sock"
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass
    received: list[dict] = []
    ready = threading.Event()
    done = threading.Event()

    def serve() -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(socket_path)
            listener.listen(1)
            listener.settimeout(2.0)
            ready.set()
            with listener.accept()[0] as conn:
                header = conn.recv(4)
                size = struct.unpack(">I", header)[0]
                payload = bytearray()
                while len(payload) < size:
                    payload.extend(conn.recv(size - len(payload)))
                received.append(json.loads(bytes(payload)))
                done.set()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(2.0)
    monkeypatch.setenv("ARES_STACK_OBSERVATION_ENABLED", "1")
    monkeypatch.setenv("ARES_STACK_OBSERVATION_SOCKET", socket_path)
    stack_monitor.shutdown()
    stack_monitor.observe_lifecycle(
        "post_api_request",
        session_id="session-1",
        turn_id="turn-1",
        api_request_id="request-1",
        model="gpt-test",
        provider="provider-test",
        prompt="must not cross the bridge",
        response="must not cross the bridge",
        usage={"total_tokens": 99},
    )
    assert done.wait(2.0)
    stack_monitor.shutdown()
    thread.join(timeout=2.0)

    assert len(received) == 1
    event = received[0]
    assert event["source_crate"] == "hermes-agent"
    assert event["adapter_id"] == "ares.lifecycle.v1"
    assert event["kind"] == "llm_call"
    assert event["correlation"]["session_id"] == "session-1"
    encoded = json.dumps(event, sort_keys=True)
    assert "must not cross the bridge" not in encoded
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass

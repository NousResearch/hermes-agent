"""Real-subprocess contract tests for the agy stream-json transport."""

from __future__ import annotations

import json
import stat
import sys

import pytest


def _fake_agy(tmp_path):
    binary = tmp_path / "agy"
    binary.write_text(
        f"#!{sys.executable}\n"
        """
import json
import os
import signal
import sys
import time

args = sys.argv[1:]
if args == ["--version"]:
    print("1.2.7")
    raise SystemExit(0)
if args == ["models"]:
    print("gemini-test\tGemini Test")
    print("claude-test-thinking\tClaude Test (Thinking)")
    raise SystemExit(0)
if "--probe-fail" in args:
    print("bad probe", file=sys.stderr)
    raise SystemExit(4)
if os.environ.get("FAKE_AGY_SLEEP"):
    time.sleep(float(os.environ["FAKE_AGY_SLEEP"]))
received = [json.loads(line) for line in sys.stdin if line.strip()]
assert received == [{"event": "user", "message": {"role": "user", "content": "hello"}}] or received == [{"event": "user", "message": {"role": "user", "content": "again"}}]
print(json.dumps({"event": "init", "conversation_id": "conv-new"}), flush=True)
for item in received:
    print(json.dumps({"event": "step_update", "step_update": {"step_index": 1, "state": "DONE", "step_type": "agent_response", "text_delta": item["message"]["content"]}}), flush=True)
print(json.dumps({"event": "tool_denied", "tool": "shell", "reason": "policy"}), flush=True)
print(json.dumps({"event": "result", "result": {"response": "done", "conversation_id": "conv-new", "status": "SUCCESS"}}), flush=True)
print("diagnostic-one", file=sys.stderr, flush=True)
""".lstrip(),
        encoding="utf-8",
    )
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return binary


def test_debug_protocol_log_redacts_sensitive_fields(tmp_path):
    import json
    from agent.transports.antigravity_cli import AntigravityClient

    path = tmp_path / "protocol.log"
    client = AntigravityClient(config_path="/missing", debug_log=path)
    client._write_debug("stdout", {"access_token": "secret", "content": "private prompt", "nested": {"cookie": "value"}, "event": "init"})

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["payload"] == {
        "access_token": "[REDACTED]",
        "content": "[REDACTED]",
        "nested": {"cookie": "[REDACTED]"},
        "event": "init",
    }
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_client_discovers_probes_and_streams_real_agy_subprocess(tmp_path, monkeypatch):
    from agent.transports.antigravity_cli import AntigravityClient

    binary = _fake_agy(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))
    received = []
    client = AntigravityClient(config_path=None, known_locations=())

    capability = client.probe()
    result = client.run_turn("hello", model="claude-test-thinking", on_event=received.append)

    assert capability.available is True
    assert capability.version == (1, 2, 7)
    assert capability.stream_json is True
    assert capability.authenticated is True
    assert capability.models == ("gemini-test", "claude-test-thinking")
    assert result.text == "done"
    assert result.conversation_id == "conv-new"
    assert result.events[0]["event"] == "init"
    assert result.events[1]["step_update"]["text_delta"] == "hello"
    assert result.events[2] == {"event": "tool_denied", "tool": "shell", "reason": "policy"}
    assert result.stderr_tail == ["diagnostic-one"]
    assert received == result.events
    assert "--input-format" in result.argv
    assert "stream-json" in result.argv
    assert result.argv[result.argv.index("--model") + 1] == "claude-test-thinking"


def test_client_yolo_mode_adds_dangerous_flag_without_sandbox(tmp_path):
    from agent.transports.antigravity_cli import AntigravityClient

    binary = _fake_agy(tmp_path)
    result = AntigravityClient(
        config_path=str(binary), known_locations=(), sandbox=False,
        dangerously_skip_permissions=True,
    ).run_turn("hello")

    assert "--dangerously-skip-permissions" in result.argv
    assert "--sandbox" not in result.argv


def test_client_uses_config_path_and_resumes_with_conversation(tmp_path):
    from agent.transports.antigravity_cli import AntigravityClient

    binary = _fake_agy(tmp_path)
    result = AntigravityClient(config_path=str(binary), known_locations=()).run_turn(
        "again", conversation_id="conv-old"
    )

    assert result.argv[:2] == [str(binary), "--input-format"]
    assert result.argv[-2:] == ["--conversation", "conv-old"]


def test_client_fails_safe_on_missing_terminal_result(tmp_path):
    from agent.transports.antigravity_cli import AntigravityProtocolError

    binary = _fake_agy(tmp_path)
    binary.write_text(
        f"#!{sys.executable}\nimport json; print(json.dumps({{'event': 'step_update', 'step_update': {{'text_delta': 'x'}}}}), flush=True)\n",
        encoding="utf-8",
    )
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    from agent.transports.antigravity_cli import AntigravityClient

    with pytest.raises(AntigravityProtocolError, match="terminal result"):
        AntigravityClient(config_path=str(binary), known_locations=()).run_turn("hello")


def test_client_surfaces_denied_actions_and_partial_timeout_success(tmp_path):
    from agent.transports.antigravity_cli import AntigravityClient, AntigravityTurnError

    binary = tmp_path / "agy"

    def write_result(payload):
        binary.write_text(
            f"#!{sys.executable}\nimport json,sys\nlist(sys.stdin)\nprint(json.dumps({{'event':'result','result':{payload!r}}}), flush=True)\n",
            encoding="utf-8",
        )
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)

    write_result({"status": "SUCCESS", "response": "", "num_turns": 1,
                  "denied_actions": [{"action": "command", "display_name": "RunCommand"}]})
    with pytest.raises(AntigravityTurnError, match="RunCommand"):
        AntigravityClient(config_path=str(binary), known_locations=()).run_turn("hello")

    write_result({"status": "SUCCESS", "response": "", "num_turns": 0})
    with pytest.raises(AntigravityTurnError, match="before executing a turn"):
        AntigravityClient(config_path=str(binary), known_locations=()).run_turn("hello")


def test_client_reports_request_timeout_as_typed_error(tmp_path, monkeypatch):
    from agent.transports.antigravity_cli import AntigravityClient, AntigravityRequestTimeout

    binary = _fake_agy(tmp_path)
    monkeypatch.setenv("FAKE_AGY_SLEEP", "1")
    with pytest.raises(AntigravityRequestTimeout):
        AntigravityClient(config_path=str(binary), known_locations=(), request_timeout=0.05).run_turn("hello")

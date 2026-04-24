import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

import agent.claude_cli_client as claude_cli_client
from agent.claude_cli_client import ClaudeCLIClient, _extract_tool_calls_from_text, _render_assistant_tool_calls
from agent.claude_cli_client import _chunk_hydration_text


@pytest.fixture(autouse=True)
def _clear_claude_cli_broker_env(monkeypatch):
    monkeypatch.delenv("HERMES_CLAUDE_CLI_BROKER", raising=False)
    monkeypatch.delenv("HERMES_CLAUDE_CLI_BROKER_SOCKET", raising=False)
    monkeypatch.delenv("HERMES_CLAUDE_CLI_BROKER_LOG", raising=False)


def test_chunk_hydration_text_prefers_tighter_packing():
    sep = "\n\n"
    text = (
        "A" * 9300
        + sep
        + "B" * 7000
        + sep
        + "C" * 7000
        + sep
        + "D" * 7000
    )

    chunks = _chunk_hydration_text(text, max_chars=16000)

    assert len(chunks) == 2
    assert len(chunks[0]) == 16000
    assert len(chunks[1]) < 16000


def test_extract_tool_calls_accepts_minimal_payload_shape():
    text = (
        'Let me inspect that.\n'
        '<tool_call>{"name":"terminal","arguments":{"command":"git status --short"}}</tool_call>\n'
        'Then I will summarize.'
    )

    tool_calls, visible_text = _extract_tool_calls_from_text(text)

    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == 'terminal'
    assert json.loads(tool_calls[0].function.arguments) == {'command': 'git status --short'}
    assert '<tool_call>' not in visible_text
    assert 'Let me inspect that.' in visible_text
    assert 'Then I will summarize.' in visible_text


def test_render_assistant_tool_calls_uses_compact_payload():
    rendered = _render_assistant_tool_calls([
        {
            'id': 'call_1',
            'type': 'function',
            'function': {
                'name': 'terminal',
                'arguments': '{"command":"git status --short"}',
            },
        }
    ])

    assert rendered == '{"name":"terminal","arguments":{"command":"git status --short"}}'


def test_extract_tool_calls_converts_action_shell_fence_to_terminal():
    text = "sorry, on it now.\n\n```bash\nls -la /tmp\n```"

    tool_calls, visible_text = _extract_tool_calls_from_text(text)

    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "terminal"
    assert json.loads(tool_calls[0].function.arguments) == {"command": "ls -la /tmp"}
    assert "```" not in visible_text
    assert "sorry, on it now" in visible_text


def test_extract_tool_calls_accepts_minimal_json_input_shape():
    text = '{"name":"read_file","input":{"path":"/tmp/example.txt"}}'

    tool_calls, visible_text = _extract_tool_calls_from_text(text)

    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "read_file"
    assert json.loads(tool_calls[0].function.arguments) == {"path": "/tmp/example.txt"}
    assert visible_text == ""


def test_extract_tool_calls_ignores_self_negated_json_shape():
    text = (
        '{"name":"read_file","input":{"path":"/tmp/example.txt"}}\n\n'
        "Wait — that's not a real tool call."
    )

    tool_calls, visible_text = _extract_tool_calls_from_text(text)

    assert tool_calls == []
    assert "not a real tool call" in visible_text


def test_stream_completion_live_falls_back_to_oneshot_before_output(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PERSISTENT", "1")
    client = ClaudeCLIClient(command="claude")

    def _broken(**_kwargs):
        def _gen():
            raise RuntimeError("boom")
            yield None

        return _gen()

    fallback_chunk = object()
    monkeypatch.setattr(client, "_stream_completion_persistent", _broken)
    monkeypatch.setattr(client, "_stream_completion_oneshot", lambda **_kwargs: iter([fallback_chunk]))

    chunks = list(
        client._stream_completion_live(
            model="claude-opus-4-7",
            prompt_text="hello",
            system_prompt="",
            structured_input=None,
            structured_system_prompt="",
            effort=None,
            timeout_seconds=1.0,
        )
    )

    assert chunks == [fallback_chunk]


def test_run_prompt_uses_persistent_worker_chunks(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PERSISTENT", "1")
    client = ClaudeCLIClient(command="claude")
    client._last_session_id = "session-123"
    client._last_total_cost_usd = 0.42
    client._last_stop_reason = "end_turn"

    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=4,
        prompt_tokens_details=SimpleNamespace(cached_tokens=3),
    )
    chunk1 = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="HEL"), finish_reason=None)],
        usage=None,
    )
    chunk2 = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="LO"), finish_reason="stop")],
        usage=usage,
    )

    monkeypatch.setattr(client, "_stream_completion_persistent", lambda **_kwargs: iter([chunk1, chunk2]))

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("one-shot path should not run")

    monkeypatch.setattr(client, "_run_prompt_oneshot", _unexpected)

    payload = client._run_prompt(
        "hello",
        system_prompt="",
        model="claude-opus-4-7",
        effort=None,
        timeout_seconds=1.0,
    )

    assert payload["result"] == "HELLO"
    assert payload["session_id"] == "session-123"
    assert payload["stop_reason"] == "end_turn"
    assert payload["total_cost_usd"] == 0.42
    assert payload["usage"] == {
        "input_tokens": 10,
        "output_tokens": 4,
        "cache_read_input_tokens": 3,
    }


def test_persistent_worker_is_opt_in(monkeypatch):
    monkeypatch.delenv("HERMES_CLAUDE_CLI_PERSISTENT", raising=False)
    client = ClaudeCLIClient(command="claude")

    assert client._persistent_enabled is False


def test_broker_is_opt_in_and_implies_persistent(monkeypatch):
    monkeypatch.delenv("HERMES_CLAUDE_CLI_PERSISTENT", raising=False)
    monkeypatch.setenv("HERMES_CLAUDE_CLI_BROKER", "1")

    client = ClaudeCLIClient(command="claude")

    assert client._broker_enabled is True
    assert client._persistent_enabled is True


def test_stream_completion_live_prefers_broker(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CLI_BROKER", "1")
    client = ClaudeCLIClient(command="claude")
    chunk = object()

    monkeypatch.setattr(client, "_stream_completion_broker", lambda **_kwargs: iter([chunk]))
    monkeypatch.setattr(
        client,
        "_stream_completion_persistent",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("local persistent should not run")),
    )

    chunks = list(client._stream_completion_live(
        model="claude-opus-4-7",
        prompt_text="hello",
        system_prompt="",
        structured_input=None,
        structured_system_prompt="",
        effort=None,
        timeout_seconds=1.0,
    ))

    assert chunks == [chunk]


def test_stream_completion_live_falls_back_from_broker_before_output(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CLI_BROKER", "1")
    client = ClaudeCLIClient(command="claude")
    fallback_chunk = object()

    def _broken_broker(**_kwargs):
        def _gen():
            raise RuntimeError("broker down")
            yield None

        return _gen()

    monkeypatch.setattr(client, "_stream_completion_broker", _broken_broker)
    monkeypatch.setattr(client, "_stream_completion_persistent", lambda **_kwargs: iter([fallback_chunk]))

    chunks = list(client._stream_completion_live(
        model="claude-opus-4-7",
        prompt_text="hello",
        system_prompt="",
        structured_input=None,
        structured_system_prompt="",
        effort=None,
        timeout_seconds=1.0,
    ))

    assert chunks == [fallback_chunk]


def test_broker_chunk_payload_roundtrip_preserves_tool_calls_and_usage():
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=4,
        total_tokens=14,
        prompt_tokens_details=SimpleNamespace(cached_tokens=3),
    )
    original = claude_cli_client._make_stream_chunk(
        model="claude-opus-4-7",
        tool_call_delta={
            "index": 1,
            "id": "call_123",
            "name": "terminal",
            "arguments": "{\"command\":\"pwd\"}",
        },
        finish_reason="tool_calls",
        usage=usage,
    )

    payload = claude_cli_client._stream_chunk_to_broker_payload(original)
    restored = claude_cli_client._stream_chunk_from_broker_payload(
        payload,
        model="fallback-model",
    )

    tool_call = restored.choices[0].delta.tool_calls[0]
    assert restored.model == "claude-opus-4-7"
    assert restored.choices[0].finish_reason == "tool_calls"
    assert tool_call.index == 1
    assert tool_call.id == "call_123"
    assert tool_call.function.name == "terminal"
    assert tool_call.function.arguments == "{\"command\":\"pwd\"}"
    assert restored.usage.prompt_tokens == 10
    assert restored.usage.completion_tokens == 4
    assert restored.usage.prompt_tokens_details.cached_tokens == 3


class _FakePipe:
    def __init__(self, lines=None):
        self.lines = list(lines or [])
        self.writes = []
        self.closed = False

    def write(self, data):
        self.writes.append(data)
        return len(data)

    def flush(self):
        return None

    def readline(self):
        if self.lines:
            return self.lines.pop(0)
        return ""


class _FakePersistentProc:
    pid = 12345

    def __init__(self):
        self.stdin = _FakePipe()
        self.stdout = _FakePipe([
            json.dumps({
                "type": "result",
                "session_id": "session-compact",
                "result": "OK",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }) + "\n"
        ])
        self.stderr = _FakePipe()

    def poll(self):
        return None


class _FakeStreamingPersistentProc:
    pid = 12346

    def __init__(self, stdout_lines):
        self.stdin = _FakePipe()
        self.stdout = _FakePipe(stdout_lines)
        self.stderr = _FakePipe()

    def poll(self):
        return 0 if not self.stdout.lines else None


def test_persistent_stream_reads_bursted_json_lines_without_select_stall(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PERSISTENT", "1")
    client = ClaudeCLIClient(command="claude")
    script = """
import json
import sys
import time

for _line in sys.stdin:
    events = [
        {"type": "system", "session_id": "sess-burst"},
        {
            "type": "stream_event",
            "event": {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "burst"},
            },
        },
        {
            "type": "result",
            "session_id": "sess-burst",
            "result": "burst",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    ]
    sys.stdout.write("".join(json.dumps(event) + "\\n" for event in events))
    sys.stdout.flush()
    time.sleep(5)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    monkeypatch.setattr(client, "_ensure_persistent_worker", lambda **_kwargs: proc)

    try:
        chunks = list(client._stream_completion_persistent(
            model="claude-opus-4-7",
            prompt_text="hello",
            system_prompt="system",
            structured_input=None,
            structured_system_prompt="system",
            effort=None,
            timeout_seconds=1.0,
        ))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except Exception:
            proc.kill()

    text = "".join(getattr(chunk.choices[0].delta, "content", None) or "" for chunk in chunks)
    assert text == "burst"
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_persistent_stream_writes_structured_delta_not_full_prompt(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PERSISTENT", "1")
    client = ClaudeCLIClient(command="claude")
    fake_proc = _FakePersistentProc()
    structured_input = json.dumps({
        "type": "user",
        "message": {"role": "user", "content": "latest delta only"},
    })

    monkeypatch.setattr(client, "_ensure_persistent_worker", lambda **_kwargs: fake_proc)
    monkeypatch.setattr(
        claude_cli_client.select,
        "select",
        lambda readers, _writers, _errors, _timeout: ([fake_proc.stdout], [], []),
    )

    chunks = list(client._stream_completion_persistent(
        model="claude-opus-4-7",
        prompt_text="FULL HISTORY " * 1000,
        system_prompt="system",
        structured_input=structured_input,
        structured_system_prompt="system with tools",
        effort="max",
        timeout_seconds=1.0,
    ))

    assert fake_proc.stdin.writes == [structured_input + "\n"]
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_persistent_stream_uses_structured_system_prompt_for_delta_turns(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PERSISTENT", "1")
    client = ClaudeCLIClient(command="claude")
    structured_input = json.dumps({
        "type": "user",
        "message": {"role": "user", "content": "latest delta only"},
    })
    captured = {}
    fake_proc = _FakePersistentProc()

    def _capture_worker(**kwargs):
        captured.update(kwargs)
        return fake_proc

    monkeypatch.setattr(client, "_ensure_persistent_worker", _capture_worker)
    monkeypatch.setattr(
        claude_cli_client.select,
        "select",
        lambda readers, _writers, _errors, _timeout: ([fake_proc.stdout], [], []),
    )

    list(client._stream_completion_persistent(
        model="claude-opus-4-7",
        prompt_text="FULL HISTORY " * 1000,
        system_prompt="system",
        structured_input=structured_input,
        structured_system_prompt="system with tools",
        effort="max",
        timeout_seconds=1.0,
    ))

    assert captured["system_prompt"] == "system with tools"


def test_persistent_stream_emits_tagged_tool_calls_before_result(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PERSISTENT", "1")
    client = ClaudeCLIClient(command="claude")
    structured_input = json.dumps({
        "type": "user",
        "message": {"role": "user", "content": "latest delta only"},
    })
    fake_proc = _FakeStreamingPersistentProc([
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text"},
            },
        }) + "\n",
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": '<tool_call>{"name":"terminal","arguments":{"command":"pwd"}}</tool_call>',
                },
            },
        }) + "\n",
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }) + "\n",
        json.dumps({
            "type": "result",
            "session_id": "session-stream",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }) + "\n",
    ])

    monkeypatch.setattr(client, "_ensure_persistent_worker", lambda **_kwargs: fake_proc)
    monkeypatch.setattr(
        claude_cli_client.select,
        "select",
        lambda readers, _writers, _errors, _timeout: ([fake_proc.stdout], [], []),
    )

    chunks = list(client._stream_completion_persistent(
        model="claude-opus-4-7",
        prompt_text="FULL HISTORY " * 1000,
        system_prompt="system",
        structured_input=structured_input,
        structured_system_prompt="system with tools",
        effort="max",
        timeout_seconds=1.0,
    ))

    tool_chunks = [chunk for chunk in chunks if chunk.choices[0].delta.tool_calls]
    finish_chunks = [chunk for chunk in chunks if chunk.choices[0].finish_reason]

    assert len(tool_chunks) == 1
    tool_call = tool_chunks[0].choices[0].delta.tool_calls[0]
    assert tool_call.function.name == "terminal"
    assert tool_call.function.arguments == '{"command":"pwd"}'
    assert [chunk.choices[0].finish_reason for chunk in finish_chunks] == ["tool_calls"]


def test_stream_structured_turn_defers_prompt_format_until_fallback(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PERSISTENT", "1")
    client = ClaudeCLIClient(command="claude")
    chunk = object()

    monkeypatch.setattr(client, "_stream_completion_persistent", lambda **_kwargs: iter([chunk]))
    monkeypatch.setattr(
        claude_cli_client,
        "_format_messages_as_prompt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full prompt should stay lazy")),
    )

    result = list(client._create_chat_completion(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    ))

    assert result == [chunk]


def test_resume_single_user_turn_stays_structured(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PERSISTENT", "1")
    client = ClaudeCLIClient(command="claude")
    client._last_session_id = "sess-123"
    captured = {}
    chunk = object()

    def _fake_persistent(**kwargs):
        captured.update(kwargs)
        return iter([chunk])

    monkeypatch.setattr(client, "_stream_completion_persistent", _fake_persistent)
    monkeypatch.setattr(
        claude_cli_client,
        "_format_messages_as_prompt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("resume single-user turn should stay structured")),
    )

    result = list(client._create_chat_completion(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": "hello again"}],
        stream=True,
    ))

    assert result == [chunk]
    assert captured["structured_input"] is not None
    assert 'hello again' in captured["structured_input"]


def test_transport_state_roundtrip():
    client = ClaudeCLIClient(command="claude", args=[], base_url="claude-cli://local")

    assert client.export_transport_state() is None

    state = {
        "claude_session_id": "sess-123",
        "hydrated_session_id": "sess-123",
        "hydrated_prompt_hash": "abc123",
    }
    client.import_transport_state(state)

    assert client.export_transport_state() == state


def test_bootstrap_hydration_uses_runtime_system_prompt_when_provided(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PERSISTENT", "1")
    client = ClaudeCLIClient(command="claude", strip_runtime=True)
    captured_system_prompts = []

    def _fake_run_prompt(_prompt_text, *, system_prompt, model, effort, timeout_seconds):
        captured_system_prompts.append(system_prompt)
        client._last_session_id = client._last_session_id or "sess-123"
        return {"result": "OK"}

    monkeypatch.setattr(client, "_run_prompt_oneshot", _fake_run_prompt)

    client._ensure_bootstrap_hydrated(
        model="claude-opus-4-7",
        full_system_prompt="FULL SYSTEM PROMPT" * 200,
        runtime_system_prompt="runtime system prompt with tool guidance",
        effort="max",
        timeout_seconds=1.0,
    )

    assert captured_system_prompts
    assert all(prompt == "runtime system prompt with tool guidance" for prompt in captured_system_prompts)


def test_bootstrap_hydration_uses_first_chunk_to_create_session(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PERSISTENT", "1")
    client = ClaudeCLIClient(command="claude", strip_runtime=True)
    captured_prompts = []

    monkeypatch.setattr(
        claude_cli_client,
        "_chunk_hydration_text",
        lambda *_args, **_kwargs: ["chunk-a", "chunk-b"],
    )

    def _fake_run_prompt(prompt_text, *, system_prompt, model, effort, timeout_seconds):
        captured_prompts.append(prompt_text)
        client._last_session_id = client._last_session_id or "sess-123"
        return {"result": "OK"}

    monkeypatch.setattr(client, "_run_prompt_oneshot", _fake_run_prompt)

    client._ensure_bootstrap_hydrated(
        model="claude-opus-4-7",
        full_system_prompt="FULL SYSTEM PROMPT" * 200,
        runtime_system_prompt="runtime system prompt with tool guidance",
        effort="max",
        timeout_seconds=1.0,
    )

    assert len(captured_prompts) == 2
    assert captured_prompts[0].startswith("Internal Hermes context block 1/2")
    assert captured_prompts[1].startswith("Internal Hermes context block 2/2")
    assert client._hydrated_session_id == "sess-123"


def test_bootstrap_hydration_uses_larger_chunk_budget(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PERSISTENT", "1")
    client = ClaudeCLIClient(command="claude", strip_runtime=True)
    captured_max_chars = []

    def _fake_chunk(text, max_chars=0):
        captured_max_chars.append(max_chars)
        return ["chunk-a"]

    def _fake_run_prompt(_prompt_text, *, system_prompt, model, effort, timeout_seconds):
        client._last_session_id = client._last_session_id or "sess-123"
        return {"result": "OK"}

    monkeypatch.setattr(claude_cli_client, "_chunk_hydration_text", _fake_chunk)
    monkeypatch.setattr(client, "_run_prompt_oneshot", _fake_run_prompt)

    client._ensure_bootstrap_hydrated(
        model="claude-opus-4-7",
        full_system_prompt="FULL SYSTEM PROMPT" * 200,
        runtime_system_prompt="runtime system prompt with tool guidance",
        effort="max",
        timeout_seconds=1.0,
    )

    assert captured_max_chars == [22000]


def test_bootstrap_hydration_uses_low_effort_for_hidden_chunks(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PERSISTENT", "1")
    client = ClaudeCLIClient(command="claude", strip_runtime=True)
    captured_efforts = []

    monkeypatch.setattr(
        claude_cli_client,
        "_chunk_hydration_text",
        lambda *_args, **_kwargs: ["chunk-a", "chunk-b"],
    )

    def _fake_run_prompt(_prompt_text, *, system_prompt, model, effort, timeout_seconds):
        captured_efforts.append(effort)
        client._last_session_id = client._last_session_id or "sess-123"
        return {"result": "OK"}

    monkeypatch.setattr(client, "_run_prompt_oneshot", _fake_run_prompt)

    client._ensure_bootstrap_hydrated(
        model="claude-opus-4-7",
        full_system_prompt="FULL SYSTEM PROMPT" * 200,
        runtime_system_prompt="runtime system prompt with tool guidance",
        effort="max",
        timeout_seconds=1.0,
    )

    assert captured_efforts == ["low", "low"]


def test_bootstrap_hydration_bypasses_persistent_worker_runner(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CLI_PERSISTENT", "1")
    client = ClaudeCLIClient(command="claude", strip_runtime=True)

    monkeypatch.setattr(
        claude_cli_client,
        "_chunk_hydration_text",
        lambda *_args, **_kwargs: ["chunk-a"],
    )

    def _fake_oneshot(_prompt_text, *, system_prompt, model, effort, timeout_seconds):
        client._last_session_id = client._last_session_id or "sess-123"
        return {"result": "OK"}

    monkeypatch.setattr(client, "_run_prompt_oneshot", _fake_oneshot)
    monkeypatch.setattr(
        client,
        "_run_prompt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("persistent helper should not be used for hydration")),
    )

    client._ensure_bootstrap_hydrated(
        model="claude-opus-4-7",
        full_system_prompt="FULL SYSTEM PROMPT" * 200,
        runtime_system_prompt="runtime system prompt with tool guidance",
        effort="max",
        timeout_seconds=1.0,
    )

    assert client._hydrated_session_id == "sess-123"

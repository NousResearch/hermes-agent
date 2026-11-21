"""Tests for agent.trace_upload — Hugging Face session-trace upload.

Covers the Claude Code JSONL converter, HF token resolution, the no-token
message path, and the upload path with a mocked ``HfApi`` (verifying repo
id, file path, and content without touching the network).
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent import trace_upload
from agent.trace_upload import (
    build_trace_jsonl,
    load_session_messages,
    upload_session_trace,
    _resolve_hf_token,
    _do_upload,
)


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

def _sample_messages():
    return [
        {"role": "system", "content": "you are hermes"},
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": "Listing.", "tool_calls": [
            {"id": "call_1", "function": {"name": "terminal", "arguments": '{"command": "ls"}'}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "tool_name": "terminal", "content": "a.txt\nb.txt"},
        {"role": "assistant", "content": "Two files."},
    ]






def test_converter_emits_tool_use_and_tool_result():
    jsonl = build_trace_jsonl(_sample_messages(), session_id="s1", model="m")
    lines = [json.loads(x) for x in jsonl.strip().split("\n")]
    # line 0 user, line 1 assistant (text + tool_use), line 2 tool_result, line 3 assistant
    assert lines[0]["type"] == "user"
    assert lines[1]["type"] == "assistant"
    blocks = lines[1]["message"]["content"]
    assert any(b.get("type") == "text" for b in blocks)
    tool_use = [b for b in blocks if b.get("type") == "tool_use"]
    assert tool_use and tool_use[0]["name"] == "terminal"
    assert tool_use[0]["input"] == {"command": "ls"}
    # tool result rides on a user turn
    assert lines[2]["type"] == "user"
    tr = lines[2]["message"]["content"][0]
    assert tr["type"] == "tool_result"
    assert tr["tool_use_id"] == "call_1"
    assert "a.txt" in tr["content"]


def test_converter_redacts_secrets_by_default():
    msgs = [{"role": "user", "content": "key OPENAI_API_KEY=sk-abc123def456ghi789jklmno end"}]
    jsonl = build_trace_jsonl(msgs, session_id="s1", redact=True)
    assert "sk-abc123def456ghi789jklmno" not in jsonl


def test_converter_refuses_unredacted_passthrough_when_redactor_fails(monkeypatch):
    def boom(_text, *, force=False):
        raise RuntimeError("redactor unavailable")

    monkeypatch.setattr("agent.redact.redact_sensitive_text", boom)
    msgs = [{"role": "user", "content": "OPENAI_API_KEY=sk-abc123def456ghi789jklmno"}]

    with pytest.raises(trace_upload.TraceRedactionError):
        build_trace_jsonl(msgs, session_id="s1", redact=True)


def test_tool_arguments_stay_structured_when_text_redaction_would_break_json():
    secret = "DB_PASSWORD=pw1longer" + "value99"
    msgs = [{
        "role": "assistant",
        "content": "done",
        "tool_calls": [{
            "id": "call_1",
            "function": {
                "name": "terminal",
                "arguments": json.dumps({"headers": secret, "safe": "keep"}),
            },
        }],
    }]

    line = json.loads(build_trace_jsonl(msgs, session_id="s1").strip())
    tool_use = line["message"]["content"][1]
    assert tool_use["input"]["safe"] == "keep"
    assert secret not in json.dumps(tool_use["input"])


def test_upload_blocks_when_redactor_fails(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")

    def boom(_text, *, force=False):
        raise RuntimeError("redactor unavailable")

    monkeypatch.setattr("agent.redact.redact_sensitive_text", boom)
    with patch.object(trace_upload, "load_session_messages", return_value=(_sample_messages(), {})), \
         patch.object(trace_upload, "_do_upload") as upload_mock:
        msg = upload_session_trace("s1")

    assert "Trace upload blocked" in msg
    upload_mock.assert_not_called()


def test_converter_keeps_secrets_when_redact_disabled():
    secret = "sk-abc123def456ghi789jklmno"
    msgs = [{"role": "user", "content": f"key OPENAI_API_KEY={secret} end"}]
    jsonl = build_trace_jsonl(msgs, session_id="s1", redact=False)
    assert secret in jsonl


def test_load_session_messages_closes_database_on_failure(monkeypatch):
    persistence = MagicMock()
    persistence.sessions.resolve_id.side_effect = RuntimeError("read failed")
    monkeypatch.setattr("persistence_assembly.open_persistence", lambda scope: persistence)

    with pytest.raises(RuntimeError, match="read failed"):
        load_session_messages("s1")

    persistence.close.assert_called_once()


def test_load_session_messages_reads_the_conversation_and_model(tmp_path):
    """A session prefix resolves; the replay projection and the session's model come from the store the path names."""
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_read import conversation
    from tests.fixtures.storage_seed import seed_messages, seed_session

    path = tmp_path / "state.db"
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, "trace-s1", "cli", model="claude-x")
        seed_messages(db, "trace-s1", {"role": "user", "content": "list files"},
                      {"role": "assistant", "content": "Two files."})
        expected = conversation(db, "trace-s1")

    messages, meta = load_session_messages("trace-s", db_path=path)

    assert messages == expected
    assert meta == {"model": "claude-x"}








# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# Top-level upload entry point
# ---------------------------------------------------------------------------







def test_upload_happy_path_mocked(monkeypatch):
    """Full upload path with a mocked HfApi — verifies repo id / path / content."""
    pytest.importorskip("huggingface_hub")  # optional dep; runtime degrades gracefully
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    messages = _sample_messages()

    fake_api = MagicMock()
    fake_api.whoami.return_value = {"name": "alice"}

    with patch.object(trace_upload, "load_session_messages",
                      return_value=(messages, {"model": "claude-x"})), \
         patch("huggingface_hub.HfApi", return_value=fake_api):
        msg = upload_session_trace("20260531_abc", cwd="/tmp")

    # Returned a viewer URL
    assert "huggingface.co/datasets/alice/hermes-traces" in msg

    # Created private dataset repo
    fake_api.create_repo.assert_called_once()
    _, kwargs = fake_api.create_repo.call_args
    assert kwargs["repo_id"] == "alice/hermes-traces"
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["private"] is True

    # Uploaded the JSONL to sessions/<id>.jsonl
    fake_api.upload_file.assert_called_once()
    _, ukwargs = fake_api.upload_file.call_args
    assert ukwargs["path_in_repo"] == "sessions/20260531_abc.jsonl"
    assert ukwargs["repo_id"] == "alice/hermes-traces"
    body = ukwargs["path_or_fileobj"]
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    # Content is valid Claude Code JSONL
    first = json.loads(body.strip().split("\n")[0])
    assert first["type"] in ("user", "assistant")
    assert first["sessionId"] == "20260531_abc"


def test_list_held_env_secret_in_tool_arguments_keeps_siblings():
    secret = "DB_PASSWORD=pw1longer" + "value99"
    msgs = [{
        "role": "assistant",
        "content": "done",
        "tool_calls": [{
            "id": "call_1",
            "function": {
                "name": "terminal",
                "arguments": json.dumps({"env": [secret], "safe": "keep"}),
            },
        }],
    }]

    line = json.loads(build_trace_jsonl(msgs, session_id="s1").strip())
    tool_use = line["message"]["content"][1]
    assert tool_use["input"]["safe"] == "keep"
    assert secret not in json.dumps(tool_use["input"])


def test_converter_unwraps_single_deferred_tool_call_and_redacts_inner_args():
    secret = "sk-abc123def456ghi789jklmno"
    msgs = [{
        "role": "assistant",
        "content": "Calling deferred MCP tool.",
        "tool_calls": [{
            "id": "call_def_1",
            "function": {
                "name": "tool_call",
                "arguments": json.dumps({
                    "calls": [{
                        "name": "mcp__demo__deploy_service",
                        "arguments": {"api_key": secret, "environment": "staging"},
                    }]
                }),
            },
        }],
    }]
    line = json.loads(build_trace_jsonl(msgs, session_id="s1", redact=True).strip())
    tool_use = next(b for b in line["message"]["content"] if b.get("type") == "tool_use")
    assert tool_use["name"] == "mcp__demo__deploy_service"
    assert tool_use["input"]["environment"] == "staging"
    assert secret not in json.dumps(tool_use["input"])


def test_converter_preserves_connector_batch_or_malformed_tool_call():
    single_connector = {
        "calls": [{"name": "connectors__gmail__list", "arguments": {}}]
    }
    batch_args = {
        "calls": [
            {"name": "connectors__gmail__search", "arguments": {"q": "invoice"}},
            {"name": "connectors__gmail__list", "arguments": {}},
        ]
    }
    legacy_single = {"name": "mcp__demo__ping", "arguments": {"host": "127.0.0.1"}}
    msgs = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_b0", "function": {"name": "tool_call", "arguments": json.dumps(single_connector)}},
            {"id": "call_b1", "function": {"name": "tool_call", "arguments": json.dumps(batch_args)}},
            {"id": "call_b2", "function": {"name": "tool_call", "arguments": "{}"}},
            {"id": "call_b3", "function": {"name": "tool_call", "arguments": json.dumps(legacy_single)}},
        ],
    }]
    line = json.loads(build_trace_jsonl(msgs, session_id="s1", redact=False).strip())
    uses = [b for b in line["message"]["content"] if b.get("type") == "tool_use"]
    assert uses[0]["name"] == "tool_call"
    assert uses[1]["name"] == "tool_call"
    assert len(uses[1]["input"]["calls"]) == 2
    assert uses[2]["name"] == "tool_call"
    assert uses[3]["name"] == "mcp__demo__ping"
    assert uses[3]["input"] == {"host": "127.0.0.1"}


def test_converter_emits_is_error_only_on_failed_tool_results():
    untrusted_err = (
        '<untrusted_tool_result source="mcp__demo__check">\n'
        'The following content was retrieved from an external source.\n\n'
        '{"error": "PRESERVATION_REFUSED: outside workspace"}\n'
        '</untrusted_tool_result>'
    )
    msgs = [
        {"role": "tool", "tool_call_id": "ok_plain", "content": "all good"},
        {"role": "tool", "tool_call_id": "ok_json", "content": json.dumps({"output": "ok", "exit_code": 0, "error": None})},
        {"role": "tool", "tool_call_id": "err_json", "content": json.dumps({"error": "Permission denied"})},
        {"role": "tool", "tool_call_id": "err_mcp", "content": untrusted_err},
        {"role": "tool", "tool_call_id": "err_timeout", "content": json.dumps({"output": "", "exit_code": 124, "error": "Command timed out", "timed_out": True})},
        {"role": "tool", "tool_call_id": "err_flag", "is_error": True, "content": "boom"},
    ]
    rows = [json.loads(x)["message"]["content"][0] for x in build_trace_jsonl(msgs, session_id="s1", redact=False).strip().splitlines()]
    by_id = {r["tool_use_id"]: r for r in rows}
    assert "is_error" not in by_id["ok_plain"]
    assert "is_error" not in by_id["ok_json"]
    assert by_id["err_json"]["is_error"] is True
    assert by_id["err_mcp"]["is_error"] is True
    assert by_id["err_timeout"]["is_error"] is True
    assert by_id["err_flag"]["is_error"] is True

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
    secret = "x-api-key: abc123"
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


def test_tool_argument_redactor_failure_is_fail_closed(monkeypatch):
    def boom(_text, *, force=False):
        raise RuntimeError("redactor unavailable")

    monkeypatch.setattr("agent.redact.redact_sensitive_text", boom)
    msgs = [{
        "role": "assistant",
        "content": "done",
        "tool_calls": [{
            "id": "call_1",
            "function": {"name": "terminal", "arguments": '{"safe": "value"}'},
        }],
    }]

    with pytest.raises(trace_upload.TraceRedactionError):
        build_trace_jsonl(msgs, session_id="s1")


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
    db = MagicMock()
    db.resolve_session_id.side_effect = RuntimeError("read failed")
    monkeypatch.setattr("hermes_state.SessionDB", lambda db_path=None: db)

    with pytest.raises(RuntimeError, match="read failed"):
        load_session_messages("s1")

    db.close.assert_called_once()








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






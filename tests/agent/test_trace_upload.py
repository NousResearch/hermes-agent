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


def test_converter_skips_system_and_counts_lines():
    jsonl = build_trace_jsonl(_sample_messages(), session_id="s1", model="m")
    lines = [json.loads(x) for x in jsonl.strip().split("\n")]
    assert len(lines) == 4  # system dropped
    assert all(o["sessionId"] == "s1" for o in lines)


def test_converter_links_turns_as_linked_list():
    jsonl = build_trace_jsonl(_sample_messages(), session_id="s1")
    lines = [json.loads(x) for x in jsonl.strip().split("\n")]
    prev = None
    for o in lines:
        assert o["parentUuid"] == prev
        prev = o["uuid"]


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


def test_converter_image_placeholder():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]
    jsonl = build_trace_jsonl(msgs, session_id="s1")
    line = json.loads(jsonl.strip())
    assert any("image omitted" in b.get("text", "") for b in line["message"]["content"])
    assert "AAAA" not in jsonl


def test_converter_empty_messages_returns_empty():
    assert build_trace_jsonl([], session_id="s1") == ""


def test_converter_handles_dict_tool_arguments():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c", "function": {"name": "f", "arguments": {"already": "dict"}}},
    ]}]
    jsonl = build_trace_jsonl(msgs, session_id="s1")
    line = json.loads(jsonl.strip())
    tu = [b for b in line["message"]["content"] if b.get("type") == "tool_use"][0]
    assert tu["input"] == {"already": "dict"}


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------

def test_resolve_token_prefers_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_primary")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_secondary")
    assert _resolve_hf_token() == "hf_primary"


def test_resolve_token_falls_back(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_fallback")
    assert _resolve_hf_token() == "hf_fallback"


def test_resolve_token_none(monkeypatch):
    for v in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    assert _resolve_hf_token() is None


# ---------------------------------------------------------------------------
# Top-level upload entry point
# ---------------------------------------------------------------------------

def test_upload_no_session_id():
    assert "No active session" in upload_session_trace("")


def test_upload_no_token(monkeypatch):
    for v in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    msg = upload_session_trace("some_session")
    assert "no Hugging Face token" in msg


def test_upload_empty_transcript(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    with patch.object(trace_upload, "load_session_messages", return_value=([], {})):
        msg = upload_session_trace("s1")
    assert "No transcript" in msg


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


def test_upload_public_flag(monkeypatch):
    pytest.importorskip("huggingface_hub")  # optional dep
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    fake_api = MagicMock()
    fake_api.whoami.return_value = {"name": "bob"}
    with patch.object(trace_upload, "load_session_messages",
                      return_value=(_sample_messages(), {})), \
         patch("huggingface_hub.HfApi", return_value=fake_api):
        upload_session_trace("s1", private=False)
    _, kwargs = fake_api.create_repo.call_args
    assert kwargs["private"] is False


def test_upload_whoami_failure(monkeypatch):
    pytest.importorskip("huggingface_hub")  # optional dep
    monkeypatch.setenv("HF_TOKEN", "hf_bad")
    fake_api = MagicMock()
    fake_api.whoami.side_effect = Exception("401 unauthorized")
    with patch.object(trace_upload, "load_session_messages",
                      return_value=(_sample_messages(), {})), \
         patch("huggingface_hub.HfApi", return_value=fake_api):
        msg = upload_session_trace("s1")
    assert "token was rejected" in msg


def test_do_upload_missing_huggingface_hub(monkeypatch):
    """If huggingface_hub import fails, return a clear install hint."""
    # Disable lazy-install so the import path deterministically fails here
    # instead of attempting a real pip install in CI.
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    msg = _do_upload("{}\n", token="t", session_id="s1")
    assert "huggingface_hub" in msg

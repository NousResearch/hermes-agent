"""Persistence-boundary redaction tests (#43666)."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


SECRET_PW = "honchorulez"
SECRET_URI = f"postgresql+psycopg://postgres:{SECRET_PW}@127.0.0.1:5432/postgres"
PG_ARGS = '{"command": "PGPASSWORD=\'honchorulez\' psql -h 127.0.0.1"}'
BEARER_ARGS = '{"command": "curl -H \'Authorization: Bearer sk-abcdef1234567890\'"}'
DB_URI_ARGS = json.dumps({"command": f"psql {SECRET_URI}"})


@pytest.fixture(autouse=True)
def _redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


def _make_agent():
    """Minimal agent exposing the real _build_assistant_message."""
    from run_agent import AIAgent

    agent = MagicMock(spec=AIAgent)
    agent._build_assistant_message = AIAgent._build_assistant_message.__get__(agent)
    agent._extract_reasoning = AIAgent._extract_reasoning.__get__(agent)
    agent._strip_think_blocks = AIAgent._strip_think_blocks.__get__(agent)
    agent.verbose_logging = False
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent._stream_callback = None
    agent._needs_thinking_reasoning_pad.return_value = False
    agent._split_responses_tool_id.return_value = (None, None)
    agent._derive_responses_function_call_id.side_effect = lambda cid, rid: rid or cid
    return agent


def _api_msg(content, **fields):
    msg = SimpleNamespace(content=content, tool_calls=fields.pop("tool_calls", None))
    for key, value in fields.items():
        setattr(msg, key, value)
    return msg


def test_reasoning_redacted():
    agent = _make_agent()
    result = agent._build_assistant_message(
        _api_msg("done", reasoning=f"I will connect via {SECRET_URI}"), "stop"
    )
    assert SECRET_PW not in result["reasoning"]
    assert ":***@" in result["reasoning"]
    assert SECRET_PW not in result["reasoning_content"]
    assert ":***@" in result["reasoning_content"]


def test_reasoning_content_from_model_extra_redacted():
    agent = _make_agent()
    msg = _api_msg("done")
    msg.model_extra = {"reasoning_content": f"model extra {SECRET_URI}"}
    result = agent._build_assistant_message(msg, "stop")
    assert SECRET_PW not in result["reasoning_content"]
    assert ":***@" in result["reasoning_content"]


def test_tool_call_reasoning_pad_uses_redacted_reasoning():
    agent = _make_agent()
    agent._needs_thinking_reasoning_pad.return_value = True
    tc = SimpleNamespace(
        id="call_1",
        call_id="call_1",
        response_item_id="fc_1",
        type="function",
        function=SimpleNamespace(name="terminal", arguments=PG_ARGS),
        extra_content=None,
    )
    result = agent._build_assistant_message(
        _api_msg("", reasoning=f"thinking {SECRET_URI}", tool_calls=[tc]),
        "tool_calls",
    )
    assert SECRET_PW not in result["reasoning_content"]
    assert result["tool_calls"][0]["function"]["arguments"] == PG_ARGS


@pytest.mark.parametrize("field", ["text", "summary", "thinking", "content"])
def test_unsigned_reasoning_detail_plain_text_fields_redacted(field):
    agent = _make_agent()
    details = [{"type": "reasoning.text", field: f"using {SECRET_URI}"}]
    result = agent._build_assistant_message(
        _api_msg("done", reasoning="r", reasoning_details=details), "stop"
    )
    assert SECRET_PW not in result["reasoning_details"][0][field]
    assert ":***@" in result["reasoning_details"][0][field]


def test_reasoning_detail_with_unknown_data_still_redacts_text():
    agent = _make_agent()
    details = [{"type": "reasoning.text", "data": "not-opaque", "text": f"using {SECRET_URI}"}]
    result = agent._build_assistant_message(
        _api_msg("done", reasoning="r", reasoning_details=details), "stop"
    )
    assert SECRET_PW not in result["reasoning_details"][0]["text"]
    assert result["reasoning_details"][0]["data"] == "not-opaque"


def test_reasoning_detail_summary_list_redacts_nested_text():
    agent = _make_agent()
    details = [
        {
            "type": "reasoning.summary",
            "summary": [{"type": "summary_text", "text": f"used {SECRET_URI}"}],
        }
    ]
    result = agent._build_assistant_message(
        _api_msg("done", reasoning="r", reasoning_details=details), "stop"
    )
    nested_text = result["reasoning_details"][0]["summary"][0]["text"]
    assert SECRET_PW not in nested_text
    assert ":***@" in nested_text


def test_signed_detail_preserved_equal_not_identity_required():
    agent = _make_agent()
    signed = {
        "type": "reasoning.text",
        "text": f"via {SECRET_URI}",
        "signature": "sig-abc123",
    }
    result = agent._build_assistant_message(
        _api_msg("done", reasoning="r", reasoning_details=[signed]), "stop"
    )
    assert result["reasoning_details"][0] == signed


def test_encrypted_content_detail_preserved_equal_not_identity_required():
    agent = _make_agent()
    encrypted = {
        "type": "reasoning.text",
        "text": f"via {SECRET_URI}",
        "encrypted_content": "opaque-provider-material",
    }
    result = agent._build_assistant_message(
        _api_msg("done", reasoning="r", reasoning_details=[encrypted]), "stop"
    )
    assert result["reasoning_details"][0] == encrypted


def test_encrypted_detail_preserved_equal_not_identity_required():
    agent = _make_agent()
    encrypted = {
        "type": "reasoning.encrypted",
        "data": f"opaque copy may contain {SECRET_URI}",
    }
    result = agent._build_assistant_message(
        _api_msg("done", reasoning="r", reasoning_details=[encrypted]), "stop"
    )
    assert result["reasoning_details"][0] == encrypted


def test_object_shaped_reasoning_details_are_redacted():
    class DictDetail:
        def __init__(self):
            self.type = "reasoning.text"
            self.text = f"using {SECRET_URI}"

    agent = _make_agent()
    result = agent._build_assistant_message(
        _api_msg("done", reasoning="r", reasoning_details=[DictDetail()]), "stop"
    )
    assert SECRET_PW not in result["reasoning_details"][0]["text"]
    assert ":***@" in result["reasoning_details"][0]["text"]


def test_model_dump_reasoning_details_are_redacted():
    class DumpDetail:
        __slots__ = ()

        def model_dump(self):
            return {"type": "reasoning.text", "text": f"using {SECRET_URI}"}

    agent = _make_agent()
    result = agent._build_assistant_message(
        _api_msg("done", reasoning="r", reasoning_details=[DumpDetail()]), "stop"
    )
    assert SECRET_PW not in result["reasoning_details"][0]["text"]
    assert ":***@" in result["reasoning_details"][0]["text"]


def test_reasoning_callback_receives_redacted_reasoning():
    agent = _make_agent()
    agent.reasoning_callback = MagicMock()
    agent._build_assistant_message(
        _api_msg("done", reasoning=f"callback {SECRET_URI}"), "stop"
    )
    callback_text = agent.reasoning_callback.call_args.args[0]
    assert SECRET_PW not in callback_text
    assert ":***@" in callback_text


def test_streamed_reasoning_callback_receives_redacted_reasoning():
    from run_agent import AIAgent

    agent = MagicMock(spec=AIAgent)
    agent.reasoning_callback = MagicMock()
    AIAgent._fire_reasoning_delta(agent, f"stream {SECRET_URI}")
    callback_text = agent.reasoning_callback.call_args.args[0]
    assert SECRET_PW not in callback_text
    assert ":***@" in callback_text


def test_redaction_disabled_is_noop_for_reasoning_fields(monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    agent = _make_agent()
    details = [{"type": "reasoning.text", "text": f"using {SECRET_URI}"}]
    result = agent._build_assistant_message(
        _api_msg(
            f"uri: {SECRET_URI}",
            reasoning=f"via {SECRET_URI}",
            reasoning_content=f"provider {SECRET_URI}",
            reasoning_details=details,
        ),
        "stop",
    )
    assert SECRET_URI in result["content"]
    assert SECRET_URI in result["reasoning"]
    assert SECRET_URI in result["reasoning_content"]
    assert SECRET_URI in result["reasoning_details"][0]["text"]


def test_builder_redacted_reasoning_fields_are_what_state_db_stores(tmp_path):
    from hermes_state import SessionDB

    agent = _make_agent()
    built = agent._build_assistant_message(
        _api_msg(
            "connected",
            reasoning=f"used {SECRET_URI}",
            reasoning_content=f"provider scratch {SECRET_URI}",
            reasoning_details=[{"type": "reasoning.text", "text": f"detail {SECRET_URI}"}],
        ),
        "stop",
    )

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = db.create_session("sess-1", "cli")
    db.append_message(session_id=sid, **built)
    replayed = db.get_messages_as_conversation(sid)
    db.close()

    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert SECRET_PW.encode() not in path.read_bytes(), path.name

    assistant = [m for m in replayed if m["role"] == "assistant"][0]
    assert assistant["reasoning"] == built["reasoning"]
    assert assistant["reasoning_content"] == built["reasoning_content"]
    assert assistant["reasoning_details"] == built["reasoning_details"]


@pytest.mark.parametrize("arguments", [PG_ARGS, BEARER_ARGS, DB_URI_ARGS])
def test_tool_call_arguments_round_trip_through_state_db_verbatim(arguments, tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = db.create_session("sess-1", "cli")
    db.append_message(
        session_id=sid,
        role="assistant",
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_1",
                "call_id": "call_1",
                "response_item_id": "fc_1",
                "type": "function",
                "function": {"name": "terminal", "arguments": arguments},
            }
        ],
    )
    replayed = db.get_messages_as_conversation(sid)
    db.close()

    assistant = [m for m in replayed if m["role"] == "assistant"][0]
    got = assistant["tool_calls"][0]["function"]["arguments"]
    assert got == arguments
    assert "***" not in got


@pytest.fixture()
def compressor():
    from agent.context_compressor import ContextCompressor

    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )


def _turns():
    return [
        {"role": "user", "content": "set up the database"},
        {
            "role": "assistant",
            "content": "running it",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": json.dumps({"command": f"psql {SECRET_URI}"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": f"connected via {SECRET_URI}",
        },
    ]


def test_summarizer_input_redacts_copy_without_mutating_replay_turns(compressor):
    turns = _turns()
    original = copy.deepcopy(turns)
    serialized = compressor._serialize_for_summary(turns)
    assert SECRET_PW not in serialized
    assert ":***@" in serialized
    assert "terminal" in serialized
    assert turns == original


def test_static_fallback_summary_redacts_copy_without_mutating_replay_turns(compressor):
    turns = _turns()
    original = copy.deepcopy(turns)
    summary = compressor._build_static_fallback_summary(turns, reason="test")
    assert SECRET_PW not in summary
    assert ":***@" in summary
    assert turns == original

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


def _make_replay_agent(monkeypatch, *, provider, base_url, model, api_mode="chat_completions"):
    """Build a real request agent without a network client."""
    import run_agent
    from run_agent import AIAgent

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.api_key = kwargs.get("api_key")
            self.base_url = kwargs.get("base_url")

        def close(self):
            pass

    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr(run_agent, "OpenAI", FakeOpenAI)
    return AIAgent(
        api_key="test-key",
        base_url=base_url,
        provider=provider,
        model=model,
        api_mode=api_mode,
        max_iterations=4,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def test_reasoning_redacted():
    agent = _make_agent()
    result = agent._build_assistant_message(
        _api_msg("done", reasoning=f"I will connect via {SECRET_URI}"), "stop"
    )
    assert SECRET_PW not in result["reasoning"]
    assert ":***@" in result["reasoning"]
    assert SECRET_PW not in result["reasoning_content"]
    assert ":***@" in result["reasoning_content"]


def test_reasoning_content_from_model_extra_preserved_for_provider_replay():
    agent = _make_agent()
    msg = _api_msg("done")
    provider_reasoning = f"model extra {SECRET_URI}"
    msg.model_extra = {"reasoning_content": provider_reasoning}
    result = agent._build_assistant_message(msg, "stop")
    assert result["reasoning_content"] == provider_reasoning


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


def test_nested_reasoning_summary_is_extracted_and_redacted():
    details = [
        {
            "type": "reasoning.summary",
            "summary": [{"type": "summary_text", "text": f"used {SECRET_URI}"}],
        }
    ]

    result = _make_agent()._build_assistant_message(
        _api_msg("done", reasoning_details=details), "stop"
    )

    assert result["reasoning"].startswith("used ")
    assert SECRET_PW not in result["reasoning"]
    assert ":***@" in result["reasoning"]


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


def test_encrypted_type_with_plaintext_shaped_field_is_preserved():
    encrypted = {
        "type": "reasoning.encrypted",
        "data": "opaque-provider-material",
        "text": f"provider bytes {SECRET_URI}",
    }

    result = _make_agent()._build_assistant_message(
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


def test_disabled_redaction_preserves_callback_input_before_storage_sanitization(monkeypatch, tmp_path):
    """Disabled redaction keeps callback bytes while persistence stays JSON-safe."""
    from hermes_state import SessionDB

    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    raw_reasoning = f"callback {SECRET_URI} \udce2"
    stored_reasoning = f"callback {SECRET_URI} \ufffd"
    agent = _make_agent()
    agent.reasoning_callback = MagicMock()

    built = agent._build_assistant_message(
        _api_msg("done", reasoning=raw_reasoning), "stop"
    )

    assert agent.reasoning_callback.call_args.args[0] == raw_reasoning
    assert built["reasoning"] == stored_reasoning
    assert built["reasoning_content"] == stored_reasoning

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = db.create_session("sess-disabled-redaction", "cli")
    db.append_message(session_id=sid, **built)
    replayed = db.get_messages_as_conversation(sid)
    db.close()

    assistant = next(message for message in replayed if message["role"] == "assistant")
    assert assistant["reasoning"] == stored_reasoning
    assert assistant["reasoning_content"] == stored_reasoning


def test_redaction_disabled_is_exact_noop_for_builder_callback_and_state_db(monkeypatch, tmp_path):
    """The explicit opt-out preserves mutable and opaque replay fields exactly."""
    from hermes_state import SessionDB

    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    raw_content = f"content {SECRET_URI}"
    raw_reasoning = f"reasoning {SECRET_URI}"
    raw_details = [
        {"type": "reasoning.text", "text": f"detail {SECRET_URI}"},
        {
            "type": "thinking",
            "thinking": f"signed {SECRET_URI}",
            "signature": "opaque-signature",
        },
    ]
    original_details = copy.deepcopy(raw_details)
    expected_reasoning = "\n\n".join(
        [raw_reasoning, raw_details[0]["text"], raw_details[1]["thinking"]]
    )
    agent = _make_agent()
    agent.reasoning_callback = MagicMock()

    built = agent._build_assistant_message(
        _api_msg(
            raw_content,
            reasoning=raw_reasoning,
            reasoning_content=raw_reasoning,
            reasoning_details=raw_details,
        ),
        "stop",
    )

    assert built["content"] == raw_content
    assert built["reasoning"] == expected_reasoning
    assert built["reasoning_content"] == raw_reasoning
    assert built["reasoning_details"] == original_details
    assert raw_details == original_details
    assert agent.reasoning_callback.call_args.args[0] == expected_reasoning

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = db.create_session("sess-exact-noop", "cli")
    db.append_message(session_id=sid, **built)
    replayed = db.get_messages_as_conversation(sid)
    db.close()

    assistant = next(message for message in replayed if message["role"] == "assistant")
    assert assistant["content"] == raw_content
    assert assistant["reasoning"] == expected_reasoning
    assert assistant["reasoning_content"] == raw_reasoning
    assert assistant["reasoning_details"] == original_details


def test_reasoning_detail_redaction_does_not_mutate_raw_input():
    """Only mutable text is copied and redacted; provider payloads stay intact."""
    agent = _make_agent()
    raw_details = [
        {
            "type": "reasoning.summary",
            "summary": [{"type": "summary_text", "text": f"detail {SECRET_URI}"}],
        },
        {
            "type": "thinking",
            "thinking": f"signed {SECRET_URI}",
            "signature": "opaque-signature",
        },
        {
            "type": "reasoning.encrypted",
            "data": f"encrypted {SECRET_URI}",
        },
    ]
    original_details = copy.deepcopy(raw_details)

    built = agent._build_assistant_message(
        _api_msg("done", reasoning="summary", reasoning_details=raw_details), "stop"
    )

    assert raw_details == original_details
    assert SECRET_PW not in built["reasoning_details"][0]["summary"][0]["text"]
    assert built["reasoning_details"][1] == original_details[1]
    assert built["reasoning_details"][2] == original_details[2]


def test_reasoning_callback_failure_does_not_block_persistence(tmp_path):
    """A display callback failure must not prevent the redacted turn from saving."""
    from hermes_state import SessionDB

    agent = _make_agent()
    agent.reasoning_callback = MagicMock(side_effect=RuntimeError("display unavailable"))
    built = agent._build_assistant_message(
        _api_msg(
            "done",
            reasoning=f"reasoning {SECRET_URI}",
            reasoning_details=[{"type": "reasoning.text", "text": f"detail {SECRET_URI}"}],
        ),
        "stop",
    )

    assert agent.reasoning_callback.call_count == 1
    assert SECRET_PW not in built["reasoning"]
    assert SECRET_PW not in built["reasoning_details"][0]["text"]

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = db.create_session("sess-callback-error", "cli")
    db.append_message(session_id=sid, **built)
    replayed = db.get_messages_as_conversation(sid)
    db.close()

    assistant = next(message for message in replayed if message["role"] == "assistant")
    assert assistant["reasoning"] == built["reasoning"]
    assert assistant["reasoning_details"] == built["reasoning_details"]


def test_persisted_reasoning_replays_through_provider_request_builders(monkeypatch, tmp_path):
    """The stored form works for both thinking-echo and strict next turns."""
    from hermes_state import SessionDB

    provider_reasoning = f"provider {SECRET_URI}"
    built = _make_agent()._build_assistant_message(
        _api_msg(
            "done",
            reasoning=f"reasoning {SECRET_URI}",
            reasoning_content=provider_reasoning,
            reasoning_details=[
                {"type": "reasoning.summary", "summary": f"detail {SECRET_URI}"},
                {
                    "type": "thinking",
                    "thinking": f"signed {SECRET_URI}",
                    "signature": "opaque-signature",
                },
            ],
        ),
        "stop",
    )
    assert SECRET_PW not in built["reasoning"]
    assert built["reasoning_content"] == provider_reasoning
    assert built["reasoning_details"][1]["signature"] == "opaque-signature"

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = db.create_session("sess-provider-replay", "cli")
    db.append_message(session_id=sid, **built)
    replayed = db.get_messages_as_conversation(sid)
    db.close()
    stored_assistant = next(message for message in replayed if message["role"] == "assistant")

    thinking_agent = _make_replay_agent(
        monkeypatch,
        provider="openrouter",
        base_url="https://api.kimi.com/coding/v1",
        model="kimi-k2-thinking",
    )
    thinking_request = stored_assistant.copy()
    thinking_agent._copy_reasoning_content_for_api(stored_assistant, thinking_request)
    thinking_request.pop("reasoning", None)
    thinking_request.pop("finish_reason", None)
    thinking_kwargs = thinking_agent._build_api_kwargs(
        [{"role": "user", "content": "continue"}, thinking_request]
    )
    thinking_message = thinking_kwargs["messages"][-1]
    assert thinking_message["reasoning_content"] == provider_reasoning
    assert thinking_message["reasoning_details"][1]["signature"] == "opaque-signature"

    strict_agent = _make_replay_agent(
        monkeypatch,
        provider="mistral",
        base_url="https://api.mistral.ai/v1",
        model="mistral-large-latest",
    )
    strict_request = stored_assistant.copy()
    strict_agent._copy_reasoning_content_for_api(stored_assistant, strict_request)
    strict_request.pop("reasoning", None)
    strict_request.pop("finish_reason", None)
    strict_kwargs = strict_agent._build_api_kwargs(
        [{"role": "user", "content": "continue"}, strict_request]
    )
    strict_message = strict_kwargs["messages"][-1]
    assert "reasoning_content" not in strict_message


@pytest.mark.parametrize(
    ("detail", "should_redact"),
    [
        (
            {
                "type": "reasoning.summary",
                "summary": [{"type": "summary_text", "text": f"summary {SECRET_URI}"}],
            },
            True,
        ),
        ({"type": "thinking", "thinking": f"unsigned {SECRET_URI}"}, True),
        ({"type": "redacted_thinking", "data": f"opaque {SECRET_URI}"}, False),
        (
            {
                "type": "thinking",
                "thinking": f"signed {SECRET_URI}",
                "signature": "opaque-signature",
            },
            False,
        ),
        (
            {
                "type": "reasoning.text",
                "text": f"encrypted {SECRET_URI}",
                "encrypted_content": "opaque-ciphertext",
            },
            False,
        ),
    ],
)
def test_reasoning_detail_provider_shape_matrix(detail, should_redact):
    """Mutable provider text is redacted without changing opaque detail shapes."""
    original = copy.deepcopy(detail)
    result = _make_agent()._build_assistant_message(
        _api_msg("done", reasoning="summary", reasoning_details=[detail]), "stop"
    )
    stored = result["reasoning_details"][0]

    assert detail == original
    if should_redact:
        assert SECRET_PW not in str(stored)
    else:
        assert stored == original


def test_codex_and_gemini_replay_carriers_survive_state_db(monkeypatch, tmp_path):
    """Encrypted Codex items and Gemini signatures survive DB-backed replay."""
    from hermes_state import SessionDB

    codex_items = [{"type": "reasoning", "id": "rs_1", "encrypted_content": "blob"}]
    thought_signature = {"google": {"thought_signature": "opaque-signature"}}
    tool_call = SimpleNamespace(
        id="call_1",
        call_id="call_1",
        response_item_id="fc_1",
        type="function",
        function=SimpleNamespace(name="terminal", arguments=DB_URI_ARGS),
        extra_content=thought_signature,
    )

    built = _make_agent()._build_assistant_message(
        _api_msg(
            "done",
            reasoning="summary",
            tool_calls=[tool_call],
            codex_reasoning_items=codex_items,
        ),
        "tool_calls",
    )

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = db.create_session("sess-codex-gemini-replay", "cli")
    db.append_message(session_id=sid, **built)
    replayed = db.get_messages_as_conversation(sid)
    db.close()
    stored_assistant = next(message for message in replayed if message["role"] == "assistant")
    assert stored_assistant["codex_reasoning_items"] == codex_items

    codex_agent = _make_replay_agent(
        monkeypatch,
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        model="gpt-5-codex",
        api_mode="codex_responses",
    )
    codex_kwargs = codex_agent._build_api_kwargs(
        [{"role": "user", "content": "continue"}, stored_assistant]
    )
    codex_reasoning = [
        item for item in codex_kwargs["input"] if item.get("type") == "reasoning"
    ]
    assert len(codex_reasoning) == 1
    assert codex_reasoning[0]["encrypted_content"] == codex_items[0]["encrypted_content"]

    gemini_agent = _make_replay_agent(
        monkeypatch,
        provider="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model="gemini-3-pro-preview",
    )
    gemini_kwargs = gemini_agent._build_api_kwargs(
        [{"role": "user", "content": "continue"}, stored_assistant]
    )
    gemini_message = next(
        message for message in gemini_kwargs["messages"] if message["role"] == "assistant"
    )
    assert gemini_message["tool_calls"][0]["extra_content"] == thought_signature
    assert gemini_message["tool_calls"][0]["function"]["arguments"] == DB_URI_ARGS


def test_builder_redacts_mutable_reasoning_while_preserving_provider_echo(tmp_path):
    from hermes_state import SessionDB

    agent = _make_agent()
    provider_reasoning = f"provider scratch {SECRET_URI}"
    built = agent._build_assistant_message(
        _api_msg(
            "connected",
            reasoning=f"used {SECRET_URI}",
            reasoning_content=provider_reasoning,
            reasoning_details=[{"type": "reasoning.text", "text": f"detail {SECRET_URI}"}],
        ),
        "stop",
    )

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = db.create_session("sess-1", "cli")
    db.append_message(session_id=sid, **built)
    replayed = db.get_messages_as_conversation(sid)
    db.close()

    assistant = [m for m in replayed if m["role"] == "assistant"][0]
    assert SECRET_PW not in assistant["reasoning"]
    assert SECRET_PW not in assistant["reasoning_details"][0]["text"]
    assert assistant["reasoning_content"] == provider_reasoning
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

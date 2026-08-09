"""Strict redaction at every compaction text boundary (issue #43666 item 2).

Compaction summaries persist across sessions and re-enter every subsequent
summarizer prompt, so ``_redact_compaction_text()`` applies strict mode
(``force=True, redact_url_credentials=True``) at each boundary:

- serializer input (``_serialize_for_summary``: message content + tool args)
- deterministic fallback summary (``_build_static_fallback_summary``)
- summarizer LLM output (``_generate_summary`` return / ``_previous_summary``)
- focus text (manual ``/compress <focus>`` and ``_derive_auto_focus_topic``)
- previous-summary re-entry into the iterative-update prompt

Every test disables the global redaction flag (simulating
``security.redact_secrets: false``) to prove ``force=True`` still redacts at
the persistence boundary, and uses an OAuth-callback-style URL to prove
``redact_url_credentials=True`` strips opaque URL tokens that default-mode
redaction deliberately passes through.
"""

from copy import deepcopy
import json
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
    SUMMARY_PREFIX,
    _redact_compaction_text,
)

SECRET = "sk-proj-" + ("a" * 40)
OAUTH_URL = (
    "https://localhost/callback?code=opaque-code-123"
    "&access_token=opaque-token-456&state=keep"
)
ALTERNATE_JSON_SECRET = "S/Key"
ALTERNATE_JSON_SPELLING = r"\u0053\/K\u0065y"


def _nested_alternate_json_spelling(layers: int) -> str:
    spelling = ALTERNATE_JSON_SPELLING
    for _ in range(layers - 1):
        spelling = json.dumps(spelling)[1:-1]
    return spelling


ALTERNATE_JSON_SPELLINGS = tuple(
    pytest.param(
        _nested_alternate_json_spelling(layers),
        id=f"json-escape-layers-{layers}",
    )
    for layers in (1, 2, 5, 10, 13)
)
# A depth-13 spelling alone exceeds the micro-compactor's exchange-selection
# budget, so that shared scanner depth is exercised at provider, memory, and
# batch boundaries instead. Micro still covers nested depth ten end-to-end.
MICRO_ALTERNATE_JSON_SPELLINGS = ALTERNATE_JSON_SPELLINGS[:-1]
MALFORMED_JSON_FRAGMENT_KINDS = (
    "unterminated",
    "invalid_escape",
    "quote_parity",
    "unquoted",
    "single_quoted",
)


def _json_with_alternate_secret(
    field: str, spelling: str = ALTERNATE_JSON_SPELLING
) -> str:
    return f'{{"{field}":"{spelling}"}}'


def _malformed_json_with_alternate_secret(field: str, kind: str) -> str:
    if kind == "unterminated":
        return f'{{"{field}":"{ALTERNATE_JSON_SPELLING}'
    if kind == "invalid_escape":
        return f'{{"{field}":"\\q{ALTERNATE_JSON_SPELLING}"}}'
    if kind == "quote_parity":
        return f'{{"broken":"prefix "{field}":"{ALTERNATE_JSON_SPELLING}"}}'
    if kind == "unquoted":
        return f"{{{field}:{ALTERNATE_JSON_SPELLING}}}"
    if kind == "single_quoted":
        return f"{{'{field}':'{ALTERNATE_JSON_SPELLING}'}}"
    raise AssertionError(f"unknown malformed JSON fragment kind: {kind}")


@pytest.fixture(autouse=True)
def _redaction_globally_disabled(monkeypatch):
    """Simulate security.redact_secrets: false — force=True must still win."""
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)


def _compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100000,
    ):
        return ContextCompressor(model="test/model", quiet_mode=True)


def _response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _assert_clean(text: str):
    assert SECRET not in text
    assert "sk-proj-" not in text
    assert "code=opaque-code-123" not in text
    assert "access_token=opaque-token-456" not in text
    assert "code=***" in text
    assert "access_token=***" in text
    assert "state=keep" in text


def test_helper_is_strict_even_when_redaction_disabled():
    result = _redact_compaction_text(f"key {SECRET} url {OAUTH_URL}")
    _assert_clean(result)
    # None-safety: helper is used on optional fields.
    assert _redact_compaction_text(None) == ""


def test_serializer_input_redacts_content_and_tool_args():
    c = _compressor()
    messages = [
        {"role": "user", "content": f"token {SECRET} url {OAUTH_URL}"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "terminal",
                        "arguments": (
                            f'{{"command": "curl {OAUTH_URL}",'
                            f' "note": "{SECRET}"}}'
                        ),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": f"got {SECRET}"},
    ]

    serialized = c._serialize_for_summary(messages)

    _assert_clean(serialized)


def test_fallback_summary_redacts_secrets():
    c = _compressor()
    turns = [
        {"role": "user", "content": f"deploy with {SECRET} via {OAUTH_URL}"},
        {"role": "assistant", "content": f"ran curl {OAUTH_URL}"},
    ]

    summary = c._build_static_fallback_summary(turns, reason="test outage")

    _assert_clean(summary)


def test_summary_output_redacts_llm_echoed_secrets():
    c = _compressor()
    leaked = f"Summary leaked OPENAI_API_KEY {SECRET} and {OAUTH_URL}"

    with patch(
        "agent.context_compressor.call_llm", return_value=_response(leaked)
    ):
        summary = c._generate_summary([{"role": "user", "content": "hi"}])

    assert summary is not None
    _assert_clean(summary)
    # The stored iterative-update seed must be clean too.
    _assert_clean(c._previous_summary)


def test_manual_focus_topic_redacted_before_summary_prompt():
    c = _compressor()
    turns = [
        {"role": "user", "content": "Summarize safely"},
        {"role": "assistant", "content": "OK"},
    ]

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response("## Goal\nSafe summary."),
    ) as mock_call:
        result = c._generate_summary(
            turns, focus_topic=f"manual focus {SECRET} {OAUTH_URL}"
        )

    assert result is not None
    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    _assert_clean(prompt)


def test_auto_focus_topic_redacted():
    c = _compressor()

    focus = c._derive_auto_focus_topic(
        [
            {"role": "assistant", "content": "older assistant turn"},
            {"role": "user", "content": f"focus has {SECRET} and {OAUTH_URL}"},
        ]
    )

    assert focus is not None
    _assert_clean(focus)


def test_previous_summary_redacted_before_iterative_prompt_reentry():
    """Legacy persisted summaries may predate compaction redaction."""
    c = _compressor()
    c._previous_summary = f"Old summary leaked {SECRET} and {OAUTH_URL}"

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response("updated summary"),
    ) as mock_call:
        result = c._generate_summary(
            [
                {"role": "user", "content": "new turn"},
                {"role": "assistant", "content": "new work"},
            ]
        )

    assert result is not None
    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:" in prompt
    _assert_clean(prompt)
    # After generation, _previous_summary holds the new (clean) LLM output —
    # the leaked secret must not have survived anywhere in it.
    assert SECRET not in c._previous_summary
    assert "access_token=opaque-token-456" not in c._previous_summary


def test_resumed_handoff_summary_redacted_before_iterative_prompt():
    """Persisted handoff messages may contain pre-fix secrets after resume."""
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100000,
    ):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=1,
            protect_last_n=1,
            quiet_mode=True,
        )
    old_summary = f"RESUMED-SUMMARY leaked {SECRET} and {OAUTH_URL}"
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": f"{SUMMARY_PREFIX}\n{old_summary}"},
        {"role": "assistant", "content": "handoff acknowledged after resume"},
        {"role": "user", "content": "new user turn after resume"},
        {"role": "assistant", "content": "new assistant work after resume"},
        {"role": "user", "content": "more new work after resume"},
        {"role": "assistant", "content": "latest tail response"},
        {"role": "user", "content": "final active request stays in tail"},
    ]

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response("updated summary"),
    ) as mock_call:
        c.compress(messages)

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:" in prompt
    _assert_clean(prompt)


def test_arbitrary_external_secret_is_forced_out_of_all_compaction_boundaries(
    monkeypatch,
):
    """Exact external values never reach the aux model or persisted summary."""
    from hermes_cli import env_loader
    from hermes_constants import get_hermes_home

    secret = 'Q7"\\Z!'
    encoded_secret = json.dumps(secret)[1:-1]
    home = get_hermes_home()
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(home.resolve()),
        {"ARBITRARY_EXTERNAL_NAME": secret},
    )
    c = _compressor()
    c._previous_summary = json.dumps({"legacy_previous_summary": secret})
    turns = [
        {"role": "user", "content": f"user text contains {secret}"},
        {
            "role": "assistant",
            "content": "running tool",
            "tool_calls": [
                {
                    "id": "call-exact-secret",
                    "function": {
                        "name": "terminal",
                        "arguments": json.dumps({"command": f"printf {secret}"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-exact-secret",
            "content": json.dumps({"tool_result": {"credential": secret}}),
        },
    ]
    original_arguments = turns[1]["tool_calls"][0]["function"]["arguments"]

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response(json.dumps({"model_output": secret})),
    ) as mock_call:
        result = c._generate_summary(
            turns,
            focus_topic=f"focus contains {secret}",
            memory_context=f"memory provider contains {secret}",
        )

    assert result is not None
    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "MEMORY PROVIDER CONTEXT:" in prompt
    assert "memory provider contains ***" in prompt
    assert secret not in prompt
    assert encoded_secret not in prompt
    assert secret not in result
    assert encoded_secret not in result
    assert secret not in (c._previous_summary or "")
    assert encoded_secret not in (c._previous_summary or "")
    assert turns[1]["tool_calls"][0]["function"]["arguments"] == original_arguments


def test_summary_reuses_one_request_local_exact_pattern(monkeypatch):
    """One compaction request snapshots active exact secrets only once."""
    import agent.redact as redact
    from hermes_cli import env_loader
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(home.resolve()),
        {"COMPACTION_LOCAL_SECRET": "p4ss"},
    )
    original_collect = redact._collect_exact_secret_values
    collections = []

    def counted_collect(secret_home):
        collections.append(secret_home)
        return original_collect(secret_home)

    monkeypatch.setattr(redact, "_collect_exact_secret_values", counted_collect)
    c = _compressor()
    c._previous_summary = "previous p4ss"
    turns = [
        {"role": "user", "content": "request p4ss"},
        {"role": "assistant", "content": "response p4ss"},
    ]

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response("summary p4ss"),
    ):
        result = c._generate_summary(
            turns,
            focus_topic="focus p4ss",
            memory_context="memory p4ss",
        )

    assert result is not None
    assert result.endswith("summary ***")
    assert len(collections) == 1


@pytest.mark.parametrize("alternate_spelling", ALTERNATE_JSON_SPELLINGS)
def test_alternate_json_escapes_leave_batch_input_output_and_state_clean(
    monkeypatch, alternate_spelling
):
    """Batch compaction decodes alternate spellings only on disposable text."""
    from hermes_cli import env_loader
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(home.resolve()),
        {"BATCH_ALTERNATE_JSON_SECRET": ALTERNATE_JSON_SECRET},
    )
    c = _compressor()
    def encoded(field: str) -> str:
        return _json_with_alternate_secret(field, alternate_spelling)

    c._previous_summary = encoded("previous_summary")
    arguments = encoded("command")
    turns = [
        {"role": "user", "content": encoded("request")},
        {
            "role": "assistant",
            "content": "running tool",
            "tool_calls": [
                {
                    "id": "call-alternate-json-secret",
                    "function": {
                        "name": "terminal",
                        "arguments": arguments,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-alternate-json-secret",
            "content": encoded("tool_result"),
        },
    ]
    original_turns = deepcopy(turns)

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response(encoded("model_output")),
    ) as mock_call:
        result = c._generate_summary(
            turns,
            focus_topic=encoded("focus"),
            memory_context=encoded("memory"),
        )

    assert result is not None
    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    for text in (prompt, result, c._previous_summary or ""):
        assert ALTERNATE_JSON_SECRET not in text
        assert ALTERNATE_JSON_SPELLING not in text
        assert alternate_spelling not in text
    assert turns == original_turns


@pytest.mark.parametrize("fragment_kind", MALFORMED_JSON_FRAGMENT_KINDS)
def test_malformed_json_escapes_leave_batch_input_output_and_state_clean(
    monkeypatch, fragment_kind
):
    """Batch masking recovers after invalid escapes and at end of input."""
    from hermes_cli import env_loader
    from hermes_constants import get_hermes_home

    def fragment(field: str) -> str:
        return _malformed_json_with_alternate_secret(field, fragment_kind)

    home = get_hermes_home()
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(home.resolve()),
        {"BATCH_MALFORMED_JSON_SECRET": ALTERNATE_JSON_SECRET},
    )
    c = _compressor()
    c._previous_summary = fragment("previous_summary")
    arguments = fragment("command")
    turns = [
        {"role": "user", "content": fragment("request")},
        {
            "role": "assistant",
            "content": "running tool",
            "tool_calls": [
                {
                    "id": "call-malformed-json-secret",
                    "function": {"name": "terminal", "arguments": arguments},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-malformed-json-secret",
            "content": fragment("tool_result"),
        },
    ]
    original_turns = deepcopy(turns)

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response(fragment("model_output")),
    ) as mock_call:
        result = c._generate_summary(
            turns,
            focus_topic=fragment("focus"),
            memory_context=fragment("memory"),
        )

    assert result is not None
    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    for text in (prompt, result, c._previous_summary or ""):
        assert "***" in text
        assert ALTERNATE_JSON_SECRET not in text
        assert ALTERNATE_JSON_SPELLING not in text
    assert turns == original_turns


def test_micro_compaction_redacts_resumed_input_output_and_persistence(
    monkeypatch,
):
    """Legacy markers and aux output cannot bypass forced compaction masking."""
    from hermes_cli import env_loader
    from hermes_constants import get_hermes_home

    secret = 'Q7"\\Z!'
    encoded_secret = json.dumps(secret)[1:-1]
    home = get_hermes_home()
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(home.resolve()),
        {"MICRO_EXTERNAL_VALUE": secret},
    )
    c = ContextCompressor(
        model="test-model",
        threshold_percent=0.75,
        protect_first_n=1,
        protect_last_n=2,
        quiet_mode=True,
        config_context_length=40960,
        provider="test",
    )
    c._micro_compact_enabled = True
    c._session_id = "micro-redaction-regression"
    c._session_db = MagicMock()

    replay_fields = {
        "tool_calls": [
            {
                "id": "call-secret",
                "type": "function",
                "function": {
                    "name": "terminal",
                    "arguments": json.dumps({"command": f"printf {secret}"}),
                },
            }
        ],
        "anthropic_content_blocks": [
            {
                "type": "thinking",
                "thinking": f"signed thought {secret}",
                "signature": f"signature-{secret}",
            },
            {
                "type": "tool_use",
                "id": "toolu-secret",
                "name": "terminal",
                "input": {"command": f"printf {secret}"},
            },
        ],
        "codex_reasoning_items": [
            {
                "type": "reasoning",
                "encrypted_content": f"sealed-{secret}",
            }
        ],
    }
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "earlier user request"},
        {
            "role": "assistant",
            "content": c._render_micro_marker_content(
                f"resumed rolling summary contains {secret}"
            ),
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        },
        {"role": "user", "content": "next user request"},
        {
            "role": "assistant",
            "content": f"exchange content contains {secret}",
            **deepcopy(replay_fields),
        },
        {
            "role": "tool",
            "tool_call_id": "call-secret",
            "content": json.dumps({"tool_result": {"credential": secret}}),
        },
        {"role": "user", "content": "later request"},
        {"role": "assistant", "content": "later answer " + "z" * 400},
        {"role": "user", "content": "protected request"},
        {"role": "assistant", "content": "protected answer " + "z" * 400},
    ]
    original_replay_fields = deepcopy(replay_fields)

    with patch(
        "agent.auxiliary_client.call_llm",
        return_value=_response(json.dumps({"rolling_summary": secret})),
    ) as mock_call:
        result = c._micro_compact(messages)

    prompt_messages = mock_call.call_args.kwargs["messages"]
    prompt_text = "\n".join(message["content"] for message in prompt_messages)
    assert secret not in prompt_text
    assert encoded_secret not in prompt_text
    assert secret not in c._micro_compact_rolling_summary
    assert encoded_secret not in c._micro_compact_rolling_summary
    markers = [m for m in result if m.get(COMPRESSED_SUMMARY_METADATA_KEY)]
    assert len(markers) == 1
    assert secret not in markers[0]["content"]
    assert encoded_secret not in markers[0]["content"]
    db_session_id, db_payload = c._session_db.archive_and_compact.call_args.args
    assert db_session_id == c._session_id
    assert secret not in str(db_payload)
    assert encoded_secret not in str(db_payload)

    # Compaction may summarize the rendered exchange, but it must not rewrite
    # compatibility-critical replay bytes in the caller-owned source message.
    for field, value in original_replay_fields.items():
        assert messages[4][field] == value


@pytest.mark.parametrize("alternate_spelling", MICRO_ALTERNATE_JSON_SPELLINGS)
def test_alternate_json_escapes_leave_micro_output_and_persistence_clean(
    monkeypatch, alternate_spelling
):
    """Micro compaction redacts alternate input/output spellings before storage."""
    from hermes_cli import env_loader
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(home.resolve()),
        {"MICRO_ALTERNATE_JSON_SECRET": ALTERNATE_JSON_SECRET},
    )
    c = ContextCompressor(
        model="test-model",
        threshold_percent=0.75,
        protect_first_n=1,
        protect_last_n=2,
        quiet_mode=True,
        config_context_length=40960,
        provider="test",
    )
    c._micro_compact_enabled = True
    c._session_id = "micro-alternate-json-redaction"
    c._session_db = MagicMock()

    def encoded(field: str) -> str:
        return _json_with_alternate_secret(field, alternate_spelling)

    arguments = encoded("command")
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "earlier user request"},
        {
            "role": "assistant",
            "content": c._render_micro_marker_content(
                encoded("resumed_summary")
            ),
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        },
        {"role": "user", "content": "next user request"},
        {
            "role": "assistant",
            "content": encoded("exchange"),
            "tool_calls": [
                {
                    "id": "call-alternate-json-secret",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": arguments},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-alternate-json-secret",
            "content": encoded("tool_result"),
        },
        {"role": "user", "content": "later request"},
        {"role": "assistant", "content": "later answer " + "z" * 400},
        {"role": "user", "content": "protected request"},
        {"role": "assistant", "content": "protected answer " + "z" * 400},
    ]
    original_messages = deepcopy(messages)

    with patch(
        "agent.auxiliary_client.call_llm",
        return_value=_response(encoded("rolling_summary")),
    ) as mock_call:
        result = c._micro_compact(messages)

    prompt_text = "\n".join(
        message["content"] for message in mock_call.call_args.kwargs["messages"]
    )
    markers = [m for m in result if m.get(COMPRESSED_SUMMARY_METADATA_KEY)]
    assert len(markers) == 1
    _db_session_id, db_payload = c._session_db.archive_and_compact.call_args.args
    for text in (
        prompt_text,
        c._micro_compact_rolling_summary,
        markers[0]["content"],
        str(db_payload),
    ):
        assert ALTERNATE_JSON_SECRET not in text
        assert ALTERNATE_JSON_SPELLING not in text
        assert alternate_spelling not in text
    assert messages[4]["content"] == original_messages[4]["content"]
    assert messages[4]["tool_calls"] == original_messages[4]["tool_calls"]
    assert messages[5]["content"] == original_messages[5]["content"]


@pytest.mark.parametrize("fragment_kind", MALFORMED_JSON_FRAGMENT_KINDS)
def test_malformed_json_escapes_leave_micro_output_and_persistence_clean(
    monkeypatch, fragment_kind
):
    """Micro masking cleans malformed fragments before model and persistence."""
    from hermes_cli import env_loader
    from hermes_constants import get_hermes_home

    def fragment(field: str) -> str:
        return _malformed_json_with_alternate_secret(field, fragment_kind)

    home = get_hermes_home()
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(home.resolve()),
        {"MICRO_MALFORMED_JSON_SECRET": ALTERNATE_JSON_SECRET},
    )
    c = ContextCompressor(
        model="test-model",
        threshold_percent=0.75,
        protect_first_n=1,
        protect_last_n=2,
        quiet_mode=True,
        config_context_length=40960,
        provider="test",
    )
    c._micro_compact_enabled = True
    c._session_id = "micro-malformed-json-redaction"
    c._session_db = MagicMock()
    arguments = fragment("command")
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "earlier user request"},
        {
            "role": "assistant",
            "content": c._render_micro_marker_content(fragment("resumed_summary")),
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        },
        {"role": "user", "content": "next user request"},
        {
            "role": "assistant",
            "content": fragment("exchange"),
            "tool_calls": [
                {
                    "id": "call-malformed-json-secret",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": arguments},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-malformed-json-secret",
            "content": fragment("tool_result"),
        },
        {"role": "user", "content": "later request"},
        {"role": "assistant", "content": "later answer " + "z" * 400},
        {"role": "user", "content": "protected request"},
        {"role": "assistant", "content": "protected answer " + "z" * 400},
    ]
    original_messages = deepcopy(messages)

    with patch(
        "agent.auxiliary_client.call_llm",
        return_value=_response(fragment("rolling_summary")),
    ) as mock_call:
        result = c._micro_compact(messages)

    prompt_text = "\n".join(
        message["content"] for message in mock_call.call_args.kwargs["messages"]
    )
    markers = [m for m in result if m.get(COMPRESSED_SUMMARY_METADATA_KEY)]
    assert len(markers) == 1
    _db_session_id, db_payload = c._session_db.archive_and_compact.call_args.args
    for text in (
        prompt_text,
        c._micro_compact_rolling_summary,
        markers[0]["content"],
        str(db_payload),
    ):
        assert "***" in text
        assert ALTERNATE_JSON_SECRET not in text
        assert ALTERNATE_JSON_SPELLING not in text
    assert messages[4] == original_messages[4]
    assert messages[5] == original_messages[5]

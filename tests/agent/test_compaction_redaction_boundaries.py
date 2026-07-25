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

from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import (
    ContextCompressor,
    SUMMARY_PREFIX,
    _redact_compaction_text,
)

SECRET = "sk-proj-" + ("a" * 40)
OAUTH_URL = (
    "https://localhost/callback?code=opaque-code-123"
    "&access_token=opaque-token-456&state=keep"
)


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

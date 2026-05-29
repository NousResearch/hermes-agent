from types import SimpleNamespace

from agent.codex_responses_adapter import _normalize_codex_response


def test_normalize_codex_response_drops_transient_rs_tmp_reasoning_items():
    response = SimpleNamespace(
        status="completed",
        output=[
            SimpleNamespace(
                type="reasoning",
                id="rs_tmp_123",
                encrypted_content="opaque-transient",
                summary=[],
            ),
            SimpleNamespace(
                type="reasoning",
                id="rs_456",
                encrypted_content="opaque-stable",
                summary=[SimpleNamespace(text="stable summary")],
            ),
            SimpleNamespace(
                type="message",
                role="assistant",
                status="completed",
                content=[SimpleNamespace(type="output_text", text="done")],
            ),
        ],
    )

    assistant_message, finish_reason = _normalize_codex_response(response)

    assert finish_reason == "stop"
    assert assistant_message.content == "done"
    assert assistant_message.codex_reasoning_items == [
        {
            "type": "reasoning",
            "encrypted_content": "opaque-stable",
            "id": "rs_456",
            "summary": [{"type": "summary_text", "text": "stable summary"}],
        }
    ]


def test_normalize_codex_response_treats_summary_only_reasoning_as_incomplete():
    response = SimpleNamespace(
        status="completed",
        output=[
            SimpleNamespace(
                type="reasoning",
                id="rs_tmp_789",
                encrypted_content="opaque-transient",
                summary=[SimpleNamespace(text="still thinking")],
            )
        ],
    )

    assistant_message, finish_reason = _normalize_codex_response(response)

    assert finish_reason == "incomplete"
    assert assistant_message.content == ""
    assert assistant_message.reasoning == "still thinking"
    assert assistant_message.codex_reasoning_items is None


def test_normalize_codex_response_guards_output_text_property_typeerror():
    """Regression: a raw SDK ``Response`` can reach ``_normalize_codex_response``
    with ``output=None`` via the compatibility passthrough in
    ``run_codex_stream`` (concrete responses returned straight through).

    ``Response.output_text`` is a *computed property* that iterates
    ``self.output`` with no None-guard, so reading it raises
    ``TypeError: 'NoneType' object is not iterable``.  ``getattr(obj, name,
    default)`` only swallows ``AttributeError`` — NOT exceptions raised inside
    a property getter — so without the guard this TypeError escapes, reaches
    run_agent with no HTTP status_code, gets misclassified as a non-retryable
    local error, and surfaces as "Non-retryable error (HTTP None)".

    With the guard, the TypeError is swallowed and normalization falls through
    to the documented "no output items" RuntimeError instead of an opaque,
    misclassified crash.
    """
    import pytest

    class _RawResponseNullOutput:
        """Mimics openai ``Response``: ``output_text`` is a computed property."""

        def __init__(self):
            self.output = None
            self.status = "completed"
            self.error = None

        @property
        def output_text(self):  # noqa: D401 - mirrors SDK property semantics
            # SDK iterates self.output (None) -> TypeError from inside getter.
            return "".join(o for o in self.output)

    with pytest.raises(RuntimeError, match="no output items"):
        _normalize_codex_response(_RawResponseNullOutput())


def test_normalize_codex_response_uses_output_text_fallback_when_present():
    """Companion to the guard test: when ``output_text`` is a plain non-empty
    string (no crash), it is still used as the last-resort fallback to
    synthesize a message item, recovering content delivered via stream events.
    """
    response = SimpleNamespace(
        output=[],
        output_text="recovered text",
        status="completed",
        error=None,
    )

    assistant_message, finish_reason = _normalize_codex_response(response)

    assert finish_reason == "stop"
    assert assistant_message.content == "recovered text"

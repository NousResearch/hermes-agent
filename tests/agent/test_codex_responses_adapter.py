from types import SimpleNamespace

import pytest

from agent.codex_responses_adapter import (
    _MAX_REPLAY_ITEM_ID_LEN,
    _chat_messages_to_responses_input,
    _format_responses_error,
    _normalize_codex_response,
    _preflight_codex_api_kwargs,
    _preflight_codex_input_items,
    _replayable_item_id,
)


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


# ---------------------------------------------------------------------------
# Server-side built-in tool calls (xAI native web_search, code interpreter,
# etc.) come back as discrete ``*_call`` output items that xAI's
# /v1/responses surface routinely leaves at ``status="in_progress"`` even
# when the overall ``response.status == "completed"``.  These must NOT mark
# the turn incomplete — otherwise grok-composer-2.5-fast research queries
# (which invoke server-side web_search) get misclassified as
# ``finish_reason="incomplete"`` and burn 3 fruitless continuation retries
# before failing with "Codex response remained incomplete after 3
# continuation attempts".  Observed live against grok-composer-2.5-fast on
# SuperGrok OAuth (2026-06).
# ---------------------------------------------------------------------------


def test_normalize_codex_response_ignores_in_progress_server_side_tool_calls():
    """A completed response with a final message + lingering in_progress
    server-side web_search_call items resolves to 'stop', not 'incomplete'."""
    response = SimpleNamespace(
        status="completed",
        incomplete_details=None,
        output=[
            SimpleNamespace(
                type="reasoning",
                id="rs_1",
                encrypted_content="opaque",
                summary=[SimpleNamespace(text="researching blades")],
            ),
            SimpleNamespace(
                type="message",
                role="assistant",
                status="completed",
                content=[SimpleNamespace(
                    type="output_text",
                    text="Milwaukee M18 blade 49-16-2734, ~$30 OEM.",
                )],
            ),
            SimpleNamespace(type="web_search_call", status="in_progress"),
            SimpleNamespace(type="web_search_call", status="in_progress"),
            SimpleNamespace(type="web_search_call", status="in_progress"),
        ],
    )

    assistant_message, finish_reason = _normalize_codex_response(response)

    assert finish_reason == "stop"
    assert assistant_message.content == "Milwaukee M18 blade 49-16-2734, ~$30 OEM."


def test_normalize_codex_response_in_progress_message_still_incomplete():
    """Guard scope: an in_progress *message* item (genuine model output that
    is still streaming) must still mark the turn incomplete — only
    server-side ``*_call`` items are exempted."""
    response = SimpleNamespace(
        status="completed",
        incomplete_details=None,
        output=[
            SimpleNamespace(
                type="message",
                role="assistant",
                status="in_progress",
                content=[SimpleNamespace(type="output_text", text="partial...")],
            ),
        ],
    )

    _assistant_message, finish_reason = _normalize_codex_response(response)

    assert finish_reason == "incomplete"


# ---------------------------------------------------------------------------
# _preflight_codex_api_kwargs — built-in (provider-executed) tools must pass
# through validation.  Regression guard for the xAI native web_search
# injection: the preflight validator previously rejected any tool whose
# ``type != "function"`` with "unsupported type", which would 400 every xAI
# turn once the native web_search tool is declared.
# ---------------------------------------------------------------------------


def test_preflight_passes_native_web_search_tool_through():
    kwargs = {
        "model": "grok-composer-2.5-fast",
        "instructions": "You are helpful.",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "store": False,
        "tools": [
            {"type": "function", "name": "read_file", "description": "Read.",
             "parameters": {"type": "object", "properties": {}}},
            {"type": "web_search"},
        ],
    }
    out = _preflight_codex_api_kwargs(kwargs, allow_stream=True)
    tools = out["tools"]
    assert {"type": "web_search"} in tools
    assert any(t.get("type") == "function" and t.get("name") == "read_file" for t in tools)


def test_preflight_still_rejects_unknown_tool_type():
    kwargs = {
        "model": "grok-composer-2.5-fast",
        "instructions": "You are helpful.",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "store": False,
        "tools": [{"type": "totally_made_up_tool"}],
    }
    with pytest.raises(ValueError, match="unsupported type"):
        _preflight_codex_api_kwargs(kwargs, allow_stream=True)


# ---------------------------------------------------------------------------
# _format_responses_error — adapted from anomalyco/opencode#28757.
# Provider failures should surface BOTH the code (rate_limit_exceeded /
# context_length_exceeded / internal_error / server_error) and the message,
# so consumers can tell rate limits apart from context-length failures and
# both apart from generic stream drops.
# ---------------------------------------------------------------------------


def test_format_responses_error_combines_code_and_message():
    err = {"code": "rate_limit_exceeded", "message": "Slow down"}
    assert _format_responses_error(err, "failed") == "rate_limit_exceeded: Slow down"


def test_format_responses_error_message_only():
    err = {"message": "Upstream model unavailable"}
    assert _format_responses_error(err, "failed") == "Upstream model unavailable"


def test_format_responses_error_code_only_when_message_empty():
    # Some providers/proxies emit a code with an empty message body. We
    # used to fall back to ``str(error_obj)`` — a dict dump — which leaked
    # ``{'code': 'internal_error', 'message': ''}`` into chat output. Now
    # the bare code is surfaced, which is the meaningful field.
    err = {"code": "internal_error", "message": ""}
    assert _format_responses_error(err, "failed") == "internal_error"


def test_format_responses_error_code_only_when_message_missing():
    err = {"code": "server_error"}
    assert _format_responses_error(err, "failed") == "server_error"


def test_format_responses_error_attribute_style_payload():
    # SDK objects expose ``code``/``message`` as attributes rather than dict
    # keys. The helper must accept both shapes since the Responses SDK
    # returns SimpleNamespace-style objects on ``response.failed``.
    err = SimpleNamespace(code="context_length_exceeded", message="too long")
    assert _format_responses_error(err, "failed") == "context_length_exceeded: too long"


def test_format_responses_error_falls_back_to_status_when_empty():
    assert (
        _format_responses_error(None, "failed")
        == "Responses API returned status 'failed'"
    )
    assert (
        _format_responses_error(None, "cancelled")
        == "Responses API returned status 'cancelled'"
    )


def test_format_responses_error_stringifies_opaque_payload():
    # Last-resort: a provider sent something that isn't a dict and has no
    # code/message attributes. Surface its repr rather than swallow it
    # silently — at least it's visible in logs.
    assert _format_responses_error("opaque sentinel", "failed") == "opaque sentinel"


def test_format_responses_error_ignores_non_string_code_message():
    # Defensive: a malformed gateway could send numbers/objects in these
    # fields. We don't want to crash; we want a best-effort string.
    err = {"code": 500, "message": None}
    assert _format_responses_error(err, "failed") == "500"


def test_normalize_codex_response_failed_includes_code_in_error():
    """Regression: response_status == 'failed' should surface the error
    code, not just the message. Used to leak a bare 'Slow down' string
    that was indistinguishable from a generic stream truncation."""
    # ``output`` non-empty so we don't trip the "no output items" guard
    # before reaching the failed-status branch. Real failed responses
    # often DO carry a partial message item alongside the error.
    response = SimpleNamespace(
        status="failed",
        output=[
            SimpleNamespace(
                type="message",
                role="assistant",
                status="incomplete",
                content=[SimpleNamespace(type="output_text", text="partial")],
            ),
        ],
        error={"code": "rate_limit_exceeded", "message": "Slow down"},
    )
    with pytest.raises(RuntimeError, match=r"^rate_limit_exceeded: Slow down$"):
        _normalize_codex_response(response)


def test_normalize_codex_response_failed_with_message_only():
    """Backwards-compat: a failed response with only a message field
    (no code) should still surface that message verbatim."""
    response = SimpleNamespace(
        status="failed",
        output=[
            SimpleNamespace(
                type="message",
                role="assistant",
                status="incomplete",
                content=[SimpleNamespace(type="output_text", text="partial")],
            ),
        ],
        error={"message": "model error"},
    )
    with pytest.raises(RuntimeError, match=r"^model error$"):
        _normalize_codex_response(response)


# ---------------------------------------------------------------------------
# Replay item id length cap
# ---------------------------------------------------------------------------
# chatgpt.com/backend-api/codex rejects input item ids longer than 64 chars
# with HTTP 400 string_above_max_length. With store=False the backend returns
# ~400-char encrypted message ids; echoing them back on replay fails the whole
# request, so over-long ids must be omitted (id is optional on replayed
# assistant messages).

_LEGAL_ID = "msg_" + "a" * 60  # exactly 64 chars
_ENCRYPTED_ID = "msg_" + "x" * 400  # store=False style encrypted id


def test_replayable_item_id_accepts_backend_legal_ids():
    assert _replayable_item_id(_LEGAL_ID) == _LEGAL_ID
    assert _replayable_item_id("  msg_123  ") == "msg_123"


def test_replayable_item_id_rejects_overlong_empty_and_non_string():
    assert len(_LEGAL_ID) == _MAX_REPLAY_ITEM_ID_LEN
    assert _replayable_item_id(_LEGAL_ID + "a") is None
    assert _replayable_item_id(_ENCRYPTED_ID) is None
    assert _replayable_item_id("   ") is None
    assert _replayable_item_id(None) is None
    assert _replayable_item_id(123) is None


def _assistant_message_with_replay_id(item_id):
    return {
        "role": "assistant",
        "content": "hello",
        "codex_message_items": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "id": item_id,
                "phase": "final",
                "content": [{"type": "output_text", "text": "hello"}],
            }
        ],
    }


def test_replay_drops_overlong_encrypted_message_ids():
    items = _chat_messages_to_responses_input(
        [_assistant_message_with_replay_id(_ENCRYPTED_ID)]
    )
    replayed = [i for i in items if i.get("type") == "message" and i.get("role") == "assistant"]
    assert len(replayed) == 1
    # The over-long id is omitted; the rest of the item replays intact.
    assert "id" not in replayed[0]
    assert replayed[0]["phase"] == "final"
    assert replayed[0]["content"] == [{"type": "output_text", "text": "hello"}]


def test_replay_preserves_backend_legal_message_ids():
    items = _chat_messages_to_responses_input(
        [_assistant_message_with_replay_id(_LEGAL_ID)]
    )
    replayed = [i for i in items if i.get("type") == "message" and i.get("role") == "assistant"]
    assert len(replayed) == 1
    assert replayed[0]["id"] == _LEGAL_ID


def test_preflight_drops_overlong_message_ids():
    normalized = _preflight_codex_input_items(
        [
            {
                "type": "message",
                "role": "assistant",
                "id": _ENCRYPTED_ID,
                "content": [{"type": "output_text", "text": "hi"}],
            }
        ]
    )
    assert len(normalized) == 1
    assert "id" not in normalized[0]


def test_preflight_preserves_backend_legal_message_ids():
    normalized = _preflight_codex_input_items(
        [
            {
                "type": "message",
                "role": "assistant",
                "id": "  msg_short  ",
                "content": [{"type": "output_text", "text": "hi"}],
            }
        ]
    )
    assert len(normalized) == 1
    assert normalized[0]["id"] == "msg_short"

from types import SimpleNamespace

import pytest

from agent.codex_responses_adapter import (
    _format_responses_error,
    _normalize_codex_response,
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


# ─────────────────────────────────────────────────────────────────────
# Regression: cross-issuer message-id replay
# ─────────────────────────────────────────────────────────────────────
#
# Symptom in the wild (May 2026, hermes-gateway production):
#
#   provider=openai-api base_url=https://api.openai.com/v1/ model=gpt-5.4-mini
#   HTTP 400: Invalid 'input[8].id': string too long.
#   Expected a string with maximum length 64, but got a string with length 408.
#
# Root cause: when a session uses Copilot/GitHub Responses primary and
# falls back to OpenAI Responses mid-conversation, hermes was replaying
# the persisted ``codex_message_items`` verbatim — including the long
# Copilot-issued ``id`` field which OpenAI's Responses API rejects with
# ``string_above_max_length`` (cap is 64). The same root cause already
# bites reasoning items via the cross-issuer encrypted_content guard.
#
# Fix is three-pronged:
#   1. _normalize_codex_response stamps each message item with an
#      ``_issuer_kind`` mirroring the existing reasoning-item stamp.
#   2. _chat_messages_to_responses_input drops the id when the active
#      endpoint's issuer differs from the stamp (principled fix).
#   3. _preflight_codex_input_items drops any id > 64 chars as a
#      defensive backstop (catches legacy unstamped items).
#
from agent.codex_responses_adapter import (
    _chat_messages_to_responses_input,
    _preflight_codex_input_items,
)


# ── Edit 3: _normalize_codex_response stamps message items ──────────

def test_normalize_codex_response_stamps_message_items_with_issuer():
    """Each persisted message item should carry an _issuer_kind so a
    later cross-provider fallback can detect that its id is sealed
    to a different Responses endpoint."""
    response = SimpleNamespace(
        status="completed",
        output=[
            SimpleNamespace(
                type="message",
                id="msg_" + "a" * 400,  # Copilot-style long opaque id
                role="assistant",
                status="completed",
                content=[SimpleNamespace(type="output_text", text="hello")],
            ),
        ],
    )
    assistant_message, _ = _normalize_codex_response(
        response, issuer_kind="github_responses"
    )
    items = assistant_message.codex_message_items
    assert len(items) == 1
    assert items[0]["_issuer_kind"] == "github_responses"
    # id is still captured for local persistence; it's the *replay* path
    # that filters based on the issuer stamp.
    assert items[0]["id"].startswith("msg_")


def test_normalize_codex_response_omits_issuer_stamp_when_unknown():
    response = SimpleNamespace(
        status="completed",
        output=[
            SimpleNamespace(
                type="message",
                id="msg_short",
                role="assistant",
                status="completed",
                content=[SimpleNamespace(type="output_text", text="hi")],
            ),
        ],
    )
    assistant_message, _ = _normalize_codex_response(response)
    items = assistant_message.codex_message_items
    assert "_issuer_kind" not in items[0]


# ── Edit 1: converter drops id on cross-issuer replay ────────────────

def _msg_with_codex_item(item_id: str, issuer: str | None = None):
    item = {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "ok"}],
    }
    if item_id is not None:
        item["id"] = item_id
    if issuer is not None:
        item["_issuer_kind"] = issuer
    return {
        "role": "assistant",
        "content": "ok",
        "codex_message_items": [item],
    }


def test_converter_drops_message_id_on_cross_issuer_replay():
    """Long Copilot-issued id replayed against OpenAI must be dropped."""
    long_copilot_id = "msg_" + "x" * 400
    messages = [_msg_with_codex_item(long_copilot_id, issuer="github_responses")]
    items = _chat_messages_to_responses_input(
        messages,
        current_issuer_kind="other:https://api.openai.com/v1/",
    )
    msg_items = [i for i in items if i.get("type") == "message"]
    assert len(msg_items) == 1
    assert "id" not in msg_items[0], (
        f"cross-issuer message id should be dropped; got {msg_items[0].get('id')!r}"
    )


def test_converter_keeps_message_id_on_same_issuer_replay():
    """Same-issuer replay (Copilot -> Copilot) must preserve the id so
    prefix caching keeps working."""
    item_id = "msg_short_id_ok"
    messages = [_msg_with_codex_item(item_id, issuer="github_responses")]
    items = _chat_messages_to_responses_input(
        messages,
        current_issuer_kind="github_responses",
    )
    msg_items = [i for i in items if i.get("type") == "message"]
    assert msg_items[0].get("id") == item_id


def test_converter_keeps_message_id_when_no_issuer_context():
    """Backwards-compat: legacy items without _issuer_kind stamp must
    pass through unchanged (matches reasoning-item behavior)."""
    item_id = "msg_legacy_unstamped"
    messages = [_msg_with_codex_item(item_id, issuer=None)]
    items = _chat_messages_to_responses_input(
        messages,
        current_issuer_kind="other:https://api.openai.com/v1/",
    )
    msg_items = [i for i in items if i.get("type") == "message"]
    assert msg_items[0].get("id") == item_id


# ── Edit 2: preflight defensive cap on oversize ids ─────────────────

def test_preflight_drops_oversize_message_id():
    """Defensive backstop — even if cross-issuer stamping is absent,
    a >64 char id must be dropped because the OpenAI Responses API
    rejects it with HTTP 400 string_above_max_length."""
    long_id = "msg_" + "y" * 200  # 204 chars total
    raw = [{
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "id": long_id,
        "content": [{"type": "output_text", "text": "ok"}],
    }]
    normalized = _preflight_codex_input_items(raw)
    assert len(normalized) == 1
    assert "id" not in normalized[0], (
        "oversize id (>64) must be dropped to avoid OpenAI 400 string_above_max_length"
    )
    # Content + role still intact
    assert normalized[0]["role"] == "assistant"
    assert normalized[0]["content"][0]["text"] == "ok"


def test_preflight_keeps_within_limit_message_id():
    """Boundary: exactly 64-char id should pass through (API spec limit)."""
    sixty_four_char_id = "m" + "a" * 63  # 64 chars
    assert len(sixty_four_char_id) == 64
    raw = [{
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "id": sixty_four_char_id,
        "content": [{"type": "output_text", "text": "ok"}],
    }]
    normalized = _preflight_codex_input_items(raw)
    assert normalized[0]["id"] == sixty_four_char_id


def test_preflight_drops_65_char_id():
    """Boundary: 65-char id (one over) must be dropped."""
    sixty_five = "m" + "a" * 64  # 65 chars
    assert len(sixty_five) == 65
    raw = [{
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "id": sixty_five,
        "content": [{"type": "output_text", "text": "ok"}],
    }]
    normalized = _preflight_codex_input_items(raw)
    assert "id" not in normalized[0]


# ── End-to-end: full Copilot→OpenAI replay shouldn't emit any oversize ids
def test_end_to_end_copilot_to_openai_fallback_strips_long_ids():
    """Full pipeline test: an assistant message persisted from Copilot
    (with a long opaque id and _issuer_kind stamp) goes through the
    converter and preflight on its way to OpenAI's Responses API.
    No item in the final input may carry an id >64 chars."""
    long_copilot_id = "msg_" + "z" * 400
    messages = [_msg_with_codex_item(long_copilot_id, issuer="github_responses")]
    converted = _chat_messages_to_responses_input(
        messages,
        current_issuer_kind="other:https://api.openai.com/v1/",
    )
    normalized = _preflight_codex_input_items(converted)
    for idx, item in enumerate(normalized):
        item_id = item.get("id")
        if isinstance(item_id, str):
            assert len(item_id) <= 64, (
                f"input[{idx}].id is {len(item_id)} chars — would trigger "
                f"OpenAI 400 string_above_max_length"
            )

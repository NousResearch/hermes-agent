"""Regression tests for the thinking-block signature recovery.

The recovery in ``agent/conversation_loop.py`` strips replayed thinking state
from ``api_messages`` (the API-call-time list rebuilt on every retry) and
leaves ``messages`` (the canonical store) untouched. The previous
implementation popped from ``messages`` directly, which never reached
``api_messages`` because each entry in ``api_messages`` was a shallow
copy of the corresponding entry in ``messages``, and the mutation also
landed in ``state.db`` on the next ``_persist_session`` call, corrupting
the conversation.

These tests cover two surfaces:

1. **Mirror tests** — verify shallow-copy semantics of the stripping loop.
2. **Real-path test** — verify that after stripping, the retry wire payload
   built by ``convert_messages_to_anthropic()`` no longer selects the
   ordered-block replay fast path, while the canonical ``messages`` list
   retains both fields for future turns.
"""

from __future__ import annotations


def _shallow_copies(messages):
    return [m.copy() for m in messages]


# ── Mirror tests: shallow-copy semantics ─────────────────────────────


def test_pop_on_shallow_copy_does_not_affect_source():
    rd = [{"type": "thinking", "thinking": "r", "signature": "s"}]
    ordered = [{"type": "thinking", "thinking": "r", "signature": "s"}]
    src = {
        "role": "assistant",
        "content": "x",
        "reasoning_details": rd,
        "anthropic_content_blocks": ordered,
    }
    cp = src.copy()

    cp.pop("reasoning_details", None)
    cp.pop("anthropic_content_blocks", None)

    assert "reasoning_details" not in cp
    assert "anthropic_content_blocks" not in cp
    assert "reasoning_details" in src
    assert "anthropic_content_blocks" in src
    assert src["reasoning_details"] is rd
    assert src["anthropic_content_blocks"] is ordered


def test_strip_api_messages_leaves_canonical_messages_intact():
    """Mirrors the recovery: pop replayed thinking state from api_messages only.

    The canonical ``messages`` list keeps both fields so future persists carry
    the original signed blocks and ordered content blocks.
    """
    rd_one = [{"type": "thinking", "thinking": "one", "signature": "sig_one"}]
    rd_two = [{"type": "thinking", "thinking": "two", "signature": "sig_two"}]
    ordered_one = [{"type": "thinking", "thinking": "one", "signature": "sig_one"}]
    ordered_two = [{"type": "thinking", "thinking": "two", "signature": "sig_two"}]
    messages = [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": "a1",
            "reasoning_details": rd_one,
            "anthropic_content_blocks": ordered_one,
        },
        {"role": "user", "content": "q2"},
        {
            "role": "assistant",
            "content": "a2",
            "reasoning_details": rd_two,
            "anthropic_content_blocks": ordered_two,
        },
    ]
    api_messages = _shallow_copies(messages)

    stripped = 0
    for m in api_messages:
        if not isinstance(m, dict):
            continue
        if "reasoning_details" in m:
            m.pop("reasoning_details", None)
            stripped += 1
        if "anthropic_content_blocks" in m:
            m.pop("anthropic_content_blocks", None)
            stripped += 1

    assert stripped == 4
    assert all("reasoning_details" not in m for m in api_messages)
    assert all("anthropic_content_blocks" not in m for m in api_messages)
    canonical_rd = [
        m.get("reasoning_details") for m in messages if m["role"] == "assistant"
    ]
    canonical_ordered = [
        m.get("anthropic_content_blocks") for m in messages if m["role"] == "assistant"
    ]
    assert canonical_rd == [rd_one, rd_two]
    assert canonical_ordered == [ordered_one, ordered_two]


def test_strip_is_idempotent_when_run_twice():
    """A second strip is a no-op when replayed state has already been removed."""
    api_messages = [
        {"role": "assistant", "content": "a", "reasoning_details": [{"x": 1}]},
        {"role": "user", "content": "q"},
    ]
    for _ in range(2):
        for m in api_messages:
            if not isinstance(m, dict):
                continue
            if "reasoning_details" in m:
                m.pop("reasoning_details", None)
            if "anthropic_content_blocks" in m:
                m.pop("anthropic_content_blocks", None)

    assert all("reasoning_details" not in m for m in api_messages)
    assert all("anthropic_content_blocks" not in m for m in api_messages)


def test_strip_skips_messages_without_reasoning_details():
    api_messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "tool", "tool_call_id": "1", "content": "ok"},
    ]
    snapshot = [dict(m) for m in api_messages]

    for m in api_messages:
        if not isinstance(m, dict):
            continue
        if "reasoning_details" in m:
            m.pop("reasoning_details", None)
        if "anthropic_content_blocks" in m:
            m.pop("anthropic_content_blocks", None)

    assert api_messages == snapshot


# ── Real-path test: production retry wire payload ───────────────────


def test_recovery_strips_ordered_blocks_from_retry_wire_payload():
    """After signature-recovery stripping, ``convert_messages_to_anthropic()``
    must NOT select the ordered-block replay fast path for the retry.

    This exercises the production path: ``api_messages`` carries both
    ``reasoning_details`` and ``anthropic_content_blocks`` on an assistant
    tool-call turn. The recovery loop strips both. The stripped list is then
    passed through ``convert_messages_to_anthropic()``, which must not replay
    the signed thinking block via the ordered-block fast path.
    """
    from agent.anthropic_adapter import convert_messages_to_anthropic

    # Simulate an assistant turn that interleaved signed thinking with tool_use,
    # as captured by normalize_response + build_assistant_message.
    signed_thinking_block = {
        "type": "thinking",
        "thinking": "I need to call a tool.",
        "signature": "sig-abc",
    }
    tool_use_block = {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "read_file",
        "input": {"path": "/tmp/x"},
    }

    canonical_messages = [
        {"role": "user", "content": "Read the file."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_details": [signed_thinking_block],
            "anthropic_content_blocks": [signed_thinking_block, tool_use_block],
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/tmp/x"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": "file contents"},
    ]

    # ── Before recovery: ordered-block fast path IS selected ───────
    _sys_before, converted_before = convert_messages_to_anthropic(
        canonical_messages, base_url=None, model="claude-opus-4-6"
    )
    asst_before = next(m for m in converted_before if m["role"] == "assistant")
    block_types_before = [b.get("type") for b in asst_before["content"]]
    assert "thinking" in block_types_before, (
        "ordered-block fast path should have replayed signed thinking"
    )

    # ── Recovery: shallow-copy, strip both fields ──────────────────
    api_messages = _shallow_copies(canonical_messages)
    for m in api_messages:
        if not isinstance(m, dict):
            continue
        m.pop("reasoning_details", None)
        m.pop("anthropic_content_blocks", None)

    # ── After recovery: ordered-block fast path is NOT selected ────
    _sys_after, converted_after = convert_messages_to_anthropic(
        api_messages, base_url=None, model="claude-opus-4-6"
    )
    asst_after = next(m for m in converted_after if m["role"] == "assistant")
    block_types_after = [b.get("type") for b in asst_after["content"] if isinstance(b, dict)]
    assert "thinking" not in block_types_after, (
        "ordered-block fast path must not fire after recovery stripped "
        "anthropic_content_blocks — retry would replay the same invalid "
        f"signed thinking block and 400 again. Got blocks: {block_types_after}"
    )
    # tool_use should still be present (reconstructed from tool_calls).
    assert "tool_use" in block_types_after, (
        "tool_use must survive the recovery — only thinking replay state "
        "is stripped, not the tool call itself."
    )

    # ── Canonical messages: both fields still intact ───────────────
    asst_canonical = next(
        m for m in canonical_messages if m["role"] == "assistant"
    )
    assert "reasoning_details" in asst_canonical
    assert "anthropic_content_blocks" in asst_canonical

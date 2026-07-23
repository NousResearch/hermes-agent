"""Regression tests for the thinking-block signature recovery.

The recovery in ``agent/conversation_loop.py`` strips replayed thinking state
from ``api_messages`` (the API-call-time list rebuilt on every retry) and
leaves ``messages`` (the canonical store) untouched. The previous
implementation popped from ``messages`` directly, which never reached
``api_messages`` because each entry in ``api_messages`` was a shallow
copy of the corresponding entry in ``messages``, and the mutation also
landed in ``state.db`` on the next ``_persist_session`` call, corrupting
the conversation.

These tests cover the surface that the recovery touches in isolation:
shallow copies share inner field references; popping a key from one dict
does not remove it from the other; and a list of shallow copies behaves
the same way.
"""


def _shallow_copies(messages):
    return [m.copy() for m in messages]


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

    The canonical ``messages`` list keeps its reasoning_details so future
    persists carry the original signed blocks and ordered content blocks.
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
        if isinstance(m, dict) and "reasoning_details" in m:
            m.pop("reasoning_details", None)
            stripped += 1
        if isinstance(m, dict) and "anthropic_content_blocks" in m:
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
    """A second strip is a no-op when reasoning_details has already been
    removed from api_messages. Guards against a duplicate firing path.
    """
    api_messages = [
        {"role": "assistant", "content": "a", "reasoning_details": [{"x": 1}]},
        {"role": "user", "content": "q"},
    ]
    for _ in range(2):
        for m in api_messages:
            if isinstance(m, dict) and "reasoning_details" in m:
                m.pop("reasoning_details", None)
            if isinstance(m, dict) and "anthropic_content_blocks" in m:
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
        if isinstance(m, dict) and "reasoning_details" in m:
            m.pop("reasoning_details", None)

    assert api_messages == snapshot

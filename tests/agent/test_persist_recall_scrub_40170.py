"""Adjustment 3 (#40170): keep an echoed recall block out of stored turns.

A single scrub at the turn-finalize boundary strips the signed injected block
from the persisted assistant turn (and the in-memory state replayed next turn),
so the echoed-then-persisted class is closed in one place rather than at every
downstream consumer. A stray <memory-context> a user types is preserved.
"""
from agent.memory_manager import build_memory_context_block, strip_injected_recall_blocks
from agent.turn_finalizer import _sanitize_current_turn_assistant_messages

SECRET = "OPERATOR_SECRET_PEER_CARD"


def _block() -> str:
    return build_memory_context_block(f"peer card: {SECRET}")


# -- strip_injected_recall_blocks --------------------------------------------

def test_strips_signed_block():
    text = f"Here is what I recalled: {_block()} — done."
    out = strip_injected_recall_blocks(text)
    assert SECRET not in out
    assert "memory-context" not in out
    assert "Here is what I recalled:" in out and "done." in out


def test_preserves_stray_tag_without_signature():
    prose = "the tag is written <memory-context> in the docs"
    assert strip_injected_recall_blocks(prose) == prose


def test_noop_on_clean_text():
    assert strip_injected_recall_blocks("a normal answer") == "a normal answer"
    assert strip_injected_recall_blocks("") == ""


# -- finalize-boundary scrub over current-turn assistant messages ------------

def test_finalize_scrub_strips_current_turn_assistant():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": f"recalled: {_block()}\nanswer"},
    ]
    _sanitize_current_turn_assistant_messages(messages)
    assert SECRET not in messages[-1]["content"]
    assert "answer" in messages[-1]["content"]


def test_finalize_scrub_stops_at_prior_user_turn():
    """Only the current turn is touched; prior turns are left as-is."""
    prior = f"old: {_block()}"
    messages = [
        {"role": "assistant", "content": prior},
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": f"new: {_block()}"},
    ]
    _sanitize_current_turn_assistant_messages(messages)
    assert SECRET not in messages[-1]["content"]
    assert messages[0]["content"] == prior  # untouched — before the current turn

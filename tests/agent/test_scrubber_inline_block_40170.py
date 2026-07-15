"""Adjustment 1 (#40170): StreamingContextScrubber must strip the real injected
memory-context block from the outbound stream even when non-blank text precedes
the open tag (model echoes it inline, tool prefix, etc.).

The REAL injected block always opens with the exact signature:
    <memory-context>\n[System note: The following is recalled memory context...
No legitimate prose contains that multi-line signature, so it is safe to scrub
regardless of block-boundary state. These tests pin that invariant.

RED on current main: the _is_block_boundary gate refuses to enter the span when
preceding text exists, so the whole block (secret included) leaks.
"""
from agent.memory_manager import build_memory_context_block, StreamingContextScrubber

SECRET = "OPERATOR_SECRET_PEER_CARD"


def _feed(text: str, size: int) -> str:
    s = StreamingContextScrubber()
    out = [s.feed(text[i:i + size]) for i in range(0, len(text), size)]
    out.append(s.flush())
    return "".join(out)


def _real_block() -> str:
    # build_memory_context_block produces the exact wrapper the agent injects.
    return build_memory_context_block(f"peer card: {SECRET}")


def test_block_alone_is_scrubbed_baseline():
    """Boundary-aligned case already works — guard against regression."""
    assert SECRET not in _feed(_real_block(), 4)


def test_inline_preceded_block_is_scrubbed():
    """The bug: text before the tag must NOT defeat scrubbing."""
    text = "Sure, here is what I recalled: " + _real_block() + " — anything else?"
    for size in (1, 4, 7, 999):
        assert SECRET not in _feed(text, size), f"leaked at size {size}"


def test_various_prefixes_do_not_leak():
    for prefix in ("x ", "tool output: ", "...", "answer:"):
        assert SECRET not in _feed(prefix + _real_block(), 4), f"leaked with prefix {prefix!r}"


def test_benign_angle_bracket_prose_is_preserved():
    """The fix must NOT over-scrub: legitimate '<' prose still passes through."""
    prose = "use a < b to compare, and <not-a-real-tag> stays visible"
    out = _feed(prose, 4)
    assert "a < b" in out
    assert "<not-a-real-tag>" in out


def test_inline_memory_context_word_without_signature_is_preserved():
    """A stray '<memory-context>' WITHOUT the system-note signature is prose,
    not the injected block — must not be silently eaten."""
    prose = "the tag is written <memory-context> in the docs"
    out = _feed(prose, 4)
    assert "memory-context" in out

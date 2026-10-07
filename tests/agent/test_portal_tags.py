"""Tests for agent.conversation_context — ambient conversation ContextVar contract."""

from __future__ import annotations






def test_ambient_context_set_none_clears():
    """set_conversation_context(None) publishes no tag (and coerces '')."""
    from agent.conversation_context import (
        get_conversation_context,
        reset_conversation_context,
        set_conversation_context,
    )

    for empty in (None, ""):
        token = set_conversation_context(empty)
        try:
            assert get_conversation_context() is None
        finally:
            reset_conversation_context(token)


def test_ambient_context_isolated_between_contexts():
    """Two copied Contexts (≈ two concurrent agents) don't leak into each other."""
    import contextvars

    from agent.conversation_context import (
        get_conversation_context,
        set_conversation_context,
    )

    def _in_conversation(cid):
        set_conversation_context(cid)
        return get_conversation_context()

    ctx_a = contextvars.copy_context().run(_in_conversation, "agent-a")
    ctx_b = contextvars.copy_context().run(_in_conversation, "agent-b")
    assert ctx_a == "agent-a"
    assert ctx_b == "agent-b"
    # The outer (test) context stays clean.
    assert get_conversation_context() is None


def test_ambient_context_propagates_via_thread_context_helper():
    """propagate_context_to_thread carries the tag onto executor workers (MoA path)."""
    from concurrent.futures import ThreadPoolExecutor

    from agent.conversation_context import (
        get_conversation_context,
        reset_conversation_context,
        set_conversation_context,
    )
    from tools.thread_context import propagate_context_to_thread

    token = set_conversation_context("moa-root")
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            plain = ex.submit(get_conversation_context).result()
            propagated = ex.submit(
                propagate_context_to_thread(get_conversation_context)
            ).result()
    finally:
        reset_conversation_context(token)

    # Bare submit loses the ContextVar; the propagation wrapper keeps it.
    assert plain is None
    assert propagated == "moa-root"

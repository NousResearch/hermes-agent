"""Regression guard for the post-tool automatic-compression attempt cap.

The pre-API and overflow compression paths share ``compression_attempts`` as a
three-pass per-turn backstop. The post-tool path must use the same counter;
otherwise a long tool loop can compact after every tool response for the
lifetime of the turn.
"""
from __future__ import annotations

import inspect


def _post_tool_compression_block() -> str:
    from agent import conversation_loop

    source = inspect.getsource(conversation_loop.run_conversation)
    start = source.index("# Use real token counts from the API response")
    end = source.index("# Save session log incrementally", start)
    return source[start:end]


def test_post_tool_compression_uses_shared_per_turn_attempt_cap():
    block = _post_tool_compression_block()

    assert "compression_attempts < 3" in block, (
        "post-tool compression must stop after the shared three attempts per turn"
    )
    assert "compression_attempts += 1" in block, (
        "post-tool compression must consume one shared per-turn attempt"
    )

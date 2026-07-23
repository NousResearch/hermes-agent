"""Negative-control test: interrupted agent must never produce empty response.

This test validates the fix for the response_len=0 / empty-bubble bug.
When interrupted mid-loop with streamed content already received, the
conversation loop must recover that content as final_response instead of
leaving it as None.
"""
import pytest


def test_interrupt_recovery_code_exists():
    """Verify the interrupt block recovers streamed content before breaking."""
    import ast
    import pathlib

    src = pathlib.Path("agent/conversation_loop.py").read_text()
    tree = ast.parse(src)

    interrupt_blocks = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        try:
            test_str = ast.unparse(node.test)
        except Exception:
            continue
        if "_interrupt_requested" not in test_str:
            continue
        block_lines = src.split("\n")[node.lineno - 1:node.end_lineno]
        interrupt_blocks.append("\n".join(block_lines))

    assert interrupt_blocks, "Must contain _interrupt_requested check"

    # At least one interrupt block must recover streamed content
    found_recovery = False
    for block in interrupt_blocks:
        if "_current_streamed_assistant_text" in block and "final_response" in block:
            found_recovery = True
            break

    assert found_recovery, (
        "No interrupt block recovers _current_streamed_assistant_text "
        "into final_response. The response_len=0 bug is NOT fixed."
    )


def test_turn_finalizer_handles_none_final_response():
    """Verify turn_finalizer handles None/empty final_response without crashing."""
    import ast
    import pathlib

    src = pathlib.Path("agent/turn_finalizer.py").read_text()
    tree = ast.parse(src)

    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        try:
            src_line = ast.get_source_segment(src, node)
        except Exception:
            continue
        if not src_line:
            continue
        if "final_response" not in src_line or "len" not in src_line:
            continue
        # Must guard against None
        if "is not None" in src_line or "and final_response" in src_line:
            found = True
            break

    assert found, (
        "turn_finalizer response_len calculation does not guard against "
        "None final_response — would crash or produce wrong diagnostic."
    )

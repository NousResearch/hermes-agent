"""Regression guard for issue #34569 — inline /steer (and /model) submit
must repaint the input area after clearing the buffer.

Mechanism of the bug
--------------------
``handle_enter`` dispatches ``/steer`` (and ``/model``) inline on the UI
thread while the agent is running.  Those branches called
``buffer.reset(append_to_history=True)`` but — unlike every *other*
early-return branch in the handler — did NOT call ``event.app.invalidate()``.
Because ``process_command()`` prints through ``patch_stdout`` (which scrolls
output above the prompt and never triggers a prompt_toolkit redraw), the
just-cleared input area could keep showing the submitted ``/steer <text>``
until some unrelated redraw fired.  The user saw their submitted text as if
it were unsent and could accidentally re-submit it.

This test pins the contract structurally: inside ``handle_enter``, any
inline-command early-return that resets the buffer must be followed by an
``event.app.invalidate()`` before its ``return``.  It is an *invariant*
(every reset-then-return repaints), not a snapshot of current source.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _load_handle_enter_node() -> ast.FunctionDef:
    """Extract the ``handle_enter`` nested function node from cli.py."""
    cli_path = Path(__file__).resolve().parents[2] / "cli.py"
    tree = ast.parse(cli_path.read_text(encoding="utf-8"))

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "handle_enter":
            target = node
            break
    assert target is not None, "handle_enter closure not found in cli.py"
    return target


def _is_buffer_reset(node: ast.stmt) -> bool:
    """True if the statement is ``...current_buffer.reset(...)``."""
    if not isinstance(node, ast.Expr):
        return False
    call = node.value
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    return isinstance(func, ast.Attribute) and func.attr == "reset"


def _is_invalidate(node: ast.stmt) -> bool:
    """True if the statement is ``event.app.invalidate()``."""
    if not isinstance(node, ast.Expr):
        return False
    call = node.value
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    return isinstance(func, ast.Attribute) and func.attr == "invalidate"


def _collect_reset_blocks(func: ast.FunctionDef) -> list[list[ast.stmt]]:
    """Find every statement sequence (a block body/orelse/finalbody) within
    ``handle_enter`` that contains a ``buffer.reset()`` call."""
    blocks: list[list[ast.stmt]] = []
    for node in ast.walk(func):
        for attr in ("body", "orelse", "finalbody"):
            seq = getattr(node, attr, None)
            if not isinstance(seq, list):
                continue
            if any(isinstance(s, ast.stmt) and _is_buffer_reset(s) for s in seq):
                blocks.append(seq)
    return blocks


def test_inline_command_reset_branches_invalidate():
    """Every handle_enter branch that resets the buffer and then returns must
    invalidate the app first (issue #34569)."""
    func = _load_handle_enter_node()
    reset_blocks = _collect_reset_blocks(func)

    assert reset_blocks, "expected to find buffer.reset() calls in handle_enter"

    offenders = []
    for seq in reset_blocks:
        for i, stmt in enumerate(seq):
            if not _is_buffer_reset(stmt):
                continue
            # Find the next return after this reset in the same block.
            ret_idx = None
            for j in range(i + 1, len(seq)):
                if isinstance(seq[j], ast.Return):
                    ret_idx = j
                    break
            if ret_idx is None:
                # reset not directly followed by a return in this block
                # (e.g. the fall-through reset at the end of the handler) —
                # the next user input naturally repaints, so skip.
                continue
            between = seq[i + 1 : ret_idx]
            if not any(_is_invalidate(s) for s in between):
                offenders.append(ast.dump(stmt))

    assert not offenders, (
        "handle_enter has reset-then-return branch(es) that never call "
        "event.app.invalidate() — the input area can keep showing the "
        "submitted text (issue #34569). Offending reset stmts:\n"
        + "\n".join(offenders)
    )


if __name__ == "__main__":  # pragma: no cover
    test_inline_command_reset_branches_invalidate()
    print("ok")

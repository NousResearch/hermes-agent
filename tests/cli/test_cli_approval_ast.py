"""Structural AST regression test for approval buffer forward (issue #35999).

Verifies the invariant inside ``handle_enter``: after resolving the
approval panel, if there is pending buffer text, the handler falls through
to the Normal Input Routing section instead of returning early with an
``event.app.invalidate()`` call.

This is an *invariant* test — it checks the control flow structure, not a
snapshot of current source.  If someone later adds an early return in the
approval-resolved branch, this test catches the regression.

See also test_steer_inline_repaint_34569.py for the same structural pattern.
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


def _collect_if_nodes(func: ast.FunctionDef, target_test: str, exact: bool = False) -> list[ast.If]:
    """Find ``if`` nodes inside *func* whose test attribute string contains
    *target_test* when rendered by ast.dump.  When *exact* is True, the
    match must be exact (the dump of the test equals *target_test*)."""
    results: list[ast.If] = []
    for node in ast.walk(func):
        if isinstance(node, ast.If):
            dump = ast.dump(node.test)
            if (exact and dump == target_test) or (not exact and target_test in dump):
                results.append(node)
    return results


def test_approval_resolved_with_buffer_does_not_return_early():
    """When approval resolves (``_approval_state is None``) AND buffer text
    is present, ``handle_enter`` must NOT return early — it must fall through
    to the Normal Input Routing section so that busy_input_mode semantics are
    preserved.

    The early return (``event.app.invalidate(); return``) is only valid when
    ``_approval_state is not None`` (view expanded in-place) OR when there is
    no buffer text to forward.
    """
    func = _load_handle_enter_node()

    # Find the outer approval-state if block (exact match avoids picking
    # up the nested ``if self._approval_state is None and ...`` inside it).
    approval_ifs = _collect_if_nodes(func, "Attribute(value=Name(id='self', ctx=Load()), attr='_approval_state', ctx=Load())", exact=True)
    assert len(approval_ifs) == 1, (
        f"expected exactly one '_approval_state' if in handle_enter, "
        f"found {len(approval_ifs)}"
    )
    approval_if = approval_ifs[0]

    # Walk the body of the approval if looking for the nested check:
    #   if self._approval_state is None and (text or has_images):
    #       pass  # fall through
    #   else:
    #       event.app.invalidate()
    #       return
    fall_through_pattern = False
    early_return_pattern = False

    for stmt in approval_if.body:
        if isinstance(stmt, ast.If):
            # This should be the nested if (approval resolved + has content)
            # Check that the else branch has return + invalidate
            if stmt.orelse:
                else_body = stmt.orelse
                # Could be a single If (elif) or a list of statements
                else_stmts = else_body if isinstance(else_body, list) else else_body.body if hasattr(else_body, 'body') else []
                # Check for invalidate + return in else
                has_invalidate = any(
                    isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                    and hasattr(s.value.func, 'attr') and s.value.func.attr == 'invalidate'
                    for s in (else_stmts if isinstance(else_stmts, list) else [else_stmts])
                )
                has_return = any(
                    isinstance(s, ast.Return) for s in (else_stmts if isinstance(else_stmts, list) else [else_stmts])
                )
                if has_invalidate and has_return:
                    early_return_pattern = True

            # Check that the if body has a 'pass' statement (fall-through)
            for s in stmt.body:
                if isinstance(s, ast.Pass):
                    fall_through_pattern = True

    assert fall_through_pattern, (
        "handle_enter approval branch missing 'pass' fall-through "
        "when approval is resolved with buffer text — the handler may "
        "return early and discard buffered input (issue #35999)"
    )
    assert early_return_pattern, (
        "handle_enter approval branch missing 'invalidate + return' in else "
        "branch for the view/no-content case"
    )


def test_approval_view_branch_returns_early():
    """When 'view' expands the command in-place (``_approval_state not None``),
    ``handle_enter`` must return early WITHOUT falling through to normal input
    routing — the buffer text stays in the buffer.
    """
    func = _load_handle_enter_node()

    approval_ifs = _collect_if_nodes(func, "Attribute(value=Name(id='self', ctx=Load()), attr='_approval_state', ctx=Load())", exact=True)
    approval_if = approval_ifs[0]

    # Find the nested if: `_approval_state is None` — the else of this
    # if is the case where approval_state IS NOT None (view mode).
    view_returns_early = False
    for stmt in approval_if.body:
        if isinstance(stmt, ast.If):
            # The else branch handles the case where approval_state is NOT None
            if stmt.orelse:
                else_body = stmt.orelse
                else_stmts = else_body if isinstance(else_body, list) else else_body.body if hasattr(else_body, 'body') else []
                for s in (else_stmts if isinstance(else_stmts, list) else [else_stmts]):
                    if isinstance(s, ast.Return):
                        view_returns_early = True

    assert view_returns_early, (
        "handle_enter missing early return in the view/no-content "
        "approval branch — buffer text may be incorrectly forwarded "
        "when approval panel stays open"
    )


if __name__ == "__main__":  # pragma: no cover
    test_approval_resolved_with_buffer_does_not_return_early()
    test_approval_view_branch_returns_early()
    print("ok")

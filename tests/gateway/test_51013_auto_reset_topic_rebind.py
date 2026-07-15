"""Regression pin for #51013 — Telegram topic lanes must keep an auto-reset child.

When ``get_or_create_session`` mints a fresh auto-reset child for a Telegram
topic lane, the stored ``(chat_id, thread_id) -> session_id`` binding still
points at the expired parent. Without a guard, the binding-heal walk
``switch_session``'s the turn straight back to that parent (re-opening the
ended session), so the reset never takes on topic lanes and the fresh-context
boundary (#35809 / #48031) is skipped.

The fix mirrors the explicit ``/new`` rebind: when ``session_entry`` was just
auto-reset, the binding is rewritten to the fresh child
(``reason="auto-reset"``) instead of honoring the stale binding.

AST invariants in the style of ``test_35809_auto_reset_clean_context.py`` —
they pin the shape of the binding-heal block in ``gateway/run.py`` without
stubbing the (heavily async) runner.
"""

from __future__ import annotations

import ast
import inspect

from gateway import run as gateway_run


def _find_topic_binding_heal_block() -> ast.If:
    """Return the ``if binding:`` block inside the topic-lane routing code."""
    tree = ast.parse(inspect.getsource(gateway_run))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not (isinstance(node.test, ast.Name) and node.test.id == "binding"):
            continue
        attrs = {
            sub.attr for sub in ast.walk(node) if isinstance(sub, ast.Attribute)
        }
        if "_sync_telegram_topic_binding" in attrs and "switch_session" in attrs:
            return node
    raise AssertionError(
        "Could not locate the Telegram topic binding-heal block "
        "(if binding: ... switch_session / _sync_telegram_topic_binding) in "
        "gateway/run.py — the structure changed or this walker is stale."
    )


def _was_auto_reset_guard(block: ast.If) -> ast.If:
    guard = block.body[0]
    assert isinstance(guard, ast.If), (
        "The binding-heal block must OPEN with the was_auto_reset guard so a "
        "stale binding can never act before the auto-reset check (#51013)."
    )
    consts = [
        n.value
        for n in ast.walk(guard.test)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    assert "was_auto_reset" in consts, (
        "The first branch of the binding-heal block no longer tests "
        "was_auto_reset — a stale topic binding can switch the turn back to "
        "the expired parent and undo the auto-reset (#51013)."
    )
    return guard


def _attrs_in(stmts) -> set:
    return {
        n.attr
        for stmt in stmts
        for n in ast.walk(stmt)
        if isinstance(n, ast.Attribute)
    }


class TestAutoResetChildKeepsTopicLane:
    def test_binding_heal_is_guarded_by_was_auto_reset(self):
        _was_auto_reset_guard(_find_topic_binding_heal_block())

    def test_auto_reset_branch_rebinds_and_never_switches(self):
        guard = _was_auto_reset_guard(_find_topic_binding_heal_block())
        body_attrs = _attrs_in(guard.body)
        assert "_sync_telegram_topic_binding" in body_attrs, (
            "The auto-reset branch must rewrite the topic binding to the "
            "fresh child (mirrors the explicit /new rebind)."
        )
        assert "switch_session" not in body_attrs, (
            "The auto-reset branch must never switch_session — that would "
            "re-open the expired parent and drop the fresh reset child."
        )
        reasons = [
            kw.value.value
            for stmt in guard.body
            for n in ast.walk(stmt)
            if isinstance(n, ast.Call)
            for kw in n.keywords
            if kw.arg == "reason" and isinstance(kw.value, ast.Constant)
        ]
        assert "auto-reset" in reasons, (
            "The rebind must be recorded with reason='auto-reset' so binding "
            "audit trails distinguish it from the compression-tip heal."
        )

    def test_stale_binding_walk_lives_in_the_else_branch(self):
        guard = _was_auto_reset_guard(_find_topic_binding_heal_block())
        else_attrs = _attrs_in(guard.orelse)
        assert "switch_session" in else_attrs, (
            "The existing stale-binding heal (switch_session to the bound "
            "session) must be preserved for ordinary, non-reset turns."
        )
        assert "get_compression_tip" in else_attrs, (
            "The compression-tip walk (#20470/#29712/#33414) must be "
            "preserved for ordinary, non-reset turns."
        )

    def test_was_auto_reset_flag_is_not_consumed_by_the_guard(self):
        """The guard must only READ was_auto_reset. Consumption (flag clear +
        conversation-boundary cleanup) stays in the dedicated block below it
        (#48031), which this guard must not preempt."""
        guard = _was_auto_reset_guard(_find_topic_binding_heal_block())
        for stmt in ast.walk(guard):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr == "was_auto_reset"
                    ):
                        raise AssertionError(
                            "The binding guard must not clear was_auto_reset; "
                            "the conversation-boundary block owns consumption."
                        )

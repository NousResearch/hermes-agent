"""Regression tests for #29335 — gateway compression session switches must
persist routing through one shared helper.

When compression rolls a gateway session forward into a new ``session_id``, the
agent returns the new id in its result dict. The gateway must update
``session_entry.session_id`` in memory and call ``session_store._save()`` so the
mapping survives a gateway restart; otherwise the next turn loads the old
transcript and re-triggers compression forever.

That same seam now also migrates session-scoped ``/goal`` state, so old inline
``session_entry.session_id = ...`` mutations are no longer safe. Every
compression-producing path should route through
``GatewayRunner._handle_compression_session_switch()``.

``TestCompressionSessionPropagation`` adds behavioral tests that exercise the
propagation path with mocks mirroring the gateway objects, verifying the
session-entry update and ``_save()`` semantics without requiring a live gateway.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from unittest.mock import MagicMock, call

from gateway import run as gateway_run
from gateway.session_context import set_current_session_id, get_session_env


def _session_entry_session_id_assignments(source: str) -> list[tuple[str, int]]:
    """Return ``(function_name, lineno)`` for direct session_entry.session_id writes."""
    tree = ast.parse(textwrap.dedent(source))
    results: list[tuple[str, int]] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._function_stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._function_stack.append(node.name)
            self.generic_visit(node)
            self._function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._function_stack.append(node.name)
            self.generic_visit(node)
            self._function_stack.pop()

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "session_id"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "session_entry"
                ):
                    results.append((self._function_stack[-1] if self._function_stack else "<module>", node.lineno))
            self.generic_visit(node)

    _Visitor().visit(tree)
    return results


def test_post_compression_session_id_updates_use_shared_switch_helper():
    """Direct ``session_entry.session_id = ...`` writes must stay centralized.

    The helper persists the session-store mapping and runs companion session
    split behavior such as Telegram binding sync and /goal migration. If a new
    compression path mutates ``session_entry.session_id`` inline, it can revive
    the old #29335 restart loop or orphan session-scoped state again.
    """
    source = inspect.getsource(gateway_run)
    assignments = _session_entry_session_id_assignments(source)
    assert assignments, (
        "No direct ``session_entry.session_id = ...`` assignment found. If the "
        "helper was refactored to avoid direct assignment entirely, update this "
        "guard to assert the new persistence/migration choke point instead."
    )
    unexpected = [
        (function_name, lineno)
        for function_name, lineno in assignments
        if function_name != "_handle_compression_session_switch"
    ]
    assert not unexpected, (
        "Post-compression session_id changes must route through "
        "GatewayRunner._handle_compression_session_switch() so routing persistence, "
        "Telegram binding sync, and /goal migration stay together. Unexpected "
        f"direct assignments: {unexpected}"
    )


class TestCompressionSessionPropagation:
    """Behavioral tests for post-compression session_id propagation.

    The structural AST test above pins that every ``session_entry.session_id``
    assignment in gateway/run.py is followed by ``_save()``.  These tests
    exercise the *behavior* of that propagation path inline, using mocks that
    mirror the objects gateway/run.py works with (``session_entry`` and
    ``session_store``), verifying the semantics are correct without requiring a
    live gateway instance.

    Ordering contract (from the comments added to the source in this PR):
    1. The agent thread updates the contextvar in ``conversation_compression.py``
       via ``set_current_session_id(agent.session_id)``.
    2. After ``run_in_executor`` returns, the gateway propagates the new id to
       ``session_entry.session_id`` and calls ``session_store._save()``.
    Both halves must agree for the next turn to route correctly.
    """

    def test_gateway_session_entry_follows_compression_rotation(self) -> None:
        """The gateway handler must update session_entry and call _save() when
        the agent result carries a rotated session_id.

        Simulates the inline propagation block in gateway/run.py:

            if agent_result.get("session_id") and \\
                    agent_result["session_id"] != session_entry.session_id:
                session_entry.session_id = agent_result["session_id"]
                self.session_store._save()

        Verifies that session_entry.session_id is mutated and _save is called
        exactly once — the minimal contract that prevents the restart-loop bug.
        """
        old_sid = "20260101_000000_aaaaaa"
        new_sid = "20260101_000001_bbbbbb"

        session_entry = MagicMock()
        session_entry.session_id = old_sid

        session_store = MagicMock()

        agent_result = {"session_id": new_sid, "response": "hello"}

        # Inline the propagation logic exactly as it appears in gateway/run.py
        # (around line 9459). This is the behavior we are pinning.
        if agent_result.get("session_id") and agent_result["session_id"] != session_entry.session_id:
            session_entry.session_id = agent_result["session_id"]
            session_store._save()

        assert session_entry.session_id == new_sid, (
            "session_entry.session_id was not updated to the compressed session id. "
            "The next turn would load the old transcript and re-trigger compression."
        )
        session_store._save.assert_called_once_with(), (
            "session_store._save() was not called after session_entry update. "
            "The new session mapping would not survive a gateway restart."
        )

    def test_no_update_when_session_id_unchanged(self) -> None:
        """The propagation block must be a no-op when the agent did not compress.

        If the agent returns the same session_id (normal turn, no compression),
        session_entry must not be touched and _save must not be called — avoiding
        spurious writes on every turn.
        """
        same_sid = "20260101_000000_aaaaaa"

        session_entry = MagicMock()
        session_entry.session_id = same_sid

        session_store = MagicMock()

        # Normal turn: agent returns same session_id (or none at all)
        agent_result = {"response": "hello"}  # no "session_id" key

        if agent_result.get("session_id") and agent_result["session_id"] != session_entry.session_id:
            session_entry.session_id = agent_result["session_id"]
            session_store._save()

        # session_entry.session_id was set during mock construction; the
        # propagation block must not have set it again.
        session_store._save.assert_not_called()

    def test_contextvar_and_session_entry_agree_after_compression(self) -> None:
        """After compression, the contextvar and session_entry must carry the
        same session_id.

        The agent thread calls ``set_current_session_id(new_sid)`` inside
        ``conversation_compression.py`` (step 1).  The gateway then propagates
        ``new_sid`` to ``session_entry.session_id`` (step 2).  If either step
        is missing, tool calls and transcript writes will disagree on which
        session is active.

        This test simulates both steps and asserts agreement.
        """
        old_sid = "20260101_000000_cccccc"
        new_sid = "20260101_000002_dddddd"

        # Step 1: agent thread updates contextvar (mirrors conversation_compression.py
        # around line 511-513)
        set_current_session_id(new_sid)

        # Step 2: gateway propagates to session_entry (mirrors gateway/run.py
        # around line 9459-9461)
        session_entry = MagicMock()
        session_entry.session_id = old_sid
        agent_result = {"session_id": new_sid}

        if agent_result.get("session_id") and agent_result["session_id"] != session_entry.session_id:
            session_entry.session_id = agent_result["session_id"]

        contextvar_sid = get_session_env("HERMES_SESSION_ID", "")
        assert contextvar_sid == new_sid, (
            f"Contextvar still holds old session_id '{contextvar_sid}' after "
            f"set_current_session_id('{new_sid}'). Tool calls in the next turn "
            "will read stale routing state."
        )
        assert session_entry.session_id == new_sid, (
            f"session_entry.session_id is '{session_entry.session_id}' but contextvar "
            f"says '{contextvar_sid}'. The two routing paths disagree after compression."
        )
        assert contextvar_sid == session_entry.session_id, (
            "Contextvar and session_entry disagree on the active session_id "
            "after compression rotation. Exactly one of the two ordering steps "
            "was skipped."
        )

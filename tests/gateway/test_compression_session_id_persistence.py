"""Regression tests for #29335 — gateway must persist ``session_entry.session_id``
after the agent's compression path mutates it.

When ``_compress_context()`` rolls the agent forward into a new session, the
agent now returns the new ``session_id`` in its result dict. The gateway
updates ``session_entry.session_id`` in memory AND must call
``session_store._save()`` so the new mapping survives a gateway restart.
Without ``_save()``, the next turn loads the OLD session's transcript and
re-triggers compression forever.

Three sites in ``gateway/run.py`` mutate ``session_entry.session_id`` after
a compression-induced session split. All three MUST be followed by a
``_save()`` call. This test pins that invariant.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from gateway import run as gateway_run


def _session_id_assignments_followed_by_save(source: str) -> list[tuple[int, bool]]:
    """For each ``session_entry.session_id = ...`` assignment in *source*,
    return ``(lineno, saved_within_5_stmts)`` — True iff a
    ``self.session_store._save()`` call appears in the same block within the
    next 5 statements (covers normal control flow without false-flagging
    cleanup that lives 200 lines away).
    """
    tree = ast.parse(textwrap.dedent(source))
    results: list[tuple[int, bool]] = []

    class _Visitor(ast.NodeVisitor):
        def _is_session_id_assign(self, node: ast.AST) -> bool:
            if not isinstance(node, ast.Assign):
                return False
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "session_id"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "session_entry"
                ):
                    return True
            return False

        def _block_has_save_after(self, body: list[ast.stmt], idx: int) -> bool:
            for stmt in body[idx : idx + 6]:
                for sub in ast.walk(stmt):
                    if (
                        isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "_save"
                    ):
                        return True
            return False

        def _walk_body(self, body: list[ast.stmt]) -> None:
            for i, stmt in enumerate(body):
                if self._is_session_id_assign(stmt):
                    results.append((stmt.lineno, self._block_has_save_after(body, i)))
                for child in ast.iter_child_nodes(stmt):
                    if isinstance(child, (ast.If, ast.For, ast.While, ast.With,
                                          ast.Try, ast.AsyncWith, ast.AsyncFor)):
                        self._walk_node(child)

        def _walk_node(self, node: ast.AST) -> None:
            for attr in ("body", "orelse", "finalbody"):
                inner = getattr(node, attr, None)
                if isinstance(inner, list):
                    self._walk_body(inner)
            if hasattr(node, "handlers"):
                for handler in node.handlers:
                    self._walk_body(handler.body)

        def visit(self, node: ast.AST) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._walk_body(node.body)
            for child in ast.iter_child_nodes(node):
                self.visit(child)

    _Visitor().visit(tree)
    return results


def test_every_post_compression_session_id_assignment_persists():
    """Every ``session_entry.session_id = ...`` in gateway/run.py must be
    followed by a ``session_store._save()`` call within the same block.

    Regression for #29335 — the assignment at the end of
    ``_handle_message_with_agent`` used to skip ``_save()`` while two sibling
    sites (hygiene rewrite, manual /compress) already persisted. The agent
    would compress correctly, the gateway would update its in-memory
    session_id, then drop it on next gateway restart.
    """
    source = inspect.getsource(gateway_run)
    assignments = _session_id_assignments_followed_by_save(source)
    assert assignments, (
        "No ``session_entry.session_id = ...`` assignments found in gateway/run.py — "
        "either the structure changed or the AST walker is broken."
    )
    missing = [lineno for lineno, saved in assignments if not saved]
    assert not missing, (
        f"{len(missing)} ``session_entry.session_id = ...`` site(s) in gateway/run.py "
        f"are not followed by ``session_store._save()`` within the same block "
        f"(lines: {missing}). Every post-compression session_id update must persist "
        f"or the next turn loads the pre-compression transcript and triggers an "
        f"infinite compression loop. See issue #29335."
    )

"""``init_agent`` must resolve reasoning_config when no caller supplies one.

Regression guard for a bug CLASS, not a single call site. Every ``AIAgent(...)``
construction site has to remember a ``reasoning_config=`` kwarg; a site that
omits it stores ``None``, and ``build_anthropic_kwargs`` gates its whole
thinking mapping on a truthiness check — so ``None`` means no ``thinking`` key
on the wire at all. Adaptive Claude then falls back to the API default
``display: "omitted"`` and returns thinking blocks whose text is empty with only
an opaque signature populated, which hosts render as garbled characters where
reasoning text should be.

Patching individual call sites has not converged; ``init_agent`` is the single
chokepoint every surface passes through, so the resolution belongs there.

The resolution lives inline inside ``init_agent``, a function far too large to
invoke in a unit test, so this asserts the *structure* of the chokepoint via
AST. Deliberately structural rather than a substring search: the explanatory
comment beside the fix names ``resolve_reasoning_config``, so a plain
``in source`` check would pass against reverted code.
"""

from __future__ import annotations

import ast
import inspect
import textwrap


def _init_agent_ast() -> ast.FunctionDef:
    from agent import agent_init

    tree = ast.parse(textwrap.dedent(inspect.getsource(agent_init)))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "init_agent":
            return node
    raise AssertionError("agent_init.init_agent not found")


def _names_bound_to_resolver(fn: ast.FunctionDef) -> set[str]:
    """Local names bound to ``resolve_reasoning_config``, alias included.

    The fix imports it inside the guard as ``_resolve_rc``; a later cleanup may
    hoist the import to module scope or drop the alias. Resolving the binding
    keeps this guard from tripping on cosmetic churn.
    """
    names = {"resolve_reasoning_config"}
    for node in ast.walk(fn):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "resolve_reasoning_config":
                    names.add(alias.asname or alias.name)
    return names


def test_chokepoint_guards_assignment_with_resolution():
    """A ``reasoning_config is None`` guard must resolve before the assignment."""
    fn = _init_agent_ast()
    resolver_names = _names_bound_to_resolver(fn)

    assignments = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Attribute)
            and t.attr == "reasoning_config"
            and isinstance(t.value, ast.Name)
            and t.value.id == "agent"
            for t in node.targets
        )
    ]
    assert assignments, "init_agent no longer assigns agent.reasoning_config"

    guards = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "reasoning_config"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Is)
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value is None
    ]
    assert guards, (
        "init_agent must guard on `reasoning_config is None` and resolve the "
        "user's configured effort there; without it, every surface that omits "
        "the kwarg silently drops thinking text"
    )

    resolving_guards = [
        guard
        for guard in guards
        if any(
            isinstance(call, ast.Call)
            and (getattr(call.func, "id", None) or getattr(call.func, "attr", None))
            in resolver_names
            for call in ast.walk(guard)
        )
    ]
    assert resolving_guards, (
        "the `reasoning_config is None` branch must call "
        "resolve_reasoning_config — a guard that leaves it None is the bug"
    )

    earliest_assignment = min(node.lineno for node in assignments)
    assert any(guard.lineno < earliest_assignment for guard in resolving_guards), (
        "the resolution must run BEFORE agent.reasoning_config is assigned, "
        "otherwise the resolved value never reaches the agent"
    )

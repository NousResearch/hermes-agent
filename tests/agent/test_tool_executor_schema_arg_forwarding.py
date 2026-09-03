"""Guard: every tool-dispatch branch in ``agent/tool_executor.py`` must forward
the parameters its own schema declares.

Why this test exists
--------------------
``execute_tool_calls_sequential`` dispatches the agent-runtime tools with one
``elif function_name == "<tool>"`` branch each, and every branch enumerates the
tool's arguments **by hand**:

    return _session_search(
        query=next_args.get("query", ""),
        ...
    )

The authoritative parameter list, however, lives in the tool's schema, and the
registry handler beside it forwards ``args.get(...)`` for each. So when a
parameter is added to a schema (and to the tool function, and to the registry
handler), the hand-written branch keeps working and silently drops it.

The failure mode is the worst kind: no error, no warning. The tool returns a
valid-looking result computed *without* the argument, and the model concludes
the feature does not work. Three instances of exactly this shipped:

* ``session_search`` dropped ``profile`` — cross-profile search silently
  returned the *active* profile's sessions, so ``@session:<profile>/<id>``
  links could not be resolved at all.
* ``memory`` dropped ``new_text`` — the documented alias for ``content``
  reached ``memory_tool`` as ``None``, so a single-op
  ``replace(old_text=..., new_text=...)`` failed with "content is required".
  Batch ops coalesce the alias themselves, which is why this stayed hidden.
* ``drive_preview`` dropped ``full`` — a request to re-read the whole page
  returned a delta instead.

Each was a one-line fix; the class is what needs a guard. This test compares
the schema's declared properties against the argument keys each branch actually
reads, so the next dropped parameter fails in CI instead of being discovered
months later by its symptom.

How it works
------------
1. Populate the registry (``discover_builtin_tools``) and take each tool's
   declared schema properties.
2. Parse ``agent/tool_executor.py`` and locate the dispatch branches
   *inside ``execute_tool_calls_sequential`` only* — ``function_name ==``
   comparisons elsewhere are bookkeeping (counter resets, checkpoints) and
   read no arguments by design.
3. Collect every key the branch reads from the tool-args mapping, in any of
   the shapes the file uses: ``next_args.get("k")``, ``next_args["k"]``,
   ``"k" in next_args``, ``next_args.pop("k")``.
4. A branch that forwards the whole mapping (``**next_args``, or a call taking
   ``next_args`` positionally as ``delegate_task`` does) cannot drop anything
   and is exempt.

Adding a genuinely-not-forwarded parameter requires an explicit entry in
``EXEMPT``, with a reason — so the decision is visible in review rather than
implicit in an omission.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

EXECUTOR_PATH = Path(__file__).resolve().parents[2] / "agent" / "tool_executor.py"

# The single function that owns the per-tool dispatch chain. Branches outside
# it (`_run_agent_tool_execution_middleware`, `_begin_tool_execution`) compare
# `function_name` for bookkeeping only.
DISPATCHER = "execute_tool_calls_sequential"

# Names the dispatch branches use for the incoming tool-arguments mapping.
ARG_MAPPINGS = {"next_args", "function_args"}

# Each branch wraps the actual tool call in a closure of this name, which the
# execution middleware invokes with the (possibly rewritten) arguments.
EXECUTE_CLOSURE = "_execute"

# Schema properties a branch is allowed not to forward, keyed by tool, with the
# reason. Empty by design: every branch currently forwards everything its schema
# declares, and an omission should have to be argued for in review rather than
# pass unnoticed. (``delegate_task`` needs no entry — it forwards the whole
# mapping and is detected automatically.)
EXEMPT: dict[str, dict[str, str]] = {}


# --------------------------------------------------------------------------- #
# Static analysis helpers
# --------------------------------------------------------------------------- #


def _dispatcher_node(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == DISPATCHER:
            return node
    raise AssertionError(f"{DISPATCHER}() not found in {EXECUTOR_PATH}")


def _branch_tool_names(test: ast.expr) -> list[str]:
    """Tool names compared against ``function_name`` in a branch test.

    Handles ``function_name == "x"``, ``function_name in ("x", "y")``, and the
    same comparison appearing inside a larger boolean expression.
    """
    names: list[str] = []
    for node in ast.walk(test):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if not (isinstance(left, ast.Name) and left.id == "function_name"):
            continue
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, ast.Eq) and isinstance(comparator, ast.Constant):
                if isinstance(comparator.value, str):
                    names.append(comparator.value)
            elif isinstance(op, ast.In) and isinstance(comparator, (ast.Tuple, ast.List, ast.Set)):
                names += [
                    e.value for e in comparator.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                ]
    return names


def _iter_dispatch_branches(tree: ast.Module):
    """Yield ``(tool_names, branch_body)`` for each dispatch branch."""
    dispatcher = _dispatcher_node(tree)
    for node in ast.walk(dispatcher):
        if not isinstance(node, ast.If):
            continue
        names = _branch_tool_names(node.test)
        if names:
            yield names, node.body


def _keys_read(body: list[ast.stmt]) -> tuple[set[str], bool]:
    """Argument keys the branch reads, and whether it forwards the whole mapping."""
    keys: set[str] = set()

    def _is_mapping(node: ast.expr) -> bool:
        return isinstance(node, ast.Name) and node.id in ARG_MAPPINGS

    for stmt in body:
        for node in ast.walk(stmt):
            # next_args.get("k") / next_args.pop("k")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if _is_mapping(node.func.value) and node.func.attr in {"get", "pop"}:
                    if node.args and isinstance(node.args[0], ast.Constant):
                        if isinstance(node.args[0].value, str):
                            keys.add(node.args[0].value)
            # next_args["k"]
            if isinstance(node, ast.Subscript) and _is_mapping(node.value):
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    keys.add(node.slice.value)
            # "k" in next_args
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant):
                if isinstance(node.left.value, str):
                    for op, comparator in zip(node.ops, node.comparators):
                        if isinstance(op, (ast.In, ast.NotIn)) and _is_mapping(comparator):
                            keys.add(node.left.value)

    return keys, _forwards_whole_mapping(body, _is_mapping)


def _forwards_whole_mapping(body: list[ast.stmt], is_mapping) -> bool:
    """True when the branch hands the entire argument mapping to the tool itself.

    Two narrowings, both learned from real false positives in this file:

    * Only the branch's ``_execute`` closure counts. The surrounding branch also
      passes the mapping around for presentation — e.g.
      ``_get_cute_tool_message_impl('memory', function_args, tool_duration)``
      renders the status line from it. Counting that would exempt every branch
      and make this guard vacuous.
    * Within the closure, only the call that *produces the tool result* counts —
      the one whose value is returned. ``memory``'s branch passes ``next_args``
      to ``agent._memory_manager.notify_memory_tool_write(result, next_args)``
      to mirror the write to external providers; that call cannot compensate for
      an argument the tool itself never received.

    ``delegate_task`` is the genuine case: its closure is
    ``return agent._dispatch_delegate_task(next_args)``, which forwards
    everything and therefore cannot drop a parameter.
    """
    for stmt in body:
        for node in ast.walk(stmt):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name != EXECUTE_CLOSURE:
                continue
            for call in _result_producing_calls(node):
                if any(kw.arg is None and is_mapping(kw.value) for kw in call.keywords):
                    return True
                if any(is_mapping(a) for a in call.args):
                    return True
    return False


def _result_producing_calls(closure: ast.FunctionDef | ast.AsyncFunctionDef):
    """Calls inside ``closure`` whose value becomes the closure's return value.

    Covers both shapes the branches use: ``return tool(...)`` directly, and
    ``result = tool(...)`` followed by ``return result``.
    """
    returned_names: set[str] = set()
    for node in ast.walk(closure):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Name):
            returned_names.add(node.value.id)

    for node in ast.walk(closure):
        if isinstance(node, ast.Return) and node.value is not None:
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Call):
                    yield sub
        elif isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if targets & returned_names:
                for sub in ast.walk(node.value):
                    if isinstance(sub, ast.Call):
                        yield sub


def _schema_properties(name: str) -> set[str] | None:
    from tools.registry import registry

    schema = registry.get_schema(name)
    if not schema:
        return None
    fn = schema.get("function", schema)
    props = (fn.get("parameters") or {}).get("properties") or {}
    return set(props)


def _dispatch_map() -> dict[str, tuple[set[str], bool]]:
    tree = ast.parse(EXECUTOR_PATH.read_text(encoding="utf-8"))
    out: dict[str, tuple[set[str], bool]] = {}
    for names, body in _iter_dispatch_branches(tree):
        keys, forwards_all = _keys_read(body)
        for name in names:
            out[name] = (keys, forwards_all)
    return out


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def _dispatched_tool_names() -> list[str]:
    from tools.registry import discover_builtin_tools

    discover_builtin_tools()
    return sorted(name for name in _dispatch_map() if _schema_properties(name) is not None)


class TestDispatchBranchesForwardSchemaArgs:
    def test_dispatcher_branches_are_found(self):
        """Sanity: the analysis actually locates branches.

        Without this, a refactor that renames the dispatcher would make the
        parametrised test below silently vacuous — passing because it checks
        nothing at all.
        """
        assert len(_dispatch_map()) >= 5

    def test_registry_resolves_dispatched_tools(self):
        assert _dispatched_tool_names(), "no dispatched tool resolved to a schema"

    @pytest.mark.parametrize("tool_name", _dispatched_tool_names())
    def test_branch_forwards_every_schema_property(self, tool_name):
        declared = _schema_properties(tool_name) or set()
        keys, forwards_all = _dispatch_map()[tool_name]

        if forwards_all:
            pytest.skip(f"{tool_name} forwards the whole argument mapping")

        exempt = set(EXEMPT.get(tool_name, {}))
        missing = sorted(declared - keys - exempt)

        assert not missing, (
            f"{tool_name}: dispatch branch in {DISPATCHER}() never reads "
            f"schema parameter(s) {missing}.\n"
            f"The model is told it may send them, so they are dropped "
            f"silently — the tool runs without them and returns a "
            f"plausible wrong result.\n"
            f"Fix: forward them in the branch's call, e.g. "
            f"`{missing[0]}=next_args.get(\"{missing[0]}\")`.\n"
            f"If the omission is deliberate, add {tool_name!r} to EXEMPT in "
            f"{Path(__file__).name} with the reason."
        )

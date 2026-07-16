"""P2.1: the finalize-path hooks must carry execution provenance too.

`build_turn_context` threads `execution_kind`/`execution_id` into the ordinary
per-turn hooks so a background-review fork's hook firings are deterministically
distinguishable from live turns. `post_llm_call` and `on_session_end` fire in
the fork just like `pre_llm_call`, so they must carry the same provenance.

`finalize_turn` is too large to unit-drive here, so this asserts the invariant
by source inspection (same technique as
tests/run_agent/test_pre_api_request_system_kwarg.py). It guards against a
future edit dropping the provenance kwargs back off either finalize-path hook.
"""

from __future__ import annotations

import ast
from pathlib import Path

import agent.turn_finalizer as turn_finalizer_module


def _invoke_hook_calls(tree):
    """Yield every ``_invoke_hook("<name>", ...)`` Call node with a str first arg."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if not (name and name.endswith("invoke_hook")):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            yield first.value, node


def _kwargs(call):
    return {kw.arg: kw.value for kw in call.keywords if kw.arg}


def _asserts_getattr(value, attr_name):
    """True if *value* is ``getattr(agent, "<attr_name>", ...)``."""
    return (
        isinstance(value, ast.Call)
        and getattr(value.func, "id", None) == "getattr"
        and len(value.args) >= 2
        and isinstance(value.args[1], ast.Constant)
        and value.args[1].value == attr_name
    )


def _load_tree():
    src = Path(turn_finalizer_module.__file__).read_text()
    return ast.parse(src)


def test_post_llm_call_carries_execution_provenance():
    tree = _load_tree()
    calls = [c for name, c in _invoke_hook_calls(tree) if name == "post_llm_call"]
    assert calls, "no invoke_hook('post_llm_call', ...) call found"
    for call in calls:
        kw = _kwargs(call)
        assert "execution_kind" in kw, "post_llm_call missing execution_kind"
        assert "execution_id" in kw, "post_llm_call missing execution_id"
        assert _asserts_getattr(kw["execution_kind"], "_execution_kind")
        assert _asserts_getattr(kw["execution_id"], "_execution_id")


def test_on_session_end_carries_execution_provenance():
    tree = _load_tree()
    calls = [c for name, c in _invoke_hook_calls(tree) if name == "on_session_end"]
    assert calls, "no invoke_hook('on_session_end', ...) call found"
    for call in calls:
        kw = _kwargs(call)
        assert "execution_kind" in kw, "on_session_end missing execution_kind"
        assert "execution_id" in kw, "on_session_end missing execution_id"
        assert _asserts_getattr(kw["execution_kind"], "_execution_kind")
        assert _asserts_getattr(kw["execution_id"], "_execution_id")

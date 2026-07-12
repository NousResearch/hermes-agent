"""``pre_api_request`` must expose the raw Anthropic ``system`` field.

On real (non-debug) requests ``request["body"]`` is sanitised to a truncated
stub, so an observer plugin cannot recover system/model/max_tokens from it. The
hook already passes ``request_messages`` as a raw, un-sanitised passthrough;
this pins that ``request_system=api_kwargs.get("system")`` is passed the same
way, at the same call site.

``run_conversation`` cannot be driven end-to-end in a unit test (it needs a live
adapter + stream cycle), so this asserts the invariant structurally over the
source AST — the same technique used by
``test_run_conversation_dict_returns_include_final_response`` in this file's
neighbour. Runtime behaviour is covered by the plugin that consumes the hook.
"""

from __future__ import annotations

import ast
import inspect

import pytest


def _pre_api_request_call(tree: ast.AST) -> ast.Call:
    """Return the ``invoke_hook("pre_api_request", ...)`` Call node."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # func is some ...invoke_hook (attribute or bare name)
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if not (name and name.endswith("invoke_hook")):
            continue
        if node.args and isinstance(node.args[0], ast.Constant) \
                and node.args[0].value == "pre_api_request":
            return node
    raise AssertionError("no invoke_hook('pre_api_request', ...) call found")


def test_pre_api_request_passes_raw_request_system():
    from agent import conversation_loop

    try:
        source = inspect.getsource(conversation_loop.run_conversation)
    except OSError as exc:  # pragma: no cover - source not shipped
        pytest.skip(f"run_conversation source is unavailable: {exc}")

    tree = ast.parse(source)
    call = _pre_api_request_call(tree)

    kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg}

    # The kwarg exists on the same call as the existing request_messages passthrough.
    assert "request_messages" in kwargs, "sanity: request_messages passthrough present"
    assert "request_system" in kwargs, (
        "pre_api_request must pass request_system so observers can recover the "
        "system prompt that request['body'] sanitisation strips"
    )

    # It is the RAW value api_kwargs.get("system") — not a sanitised/derived form.
    value = kwargs["request_system"]
    assert isinstance(value, ast.Call), "request_system must be api_kwargs.get('system')"
    assert isinstance(value.func, ast.Attribute) and value.func.attr == "get"
    assert isinstance(value.func.value, ast.Name) and value.func.value.id == "api_kwargs"
    assert len(value.args) == 1
    assert isinstance(value.args[0], ast.Constant) and value.args[0].value == "system"

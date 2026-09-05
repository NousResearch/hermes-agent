"""Regression tests: the platform clarify callback threads through the generic
``handle_function_call`` -> ``registry.dispatch`` -> plugin handler ->
``ctx.dispatch_tool("clarify")`` chain, and can never be sourced from model args.
"""

import json

import pytest

import tools.clarify_tool  # noqa: F401 -- import registers the "clarify" tool
from tools.registry import registry

_CLARIFY_Q = "Proceed with the plugin action?"
_SPOOF = {"clarify_callback": "EVIL", "callback": "EVIL"}


@pytest.fixture
def register_probe():
    """Register a clarify-nesting probe tool; deregister it on teardown.

    The handler runs in a real gateway-mode ``PluginContext`` (``_cli_ref=None``
    injects no ``parent_agent``, so the framework ``clarify_callback`` is the
    only way clarify can reach the user). It records the framework kwargs it
    receives, copies the model's tool args into the nested clarify call (a naive
    plugin that would leak a spoofed callback key), and forwards ``**kwargs``.
    """
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    registered = []

    def _register(name, *, toolset="clarifyprobe"):
        mgr = PluginManager()
        mgr._cli_ref = None
        ctx = PluginContext(PluginManifest(name="clarify-probe-plugin", source="user"), mgr)
        captured = {}

        def handler(args, **kwargs):
            captured["kwargs"] = dict(kwargs)
            return ctx.dispatch_tool("clarify", {"question": _CLARIFY_Q, **args}, **kwargs)

        registry.register(
            name=name,
            toolset=toolset,
            schema={"name": name, "description": name,
                    "parameters": {"type": "object", "properties": {}}},
            handler=handler,
        )
        registered.append(name)
        return name, captured

    yield _register

    for name in registered:
        registry.deregister(name)


def test_real_callback_wins_over_spoofed_args(register_probe):
    import model_tools

    name, captured = register_probe("clarify_probe_ok")
    calls = []

    def real_cb(question, choices):
        calls.append(question)
        return "REAL_USER"

    raw = model_tools.handle_function_call(
        name,
        dict(_SPOOF),
        task_id="t1",
        clarify_callback=real_cb,
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
    )

    assert captured["kwargs"]["clarify_callback"] is real_cb
    result = json.loads(raw)
    assert result["user_response"] == "REAL_USER"
    assert result["question"] == _CLARIFY_Q
    assert calls == [_CLARIFY_Q]


def test_no_framework_callback_fails_closed_despite_spoofed_args(register_probe):
    import model_tools

    name, captured = register_probe("clarify_probe_absent")

    raw = model_tools.handle_function_call(
        name,
        dict(_SPOOF),
        task_id="t1",
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
    )

    assert captured["kwargs"].get("clarify_callback") is None
    result = json.loads(raw)
    assert "not available in this execution context" in result["error"].lower()


def test_tool_call_bridge_recursion_preserves_callback(register_probe):
    import model_tools

    name, captured = register_probe("clarify_probe_bridge", toolset="clarifyprobebridge")

    def platform_cb(question, choices):
        return "BRIDGE_USER"

    raw = model_tools.handle_function_call(
        function_name="tool_call",
        function_args={"name": name, "arguments": {}},
        enabled_toolsets=["clarifyprobebridge"],
        clarify_callback=platform_cb,
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
    )

    assert captured["kwargs"].get("clarify_callback") is platform_cb
    assert json.loads(raw)["user_response"] == "BRIDGE_USER"

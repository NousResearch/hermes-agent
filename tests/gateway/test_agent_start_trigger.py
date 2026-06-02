"""The MAIN-dispatch ``agent:start`` payload must carry the turn discriminator.

``agent:start`` is emitted at two sites in ``gateway/run.py``: the MAIN inbound
dispatch (a fresh user message) and the interrupt/drain follow-up path.  To let
hooks tell the two apart, each payload carries ``trigger`` (a string, kept open
for future turn kinds like ``"goal"``/``"schedule"``) and ``interrupt_depth``
(an int).

The drain emit is exercised end-to-end in ``test_drain_emits_agent_start.py``,
which drives the real ``_run_agent`` drain path and asserts
``trigger="interrupt"`` with the live ``_interrupt_depth + 1`` value.  The MAIN
emit, by contrast, sits ~630 lines deep inside ``_handle_message_with_agent``,
behind session-store, DB and env I/O that make the method impractical to drive
in isolation.  Rather than mock that whole world, this test statically inspects
the actual dict literal the production code hands to ``hooks.emit("agent:start",
...)`` and pins ``trigger="message"``/``interrupt_depth=0`` — a fresh inbound
turn is never an interrupt.  It asserts against the real construction, not a
copy of it.
"""

import ast
import inspect

import gateway.run


def _module_tree():
    return ast.parse(inspect.getsource(gateway.run))


def _find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    return None


def _is_emit_agent_start(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "emit"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "agent:start"
    )


def _agent_start_payload_dict(func):
    """Return the ``ast.Dict`` literal emitted as the agent:start payload.

    Handles both an inline dict argument and a payload passed by name (the main
    path builds ``hook_ctx = {...}`` then emits ``hook_ctx``); for the latter we
    resolve the last in-function assignment of that name to a dict literal.
    """
    payload = None
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and _is_emit_agent_start(node) and len(node.args) >= 2:
            payload = node.args[1]
            break
    if isinstance(payload, ast.Dict):
        return payload
    if isinstance(payload, ast.Name):
        resolved = None
        for node in ast.walk(func):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                if any(isinstance(t, ast.Name) and t.id == payload.id for t in node.targets):
                    resolved = node.value
        return resolved
    return None


def _string_keys(dict_node):
    return {k.value for k in dict_node.keys if isinstance(k, ast.Constant)}


def _const_items(dict_node):
    items = {}
    for key, value in zip(dict_node.keys, dict_node.values):
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
            items[key.value] = value.value
    return items


def test_main_dispatch_agent_start_payload_is_message_trigger_depth0():
    func = _find_function(_module_tree(), "_handle_message_with_agent")
    assert func is not None, "could not locate _handle_message_with_agent"

    payload = _agent_start_payload_dict(func)
    assert payload is not None, "could not locate the main-dispatch agent:start payload dict"

    keys = _string_keys(payload)
    assert {
        "platform",
        "user_id",
        "chat_id",
        "session_id",
        "message",
        "trigger",
        "interrupt_depth",
    } <= keys, f"main agent:start payload is missing discriminator keys; has {sorted(map(str, keys))}"

    consts = _const_items(payload)
    assert consts.get("trigger") == "message"
    assert consts.get("interrupt_depth") == 0

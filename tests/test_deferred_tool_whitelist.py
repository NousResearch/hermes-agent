"""Resolved tool authority is independent of the pre-hook single-fire flag.

The worker installs policy explicitly: context propagation is separate work in
PRs #136632 / #119747, not something these dispatch regressions claim to fix.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest


@pytest.fixture
def dispatch_probe(monkeypatch, hook_action):
    import model_tools
    from hermes_cli import lifecycle, plugins
    from tools.registry import registry

    name = "mcp_whitelist_probe_read"
    toolset = "mcp-whitelist-probe"
    ran, hooks, approvals = [], [], []
    schema = {
        "name": name, "description": "Side-effect-free dispatch probe.",
        "parameters": {"type": "object", "properties": {"value": {"type": "string"}},
                       "required": ["value"]},
    }

    def handler(args, **kwargs):
        ran.append(dict(args))
        return json.dumps({"ok": args["value"]})

    def hook(event, **kwargs):
        if event == "pre_tool_call":
            hooks.append(kwargs["tool_name"])
            if kwargs["tool_name"] != "tool_call" and hook_action != "modify":
                return [{"action": hook_action, "message": "plugin decision"}]
            return [{"action": "modify", "args": {"value": "modified"}}]
        return []

    def approve(tool_name, message, **kwargs):
        approvals.append(tool_name)
        return {"approved": True}

    monkeypatch.setattr("tools.approval.request_tool_approval", approve)
    registry.register(name=name, toolset=toolset, schema=schema, handler=handler)
    monkeypatch.setattr(lifecycle, "invoke_hook", hook)
    monkeypatch.setattr(lifecycle, "has_hook", lambda event: False)
    try:
        yield model_tools, plugins, name, toolset, ran, hooks, approvals
    finally:
        plugins.clear_thread_tool_whitelist()
        registry.deregister(name)


@pytest.mark.parametrize("policy,in_scope,arguments,expected", [
    ("denied", True, {"value": "original"}, "denied"),
    ("empty", True, {"value": "original"}, "denied"),
    ("allowed", True, {"value": "original"}, "ok"),
    ("unrestricted", True, {"value": "original"}, "ok"),
    ("allowed", False, {"value": "original"}, "scope"),
    ("allowed", True, {}, "schema"),
])
@pytest.mark.parametrize("prechecked,hook_action", [
    (True, "modify"), (False, "modify"), (False, "block"), (False, "approve"),
], ids=["executor-prechecked", "rpc-modify", "rpc-block", "rpc-approve"])
def test_resolved_target_authority_and_single_hook(
    dispatch_probe, policy, in_scope, arguments, expected, prechecked, hook_action,
):
    model_tools, plugins, name, toolset, ran, hooks, approvals = dispatch_probe
    allowed = {"tool_call"} | ({name} if policy == "allowed" else set())
    if policy == "empty":
        allowed = set()
    if policy == "unrestricted":
        allowed = None

    def worker():
        plugins.set_thread_tool_whitelist(allowed, "denied {tool_name}")
        try:
            # The upstream executor may already have fired the wrapper hook.
            # Do not fire it again to authorize the resolved target.
            if prechecked and policy != "empty":
                block, _ = plugins._dispatch_pre_tool_call_hooks("tool_call", {})
                assert block is None
            return json.loads(model_tools.handle_function_call(
                "tool_call", {"name": name, "arguments": arguments},
                enabled_toolsets=[toolset] if in_scope else [],
                skip_pre_tool_call_hook=prechecked,
                session_id="whitelist-probe", tool_call_id="probe-call"))
        finally:
            plugins.clear_thread_tool_whitelist()

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(worker).result(timeout=20)
    if expected == "ok" and hook_action == "block":
        assert result["error"] == "plugin decision"
        assert not ran
    elif expected == "ok":
        value = "modified" if not prechecked and hook_action == "modify" else "original"
        assert result == {"ok": value}
        assert ran == [{"value": value}]
    else:
        assert not ran, f"denied target body executed: {result}"
        assert result.get("error")
        if expected == "denied":
            assert result["error"] == f"denied {name}"
        elif expected == "scope":
            assert "not available in this session" in result["error"]
        else:
            assert "value" in result["error"]
    expected_hooks = (["tool_call"] if prechecked and policy != "empty" else
                      [name] if not prechecked and expected == "ok" else [])
    assert hooks == expected_hooks
    assert approvals == ([name] if expected == "ok" and hook_action == "approve" else [])


@pytest.mark.parametrize("policy", ["restricted", "sentinel-only", "unrestricted"])
@pytest.mark.parametrize("prechecked", [True, False], ids=["executor-prechecked", "rpc-hooks"])
def test_connector_batch_checks_each_real_target(monkeypatch, policy, prechecked):
    import model_tools
    from hermes_cli import lifecycle, plugins
    from tools.connectors import CONNECTOR_BATCH_SENTINEL
    from tools.connectors.gateway import bridge, config
    from tools.registry import invalidate_check_fn_cache

    permitted, denied = "connectors__gmail__FETCH_EMAILS", "connectors__gmail__SEND_EMAIL"
    wire, hooks = [], []
    monkeypatch.setattr(config, "connectors_available", lambda: True)
    monkeypatch.setattr(bridge, "connectors_available", lambda: True)
    invalidate_check_fn_cache()

    class Client:
        def execute(self, planned):
            wire.extend(p.name for p in planned)
            return [{"data": "remote-ok", "error": None} for _ in planned]

    def hook(event, **kwargs):
        if event == "pre_tool_call":
            hooks.append(kwargs["tool_name"])
        return []

    monkeypatch.setattr(bridge, "_default_client_factory", lambda call=None: Client())
    monkeypatch.setattr(lifecycle, "invoke_hook", hook)
    monkeypatch.setattr(lifecycle, "has_hook", lambda event: False)
    allowed = {"tool_call", permitted} if policy == "restricted" else {"tool_call", CONNECTOR_BATCH_SENTINEL}
    plugins.set_thread_tool_whitelist(None if policy == "unrestricted" else allowed, "denied {tool_name}")
    try:
        result = json.loads(model_tools.handle_function_call(
            "tool_call", {"calls": [{"name": n, "arguments": {}} for n in (permitted, denied)]},
            enabled_toolsets=["connections"], session_id="connector-probe", tool_call_id="batch-probe",
            skip_pre_tool_call_hook=prechecked))
    finally:
        plugins.clear_thread_tool_whitelist()
    expected = [permitted, denied] if policy == "unrestricted" else [permitted] if policy == "restricted" else []
    assert wire == expected
    assert hooks == expected  # Exactly one hook per permitted entry, never for the sentinel.
    assert result["total_count"] == 2
    assert result["success_count"] == len(expected)
    for entry in result["results"]:
        if entry["name"] in expected:
            assert entry["response"] == "remote-ok"
        else:
            assert entry["error"]["message"] == f"denied {entry['name']}"

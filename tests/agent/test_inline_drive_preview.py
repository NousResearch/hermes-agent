"""Native drag must cross both dispatch paths without lossy argument repair."""
import json
from types import SimpleNamespace

import pytest

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from tools.drive_preview_tool import drive_preview_tool
from tools.registry import registry
from tui_gateway.contracts import SERVER_REQUESTS
from tui_gateway.contracts.registry import validate_params


VALID = {"action": "drag", "ref": "btn-handle", "dx": 40, "dy": -0.5}
INVALID = [
    {k: v for k, v in VALID.items() if k != missing} for missing in ("ref", "dx", "dy")
] + [
    {**VALID, **change} for change in [
        {"selector": "#handle"}, {"selector": None}, {"ref": ""}, {"ref": "  "},
        {"ref": None}, {"ref": 42}, {"dx": True}, {"dy": False}, {"dx": "40"},
        {"dx": " 40 "}, {"dy": None}, {"dx": float("nan")}, {"dy": float("inf")},
        {"dx": 2000.01}, {"dy": -2001}, {"dx": 0, "dy": 0}, {"submit": None},
        {"text": "x"}, {"key": "Enter"}, {"full": True}, {"max": 2}, {"unknown": 1},
        {"allow_shortcut": False}, {"amount": 1}, {"to": "top"}, {"action": "click"},
    ]
]


def dispatch(route, args, callback):
    if route == "inline":
        return INLINE_TOOL_EXECUTORS["drive_preview"](
            SimpleNamespace(drive_preview_callback=callback), args.copy(), InlineToolContext("test"))
    if route == "keyword":
        return drive_preview_tool(**args, callback=callback)
    return registry.dispatch("drive_preview", args.copy(), callback=callback)


@pytest.mark.parametrize("route", ["registry", "inline", "keyword"])
@pytest.mark.parametrize("args", INVALID)
def test_invalid_raw_input_never_reaches_callback(route, args):
    calls = []
    result = json.loads(dispatch(route, args, lambda p: calls.append(p) or '{"success": true}'))
    assert result.get("error")
    assert calls == []
    params, error = validate_params(SERVER_REQUESTS["preview.act"], {"session_id": "test", **args})
    assert params is None and error
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        SERVER_REQUESTS["preview.act"].params.model_validate({"session_id": "test", **args})


@pytest.mark.parametrize("route", ["registry", "inline", "keyword"])
@pytest.mark.parametrize("args", [
    VALID, {"action": "drag", "selector": "svg circle", "dx": -2000, "dy": 2000},
    {"action": "drag", "ref": "btn-handle", "dx": 0, "dy": 0.5},
    {"action": "elements", "full": True, "max": 7},
    {"action": "press", "selector": "body", "key": "a", "allow_shortcut": True},
])
def test_valid_payload_preserved_at_real_contract_boundary(route, args):
    calls = []
    def callback(payload):
        wire = {"session_id": "test", **payload}
        assert validate_params(SERVER_REQUESTS["preview.act"], wire) == (wire, None)
        calls.append(payload)
        return '{"success": true}'
    assert json.loads(dispatch(route, args, callback))["success"]
    assert calls == [args]


def test_model_dispatch_cannot_repair_invalid_delta_into_authority(monkeypatch):
    def must_not_coerce(*args):
        raise AssertionError("raw invalid drag must be rejected before coercion")
    monkeypatch.setattr("model_tools.coerce_tool_args", must_not_coerce)
    monkeypatch.setattr("agent.inline_tool_executors.coerce_tool_args", must_not_coerce)
    from model_tools import handle_function_call
    for delta in ("40", True, None):
        result = json.loads(handle_function_call("drive_preview", {**VALID, "dx": delta}))
        assert "finite numbers" in result["error"]
        result = json.loads(dispatch("inline", {**VALID, "dx": delta}, lambda args: pytest.fail("callback reached")))
        assert "finite numbers" in result["error"]


def test_annotate_and_unrelated_contracts_keep_their_existing_domains():
    wire = {"session_id": "test", "action": "pin", "selector": "#handle", "text": "label"}
    assert validate_params(SERVER_REQUESTS["preview.act"], wire) == (wire, None)
    wire = {"session_id": "test", "start": "legacy-handler-owned"}
    assert validate_params(SERVER_REQUESTS["preview.read"], wire) == (wire, None)

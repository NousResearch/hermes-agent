"""Regression tests for targeting a field before typing into it."""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def noop_backend():
    from tools.computer_use import tool as cu_tool

    cu_tool.reset_backend_for_tests()
    with patch.dict(os.environ, {"HERMES_COMPUTER_USE_BACKEND": "noop"}, clear=False):
        yield cu_tool._get_backend()
    cu_tool.reset_backend_for_tests()


def test_schema_exposes_type_delay_bounds():
    from tools.computer_use.schema import COMPUTER_USE_SCHEMA

    prop = COMPUTER_USE_SCHEMA["parameters"]["properties"]["delay_ms"]
    assert prop["minimum"] == 0
    assert prop["maximum"] == 200


def test_dispatch_forwards_element_coordinates_and_delay(noop_backend):
    from tools.computer_use.tool import handle_computer_use

    handle_computer_use({
        "action": "type",
        "text": "hello",
        "element": 4,
        "coordinate": [30, 40],
        "delay_ms": 12,
        "delivery_mode": "foreground",
    })
    first = next(c[1] for c in noop_backend.calls if c[0] == "type")
    assert first == {
        "text": "hello",
        "element": 4,
        "delay_ms": 12,
        "delivery_mode": "foreground",
        "bring_to_front": False,
    }

    noop_backend.calls.clear()
    handle_computer_use({"action": "type", "text": "world", "coordinate": [30, 40]})
    second = next(c[1] for c in noop_backend.calls if c[0] == "type")
    assert second == {
        "text": "world",
        "x": 30,
        "y": 40,
        "delivery_mode": None,
        "bring_to_front": False,
    }


def _cua_backend():
    from tools.computer_use.cua_backend import CuaDriverBackend

    session = MagicMock()
    session.supports_capability.return_value = True
    session.supports_input_property.return_value = True
    session.call_tool.return_value = {
        "data": {"message": "ok"},
        "structuredContent": {},
        "isError": False,
    }
    backend = CuaDriverBackend.__new__(CuaDriverBackend)
    backend._session = session
    backend._session_id = "hermes-test"
    backend._active_pid = 42
    backend._active_window_id = 99
    backend._snapshot_tokens = {3: "snapshot:3"}
    return backend


def test_cua_element_target_forwards_snapshot_token_and_delivery():
    backend = _cua_backend()

    result = backend.type_text(
        "hello", element=3, delay_ms=17, delivery_mode="foreground"
    )

    name, args = backend._session.call_tool.call_args.args
    assert result.ok is True
    assert name == "type_text"
    assert args["pid"] == 42
    assert args["window_id"] == 99
    assert args["element_index"] == 3
    assert args["element_token"] == "snapshot:3"
    assert args["delay_ms"] == 17
    assert args["delivery_mode"] == "foreground"


def test_cua_coordinate_target_is_used_when_element_is_absent():
    backend = _cua_backend()

    backend.type_text("hello", x=30, y=40, delay_ms=500)

    _, args = backend._session.call_tool.call_args.args
    assert args["x"] == 30
    assert args["y"] == 40
    assert args["delay_ms"] == 200
    assert "element_index" not in args

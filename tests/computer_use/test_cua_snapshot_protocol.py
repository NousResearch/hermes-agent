"""Structured snapshots and index-based Hermes actions survive token-only driver schemas."""

import pytest

from tools.computer_use.cua_backend import CuaDriverBackend
from tools.computer_use.cua_backend_session import _AsyncBridge, _CuaDriverSession


class SnapshotSession(_CuaDriverSession):
    def __init__(self, protocol):
        super().__init__(_AsyncBridge())
        self.protocol = protocol
        self.calls = []
        element_props = {"element_index": {}} if protocol == "index" else {"element_index": {}, "element_token": {}}
        if protocol == "token":
            element_props = {"element_token": {}}
        self._tool_schemas = {
            "get_window_state": {"properties": {"full_output": {}} if protocol == "token" else {}},
            **{name: {"properties": dict(element_props)} for name in ["click", "double_click", "scroll", "set_value"]},
        }
        self._capabilities = {name: set() for name in ["list_windows", *self._tool_schemas]}

    def call_tool(self, name, args, **kwargs):
        self.calls.append((name, dict(args)))
        sc = {}
        if name == "list_windows":
            sc = {"windows": [{"pid": 111, "window_id": 222, "app_name": "Fixture", "title": "Fixture",
                               "off_screen": False, "z_index": 1}]}
        elif name == "get_window_state":
            sc = {"tree_markdown": '[5] AXButton "Confirm"', "window_title": "Fixture"}
            if self.protocol != "token" or args.get("full_output") is True:
                sc["elements"] = [{"element_index": 5, "role": "AXButton", "label": "Confirm",
                                   "frame": {"x": -1450, "y": 48, "w": 80, "h": 24},
                                   **({"element_token": "s00000001:5"} if self.protocol != "index" else {})}]
        elif self.protocol == "token" and "element_index" in args:
            return {"data": "unknown argument element_index", "isError": True, "structuredContent": {}}
        return {"data": 'Fixture\n[5] AXButton "Confirm"', "images": [], "image_mime_types": [],
                "structuredContent": sc, "isError": False}


@pytest.mark.parametrize("protocol", ["index", "mixed", "token"])
def test_capture_preserves_structured_geometry_and_dispatches_advertised_element_address(protocol):
    backend = CuaDriverBackend()
    backend._session = SnapshotSession(protocol)
    capture = backend.capture(mode="ax", app="Fixture")
    assert capture.elements[0].bounds == (-1450, 48, 80, 24)
    assert (capture.elements[0].element_token is not None) == (protocol != "index")
    gws = next(args for name, args in backend._session.calls if name == "get_window_state")
    assert (gws.get("full_output") is True) == (protocol == "token")
    for action in [lambda: backend.click(element=5), lambda: backend.click(element=5, click_count=2),
                   lambda: backend.scroll(direction="down", element=5), lambda: backend.set_value("value", element=5)]:
        assert action().ok is True
        _, args = backend._session.calls[-1]
        assert ("element_index" in args) == (protocol != "token")
        assert args.get("element_token") == ("s00000001:5" if protocol != "index" else None)
    backend.click(x=31, y=42)
    _, args = backend._session.calls[-1]
    assert {key: args[key] for key in ["pid", "window_id", "x", "y"]} == {
        "pid": 111, "window_id": 222, "x": 31, "y": 42,
    }
    assert capture.window_title == "Fixture"


def test_token_only_actions_without_a_current_handle_fail_before_dispatch():
    backend = CuaDriverBackend()
    backend._session = SnapshotSession("token")
    backend._active_pid, backend._active_window_id = 111, 222
    backend._snapshot_tokens = {}
    for action in [lambda: backend.click(element=5), lambda: backend.scroll(direction="down", element=5),
                   lambda: backend.set_value("value", element=5)]:
        result = action()
        assert result.ok is False
        assert result.code == "element_token_unavailable"
    assert backend._session.calls == []

"""Regression tests for whole-desktop capture routing."""

from types import SimpleNamespace

import pytest

from tools.computer_use.cua_backend import CuaDriverBackend


_ONE_PIXEL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+AcAAQUBAScY42YAAAAASUVORK5CYII="
)


def _backend():
    backend = object.__new__(CuaDriverBackend)
    backend._session_id = "test-session"
    backend._session = SimpleNamespace(
        _has_tool=lambda name: False,
        capabilities_discovered=True,
    )
    backend._active_pid = None
    backend._active_window_id = None
    backend._last_app = None
    backend._last_target = None
    backend._snapshot_tokens = {}
    return backend


def test_windows_whole_screen_uses_scoped_desktop_capture_without_uia(monkeypatch):
    backend = _backend()
    capture_calls = []
    lifecycle_calls = []
    backend._session.call_tool = lambda name, args: (
        lifecycle_calls.append((name, args)) or {"isError": False}
    )

    def fake_capture_tool(name, args):
        capture_calls.append((name, args))
        if name == "start_session":
            return {"isError": False}
        assert name == "get_desktop_state"
        return {
            "images": [],
            "structuredContent": {
                "screenshot_png_b64": _ONE_PIXEL_PNG,
                "screenshot_mime_type": "image/png",
            },
            "isError": False,
        }

    monkeypatch.setattr("tools.computer_use.cua_backend.sys.platform", "win32")
    monkeypatch.setattr(backend, "_call_capture_tool", fake_capture_tool)
    monkeypatch.setattr(
        backend,
        "_load_windows",
        lambda: (_ for _ in ()).throw(AssertionError("desktop capture must not enumerate UIA windows")),
    )

    result = backend.capture(mode="som", app="screen")

    assert capture_calls[0][0] == "start_session"
    desktop_session = capture_calls[0][1]["session"]
    assert desktop_session.startswith("test-session-desktop-")
    assert capture_calls[0][1]["capture_scope"] == "desktop"
    assert capture_calls[1] == (
        "get_desktop_state",
        {"session": desktop_session},
    )
    assert lifecycle_calls == [
        ("end_session", {"session": desktop_session}),
        ("start_session", {"session": "test-session"}),
    ]
    assert result.mode == "vision"
    assert result.app == "screen"
    assert result.window_title == "Desktop"
    assert result.png_b64 == _ONE_PIXEL_PNG
    assert result.elements == []
    assert result.width == 1
    assert result.height == 1


def test_windows_whole_screen_restores_primary_session_after_capture_failure(monkeypatch):
    backend = _backend()
    lifecycle_calls = []
    backend._session.call_tool = lambda name, args: (
        lifecycle_calls.append((name, args)) or {"isError": False}
    )

    def fake_capture_tool(name, args):
        if name == "start_session":
            return {"isError": False}
        raise RuntimeError("desktop capture failed")

    monkeypatch.setattr("tools.computer_use.cua_backend.sys.platform", "win32")
    monkeypatch.setattr(backend, "_call_capture_tool", fake_capture_tool)

    with pytest.raises(RuntimeError, match="desktop capture failed"):
        backend.capture(mode="vision", app="screen")

    desktop_session = lifecycle_calls[0][1]["session"]
    assert lifecycle_calls == [
        ("end_session", {"session": desktop_session}),
        ("start_session", {"session": "test-session"}),
    ]


def test_application_capture_keeps_window_path(monkeypatch):
    backend = _backend()
    monkeypatch.setattr("tools.computer_use.cua_backend.sys.platform", "win32")
    monkeypatch.setattr(
        backend,
        "_load_windows",
        lambda: [{
            "app_name": "Hermes",
            "title": "Hermes",
            "pid": 123,
            "window_id": 456,
            "off_screen": False,
            "z_index": 10,
        }],
    )
    monkeypatch.setattr(
        backend,
        "_call_capture_tool",
        lambda name, args: {
            "images": [_ONE_PIXEL_PNG],
            "image_mime_types": ["image/png"],
            "data": "AXWindow \"Hermes\"",
            "structuredContent": {"elements": []},
            "isError": False,
        },
    )

    result = backend.capture(mode="vision", app="Hermes")

    assert result.app == "Hermes"
    assert result.png_b64 == _ONE_PIXEL_PNG
    assert result.width == 1
    assert result.height == 1

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hermes_runtime.computer_use import ActionKind, ComputerUseCommandCenter, NativeBackend


@pytest.fixture()
def native_backend(monkeypatch):
    backend = NativeBackend()
    monkeypatch.setattr(backend, "max_x", 1920)
    monkeypatch.setattr(backend, "max_y", 1080)
    user32_mock = MagicMock()
    monkeypatch.setattr("hermes_runtime.computer_use.user32", user32_mock)
    monkeypatch.setattr("hermes_runtime.computer_use.kernel32", MagicMock())
    user32_mock.FindWindowW.side_effect = lambda *a, **k: 123 if a[-1] == "vlc" else 0
    user32_mock.SendInput.return_value = 1
    return backend, user32_mock


def test_move_normalizes(native_backend):
    backend, user32_mock = native_backend
    backend.max_x = 10
    backend.max_y = 10
    result = backend.move(15, 9, duration=0.0)
    assert result.ok is True
    assert result.action == "move"
    assert user32_mock.SetCursorPos.called


def test_click_sends_input(native_backend):
    backend, user32_mock = native_backend
    assert backend.click(button="left", count=2).ok is True
    assert user32_mock.SendInput.call_count >= 3


def test_right_click_and_scroll(native_backend):
    backend, user32_mock = native_backend
    assert backend.click(button="right").ok is True
    assert backend.scroll(1).ok is True


def test_type_and_press(native_backend):
    backend, user32_mock = native_backend
    assert backend.type_text("ab").ok is True
    assert backend.press("enter").ok is True


def test_hotkey_sequence(native_backend):
    backend, user32_mock = native_backend
    assert backend.hotkey(("ctrl", "c")).ok is True


def test_dispatch_parse():
    center = ComputerUseCommandCenter(dry_run=True)
    move = center.dispatch("computer://move 100,200")
    assert move["ok"] is True
    assert move["action"] == "move"
    click = center.dispatch("computer://click left,100,200")
    assert click["action"] == "click"
    assert click["surface"]["button"] == "left"
    invalid = center.dispatch("computer://unknown")
    assert invalid["ok"] is False
    assert invalid["rc"] == 2


def test_execute_focus_success(native_backend):
    backend, user32_mock = native_backend
    center = ComputerUseCommandCenter(backend=backend, dry_run=False)
    result = center.execute(ActionKind.FOCUS, {"title": "vlc"})
    assert result.ok is True
    assert result.surface["action"] == "focus"


def test_execute_type_uses_payload_text():
    center = ComputerUseCommandCenter(dry_run=True)
    result = center.execute(ActionKind.TYPE, {"text": "hello"})
    assert result.ok is True
    assert result.stdout == "{'text': 'hello'}"

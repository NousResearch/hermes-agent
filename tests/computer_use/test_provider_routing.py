"""Tests for the computer-use provider registry + tool.py routing hook.

PR: ``computer_use: pluggable per-task backend provider``.

The provider ABC + registry mirror the browser provider surface (PR #25214).
These tests pin the contract the tool.py dispatcher relies on:

* ``check_computer_use_requirements()`` returns True when an active provider
  is available, even on a headless host (no DISPLAY) — the provider supplies
  its own per-task displays.
* ``_get_backend(task_id)`` delegates to ``provider.get_backend(task_id)``
  when a provider is active and task_id is supplied; different task_ids get
  independent backends; the provider owns start().
* ``reset_backend_for_tests()`` clears the provider cache so tests don't
  leak the previous test's provider.
* Legacy mode (no provider configured) falls back to the host singleton.
"""

from __future__ import annotations

import os
from typing import Dict
from unittest.mock import patch

import pytest

from agent.computer_use_provider import ComputerUseProvider
from agent.computer_use_registry import _reset_for_tests, register_provider
from tools.computer_use.backend import ComputerUseBackend


class _StubBackend(ComputerUseBackend):
    """Minimal started-backend stub recording the task_id it was minted for."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:  # pragma: no cover
        self.started = False

    def is_available(self) -> bool:
        return True

    def capture(self, mode="som", app=None):
        from tools.computer_use.backend import CaptureResult
        return CaptureResult(mode=mode, width=1024, height=768, png_b64=None,
                             elements=[], app=app or "", window_title="")

    def click(self, **kw):
        from tools.computer_use.backend import ActionResult
        return ActionResult(ok=True, action="click")

    def drag(self, **kw):
        from tools.computer_use.backend import ActionResult
        return ActionResult(ok=True, action="drag")

    def scroll(self, **kw):
        from tools.computer_use.backend import ActionResult
        return ActionResult(ok=True, action="scroll")

    def type_text(self, text):
        from tools.computer_use.backend import ActionResult
        return ActionResult(ok=True, action="type")

    def key(self, keys):
        from tools.computer_use.backend import ActionResult
        return ActionResult(ok=True, action="key")

    def list_apps(self):
        return []

    def focus_app(self, app, raise_window=False):
        from tools.computer_use.backend import ActionResult
        return ActionResult(ok=True, action="focus_app")

    def set_value(self, value, element=None):
        from tools.computer_use.backend import ActionResult
        return ActionResult(ok=True, action="set_value")

    def wait(self, seconds):
        from tools.computer_use.backend import ActionResult
        return ActionResult(ok=True, action="wait")


class _StubProvider(ComputerUseProvider):
    """Records every get_backend call + caches a started backend per task_id.

    Mirrors the real per-task contract: the same task_id returns the same
    backend instance across calls (stable per-task binding), different
    task_ids get independent backends. The provider owns start().
    """

    name = "stub-cu"
    available = True
    get_calls: list = []
    _backends: dict = {}

    def is_available(self) -> bool:
        return self.available

    def get_backend(self, task_id: str) -> ComputerUseBackend:
        self.get_calls.append(task_id)
        b = self._backends.get(task_id)
        if b is None:
            b = _StubBackend(task_id)
            b.start()
            self._backends[task_id] = b
        return b

    def close_backend(self, task_id: str) -> bool:
        return True

    def emergency_cleanup(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _isolated_registry():
    _reset_for_tests()
    from tools.computer_use import tool as cu_tool
    cu_tool.reset_backend_for_tests()
    _StubProvider.get_calls = []
    _StubProvider._backends = {}
    _StubProvider.available = True
    yield
    _reset_for_tests()
    cu_tool.reset_backend_for_tests()


def _cfg(provider_value):
    """Patch load_config to return a computer_use.provider setting."""
    cfg: Dict = {"computer_use": {"provider": provider_value}} if provider_value is not None else {}
    return patch("hermes_cli.config.load_config", return_value=cfg)


def test_gate_true_on_headless_linux_when_provider_available():
    # No DISPLAY, no host binary — but an active available provider supplies
    # its own displays, so the tool is surfaced.
    register_provider(_StubProvider())
    from tools.computer_use import tool as cu_tool
    with patch("tools.computer_use.tool.sys.platform", "linux"), \
         patch("tools.computer_use.cua_backend.cua_driver_binary_available", return_value=False), \
         patch.dict(os.environ, {"DISPLAY": ""}, clear=False), \
         _cfg("stub-cu"):
        assert cu_tool.check_computer_use_requirements() is True


def test_gate_false_when_provider_configured_but_unavailable():
    register_provider(_StubProvider())
    _StubProvider.available = False
    from tools.computer_use import tool as cu_tool
    with patch("tools.computer_use.tool.sys.platform", "linux"), \
         patch.dict(os.environ, {"DISPLAY": ""}, clear=False), \
         _cfg("stub-cu"):
        # Unavailable provider on a headless host → tool hidden (no thrash).
        assert cu_tool.check_computer_use_requirements() is False


def test_get_backend_routes_per_task_to_provider():
    register_provider(_StubProvider())
    from tools.computer_use import tool as cu_tool
    with _cfg("stub-cu"):
        b1 = cu_tool._get_backend("task-A")
        b2 = cu_tool._get_backend("task-B")
        b1_again = cu_tool._get_backend("task-A")
    assert isinstance(b1, _StubBackend) and b1.started
    assert isinstance(b2, _StubBackend) and b2.started
    assert b1.task_id == "task-A"
    assert b2.task_id == "task-B"
    assert b1 is b1_again  # provider returns same instance per task_id
    assert _StubProvider.get_calls == ["task-A", "task-B", "task-A"]


def test_get_backend_without_task_id_uses_legacy_singleton():
    # No task_id → never consults the provider, even if one is registered.
    register_provider(_StubProvider())
    from tools.computer_use import tool as cu_tool
    with patch.dict(os.environ, {"HERMES_COMPUTER_USE_BACKEND": "noop"}, clear=False), \
         _cfg("stub-cu"):
        b = cu_tool._get_backend(None)
    from tools.computer_use.tool import _NoopBackend
    assert isinstance(b, _NoopBackend)
    assert _StubProvider.get_calls == []


def test_legacy_provider_value_falls_back_to_singleton():
    # "local" / "cua" / unset are legacy sentinels → no provider, singleton path.
    from tools.computer_use import tool as cu_tool
    for legacy in ("local", "cua", "cua-driver", None):
        _reset_for_tests()
        cu_tool.reset_backend_for_tests()
        with patch.dict(os.environ, {"HERMES_COMPUTER_USE_BACKEND": "noop"}, clear=False), \
             _cfg(legacy):
            b = cu_tool._get_backend("task-X")
        from tools.computer_use.tool import _NoopBackend
        assert isinstance(b, _NoopBackend), f"legacy={legacy!r} should use singleton"


def test_reset_clears_provider_cache():
    register_provider(_StubProvider())
    from tools.computer_use import tool as cu_tool
    with _cfg("stub-cu"):
        cu_tool._get_active_cu_provider()
    assert cu_tool._cu_provider_cache  # cached
    cu_tool.reset_backend_for_tests()
    assert not cu_tool._cu_provider_cache  # cleared

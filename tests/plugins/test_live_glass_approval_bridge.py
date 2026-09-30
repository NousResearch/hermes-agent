"""Tests for the live-glass approval bridge (AVA-18)."""
from __future__ import annotations

import json
import sys
import importlib

import pytest


def _fresh_bridge():
    sys.modules.pop("plugins.observability.live_glass.approval_bridge", None)
    return importlib.import_module(
        "plugins.observability.live_glass.approval_bridge"
    )


def _fresh_cu_tool():
    """Re-import computer_use tool module with a clean _approval_callback."""
    import tools.computer_use.tool as cu

    cu._approval_callback = None
    cu._session_auto_approve = False
    cu._always_allow = set()
    return cu


class TestApprovalBridge:
    def test_wrapped_callback_fires_hooks_and_preserves_verdict(self):
        """The wrapper must fire pre/post hooks and return the original verdict."""
        bridge = _fresh_bridge()
        cu = _fresh_cu_tool()

        # Track what the original callback received and returned.
        captured = {}

        def original_callback(action, args, summary):
            captured["action"] = action
            captured["args"] = args
            captured["summary"] = summary
            return "approve_once"

        # Wrap the callback.
        wrapped = bridge.wrap_approval_callback(original_callback)

        # Call the wrapped callback (simulating _request_approval).
        verdict = wrapped("click", {"element": 5}, "click element #5")

        assert verdict == "approve_once"
        assert captured["action"] == "click"
        assert captured["args"] == {"element": 5}
        assert captured["summary"] == "click element #5"

    def test_wrapped_callback_handles_deny_verdict(self):
        bridge = _fresh_bridge()
        original = lambda action, args, summary: "deny"
        wrapped = bridge.wrap_approval_callback(original)
        verdict = wrapped("key", {"keys": "cmd+q"}, "key 'cmd+q'")
        assert verdict == "deny"

    def test_wrapped_callback_handles_all_verdicts(self, monkeypatch):
        bridge = _fresh_bridge()
        for verdict in ("approve_once", "approve_session", "always_approve", "deny"):
            original = lambda *a, v=verdict: v
            wrapped = bridge.wrap_approval_callback(original)
            assert wrapped("click", {}, "click") == verdict

    def test_original_callback_exception_is_caught_and_defaults_to_deny(self):
        bridge = _fresh_bridge()

        def broken(action, args, summary):
            raise RuntimeError("callback crash")

        wrapped = bridge.wrap_approval_callback(broken)
        verdict = wrapped("click", {}, "click")
        assert verdict == "deny"

    def test_wrap_approval_callback_returns_original_if_none(self):
        bridge = _fresh_bridge()
        assert bridge.wrap_approval_callback(None) is None

    def test_install_bridge_installs_wrapped_callback(self, monkeypatch):
        bridge = _fresh_bridge()
        cu = _fresh_cu_tool()

        original = lambda action, args, summary: "approve_once"
        cu.set_approval_callback(original)

        bridge.install_bridge(callback_getter=lambda: cu._approval_callback,
                              callback_setter=cu.set_approval_callback)

        # The stored callback should now be a wrapper.
        wrapped = cu._approval_callback
        assert wrapped is not None
        assert wrapped is not original  # It's a different object
        verdict = wrapped("click", {"element": 1}, "click element #1")
        assert verdict == "approve_once"

    def test_install_bridge_is_idempotent(self, monkeypatch):
        bridge = _fresh_bridge()
        cu = _fresh_cu_tool()

        original = lambda action, args, summary: "approve_once"
        cu.set_approval_callback(original)

        bridge.install_bridge(callback_getter=lambda: cu._approval_callback,
                              callback_setter=cu.set_approval_callback)
        first = cu._approval_callback

        bridge.install_bridge(callback_getter=lambda: cu._approval_callback,
                              callback_setter=cu.set_approval_callback)
        second = cu._approval_callback

        assert first is second  # Same wrapper, not double-wrapped

    def test_install_bridge_noops_when_no_callback_set(self):
        bridge = _fresh_bridge()
        cu = _fresh_cu_tool()
        # No callback set yet
        bridge.install_bridge(callback_getter=lambda: cu._approval_callback,
                              callback_setter=cu.set_approval_callback)
        assert cu._approval_callback is None  # No-op

    def test_register_hooks_via_plugin_context(self, monkeypatch):
        bridge = _fresh_bridge()
        cu = _fresh_cu_tool()

        original = lambda action, args, summary: "approve_once"
        cu.set_approval_callback(original)

        calls = []

        class Ctx:
            def register_hook(self, name, fn):
                calls.append((name, fn.__name__))

        bridge.register_approval_bridge(Ctx())
        assert calls == [
            ("pre_approval_request", "on_pre_approval_request"),
            ("post_approval_response", "on_post_approval_response"),
        ]

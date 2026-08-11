"""RED tests for the generic plugin command-context seam (REM-302/303).

The historical dispatch calls plugin command handlers as ``handler(raw_args)``
only, so a plugin cannot know which exact session invoked a slash command.
This pins the smallest backwards-compatible public seam:

- legacy one-argument handlers (``fn(raw_args)``) remain unchanged;
- an opt-in handler may accept keyword-only ``session_id``, ``platform`` and
  an opaque session target; the host passes them only when the handler
  accepts them (via a compatibility wrapper);
- no plugin code is special-cased in the core; the wrapper is generic.

The tests drive the real ``dispatch_plugin_command`` entry point (the public
seam added by this phase) with disposable fake plugin managers.
"""

from __future__ import annotations

import pytest


class LegacyHandler:
    """A legacy handler that takes only raw_args (unchanged contract)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, raw: str) -> str:
        self.calls.append(raw)
        return f"legacy:{raw}"


class OptInHandler:
    """An opt-in handler that accepts exact session context kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, raw: str, **kwargs) -> str:
        self.calls.append({"raw": raw, **kwargs})
        return f"opt:{raw}:{kwargs.get('session_id', '')}:{kwargs.get('platform', '')}"


def _make_manager(handlers: dict[str, object]) -> object:
    """A minimal plugin-manager stand-in exposing _plugin_commands."""

    class M:
        def __init__(self, h: dict[str, object]) -> None:
            self._plugin_commands = {
                name: {"handler": handler, "description": "", "plugin": "test", "args_hint": ""}
                for name, handler in h.items()
            }

    return M(handlers)


class TestLegacyCompatibility:
    def test_legacy_one_arg_handler_unchanged(self):
        """A handler with signature fn(raw_args) must be called exactly as
        before — one positional arg, no kwargs injected."""
        from hermes_cli.plugins import dispatch_plugin_command

        h = LegacyHandler()
        mgr = _make_manager({"peers": h})
        result = dispatch_plugin_command(mgr, "peers", "raw text", session_id="sess-1", platform="cli")
        assert result == "legacy:raw text"
        assert h.calls == ["raw text"]

    def test_legacy_handler_does_not_crash_on_kwargs_available(self):
        """Even when session context exists at the host, the legacy handler
        must not receive it (no unexpected-keyword failure)."""
        from hermes_cli.plugins import dispatch_plugin_command

        h = LegacyHandler()
        mgr = _make_manager({"peers": h})
        dispatch_plugin_command(mgr, "peers", "x", session_id="sess-9", platform="gateway")
        assert h.calls == ["x"]


class TestOptInContext:
    def test_opt_in_handler_receives_exact_session_context(self):
        """An opt-in handler with **kwargs receives session_id/platform."""
        from hermes_cli.plugins import dispatch_plugin_command

        h = OptInHandler()
        mgr = _make_manager({"peer-name": h})
        result = dispatch_plugin_command(mgr, "peer-name", "backend", session_id="sess-7", platform="cli")
        assert result == "opt:backend:sess-7:cli"
        assert h.calls == [{"raw": "backend", "session_id": "sess-7", "platform": "cli", "session_target": None}]

    def test_opt_in_handler_receives_platform_gateway(self):
        from hermes_cli.plugins import dispatch_plugin_command

        h = OptInHandler()
        mgr = _make_manager({"peers": h})
        dispatch_plugin_command(mgr, "peers", "", session_id="gw-1", platform="gateway")
        assert h.calls[0]["session_id"] == "gw-1"
        assert h.calls[0]["platform"] == "gateway"

    def test_unknown_command_returns_none(self):
        from hermes_cli.plugins import dispatch_plugin_command

        mgr = _make_manager({})
        assert dispatch_plugin_command(mgr, "nope", "") is None


class TestClISurface:
    def test_cli_dispatch_uses_exact_session(self, monkeypatch):
        """The CLI slash-command path must pass the CLI's session id."""
        from hermes_cli.plugins import dispatch_plugin_command

        h = OptInHandler()
        mgr = _make_manager({"peers": h})
        # Simulate the CLI invocation: process_command calls dispatch with
        # the CLI's current session_id.
        dispatch_plugin_command(mgr, "peers", "", session_id="cli-sess-abc", platform="cli")
        assert h.calls[0]["session_id"] == "cli-sess-abc"

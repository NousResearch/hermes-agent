"""Tests for hermes_cli/plugin_command_dispatch.py result resolution.

Moved verbatim from tests/hermes_cli/test_plugins.py (FILE_LINES ratchet) with
the monkeypatch targets updated to the new module: the resolver and its timeout
constant now live in hermes_cli.plugin_command_dispatch.
"""

import pytest

from hermes_cli.plugins import resolve_plugin_command_result


class TestPluginCommandResultResolution:

    def test_awaits_async_result_with_running_loop(self, monkeypatch):
        class _Loop:
            pass

        async def _handler():
            return "threaded-ok"

        monkeypatch.setattr("hermes_cli.plugin_command_dispatch.asyncio.get_running_loop", lambda: _Loop())
        assert resolve_plugin_command_result(_handler()) == "threaded-ok"

    def test_running_loop_timeout_does_not_hang_forever(self, monkeypatch):
        """Threaded path must abort a hung async handler instead of blocking the caller."""
        import asyncio as _asyncio

        class _Loop:
            pass

        async def _slow_handler():
            await _asyncio.sleep(10)
            return "should-not-reach"

        monkeypatch.setattr("hermes_cli.plugin_command_dispatch.asyncio.get_running_loop", lambda: _Loop())
        monkeypatch.setattr("hermes_cli.plugin_command_dispatch._PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS", 0.1)

        with pytest.raises(TimeoutError):
            resolve_plugin_command_result(_slow_handler())

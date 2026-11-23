"""A slow MCP connection must not hold up platform-handler rewiring."""

import asyncio
import contextlib
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from gateway.run_plugin_rewire import GatewayPluginRewireMixin, reload_plugins_verb


class SlowMcpRewireTest(unittest.IsolatedAsyncioTestCase):
    async def test_rewire_while_mcp_is_blocked_but_answer_waits(self):
        mcp_started = threading.Event()
        release_mcp = threading.Event()
        rewired = threading.Event()
        observed_events = []
        calls = []

        class Manager:
            scope_key = "secondary"
            _plugins = {"new": object()}

            def __init__(self):
                self.listeners = []

            def on_plugin_loaded(self, listener):
                self.listeners.append(listener)
                return lambda: self.listeners.remove(listener)

        manager = Manager()

        def discover_plugins(*, force):
            assert force
            for listener in manager.listeners:
                listener(["new"])

        def reconcile():
            mcp_started.set()
            if not release_mcp.wait(8):
                raise AssertionError("test did not release MCP")
            return {"removed": [], "added": ["new"], "pending": []}

        def module(name, **attrs):
            result = ModuleType(name)
            result.__dict__.update(attrs)
            return result

        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp) / "secondary"
            modules = {
                "hermes_constants": module("hermes_constants", get_hermes_home=lambda: home,
                                           hermes_home_key=lambda path: str(path)),
                "hermes_cli.plugins": module("hermes_cli.plugins", discover_plugins=discover_plugins,
                                             get_plugin_manager=lambda: manager),
                "hermes_cli.plugins_activation": module("hermes_cli.plugins_activation",
                    activation_summaries=lambda _: [], dependency_blocked_summaries=lambda _: []),
                "gateway.run": module("gateway.run", _profile_runtime_scope=lambda *a, **k: contextlib.nullcontext()),
                "gateway.run_turn": module("gateway.run_turn", GatewayTurnMixin=SimpleNamespace(
                    _standalone_launch_scope=lambda: contextlib.nullcontext())),
                "tools.mcp_oauth": module("tools.mcp_oauth", suppress_interactive_oauth=contextlib.nullcontext),
                "tools.mcp_tool_discovery": module("tools.mcp_tool_discovery", reconcile_mcp_servers_with_config=reconcile),
            }

            class Adapter:
                def rewire_plugin_handlers(self):
                    calls.append("secondary")
                    rewired.set()

            class OtherAdapter:
                def rewire_plugin_handlers(self):
                    calls.append("primary")

            class Runner(GatewayPluginRewireMixin):
                config = SimpleNamespace(multiplex_profiles=True)
                _served_profile_homes = {"secondary": home}
                adapters = {"platform": OtherAdapter()}
                _profile_adapters = {"secondary": {"platform": Adapter()}}

            with patch.dict(sys.modules, modules):
                runner = Runner()
                runner._subscribe_plugin_rewire(manager, "secondary", home)
                manager.on_plugin_loaded(observed_events.append)
                task = asyncio.create_task(asyncio.to_thread(
                    reload_plugins_verb(runner, asyncio.get_running_loop()), {"home": str(home)}))
                try:
                    self.assertTrue(await asyncio.to_thread(mcp_started.wait, 2), "MCP never started")
                    self.assertTrue(await asyncio.to_thread(rewired.wait, 2), "rewire waited for MCP")
                    self.assertFalse(task.done(), "response must wait for MCP")
                    self.assertEqual(calls, ["secondary"])
                    self.assertEqual(observed_events, [["new"]])
                finally:
                    release_mcp.set()
                    answer = await task
                self.assertTrue(answer["reloaded"])
                self.assertEqual(answer["adapters_rewired"], 1)
                self.assertEqual(calls, ["secondary"])


if __name__ == "__main__":
    unittest.main()

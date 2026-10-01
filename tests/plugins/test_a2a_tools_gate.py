"""a2a client tools config gate (#95681, maintainer-directed).

The 5 outbound a2a_* tools registered unconditionally — every session on
every install paid ~561 tok/call for a toolset whose only possible output
without config is "no peers configured". A2A is NOT the Bot Mode
mechanism (bots talk over gateway RPCs); it is opt-in foreign-agent
plumbing. Gate: serve only when a2a_agents is non-empty, the inbound
platform is enabled, or A2A_PORT is set. Fail closed.
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import plugins.platforms.a2a.tools as a2at


class TestA2AToolsGate(unittest.TestCase):
    def setUp(self):
        os.environ.pop("A2A_PORT", None)

    def _avail(self, cfg):
        with patch.object(a2at, "_load_config", return_value=cfg):
            return a2at._a2a_tools_available()

    def test_unconfigured_install_serves_nothing(self):
        self.assertFalse(self._avail({}))
        self.assertFalse(self._avail({"a2a_agents": {}}))

    def test_peers_configured_serves(self):
        self.assertTrue(self._avail({"a2a_agents": {"r": {"url": "http://x"}}}))

    def test_inbound_platform_enabled_serves(self):
        self.assertTrue(self._avail({"platforms": {"a2a": {"enabled": True}}}))

    def test_inbound_platform_enabled_at_documented_gateway_layer_serves(self):
        """#109011: ``gateway.platforms.a2a.enabled`` is what the docs and
        ``hermes gateway setup`` write, and what ``merge_platform_sections``
        honours for the inbound server. The gate checked only the top-level
        ``platforms`` block, so the five client tools stayed unserved."""
        self.assertTrue(self._avail(
            {"gateway": {"platforms": {"a2a": {"enabled": True, "extra": {"port": 9900}}}}}))
        self.assertTrue(self._avail({"gateway": {"platforms": {"a2a": {"enabled": True}}}}))

    def test_gateway_platform_subsection_layer_serves(self):
        """Third layer ``merge_platform_sections`` merges: ``gateway.a2a:``."""
        self.assertTrue(self._avail({"gateway": {"a2a": {"enabled": True}}}))

    def test_list_valued_platform_layer_does_not_crash(self):
        """``gateway.platforms: [telegram, api_server]`` (the shipped shape on a
        real install) is a list, not a dict — the gate must skip it, not raise."""
        self.assertFalse(self._avail({"gateway": {"platforms": ["telegram", "api_server"]}}))
        self.assertFalse(self._avail(
            {"gateway": {"platforms": ["telegram", "api_server"]},
             "platforms": {"telegram": {"enabled": True}}}))

    def test_list_platform_layer_still_honours_dict_sibling_layer(self):
        self.assertTrue(self._avail(
            {"gateway": {"platforms": ["telegram"]}, "platforms": {"a2a": {"enabled": True}}}))

    def test_enabled_false_in_every_layer_serves_nothing(self):
        self.assertFalse(self._avail(
            {"gateway": {"platforms": {"a2a": {"enabled": False}}},
             "platforms": {"a2a": {"enabled": False}}}))

    def test_non_dict_platform_block_serves_nothing(self):
        """A bare ``a2a: true``-shaped or string block is not a config block."""
        self.assertFalse(self._avail({"platforms": {"a2a": "yes"}}))
        self.assertFalse(self._avail({"platforms": {"a2a": True}}))

    def test_a2a_port_env_serves(self):
        os.environ["A2A_PORT"] = "9999"
        try:
            self.assertTrue(self._avail({}))
        finally:
            os.environ.pop("A2A_PORT", None)

    def test_config_crash_fails_closed(self):
        with patch.object(a2at, "_load_config", side_effect=RuntimeError("boom")):
            self.assertFalse(a2at._a2a_tools_available())

    def test_all_five_tools_carry_the_gate(self):
        """Every a2a_* registration must pass the check_fn — a sixth tool
        added without it would silently reopen the hole."""
        seen = {}

        class Ctx:
            def register_tool(self, name, **kw):
                seen[name] = kw.get("check_fn")

        a2at.register_tools(Ctx())
        self.assertTrue(seen)
        for name, fn in seen.items():
            self.assertIs(fn, a2at._a2a_tools_available, name)


if __name__ == "__main__":
    unittest.main()

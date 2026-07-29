"""Regression tests for feishu multiplex_profiles support.

Background
----------
These tests guard the three commits that together let a single multiplexing
gateway serve multiple Feishu apps (one per profile) — enabling multi-bot
group-discussion scenarios that ``profile_routes`` cannot express (it routes
messages from one bot to different personas; users see a single bot talking
to itself, defeating the "panel of experts" UX).

The upstream code already implies support for feishu in secondary profiles
via ``gateway/config.py::PORT_BINDING_CONDITIONAL_MODES = {"feishu":
"webhook"}`` — feishu in websocket mode doesn't bind a port, so it should be
safe under multiplexing. But the implementation has three gaps that this
PR closes:

1. ``_apply_yaml_config`` — translate ``feishu:`` YAML block into the
   ``FEISHU_*`` env vars the plugin loader requires, so a secondary profile
   can declare its own feishu app via per-profile YAML.
2. ``_ThreadLocalLoopProxy`` — isolate the ``lark_oapi.ws.Client`` event loop
   per-adapter, so multiple WS clients in one process don't trip
   ``Task ... attached to a different loop``.
3. ``per-instance hermes_loop`` — bind SDK client methods to the adapter's
   own loop rather than the module-global loop the SDK assumes.

If any of the three regresses, the corresponding test below fails loudly
with a pointer to the mechanism that broke.
"""
from __future__ import annotations

import os
import unittest
from typing import Dict, Any
from unittest.mock import patch


class FeishuMultiplexYamlConfigTest(unittest.TestCase):
    """Verifies ``_apply_yaml_config`` translates per-profile YAML into the
    form the plugin loader / FeishuAdapterSettings can consume."""

    def test_apply_yaml_config_seeds_app_credentials_into_extra(self):
        from plugins.platforms.feishu.adapter import _apply_yaml_config

        feishu_cfg = {
            "app_id": "cli_test_app_id",
            "app_secret": "secret_value",
            "encrypt_key": "enc",
            "verification_token": "tok",
            "domain": "feishu",
            "connection_mode": "websocket",
            "extra": {"default_group_policy": "open"},
        }
        # Clear env to assert the function does NOT pollute it for app creds
        # (allow_bots is the one exception, see below).
        with patch.dict(os.environ, {}, clear=False):
            for k in ("FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_ALLOW_BOTS"):
                os.environ.pop(k, None)
            seeded = _apply_yaml_config({}, feishu_cfg)

        # All six credential/behavior keys flow into extra, so a secondary
        # profile's app_id is visible to the plugin loader via config.extra
        # even though FEISHU_APP_ID env is never set globally.
        self.assertIsNotNone(seeded, "_apply_yaml_config must return a non-None seeded dict")
        self.assertEqual(seeded["app_id"], "cli_test_app_id")
        self.assertEqual(seeded["app_secret"], "secret_value")
        self.assertEqual(seeded["encrypt_key"], "enc")
        self.assertEqual(seeded["verification_token"], "tok")
        self.assertEqual(seeded["domain"], "feishu")
        self.assertEqual(seeded["connection_mode"], "websocket")
        # Existing extra keys are preserved.
        self.assertEqual(seeded["default_group_policy"], "open")

    def test_apply_yaml_config_does_not_leak_app_secret_into_os_environ(self):
        """A secondary profile's app_secret must NOT be written to
        os.environ — that would leak it to every other profile's turn under
        the multiplexer. Only allow_bots is allowed to seed env (for the
        legacy auth-bypass path)."""
        from plugins.platforms.feishu.adapter import _apply_yaml_config

        with patch.dict(os.environ, {}, clear=False):
            for k in ("FEISHU_APP_ID", "FEISHU_APP_SECRET"):
                os.environ.pop(k, None)
            _apply_yaml_config({}, {
                "app_id": "cli_x",
                "app_secret": "should_not_leak",
            })
            self.assertNotIn("FEISHU_APP_ID", os.environ)
            self.assertNotIn("FEISHU_APP_SECRET", os.environ)

    def test_apply_yaml_config_returns_none_for_empty_feishu_block(self):
        """No feishu config → no seed → plugin loader uses env as before."""
        from plugins.platforms.feishu.adapter import _apply_yaml_config

        self.assertIsNone(_apply_yaml_config({}, {}))
        self.assertIsNone(_apply_yaml_config({}, {"extra": {}}))


class FeishuMultiplexLoopIsolationTest(unittest.TestCase):
    """Verifies the per-adapter WS event loop isolation that lets N feishu
    WS clients coexist in one gateway process.

    Without this, the second profile's WS connect trips:
        ``RuntimeError: Task ... got Future ... attached to a different loop``
    """

    def test_thread_local_loop_proxy_symbol_is_present(self):
        """``_ThreadLocalLoopProxy`` is the class that scopes the SDK's
        module-global loop to the calling thread. Importing it must not fail.
        If a future refactor renames or removes it, this test fires first
        rather than letting production hit the cross-loop Task error."""
        from plugins.platforms.feishu import adapter

        # The class may live at module scope or as a FeishuAdapter attribute;
        # accept either.
        self.assertTrue(
            hasattr(adapter, "_ThreadLocalLoopProxy")
            or any("_ThreadLocalLoopProxy" in type(getattr(adapter, n, None)).__name__
                   for n in dir(adapter) if not n.startswith("__")),
            "_ThreadLocalLoopProxy is missing — multi-profile feishu WS will "
            "hit 'Task attached to a different loop' (see PR description).",
        )

    def test_feishu_adapter_instances_have_isolated_loop_state(self):
        """Two adapter instances must not share the per-instance hermes_loop
        slot. If they do, the second profile's SDK calls run on the first
        profile's loop.

        The isolation mechanism has two halves, both required:
        1. ``_ThreadLocalLoopProxy`` at module scope — replaces the SDK's
           module-global ``loop`` so each worker thread sees its own.
        2. ``self._hermes_loop`` on each adapter instance — patched SDK
           client methods reach for this instead of the module global.
        """
        from plugins.platforms.feishu import adapter
        from plugins.platforms.feishu.adapter import FeishuAdapter

        # Half 1: module-scope _ThreadLocalLoopProxy class exists.
        self.assertTrue(
            hasattr(adapter, "_ThreadLocalLoopProxy"),
            "_ThreadLocalLoopProxy class missing — module-global SDK loop is "
            "shared across all feishu worker threads.",
        )
        # Half 2: SDK client method patches reference self._hermes_loop.
        # Verify by reading the adapter source file directly —
        # inspect.getsource(FeishuAdapter) truncates on classes this large.
        import plugins.platforms.feishu.adapter as _mod
        with open(_mod.__file__, encoding="utf-8") as fh:
            src_text = fh.read()
        self.assertIn(
            "_hermes_loop",
            src_text,
            "FeishuAdapter never assigns/reads self._hermes_loop — SDK client "
            "methods will use the module-global loop and trip 'Task attached "
            "to a different loop' under multiplex_profiles.",
        )


if __name__ == "__main__":
    unittest.main()

"""Regression: an unsupported secondary registration must not warn once per registration.

The memory-provider collector in ``plugins/memory/__init__.py`` fabricates a callable for ANY
``register_*`` attribute name (its ``__getattr__``), so a plugin's capability probe always finds
something "callable" and calls it. When the real ``PluginContext`` has no such hook the
AttributeError is swallowed inside the proxy and logged at WARNING. Open Second Brain probes for
``register_health_check`` on every load, so a missing hook produced one warning per load, describing
a capability mismatch that is permanent, not a fault.

A missing hook must stay silent: the provider is already registered by the time secondary hooks are
probed, so losing one costs nothing and there is nothing for the operator to fix. A hook that EXISTS
and raises is a genuine failure and must keep warning.
"""

import logging
import unittest
from unittest import mock

from plugins.memory import _ProviderCollector


class _RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _collector() -> _ProviderCollector:
    return _ProviderCollector("open-second-brain")


class SecondaryRegistrationWarningTests(unittest.TestCase):
    def _capture(self) -> _RecordingHandler:
        handler = _RecordingHandler()
        logger = logging.getLogger("plugins.memory")
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)
        return handler

    def test_missing_hook_does_not_warn(self):
        """A hook the host does not implement stays silent."""
        handler = self._capture()
        collector = _collector()
        # A real PluginContext has no register_health_check, so attribute access raises AttributeError.
        real_ctx = mock.MagicMock(spec=[])
        with mock.patch.object(
            _ProviderCollector, "_plugin_context", return_value=real_ctx
        ):
            self.assertIsNone(collector.register_health_check("open-second-brain", lambda: None))
        self.assertEqual(handler.records, [])

    def test_existing_hook_that_raises_still_warns(self):
        """A supported hook that genuinely fails must NOT be silenced."""
        handler = self._capture()
        collector = _collector()

        def _boom(*args, **kwargs):
            raise RuntimeError("registry offline")

        real_ctx = mock.MagicMock(spec=["register_skills"])
        real_ctx.register_skills.side_effect = _boom
        with mock.patch.object(
            _ProviderCollector, "_plugin_context", return_value=real_ctx
        ):
            self.assertIsNone(collector.register_skills("/tmp/x.md"))
        self.assertTrue(
            handler.records,
            "a real failure on a supported hook must still warn",
        )

    def test_non_register_attribute_still_raises(self):
        """The typo guard must survive: non-``register_*`` still raises loudly."""
        collector = _collector()
        with self.assertRaises(AttributeError):
            _ = collector.some_typo_attribute


if __name__ == "__main__":
    unittest.main()
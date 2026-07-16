"""Oneshot (`hermes -z`) must declare itself unable to deliver async results.

A oneshot process exits right after printing its single response. A
``delegate_task background=true`` dispatched during that turn runs on a daemon
thread in the SAME process, so it dies mid-flight and its row in
``async_delegations`` is orphaned at ``state: running`` forever (observed live
as deleg_3725e69f, deleg_4d9013c4, deleg_9dd323d8, deleg_a9651f0a).

The fix binds the session async-delivery contextvar to False in
``run_oneshot`` — the same capability gate the stateless API server adapter
uses — so ``delegate_task`` downgrades background batches to synchronous
execution and ``terminal`` refuses notify_on_complete watchers.
"""

import logging
import unittest
from unittest.mock import patch

from gateway.session_context import (
    async_delivery_supported,
    reset_session_vars,
    set_async_delivery_supported,
)


class TestSetAsyncDeliverySupported(unittest.TestCase):
    """The standalone setter added for non-adapter callers (oneshot)."""

    def tearDown(self):
        reset_session_vars()

    def test_set_false_is_unsupported(self):
        set_async_delivery_supported(False)
        self.assertFalse(async_delivery_supported())

    def test_set_true_is_supported(self):
        set_async_delivery_supported(True)
        self.assertTrue(async_delivery_supported())

    def test_reset_returns_to_default_supported(self):
        set_async_delivery_supported(False)
        reset_session_vars()
        self.assertTrue(async_delivery_supported())


class TestRunOneshotBindsNoAsyncDelivery(unittest.TestCase):
    """run_oneshot must bind the capability BEFORE the agent turn starts."""

    def tearDown(self):
        reset_session_vars()
        # run_oneshot silences the root logger for the whole process; undo so
        # later tests keep their logging behavior.
        logging.disable(logging.NOTSET)

    def test_agent_turn_sees_async_delivery_unsupported(self):
        seen = {}

        def _fake_run_agent(prompt, **kwargs):
            seen["async_ok"] = async_delivery_supported()
            return "ok", {}

        from hermes_cli import oneshot

        with patch.object(oneshot, "_run_agent", side_effect=_fake_run_agent):
            exit_code = oneshot.run_oneshot("hi")

        self.assertEqual(exit_code, 0)
        self.assertIn("async_ok", seen, "the stubbed agent turn never ran")
        self.assertFalse(
            seen["async_ok"],
            "oneshot turn still believes it can deliver async results — "
            "background delegations would be orphaned when the process exits",
        )


if __name__ == "__main__":
    unittest.main()

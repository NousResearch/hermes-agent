"""A wedged native Relay call inside a shared-metrics hook must never stall the
instrumented path: delegation returns a child's result only after these hooks run,
so an unbounded call holds the parent's wait for its whole budget."""

import sys
import os
import threading
import time
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import hermes_cli.observability.relay_shared_metrics as rsm_mod
import importlib
_real = importlib.import_module("hermes_cli.observability.relay_shared_metrics")
rsm = rsm_mod


class _FakeRelay:
    def __init__(self):
        self.stack = MagicMock()

    def get_scope_stack(self):
        return self.stack


class TestBoundedTaskScopeOps:
    def _runtime(self):
        rt = object.__new__(rsm_mod._Runtime)
        rt.host = MagicMock()
        rt.relay = _FakeRelay()
        return rt

    def test_task_scope_op_is_bounded(self):
        rt = self._runtime()
        task = MagicMock()
        task.context = threading.local.__dict__ if False else None
        # a real Context: the invoke must run inside it
        import contextvars
        task.context = contextvars.Context()

        def wedged_callback():
            time.sleep(60)

        started = time.monotonic()
        try:
            rt._run_in_task(task, wedged_callback)
            raised = False
        except TimeoutError:
            raised = True
        elapsed = time.monotonic() - started
        assert raised, "a wedged native call must raise TimeoutError, not hang"
        assert elapsed < 30, f"bound not enforced: waited {elapsed:.1f}s"

    def test_task_scope_op_returns_value(self):
        rt = self._runtime()
        task = MagicMock()
        import contextvars
        task.context = contextvars.Context()
        out = rt._run_in_task(task, lambda: "ok")
        assert out == "ok"

    def test_session_scoped_op_passes_timeout(self):
        rt = self._runtime()
        session = MagicMock()
        session.relay_session = MagicMock()
        rt._run_scoped(session, None, lambda: "fine")
        _, kwargs = rt.host.run_in_session.call_args
        assert kwargs.get("timeout") == rsm_mod._NATIVE_OP_TIMEOUT

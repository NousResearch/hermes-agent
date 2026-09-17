"""Unit tests for lifecycle-capture plugin — timeout guards and flush paths."""

from __future__ import annotations

import queue
import sys
import tempfile
import shutil
import os
import threading
import time
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Plugin loader — imports the patched plugin from the real tree
# ---------------------------------------------------------------------------

def _load_plugin() -> any:
    """Load the lifecycle-capture plugin from /Users/sharpe/.hermes/plugins/."""
    tmp = tempfile.mkdtemp(prefix="lc_test_")
    pkg_dir = os.path.join(tmp, "lifecycle_capture")
    os.makedirs(pkg_dir)

    src = "/Users/sharpe/.hermes/plugins/lifecycle-capture/__init__.py"
    priv = "/Users/sharpe/.hermes/plugins/lifecycle-capture/privacy.py"

    shutil.copyfile(priv, os.path.join(pkg_dir, "privacy.py"))

    with open(src) as f:
        init_src = f.read()
    init_src = init_src.replace("from . import privacy", "from lifecycle_capture import privacy")
    with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
        f.write(init_src)
    with open(os.path.join(tmp, "__init__.py"), "w") as f:
        f.write("# empty\n")

    sys.path.insert(0, tmp)
    try:
        import lifecycle_capture as lc
        import importlib
        importlib.reload(lc)
    finally:
        sys.path.remove(tmp)

    return lc


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

lc = None  # loaded in setUpModule


def setUpModule():
    global lc
    lc = _load_plugin()


def _mock_ctx(overrides=None):
    defaults = {
        "compression": "synthetic",
        "target_provider": "hindsight",
        "capture_tool_results": True,
        "capture_llm_turns": True,
        "honcho_turn_capture": False,
        "project_context_inject": False,
        "max_payload_chars": 4096,
        "subagent_capture": False,
        "api_request_capture": False,
        "max_buffer_observations": 500,
        "flush_content_chars": 8000,
    }
    if overrides:
        defaults.update(overrides)
    ctx = MagicMock()
    ctx.get_config.side_effect = lambda k, d=None: defaults.get(k, d)
    return ctx


def _fresh_engine():
    lc._ENGINE = None
    lc.register(_mock_ctx())
    eng = lc._ENGINE
    while eng._flush_q.qsize():
        try:
            eng._flush_q.get_nowait()
        except queue.Empty:
            break
    return eng


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTimeoutBoundRealtimeSink(unittest.TestCase):
    """B-1 regression: inline flush paths must never hang on a slow backend."""

    SLOW = 8.0       # must be > _INLINE_RETAIN_TIMEOUT (5s)
    BOUND = 5.5      # max allowed inline return time

    def _slow_retain(self, content, context, tags):
        time.sleep(self.SLOW)
        return True

    def test_on_session_end_returns_within_bound(self):
        """on_session_end → _bounded_flush must not block > 5.5s."""
        eng = _fresh_engine()
        eng._add("[probe] test")
        eng._session_id = "s"
        eng._project = "p"
        eng.observe_event("session_end", {"session_id": "s"})

        orig = lc._ProviderSink.retain
        lc._ProviderSink.retain = self._slow_retain
        try:
            start = time.monotonic()
            eng._bounded_flush(trigger="on_session_end")
            elapsed = time.monotonic() - start
        finally:
            lc._ProviderSink.retain = orig

        self.assertLess(elapsed, self.BOUND,
            f"on_session_end hung for {elapsed:.2f}s (slow_retain was {self.SLOW}s)")

    def test_on_session_finalize_returns_within_bound(self):
        """on_session_finalize → _bounded_flush must not block > 5.5s."""
        eng = _fresh_engine()
        eng._add("[probe] test")

        orig = lc._ProviderSink.retain
        lc._ProviderSink.retain = self._slow_retain
        try:
            start = time.monotonic()
            eng._bounded_flush(trigger="on_session_finalize")
            elapsed = time.monotonic() - start
        finally:
            lc._ProviderSink.retain = orig

        self.assertLess(elapsed, self.BOUND)

    def test_kanban_task_completed_returns_within_bound(self):
        """kanban_task_completed → _bounded_flush must not block > 5.5s."""
        eng = _fresh_engine()
        eng._add("[probe] test")
        eng.observe_event("kanban_completed", {
            "task_id": "t", "board": "b", "assignee": "a",
            "summary": "s", "reason": None})

        orig = lc._ProviderSink.retain
        lc._ProviderSink.retain = self._slow_retain
        try:
            start = time.monotonic()
            eng._bounded_flush(trigger="kanban_task_completed")
            elapsed = time.monotonic() - start
        finally:
            lc._ProviderSink.retain = orig

        self.assertLess(elapsed, self.BOUND)

    def test_timed_out_chunks_go_to_daemon_queue(self):
        """When retain times out, the chunk must be enqueued for async retry."""
        eng = _fresh_engine()
        eng._add("[probe] should be retried")

        orig = lc._ProviderSink.retain
        lc._ProviderSink.retain = self._slow_retain
        try:
            eng._bounded_flush(trigger="test")
        finally:
            lc._ProviderSink.retain = orig

        # The daemon queue should now contain the retry dict
        self.assertEqual(eng._flush_q.qsize(), 1)
        item = eng._flush_q.get_nowait()
        self.assertIsInstance(item, dict)
        self.assertIn("__retry__", item)
        self.assertIn("[probe] should be retried", item["__retry__"])


class TestAutoBackendCachingBug(unittest.TestCase):
    """Minor: 'auto' sentinel must not be cached as a resolved backend name."""

    def test_auto_not_left_as_resolved_name(self):
        """name() must never return the literal string 'auto' — it must resolve."""
        eng = _fresh_engine()
        # Cache is empty (fresh engine), target is 'auto', backend resolves to 'hindsight'
        name = eng._sink.name()
        self.assertNotEqual(name, "auto",
            "name() returned literal 'auto' instead of resolving to a backend")
        # Should be one of the real backends
        self.assertIn(name, ("hindsight", "honcho", "none"))

    def test_auto_forces_re_resolution_on_subsequent_calls(self):
        """When settings=auto and _name=auto (pre-resolve), name() re-resolves."""
        eng = _fresh_engine()
        # Force the cached state that existed before the fix:
        eng._settings._cache["target_provider"] = "auto"
        eng._sink._name = "auto"
        eng._sink._resolved_at = time.monotonic()  # within 60s window

        resolve_count = [0]

        def counting_configured_name():
            resolve_count[0] += 1
            return "hindsight"

        orig = lc._ProviderSink._configured_name_uncached
        lc._ProviderSink._configured_name_uncached = staticmethod(counting_configured_name)
        try:
            name1 = eng._sink.name()
            name2 = eng._sink.name()
        finally:
            lc._ProviderSink._configured_name_uncached = orig

        # With the fix: "auto" resets _resolved_at, so next call re-resolves
        # verify name1 is not 'auto' and re-resolution happened
        self.assertNotEqual(name1, "auto")
        self.assertGreater(resolve_count[0], 0,
            "Re-resolution did not happen even though 'auto' was cached")


class TestEngineLifecycle(unittest.TestCase):
    """Basic engine sanity — buffer, drain, observe_*."""

    def test_observe_event_then_drain(self):
        eng = _fresh_engine()
        eng._session_id = "s"
        eng._project = "p"
        eng.observe_event("session_start", {"session_id": "s"})
        batch = eng._drain()
        self.assertEqual(len(batch), 1)
        self.assertIn("session_start", batch[0])

    def test_observe_tool_call_sanitized(self):
        eng = _fresh_engine()
        eng._session_id = "s"
        eng._project = "p"
        eng.observe_tool_call(
            tool_name="terminal", args={"command": "secret_key=abc123"}, result=None,
            duration_ms=100, status="ok", error_type=None, error_message=None)
        batch = eng._drain()
        self.assertEqual(len(batch), 1)
        # The args should be replaced with FILE_ACCESS_NOTE (dotfile guard).
        # The raw secret value must not appear in the args field.
        self.assertIn("[FILE_ACCESS: secret-bearing path touched", batch[0])
        self.assertNotIn('"command":', batch[0])

    def test_flush_async_never_blocks(self):
        """flush_async must return immediately without waiting for retain."""
        eng = _fresh_engine()
        eng._add("[x] test")

        def slow_retain(c, ctx, tags):
            time.sleep(20.0)
            return True

        orig = lc._ProviderSink.retain
        lc._ProviderSink.retain = slow_retain
        try:
            start = time.monotonic()
            eng.flush_async(trigger="test")
            elapsed = time.monotonic() - start
        finally:
            lc._ProviderSink.retain = orig

        self.assertLess(elapsed, 1.0,
            "flush_async should return near-instantly")


class TestBoundedRetainCall(unittest.TestCase):
    """Unit test for _bounded_retain itself."""

    def test_bounded_retain_returns_true_on_fast_retain(self):
        eng = _fresh_engine()
        eng._add("[x]")
        ok = eng._bounded_retain("content", "ctx", ["tag"])
        self.assertTrue(ok)

    def test_bounded_retain_returns_false_on_timeout(self):
        eng = _fresh_engine()

        def hang(c, ctx, tags):
            time.sleep(20.0)
            return True

        orig = lc._ProviderSink.retain
        lc._ProviderSink.retain = hang
        try:
            ok = eng._bounded_retain("content", "ctx", ["tag"])
        finally:
            lc._ProviderSink.retain = orig

        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)

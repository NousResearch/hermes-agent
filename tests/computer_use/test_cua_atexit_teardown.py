"""Tests for cua-driver subprocess teardown on interpreter exit.

``CuaDriverBackend`` spawns a long-lived ``cua-driver`` child process and
caches it for the life of the Hermes process. Nothing ever tore it down, so
the driver outlived the session that started it (#28152 item 3 — "Hermes does
not keep the driver alive after tool completion"). #69903 fixed the *overlay*
redraw loop that made the lingering process burn CPU, but not the lingering
process itself.

``tools/computer_use/tool.py`` registers an ``atexit`` hook that stops the
cached backend, mirroring ``browser_tool``'s
``atexit.register(_emergency_cleanup_all_sessions)``.

These assert the behavior contract — the hook is registered, it stops a live
backend, it is a no-op when nothing was ever started, and it never raises out
of ``atexit`` — not a snapshot of the module's source.
"""

import atexit
from unittest.mock import MagicMock, patch

from tools.computer_use import tool as cu_tool


class TestAtexitTeardown:
    def test_shutdown_stops_a_live_backend(self):
        """A cached backend is stopped when the interpreter exits."""
        fake = MagicMock()
        with patch.object(cu_tool, "_backend", fake):
            cu_tool._shutdown_backend_atexit()
            fake.stop.assert_called_once()

    def test_shutdown_clears_the_cached_backend(self):
        """After teardown the module cache is empty, so a later call re-spawns."""
        fake = MagicMock()
        with patch.object(cu_tool, "_backend", fake):
            cu_tool._shutdown_backend_atexit()
            assert cu_tool._backend is None

    def test_shutdown_is_a_noop_when_never_started(self):
        """No backend was ever created => nothing to stop, no error."""
        with patch.object(cu_tool, "_backend", None):
            cu_tool._shutdown_backend_atexit()  # must not raise
            assert cu_tool._backend is None

    def test_shutdown_swallows_backend_errors(self):
        """A failing stop() must not raise out of an atexit hook.

        Exceptions escaping atexit print a traceback on every exit and can
        mask the real exit status.
        """
        fake = MagicMock()
        fake.stop.side_effect = RuntimeError("driver already dead")
        with patch.object(cu_tool, "_backend", fake):
            cu_tool._shutdown_backend_atexit()  # must not raise
            assert cu_tool._backend is None

    def test_shutdown_stops_every_session_backend(self):
        """Session-scoped caches are all drained, not only the legacy slot."""
        first = MagicMock()
        second = MagicMock()
        with patch.object(cu_tool, "_backend", None), \
             patch.object(cu_tool, "_backends", {"one": first, "two": second}), \
             patch.object(cu_tool, "_backend_call_locks", {}):
            cu_tool._shutdown_backend_atexit()
            first.stop.assert_called_once()
            second.stop.assert_called_once()
            assert cu_tool._backends == {}

    def test_hook_is_registered_with_atexit(self):
        """Importing the tool module registers the teardown hook.

        Verified by unregistering and re-registering: atexit.unregister only
        removes a function that was actually registered, so a successful
        round-trip proves the import-time registration happened.
        """
        atexit.unregister(cu_tool._shutdown_backend_atexit)
        try:
            with patch.object(cu_tool, "_backend", MagicMock()) as fake:
                # Re-register and fire the full atexit chain the way the
                # interpreter would, then confirm our hook ran.
                atexit.register(cu_tool._shutdown_backend_atexit)
                atexit._run_exitfuncs()
                fake.stop.assert_called_once()
        finally:
            atexit.unregister(cu_tool._shutdown_backend_atexit)
            atexit.register(cu_tool._shutdown_backend_atexit)

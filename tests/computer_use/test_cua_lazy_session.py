"""The cua-driver run session (agent cursor) must be declared lazily.

On Linux/X11, cua-driver 0.8.3 backs the per-session agent cursor declared by
``start_session`` with a temporary uinput virtual pointer whose creation can
crash a KDE Plasma/Qt X11 session (NousResearch/hermes-agent#66392). A pure
read-only request (``capture``/``list_apps``/``wait``) must therefore NOT
declare that input-capable session; the declaration has to defer until the
first genuine input/cursor action routed through ``_action``.

These assert the ordering contract (no ``start_session`` for read-only flows;
exactly one on the first input action, idempotent afterwards; ``end_session``
only when the session was actually declared), not any real display behaviour —
the session is faked so the test needs neither a DISPLAY nor uinput.
"""

from unittest.mock import patch

from tools.computer_use import cua_backend
from tools.computer_use.cua_backend import CuaDriverBackend


class _FakeSession:
    """Records every call_tool name/args; no real cua-driver involved."""

    def __init__(self) -> None:
        self.calls = []
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def call_tool(self, name, args, timeout=30.0):
        self.calls.append((name, args))
        return {
            "data": None,
            "images": [],
            "structuredContent": {},
            "isError": False,
        }

    def supports_capability(self, capability, tool=None):
        return False

    @property
    def call_names(self):
        return [name for name, _ in self.calls]


def _make_backend():
    backend = CuaDriverBackend()
    fake = _FakeSession()
    backend._session = fake
    return backend, fake


def _start(backend):
    # Neutralise the optional-dep install + update nudge so start() only
    # exercises the session-declaration logic under test.
    with patch.object(cua_backend, "_maybe_nudge_update", lambda: None), \
         patch("tools.lazy_deps.ensure", lambda *a, **k: None):
        backend.start()


class TestLazyRunSession:
    def test_start_does_not_declare_session(self):
        backend, fake = _make_backend()
        _start(backend)
        # start() alone must not create the input-capable agent-cursor session.
        assert "start_session" not in fake.call_names

    def test_read_only_actions_do_not_declare_session(self):
        backend, fake = _make_backend()
        _start(backend)
        backend.list_apps()
        assert "start_session" not in fake.call_names

    def test_first_input_action_declares_session_once(self):
        backend, fake = _make_backend()
        _start(backend)
        # Prime an active target so click() routes through _action().
        backend._active_pid = 1234
        backend._active_window_id = 5678

        backend.click(element=0)
        assert fake.call_names.count("start_session") == 1

        # A second input action must not re-declare (idempotent flag).
        backend.click(element=0)
        assert fake.call_names.count("start_session") == 1

    def test_stop_ends_session_only_when_declared(self):
        # No input action => no start_session => no end_session.
        backend, fake = _make_backend()
        _start(backend)
        backend.list_apps()
        backend.stop()
        assert "end_session" not in fake.call_names

    def test_stop_ends_session_after_input_action(self):
        backend, fake = _make_backend()
        _start(backend)
        backend._active_pid = 1234
        backend._active_window_id = 5678
        backend.click(element=0)
        backend.stop()
        assert fake.call_names.count("start_session") == 1
        assert fake.call_names.count("end_session") == 1

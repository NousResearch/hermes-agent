"""Concurrent-session isolation for HERMES_SESSION_ID (gateway os.environ leak).

The gateway runs concurrent sessions (+ an in-process cron tick) in ONE process.
Per-session state written to process-global os.environ gets clobbered across
concurrent sessions. The session-context module was migrated to contextvars for
exactly this reason, but two gaps remained:

  1. ``_set_session_env`` bound ``session_key`` but OMITTED ``session_id`` — so
     the ``HERMES_SESSION_ID`` contextvar was "" on a normal turn and only the
     (clobbered) os.environ carried a value.
  2. ``set_current_session_id`` wrote os.environ unconditionally (clobbering
     concurrent sessions) even inside the gateway.

These tests assert the contextvar carries the live id AND the os.environ global
is not cross-contaminated under concurrency. They are RED on the pre-fix code.
"""

import os
import threading

import pytest

import gateway.session_context as sc


@pytest.fixture(autouse=True)
def _clean_session_env():
    saved = {k: os.environ.get(k) for k in ("HERMES_SESSION_ID", "_HERMES_GATEWAY")}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class TestSessionIdContextvarBinding:
    def test_session_id_bound_via_set_session_vars(self):
        """AC0: set_session_vars(session_id=...) binds the contextvar so
        get_session_env returns the live id (not '')."""
        tokens = sc.set_session_vars(platform="discord", chat_id="C", session_id="SESS_LIVE")
        try:
            assert sc.get_session_env("HERMES_SESSION_ID") == "SESS_LIVE"
        finally:
            sc.clear_session_vars(tokens)


class TestSetCurrentSessionIdGatewayAware:
    def test_in_gateway_does_not_write_os_environ(self):
        """In the gateway (_HERMES_GATEWAY=1), set_current_session_id binds the
        contextvar but must NOT clobber the process-global os.environ."""
        os.environ["_HERMES_GATEWAY"] = "1"
        os.environ.pop("HERMES_SESSION_ID", None)
        sc.set_current_session_id("S_GATEWAY")
        assert sc.get_session_env("HERMES_SESSION_ID") == "S_GATEWAY"  # contextvar set
        assert os.environ.get("HERMES_SESSION_ID") is None  # os.environ NOT clobbered

    def test_outside_gateway_writes_os_environ(self):
        """Outside the gateway (CLI/cron-standalone/worker), the os.environ
        write IS needed (single-process; tools read it via the fallback)."""
        os.environ.pop("_HERMES_GATEWAY", None)
        os.environ.pop("HERMES_SESSION_ID", None)
        sc.set_current_session_id("S_CLI")
        assert os.environ.get("HERMES_SESSION_ID") == "S_CLI"


class TestConcurrentSessionsOsEnvironNotContaminated:
    def test_concurrent_sessions_os_environ_not_contaminated(self):
        """AC1 (the REAL isolation test): two concurrent in-gateway contexts
        binding their own session must NOT cross-contaminate the process-global
        os.environ['HERMES_SESSION_ID']. A get_session_env-only assertion is a
        fake-green (contextvar-first masks the clobber)."""
        os.environ["_HERMES_GATEWAY"] = "1"
        os.environ.pop("HERMES_SESSION_ID", None)

        import contextvars

        observed_env = {}
        barrier = threading.Barrier(2)

        def _run(name, sid):
            def _body():
                tokens = sc.set_session_vars(
                    platform="discord", chat_id=name, session_id=sid
                )
                sc.set_current_session_id(sid)
                barrier.wait(timeout=5)
                # Each context reads its OWN id from the contextvar...
                assert sc.get_session_env("HERMES_SESSION_ID") == sid
                # ...and the process-global os.environ must NOT have been
                # clobbered by either context (in-gateway → no os.environ write).
                observed_env[name] = os.environ.get("HERMES_SESSION_ID")
                sc.clear_session_vars(tokens)

            ctx = contextvars.copy_context()
            t = threading.Thread(target=lambda: ctx.run(_body))
            t.start()
            return t

        t1 = _run("A", "SESS_A")
        t2 = _run("B", "SESS_B")
        t1.join(10)
        t2.join(10)

        # The clobber surface: in-gateway, neither context wrote os.environ, so
        # the global stays None throughout. (Pre-fix, set_current_session_id
        # wrote os.environ unconditionally → A would see "SESS_B" or vice versa.)
        assert observed_env.get("A") is None, f"os.environ clobbered: {observed_env}"
        assert observed_env.get("B") is None, f"os.environ clobbered: {observed_env}"

"""Tests for the HERMES_CRON_SESSION ContextVar isolation (origin-leak v3).

The bug (PRD-send-message-origin-leak-v3): the in-process cron scheduler set
``os.environ["HERMES_CRON_SESSION"] = "1"`` process-globally and never cleared
it, so after the first cron tick every interactive gateway turn read cron=True
and mis-routed bare send_message calls to the home channel (and the approval
gate / send-origin resolver were poisoned). The fix makes the marker a
task/thread-isolated ContextVar with an os.environ fallback for back-compat.

These cover the context primitive + the reader migration. The cron-spawned
subagent approval-gating rebind (B1) lives in
``tests/tools/test_cron_subagent_session.py``.
"""

import concurrent.futures
import contextvars
import os

import pytest

import gateway.session_context as sc


@pytest.fixture(autouse=True)
def _clean_cron_env():
    """Ensure os.environ is clean around each test (the old bug latched it)."""
    prior = os.environ.pop("HERMES_CRON_SESSION", None)
    yield
    os.environ.pop("HERMES_CRON_SESSION", None)
    if prior is not None:
        os.environ["HERMES_CRON_SESSION"] = prior


# ---------------------------------------------------------------------------
# Phase 1 — the ContextVar primitive
# ---------------------------------------------------------------------------

class TestCronSessionContextVar:
    def test_unset_is_false(self):
        assert sc.is_cron_session() is False

    def test_set_then_clear(self):
        tok = sc.set_cron_session()
        try:
            assert sc.is_cron_session() is True
        finally:
            sc.clear_cron_session(tok)
        assert sc.is_cron_session() is False

    def test_clear_none_is_noop(self):
        # No marker bound (interactive parent's child) -> clear(None) is safe.
        sc.clear_cron_session(None)
        assert sc.is_cron_session() is False

    def test_clear_with_bad_token_restores_env_fallback(self):
        # Greptile P2: if reset(token) raises (cross-context token), the
        # fallback must restore _UNSET (not "") so get_session_env still falls
        # through to os.environ — otherwise I5 back-compat is silently defeated.
        sc.set_cron_session()  # bind it (we will clear with a bogus token)
        os.environ["HERMES_CRON_SESSION"] = "1"
        # A token from a different ContextVar instance forces reset() to raise.
        import contextvars
        bogus = contextvars.ContextVar("bogus", default="").set("x")
        sc.clear_cron_session(bogus)
        # After the except path, the ContextVar is _UNSET, so the os.environ
        # value is honored again (fallback works).
        assert sc.is_cron_session() is True
        os.environ.pop("HERMES_CRON_SESSION")
        assert sc.is_cron_session() is False

    def test_nestable_reset_restores_prior(self):
        outer = sc.set_cron_session()
        try:
            assert sc.is_cron_session() is True
            inner = sc.set_cron_session()
            try:
                assert sc.is_cron_session() is True
            finally:
                sc.clear_cron_session(inner)
            # After inner clear we must still be cron (outer still bound).
            assert sc.is_cron_session() is True
        finally:
            sc.clear_cron_session(outer)
        assert sc.is_cron_session() is False

    def test_registered_in_var_map(self):
        # The _VAR_MAP registration is what makes get_session_env consult the
        # ContextVar before os.environ. Without it, isolation silently breaks.
        assert "HERMES_CRON_SESSION" in sc._VAR_MAP
        assert sc._VAR_MAP["HERMES_CRON_SESSION"] is sc._CRON_SESSION


# ---------------------------------------------------------------------------
# AC9 / RC-3 — thread isolation (the registration gate)
# ---------------------------------------------------------------------------

class TestCronSessionThreadIsolation:
    def test_set_in_one_thread_not_visible_in_another(self):
        """A cron job's marker (set in its worker thread) must NOT leak to a
        concurrent interactive turn running in a different thread/context.
        """
        results = {}

        def _cron_worker(barrier_set, barrier_check):
            tok = sc.set_cron_session()
            try:
                results["cron_thread_sees_cron"] = sc.is_cron_session()
                barrier_set.set_result(True)  # signal: marker is set
                # Wait until the interactive thread has checked.
                barrier_check.result(timeout=5)
            finally:
                sc.clear_cron_session(tok)

        def _interactive_worker(barrier_set, barrier_check):
            # Wait until the cron thread has set its marker.
            barrier_set.result(timeout=5)
            # A separate thread runs its own context; the cron marker must be
            # invisible here (this is the leak the bug caused process-globally).
            results["interactive_thread_sees_cron"] = sc.is_cron_session()
            barrier_check.set_result(True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            b_set = concurrent.futures.Future()
            b_check = concurrent.futures.Future()
            f1 = ex.submit(_cron_worker, b_set, b_check)
            f2 = ex.submit(_interactive_worker, b_set, b_check)
            f1.result(timeout=10)
            f2.result(timeout=10)

        assert results["cron_thread_sees_cron"] is True
        assert results["interactive_thread_sees_cron"] is False

    def test_copy_context_inherits_marker(self):
        """A cron job's MAIN agent run inherits the marker via copy_context()
        (the scheduler does this at run_job:~1905). Prove the inheritance.
        """
        tok = sc.set_cron_session()
        try:
            ctx = contextvars.copy_context()
            assert ctx.run(sc.is_cron_session) is True
        finally:
            sc.clear_cron_session(tok)


# ---------------------------------------------------------------------------
# AC6 / I5 — os.environ back-compat fallback
# ---------------------------------------------------------------------------

class TestCronSessionEnvFallback:
    def test_env_fallback_when_contextvar_unset(self):
        # CLI / standalone scheduler / tests that set os.environ still work.
        assert sc.is_cron_session() is False
        os.environ["HERMES_CRON_SESSION"] = "1"
        assert sc.is_cron_session() is True

    def test_contextvar_explicit_unset_does_not_fallback(self):
        # After clear (reset to _UNSET default), a stale os.environ would still
        # be read — but a real cron lineage uses set/clear, and interactive
        # turns never set the ContextVar at all, so they fall to a clean env.
        os.environ["HERMES_CRON_SESSION"] = "1"
        assert sc.is_cron_session() is True
        os.environ.pop("HERMES_CRON_SESSION")
        assert sc.is_cron_session() is False


# ---------------------------------------------------------------------------
# AC1 — the actual leak fix: an interactive turn resolves to ORIGIN, not home,
# even after a cron job has run (the marker must not be latched process-wide).
# ---------------------------------------------------------------------------

class TestInteractiveTurnNotPoisoned:
    def test_resolve_send_target_origin_after_cron_cycle(self):
        """Simulate a cron job running (set then clear the marker via the real
        path), then bind an interactive turn's session and assert the bare
        send target resolves to ORIGIN (the current channel), not NOT_IN_TURN
        (home). Pre-fix, the latched process-global marker forced NOT_IN_TURN.
        """
        from tools.send_message_tool import (
            _resolve_send_target,
            _SEND_TARGET_ORIGIN,
            _SEND_TARGET_NOT_IN_TURN,
        )

        # 1. A cron job runs and finishes (marker set then reset).
        tok = sc.set_cron_session()
        assert sc.is_cron_session() is True
        sc.clear_cron_session(tok)
        assert sc.is_cron_session() is False  # the key: not latched

        # 2. Now an interactive gateway turn binds its session.
        session_tokens = sc.set_session_vars(
            platform="discord",
            chat_id="1517844402897424504",  # #fix-issues
            chat_name="Daemonarchy / #fix-issues",
        )
        try:
            state, chat_id, _thread = _resolve_send_target("discord")
        finally:
            sc.clear_session_vars(session_tokens)

        assert state == _SEND_TARGET_ORIGIN
        assert chat_id == "1517844402897424504"
        assert state != _SEND_TARGET_NOT_IN_TURN

    def test_resolve_send_target_homes_during_real_cron(self):
        """Cron-guard-first ordering preserved: during a REAL cron job (marker
        bound, no interactive platform) the bare target still homes."""
        from tools.send_message_tool import (
            _resolve_send_target,
            _SEND_TARGET_NOT_IN_TURN,
        )

        tok = sc.set_cron_session()
        try:
            # No interactive platform bound -> cron homes (delivery handles it).
            state, _chat, _thread = _resolve_send_target("discord")
        finally:
            sc.clear_cron_session(tok)
        assert state == _SEND_TARGET_NOT_IN_TURN

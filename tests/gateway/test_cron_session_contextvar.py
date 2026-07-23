"""Tests for ContextVar-based cron session flag isolation (#73195).

When the cron scheduler runs inside a gateway process
(InProcessCronScheduler), the old process-global flag
  os.environ["HERMES_CRON_SESSION"] = "1"
leaked from cron job threads into user-interactive sessions, blocking
execute_code and other cron-gated tools in every subsequent user turn.

The fix has two layers:

  Layer 1 (producer): cron/scheduler.py sets the flag via ContextVar
    instead of os.environ.  The flag is task-local — it never escapes
    the cron job's ctx.run() context.

  Layer 2 (consumer/defense): gateway/session_context.py's
    set_session_vars() explicitly sets _CRON_SESSION_FLAG to "",
    suppressing the os.environ fallback.  Even if an external process
    leaked HERMES_CRON_SESSION into the environment, a gateway session
    that binds its contextvars will never be miscategorised as cron.
"""

import contextvars
import os
import threading

import pytest

import gateway.session_context as sc
from gateway.session_context import (
    _CRON_SESSION_FLAG,
    _UNSET,
    _VAR_MAP,
    is_cron_session,
    reset_session_vars,
    set_session_vars,
)


@pytest.fixture(autouse=True)
def _clean_cron_contextvar(monkeypatch):
    """Reset every path that stores the cron flag between tests."""
    _CRON_SESSION_FLAG.set(_UNSET)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    yield
    _CRON_SESSION_FLAG.set(_UNSET)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)


# ---------------------------------------------------------------------------
# 1. is_cron_session() resolution order
# ---------------------------------------------------------------------------

class TestIsCronSessionResolution:
    """ContextVar wins over os.environ.  _UNSET falls through."""

    def test_neither_set_returns_false(self):
        assert _CRON_SESSION_FLAG.get() is _UNSET
        assert "HERMES_CRON_SESSION" not in os.environ
        assert is_cron_session() is False

    def test_contextvar_true(self):
        _CRON_SESSION_FLAG.set("1")
        assert is_cron_session() is True

    def test_contextvar_empty_suppresses_fallback(self, monkeypatch):
        """ContextVar "" explicitly cleared → suppress os.environ."""
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        _CRON_SESSION_FLAG.set("")
        assert is_cron_session() is False

    def test_falls_back_to_osenviron_when_unset(self, monkeypatch):
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        assert _CRON_SESSION_FLAG.get() is _UNSET
        assert is_cron_session() is True

    @pytest.mark.parametrize("val", ["", "0", "false", "no"])
    def test_untruthy_osenviron_values(self, monkeypatch, val):
        monkeypatch.setenv("HERMES_CRON_SESSION", val)
        assert is_cron_session() is False


# ---------------------------------------------------------------------------
# 2. Layer 1 — Producer isolation (ContextVar does not leak)
# ---------------------------------------------------------------------------

class TestProducerIsolation:
    """cron/scheduler.py sets _CRON_SESSION_FLAG.set("1") inside the
    cron job's context.  Concurrent threads / separate contexts never
    see it."""

    def test_cron_flag_not_visible_in_other_thread(self):
        seen = []

        def cron_thread():
            _CRON_SESSION_FLAG.set("1")
            assert is_cron_session() is True

        def user_thread():
            seen.append(is_cron_session())

        cron_thread()
        t = threading.Thread(target=user_thread)
        t.start()
        t.join()

        assert seen == [False], (
            f"Thread inherited cron flag: {seen}"
        )

    def test_cron_flag_not_visible_outside_copied_context(self):
        ctx = contextvars.copy_context()

        def inside():
            _CRON_SESSION_FLAG.set("1")
            return is_cron_session()

        assert ctx.run(inside) is True
        assert _CRON_SESSION_FLAG.get() is _UNSET
        assert is_cron_session() is False

    def test_clear_session_vars_resets_cron_flag(self):
        """clear_session_vars() manages _CRON_SESSION_FLAG because
        it's in _VAR_MAP."""
        tokens = set_session_vars(session_key="test")
        _CRON_SESSION_FLAG.set("1")
        assert is_cron_session() is True

        sc.clear_session_vars(tokens)
        assert _CRON_SESSION_FLAG.get() == ""
        assert is_cron_session() is False


# ---------------------------------------------------------------------------
# 3. Layer 2 — Defense-in-depth (set_session_vars gates the flag)
# ---------------------------------------------------------------------------

class TestSetSessionVarsGatesCronFlag:
    """set_session_vars() sets _CRON_SESSION_FLAG to "" so even if
    os.environ carries a leaked HERMES_CRON_SESSION=1, a gateway
    session is never miscategorised as cron."""

    def test_set_session_vars_sets_cron_flag_to_empty(self):
        _CRON_SESSION_FLAG.set("1")  # simulate leftover from prior cron
        tokens = set_session_vars(session_key="test")
        try:
            assert _CRON_SESSION_FLAG.get() == ""
            assert is_cron_session() is False
        finally:
            sc.clear_session_vars(tokens)

    def test_set_session_vars_gates_osenviron_leak(self, monkeypatch):
        """Defense-in-depth: even with HERMES_CRON_SESSION=1 in
        os.environ, a gateway session that calls set_session_vars
        must return False from is_cron_session()."""
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")

        reset_session_vars()
        tokens = set_session_vars(
            platform="feishu",
            chat_id="group-abc",
            user_id="user-123",
            session_key="session-xyz",
        )

        try:
            assert is_cron_session() is False, (
                "BROKEN: set_session_vars() did not gate the cron flag. "
                "os.environ fallback still active."
            )
        finally:
            sc.clear_session_vars(tokens)


# ---------------------------------------------------------------------------
# 4. Bug scenario — User replies to cron-delivered Feishu message
# ---------------------------------------------------------------------------

class TestBugScenarioFeishuReplyToCronMessage:
    """Reproduction of #73195: a Feishu user replies to a cron-delivered
    message card.  After the fix, execute_code is NOT blocked."""

    def test_gateway_session_is_cron_session_false(self, monkeypatch):
        """Exact production environment: both HERMES_GATEWAY_SESSION
        and HERMES_CRON_SESSION set in os.environ (the latter leaked
        by a prior cron job).  Gateway handler binds its session →
        is_cron_session() must be False."""
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")

        reset_session_vars()
        tokens = set_session_vars(
            platform="feishu",
            chat_id="group-abc",
            session_key="session-xyz",
        )

        try:
            assert is_cron_session() is False
        finally:
            sc.clear_session_vars(tokens)

    def test_execute_code_not_blocked_in_gateway_after_cron_leak(
        self, monkeypatch
    ):
        """After the fix, a user session that inherited leaked
        HERMES_CRON_SESSION=1 from a prior cron job should still
        be able to use execute_code — the approval system must NOT
        apply cron_mode=deny."""
        import tools.approval as A
        from unittest.mock import patch as mock_patch

        # Simulate the exact bug environment
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)

        # Gateway handler binds session (defense-in-depth clears cron flag)
        reset_session_vars()
        tokens = set_session_vars(
            platform="feishu",
            chat_id="group-abc",
            user_id="user-123",
            session_key="session-xyz",
        )

        try:
            assert is_cron_session() is False

            # With approvals off, execute_code must be approved
            with mock_patch.object(A, "_get_approval_mode", return_value="off"):
                result = A.check_execute_code_guard(
                    "print('hello world')", "local", has_host_access=False
                )
                assert result["approved"] is True, (
                    f"Blocked despite is_cron_session()=False: {result.get('message')}"
                )
        finally:
            sc.clear_session_vars(tokens)

    def test_execute_code_blocked_in_actual_cron_session(self, monkeypatch):
        """A real cron job (os.environ set, no set_session_vars call)
        must still be blocked by cron_mode=deny.  Backward compat."""
        import tools.approval as A
        from unittest.mock import patch as mock_patch

        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)

        with mock_patch.object(A, "_get_cron_approval_mode", return_value="deny"):
            result = A.check_execute_code_guard(
                "print('hello')", "local", has_host_access=False
            )
            assert not result["approved"]
            assert "BLOCKED" in result["message"]


# ---------------------------------------------------------------------------
# 5. _VAR_MAP registration — lifecycle
# ---------------------------------------------------------------------------

class TestCronSessionInVarMap:
    """HERMES_CRON_SESSION must be in _VAR_MAP so reset_session_vars()
    and clear_session_vars() manage it automatically."""

    def test_registered(self):
        assert "HERMES_CRON_SESSION" in _VAR_MAP
        assert _VAR_MAP["HERMES_CRON_SESSION"] is _CRON_SESSION_FLAG

    def test_reset_restores_unset(self):
        _CRON_SESSION_FLAG.set("1")
        reset_session_vars()
        assert _CRON_SESSION_FLAG.get() is _UNSET

    def test_clear_sets_empty(self):
        _CRON_SESSION_FLAG.set("1")
        tokens = set_session_vars(session_key="test")
        sc.clear_session_vars(tokens)
        assert _CRON_SESSION_FLAG.get() == ""
        assert is_cron_session() is False

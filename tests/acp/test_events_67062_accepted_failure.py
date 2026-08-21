"""Behavior tests for PR #67062 — accepted-future ACP delivery gap.

Adjudicated contract (from adjudication-lane-b.md §4 and live-dedupe-wave456.json):

    In ``_observe_delivery``, when the accepted future fails, attempt a bounded
    retry of the same completion update (the update carries the same result,
    so retrying cannot produce duplicate/crossed results). If retries are
    exhausted, retain a "failed delivery" marker in ``tool_call_meta`` so a
    follow-up reconciliation (e.g., heartbeat or session close) can surface the
    lost completion.

These tests exercise the canonical completion path only (``make_tool_complete_cb``).
The generic transport (``_send_update``) must keep its existing acceptance contract
(an accepted Future stays accepted — at-most-once canonical completion semantics).
The recovery logic is layered on top, bounded, and does not recurse into a fresh
retry loop on a fresh failure.

Tests are behavior tests: they drive ``make_tool_complete_cb`` end-to-end,
inspect ``tool_call_meta`` state, and count ``_send_update`` invocations on
the canonical completion update. They do NOT inspect source text.
"""

import asyncio
from collections import deque
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

import acp

from acp_adapter.events import (
    _send_update,
    make_tool_complete_cb,
)


@pytest.fixture()
def mock_conn():
    conn = MagicMock(spec=acp.Client)
    conn.session_update = MagicMock()
    return conn


@pytest.fixture()
def event_loop_fixture():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def _build_tool_complete_update(tc_id, tool_name, result="ok"):
    """Build a real ToolCallUpdate so the canonical-ID contract is preserved.

    Importing build_tool_complete here (rather than patching it out) means the
    tc_id and update payload used by the retry are exactly the same Python
    object the canonical completion path produces — the retry's identity
    invariant the tests assert is meaningful.
    """
    from acp_adapter.events import build_tool_complete

    return build_tool_complete(
        tc_id, tool_name, result=result, function_args=None, snapshot=None
    )


class TestAcceptedFutureFailureRecovery:
    """Bounded retry on observed-future failure for canonical completion updates."""

    def test_accepted_future_failure_triggers_one_retry_with_same_update(
        self, mock_conn, event_loop_fixture
    ):
        """When the loop-owned Future later fails, the canonical completion must be
        resubmitted once with the SAME update object (canonical tc_id preserved).

        The retry MUST reuse the original update (the result is immutable and the
        tc_id is canonical) — a fresh ``build_tool_complete`` would risk pairing
        the retry with a different tool call's result.
        """
        tc_id = "tc-canonical-1"
        update = _build_tool_complete_update(tc_id, "terminal", result="payload-A")
        tool_call_ids = {"terminal": deque([tc_id])}
        tool_call_meta = {tc_id: {"args": {"command": "ls"}, "snapshot": None}}

        # First safe_schedule_threadsafe returns an accepted future that later
        # fails; the retry's safe_schedule_threadsafe returns a fresh accepted
        # future that succeeds.
        initial_future = Future()
        retry_future = Future()
        initial_future.set_exception(RuntimeError("loop dropped it"))
        retry_future.set_result(None)

        with patch(
            "agent.async_utils.safe_schedule_threadsafe",
            side_effect=[initial_future, retry_future],
        ) as mock_sched:
            cb = make_tool_complete_cb(
                mock_conn,
                "session-1",
                event_loop_fixture,
                tool_call_ids,
                tool_call_meta,
            )
            cb(tc_id, "terminal", {"command": "ls"}, "payload-A")

        # Two scheduling attempts: initial + 1 retry. Bounded.
        assert mock_sched.call_count == 2
        # Inspect the real transport arguments, not the scheduler's mocked
        # coroutine return value. Both sends must target the same session and
        # reuse the exact canonical update object.
        first_session, first_update = mock_conn.session_update.call_args_list[0].args
        second_session, second_update = mock_conn.session_update.call_args_list[1].args
        assert first_session == second_session == "session-1"
        assert first_update is second_update
        assert first_update == update
        assert first_update.tool_call_id == tc_id

    def test_retry_exhaustion_marks_failed_delivery_in_tool_call_meta(
        self, mock_conn, event_loop_fixture
    ):
        """When the initial future fails AND the retry's scheduler rejects,
        a ``failed_delivery`` marker must be retained in ``tool_call_meta`` so a
        later reconciliation (heartbeat / session close) can surface the loss.
        """
        tc_id = "tc-canonical-2"
        tool_call_ids = {"terminal": deque([tc_id])}
        tool_call_meta = {tc_id: {"args": {"command": "ls"}, "snapshot": None}}

        # Initial: accepted future that later fails.
        # Retry: scheduler REJECTS (safe_schedule_threadsafe returns None).
        initial_future = Future()
        initial_future.set_exception(RuntimeError("first attempt dropped"))

        with patch(
            "agent.async_utils.safe_schedule_threadsafe",
            side_effect=[initial_future, None],
        ):
            cb = make_tool_complete_cb(
                mock_conn,
                "session-1",
                event_loop_fixture,
                tool_call_ids,
                tool_call_meta,
            )
            cb(tc_id, "terminal", {"command": "ls"}, "payload-B")

        # Marker retained for follow-up reconciliation.
        assert tc_id in tool_call_meta, (
            "tool_call_meta entry must persist after a failed delivery so a "
            "later heartbeat / session-close reconciliation can surface the loss"
        )
        meta = tool_call_meta[tc_id]
        assert meta.get("failed_delivery") is True, (
            f"expected failed_delivery=True marker, got meta={meta!r}"
        )

    def test_retry_exhaustion_marker_carries_diagnostic_context(
        self, mock_conn, event_loop_fixture
    ):
        """The failed_delivery marker must carry enough context to diagnose the loss.

        At minimum: session id, tool name, tc_id, and the result string that was
        being delivered. Without these, follow-up reconciliation cannot
        meaningfully replay / surface the completion.
        """
        tc_id = "tc-canonical-3"
        tool_call_ids = {"terminal": deque([tc_id])}
        tool_call_meta = {tc_id: {"args": {"command": "ls"}, "snapshot": None}}

        initial_future = Future()
        initial_future.set_exception(RuntimeError("loop down"))

        with patch(
            "agent.async_utils.safe_schedule_threadsafe",
            side_effect=[initial_future, None],
        ):
            cb = make_tool_complete_cb(
                mock_conn,
                "session-1",
                event_loop_fixture,
                tool_call_ids,
                tool_call_meta,
            )
            cb(tc_id, "terminal", {"command": "ls"}, "the-real-result")

        meta = tool_call_meta[tc_id]
        assert meta.get("failed_delivery") is True
        # Diagnostic context sufficient to surface the lost completion.
        for required in ("session_id", "tool_name", "tc_id"):
            assert required in meta, (
                f"failed_delivery marker missing diagnostic field {required!r}; "
                f"got meta={meta!r}"
            )
        assert meta["session_id"] == "session-1"
        assert meta["tool_name"] == "terminal"
        assert meta["tc_id"] == tc_id
        # The actual result that was being delivered, so reconciliation can
        # replay it without consulting the executor again.
        assert meta.get("result") == "the-real-result"

    def test_observed_failure_logs_warning_with_session_context(
        self, mock_conn, event_loop_fixture
    ):
        """The observed-future failure must remain visible (WARNING) with session id."""
        tc_id = "tc-canonical-4"
        tool_call_ids = {"terminal": deque([tc_id])}
        tool_call_meta = {tc_id: {"args": {"command": "ls"}, "snapshot": None}}

        initial_future = Future()
        initial_future.set_exception(RuntimeError("delayed failure"))

        with (
            patch(
                "agent.async_utils.safe_schedule_threadsafe",
                side_effect=[initial_future, None],
            ),
            patch("acp_adapter.events.logger") as mock_logger,
        ):
            cb = make_tool_complete_cb(
                mock_conn,
                "session-1",
                event_loop_fixture,
                tool_call_ids,
                tool_call_meta,
            )
            cb(tc_id, "terminal", {"command": "ls"}, "payload-D")

        # The warning is the same WARNING observability the existing
        # TestSendUpdateDeliveryStatus tests pin. It must remain so the loss
        # is diagnosable in production logs.
        assert mock_logger.warning.called, (
            "WARNING must be emitted on observed-future failure"
        )
        warning_str = str(mock_logger.warning.call_args)
        assert "session-1" in warning_str, (
            f"warning must carry session id for diagnosability; got {warning_str!r}"
        )

    def test_successful_observed_future_does_not_retry(
        self, mock_conn, event_loop_fixture
    ):
        """When the accepted future completes normally, no retry must fire.

        The retry only triggers on the future later failing. A successful
        observed completion is canonical: retried would risk double-delivery
        of the same completion update to the ACP client.
        """
        tc_id = "tc-canonical-5"
        tool_call_ids = {"terminal": deque([tc_id])}
        tool_call_meta = {tc_id: {"args": {"command": "ls"}, "snapshot": None}}

        ok_future = Future()
        ok_future.set_result(None)

        with patch(
            "agent.async_utils.safe_schedule_threadsafe",
            return_value=ok_future,
        ) as mock_sched:
            cb = make_tool_complete_cb(
                mock_conn,
                "session-1",
                event_loop_fixture,
                tool_call_ids,
                tool_call_meta,
            )
            cb(tc_id, "terminal", {"command": "ls"}, "payload-E")

        # Exactly one scheduling attempt — no retry on a successful future.
        assert mock_sched.call_count == 1

    def test_successful_observed_future_clears_meta_with_no_failed_marker(
        self, mock_conn, event_loop_fixture
    ):
        """A normally-delivered completion must not leave a failed_delivery marker."""
        tc_id = "tc-canonical-6"
        tool_call_ids = {"terminal": deque([tc_id])}
        tool_call_meta = {tc_id: {"args": {"command": "ls"}, "snapshot": None}}

        ok_future = Future()
        ok_future.set_result(None)

        with patch(
            "agent.async_utils.safe_schedule_threadsafe",
            return_value=ok_future,
        ):
            cb = make_tool_complete_cb(
                mock_conn,
                "session-1",
                event_loop_fixture,
                tool_call_ids,
                tool_call_meta,
            )
            cb(tc_id, "terminal", {"command": "ls"}, "payload-F")

        # Either the meta is cleared (canonical cleanup), or it remains without
        # any failed_delivery marker. The marker must NOT be set on success.
        if tc_id in tool_call_meta:
            assert "failed_delivery" not in tool_call_meta[tc_id] or (
                tool_call_meta[tc_id].get("failed_delivery") is False
            ), (
                f"failed_delivery marker must NOT be set on a successful "
                f"delivery; got meta={tool_call_meta[tc_id]!r}"
            )

    def test_retry_does_not_infinite_loop_when_retry_future_also_fails(
        self, mock_conn, event_loop_fixture
    ):
        """When the retry's own Future ALSO fails, the recovery must NOT recurse.

        Bounded = at most one retry attempt on the canonical completion path.
        A fresh failure on the retry's future must not trigger another retry.
        """
        tc_id = "tc-canonical-7"
        tool_call_ids = {"terminal": deque([tc_id])}
        tool_call_meta = {tc_id: {"args": {"command": "ls"}, "snapshot": None}}

        # Both the initial and retry are accepted by the scheduler, but BOTH
        # later fail. The retry's failure must not trigger another retry.
        initial_future = Future()
        retry_future = Future()
        initial_future.set_exception(RuntimeError("first failed"))
        retry_future.set_exception(RuntimeError("retry failed"))

        with patch(
            "agent.async_utils.safe_schedule_threadsafe",
            side_effect=[initial_future, retry_future],
        ) as mock_sched:
            cb = make_tool_complete_cb(
                mock_conn,
                "session-1",
                event_loop_fixture,
                tool_call_ids,
                tool_call_meta,
            )
            cb(tc_id, "terminal", {"command": "ls"}, "payload-G")

        # Initial + 1 retry, NO third attempt.
        assert mock_sched.call_count == 2, (
            f"recovery must be bounded; expected exactly 2 scheduling attempts, "
            f"got {mock_sched.call_count}"
        )
        # Marker retained (the canonical completion could not be delivered).
        assert tool_call_meta.get(tc_id, {}).get("failed_delivery") is True

    def test_late_double_failure_rebuilds_marker_after_meta_cleanup(
        self, mock_conn, event_loop_fixture
    ):
        """A genuinely late failure retains replay context after normal cleanup.

        The callback returns while the first Future is still pending, so the
        canonical path removes ``tool_call_meta``. Both failures are then fired
        later. The terminal marker must be rebuilt from closure-captured state,
        not from metadata that no longer exists.
        """
        tc_id = "tc-canonical-late"
        tool_call_ids = {"terminal": deque([tc_id])}
        expected_args = {"command": "ls", "cwd": "/tmp"}
        tool_call_meta = {tc_id: {"args": expected_args, "snapshot": "final snapshot"}}
        initial_future = Future()
        retry_future = Future()

        with patch(
            "agent.async_utils.safe_schedule_threadsafe",
            side_effect=[initial_future, retry_future],
        ) as mock_sched:
            cb = make_tool_complete_cb(
                mock_conn,
                "session-late",
                event_loop_fixture,
                tool_call_ids,
                tool_call_meta,
            )
            cb(tc_id, "terminal", None, "late-result")

            assert tc_id not in tool_call_meta
            initial_future.set_exception(RuntimeError("first failed late"))
            assert mock_sched.call_count == 2
            retry_future.set_exception(RuntimeError("retry failed late"))

        marker = tool_call_meta[tc_id]
        assert marker["failed_delivery"] is True
        assert marker["session_id"] == "session-late"
        assert marker["result"] == "late-result"
        assert marker["args"] == expected_args
        assert marker["snapshot"] == "final snapshot"
        assert mock_sched.call_count == 2

    def test_scheduler_rejection_path_unchanged_unaffected_by_observed_failure_logic(
        self, mock_conn, event_loop_fixture
    ):
        """The pre-existing scheduler-rejection retry (per make_tool_complete_cb)
        must keep working unchanged when safe_schedule_threadsafe returns None.

        This is the existing bounded-retry path the PR originally shipped — it
        must still produce the same call shape: initial + 1 retry, both
        rejected at the scheduler, then ERROR log + tool tracking cleanup.
        """
        tc_id = "tc-canonical-8"
        tool_call_ids = {"terminal": deque([tc_id])}
        tool_call_meta = {tc_id: {"args": {"command": "ls"}, "snapshot": None}}

        with (
            patch(
                "agent.async_utils.safe_schedule_threadsafe",
                return_value=None,
            ) as mock_sched,
            patch("acp_adapter.events.logger") as mock_logger,
        ):
            cb = make_tool_complete_cb(
                mock_conn,
                "session-1",
                event_loop_fixture,
                tool_call_ids,
                tool_call_meta,
            )
            cb(tc_id, "terminal", {"command": "ls"}, "payload-H")

        # Two scheduling attempts on scheduler rejection (initial + 1 retry).
        assert mock_sched.call_count == 2
        # ERROR log for permanent undelivery (scheduler path).
        assert mock_logger.error.called, (
            "permanent scheduler-rejection must still surface at ERROR"
        )
        # The canonical completion path's scheduler-rejection recovery does
        # NOT use the new failed_delivery marker — that's only for the
        # observed-future-failure case. Marker absence here is intentional.
        if tc_id in tool_call_meta:
            assert "failed_delivery" not in tool_call_meta[tc_id], (
                f"scheduler-rejection retry must not set failed_delivery "
                f"(that's the observed-failure marker); got "
                f"{tool_call_meta[tc_id]!r}"
            )

    def test_generic_send_update_recovery_is_optional_and_defaults_to_none(
        self, event_loop_fixture
    ):
        """``_send_update`` must accept an optional recovery hook with a default
        of ``None`` and remain backward-compatible for existing callers (legacy
        step callback, plan update, message callback, thinking callback).

        The existing call signature ``_send_update(conn, session_id, loop, update)``
        must continue to work unchanged so we do not regress the legacy step
        callback retry-on-scheduler-rejection behaviour.
        """
        ok_future = Future()
        ok_future.set_result(None)
        conn = MagicMock()

        # Plain 4-arg call — no recovery hook — must work and return True.
        with patch(
            "agent.async_utils.safe_schedule_threadsafe", return_value=ok_future
        ):
            result = _send_update(conn, "session-1", event_loop_fixture, {"type": "x"})

        assert result is True

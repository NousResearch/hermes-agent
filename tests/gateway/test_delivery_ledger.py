"""Tests for the gateway delivery-obligation ledger (gateway/delivery_ledger.py).

State machine, dead-owner claiming, attempts cap, stale cutoff, retention,
id stability, and the startup redelivery sweep's contract:
- pending rows redeliver plainly (send never started, no dup risk)
- attempting/failed rows carry the recovered-reply marker (honest
  at-least-once; ambiguity is labeled, never silently resent)
- rows owned by a LIVE process are never claimed
- poison rows abandon at the attempts cap / stale cutoff
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway import delivery_ledger as dl


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    """Isolated state.db per test (autouse HERMES_HOME isolation already
    redirects get_hermes_home; make the redirect explicit and per-test)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    yield


def _record(oid="ob-1", session_key="agent:main:slack:channel:C1", **kw):
    dl.record_obligation(
        obligation_id=oid,
        session_key=session_key,
        platform=kw.get("platform", "slack"),
        chat_id=kw.get("chat_id", "C1"),
        thread_id=kw.get("thread_id", "171.001"),
        content=kw.get("content", "the final answer"),
    )


def _row(oid):
    with dl._connect() as conn:
        r = conn.execute(
            """SELECT state, attempts, owner_pid, content
               FROM delivery_obligations WHERE obligation_id=?""",
            (oid,),
        ).fetchone()
    return None if r is None else {
        "state": r[0], "attempts": r[1], "owner_pid": r[2], "content": r[3],
    }


def _orphan(oid):
    """Make the row look like it belongs to a dead process."""
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, "
            "owner_started_at=1 WHERE obligation_id=?",
            (oid,),
        )


class TestStateMachine:
    def test_record_starts_pending(self):
        _record()
        assert _row("ob-1")["state"] == "pending"

    def test_full_happy_path(self):
        _record()
        dl.mark_attempting("ob-1")
        assert _row("ob-1")["state"] == "attempting"
        dl.mark_delivered("ob-1")
        assert _row("ob-1")["state"] == "delivered"

    def test_failed_records_error(self):
        _record()
        dl.mark_attempting("ob-1")
        dl.mark_failed("ob-1", "chat_not_found")
        assert _row("ob-1")["state"] == "failed"

    def test_rerecord_same_id_is_idempotent(self):
        _record()
        dl.mark_attempting("ob-1")
        _record()  # INSERT OR REPLACE resets to pending — same turn re-record
        assert _row("ob-1")["state"] == "pending"


class TestObligationId:
    def test_stable_and_distinct(self):
        a = dl.compute_obligation_id("sk1", "msg1", "hello")
        assert a == dl.compute_obligation_id("sk1", "msg1", "hello")
        # Different thread (baked into session_key) → different id. This is
        # the cron-topic collision class from the earlier outbox attempt.
        assert a != dl.compute_obligation_id("sk1:threadB", "msg1", "hello")
        assert a != dl.compute_obligation_id("sk1", "msg2", "hello")
        assert a != dl.compute_obligation_id("sk1", "msg1", "other")
        assert len(a) == 24


class TestSweep:
    def test_live_owner_rows_never_claimed(self):
        _record()  # owner = this (live) process
        assert dl.sweep_recoverable() == []

    def test_dead_owner_pending_claimed_without_marker(self):
        _record()
        _orphan("ob-1")
        claimed = dl.sweep_recoverable()
        assert len(claimed) == 1
        assert claimed[0]["needs_marker"] is False
        assert claimed[0]["attempts"] == 1
        # Claim re-stamps ownership: a second sweep in the same (live)
        # process must not double-claim.
        assert dl.sweep_recoverable() == []

    def test_dead_owner_attempting_needs_marker(self):
        _record()
        dl.mark_attempting("ob-1")
        _orphan("ob-1")
        claimed = dl.sweep_recoverable()
        assert claimed[0]["needs_marker"] is True

    def test_dead_owner_failed_needs_marker(self):
        _record()
        dl.mark_failed("ob-1", "boom")
        _orphan("ob-1")
        claimed = dl.sweep_recoverable()
        assert claimed[0]["needs_marker"] is True

    def test_delivered_rows_ignored(self):
        _record()
        dl.mark_delivered("ob-1")
        _orphan("ob-1")
        assert dl.sweep_recoverable() == []

    def test_attempts_cap_abandons(self):
        _record()
        _orphan("ob-1")
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET attempts=? WHERE obligation_id=?",
                (dl.MAX_ATTEMPTS, "ob-1"),
            )
        assert dl.sweep_recoverable() == []
        assert _row("ob-1")["state"] == "abandoned"

    def test_stale_cutoff_abandons(self):
        _record()
        _orphan("ob-1")
        future = time.time() + dl.STALE_AFTER_SECONDS + 60
        assert dl.sweep_recoverable(now=future) == []
        assert _row("ob-1")["state"] == "abandoned"


class TestPrune:
    def test_old_delivered_rows_pruned(self):
        _record()
        dl.mark_delivered("ob-1")
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET updated_at=? WHERE obligation_id=?",
                (time.time() - dl._RETENTION_SECONDS - 60, "ob-1"),
            )
        dl._prune()
        assert _row("ob-1") is None

    def test_undelivered_rows_survive_retention(self):
        _record()
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET updated_at=? WHERE obligation_id=?",
                (time.time() - dl._RETENTION_SECONDS - 60, "ob-1"),
            )
        dl._prune()
        assert _row("ob-1") is not None


class TestLedgerEnabled:
    def test_default_on(self):
        assert dl.ledger_enabled({}) is True
        assert dl.ledger_enabled({"gateway": {}}) is True

    def test_explicit_off(self):
        assert dl.ledger_enabled({"gateway": {"delivery_ledger": False}}) is False
        assert dl.ledger_enabled({"gateway": {"delivery_ledger": "off"}}) is False

    def test_truthy_strings(self):
        assert dl.ledger_enabled({"gateway": {"delivery_ledger": "true"}}) is True


class TestGatewayRedeliverySweep:
    """Drive the real GatewayRunner._redeliver_pending_obligations."""

    @staticmethod
    def _runner(adapter=None):
        from gateway.config import Platform
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.adapters = {Platform.SLACK: adapter} if adapter else {}
        _store = MagicMock()
        _store.clear_resume_pending = AsyncMock()
        _store._store = None
        runner.session_store = None
        runner._async_session_store = _store
        return runner

    @staticmethod
    def _adapter(success=True):
        adapter = MagicMock()
        adapter.send = AsyncMock(
            return_value=MagicMock(success=success, error="" if success else "nope")
        )
        return adapter

    @pytest.mark.asyncio
    async def test_pending_redelivers_plain_and_clears_resume(self):
        _record()  # pending
        _orphan("ob-1")
        adapter = self._adapter()
        runner = self._runner(adapter)

        n = await runner._redeliver_pending_obligations()

        assert n == 1
        sent = adapter.send.call_args.kwargs
        assert sent["content"] == "the final answer"  # no marker
        assert sent["metadata"] == {"thread_id": "171.001"}
        assert _row("ob-1")["state"] == "delivered"
        runner._async_session_store.clear_resume_pending.assert_awaited_once_with(
            "agent:main:slack:channel:C1"
        )

    @pytest.mark.asyncio
    async def test_attempting_redelivers_with_marker(self):
        _record()
        dl.mark_attempting("ob-1")
        _orphan("ob-1")
        adapter = self._adapter()
        runner = self._runner(adapter)

        await runner._redeliver_pending_obligations()

        sent = adapter.send.call_args.kwargs
        assert sent["content"].startswith(dl.RECOVERED_MARKER)
        assert sent["content"].endswith("the final answer")

    @pytest.mark.asyncio
    async def test_send_failure_marks_failed_for_next_boot(self):
        _record()
        _orphan("ob-1")
        runner = self._runner(self._adapter(success=False))

        n = await runner._redeliver_pending_obligations()

        assert n == 0
        assert _row("ob-1")["state"] == "failed"

    @pytest.mark.asyncio
    async def test_missing_adapter_leaves_row_recoverable(self):
        _record()
        _orphan("ob-1")
        runner = self._runner(adapter=None)  # slack not connected

        n = await runner._redeliver_pending_obligations()

        assert n == 0
        # Row still claimed by us but NOT delivered/abandoned — a later boot
        # (attempts cap permitting) can retry once the platform connects.
        assert _row("ob-1")["state"] == "pending"

    @pytest.mark.asyncio
    async def test_disabled_gate_short_circuits(self):
        _record()
        _orphan("ob-1")
        adapter = self._adapter()
        runner = self._runner(adapter)
        with patch.object(dl, "ledger_enabled", return_value=False), patch(
            "gateway.delivery_ledger.ledger_enabled", return_value=False
        ):
            n = await runner._redeliver_pending_obligations()
        assert n == 0
        adapter.send.assert_not_awaited()


class TestAttemptsOnlySpentOnRealSends:
    """``attempts`` is the redelivery budget — it must buy a send.

    ``self.adapters`` only holds a platform after its ``connect()`` succeeded,
    and the sweep claimed every dead-owner row regardless. A platform that
    failed to connect this boot therefore burned one attempt per boot while
    the caller's ``adapter is None`` branch skipped it without sending — so
    after MAX_ATTEMPTS boots the row abandoned having never been sent once,
    losing exactly the response the ledger exists to guarantee. That failure
    correlates with the crash that created the obligation: the network
    trouble that killed the send tends to still be there on the next boot.
    """

    def test_absent_platform_does_not_burn_attempts(self):
        _record(platform="telegram")
        dl.mark_attempting("ob-1")

        for _ in range(dl.MAX_ATTEMPTS + 2):
            _orphan("ob-1")
            assert dl.sweep_recoverable(deliverable_platforms={"discord"}) == []

        row = dl.debug_rows()
        assert "abandoned" not in row
        with dl._connect() as conn:
            state, attempts = conn.execute(
                "SELECT state, attempts FROM delivery_obligations "
                "WHERE obligation_id=?", ("ob-1",),
            ).fetchone()
        assert attempts == 0, "an unsendable boot must not spend the budget"
        assert state == "attempting"

    def test_row_still_delivers_once_its_platform_returns(self):
        _record(platform="telegram")
        for _ in range(dl.MAX_ATTEMPTS + 2):
            _orphan("ob-1")
            dl.sweep_recoverable(deliverable_platforms={"discord"})

        _orphan("ob-1")
        claimed = dl.sweep_recoverable(deliverable_platforms={"telegram"})
        assert len(claimed) == 1
        assert claimed[0]["attempts"] == 1

    def test_present_platform_still_claims(self):
        _record(platform="slack")
        _orphan("ob-1")
        claimed = dl.sweep_recoverable(deliverable_platforms={"slack"})
        assert len(claimed) == 1

    def test_omitting_the_filter_claims_everything(self):
        """Back-compat: existing callers pass no platform set."""
        _record(platform="telegram")
        _orphan("ob-1")
        assert len(dl.sweep_recoverable()) == 1

    def test_stale_rows_abandon_even_when_undeliverable(self):
        """The cutoff still bounds rows whose platform never returns."""
        _record(platform="telegram")
        _orphan("ob-1")
        future = time.time() + dl.STALE_AFTER_SECONDS + 10
        assert dl.sweep_recoverable(
            now=future, deliverable_platforms={"discord"}
        ) == []
        with dl._connect() as conn:
            state = conn.execute(
                "SELECT state FROM delivery_obligations WHERE obligation_id=?",
                ("ob-1",),
            ).fetchone()[0]
        assert state == "abandoned"


class TestUnconnectedPlatformKeepsItsBudget:
    """End-to-end through the real runner: boots where the platform failed to
    connect must not consume the row's redelivery budget."""

    @staticmethod
    def _runner_without_slack():
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.adapters = {}  # slack failed to connect this boot
        _store = MagicMock()
        _store.clear_resume_pending = AsyncMock()
        _store._store = None
        runner.session_store = None
        runner._async_session_store = _store
        return runner

    @pytest.mark.asyncio
    async def test_row_survives_boots_where_its_platform_is_down(self):
        _record(platform="slack")
        dl.mark_attempting("ob-1")

        for _ in range(dl.MAX_ATTEMPTS + 1):
            _orphan("ob-1")
            runner = self._runner_without_slack()
            assert await runner._redeliver_pending_obligations() == 0

        assert _row("ob-1")["state"] != "abandoned", (
            "the obligation was abandoned without a single send being attempted"
        )
        assert _row("ob-1")["attempts"] == 0

    @pytest.mark.asyncio
    async def test_delivers_when_the_platform_comes_back(self):
        from gateway.config import Platform

        _record(platform="slack")
        for _ in range(dl.MAX_ATTEMPTS + 1):
            _orphan("ob-1")
            await self._runner_without_slack()._redeliver_pending_obligations()

        _orphan("ob-1")
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=MagicMock(success=True, error=""))
        runner = self._runner_without_slack()
        runner.adapters = {Platform.SLACK: adapter}

        assert await runner._redeliver_pending_obligations() == 1
        assert _row("ob-1")["state"] == "delivered"

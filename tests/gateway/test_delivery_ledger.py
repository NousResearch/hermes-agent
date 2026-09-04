"""Tests for the gateway delivery-obligation ledger (gateway/delivery_ledger.py).

State machine, dead-owner claiming, attempts cap, stale cutoff, retention,
id stability, and the startup redelivery sweep's contract:
- pending rows redeliver plainly (send never started, no dup risk)
- attempting/failed rows carry the recovered-reply marker (honest
  at-least-once; ambiguity is labeled, never silently resent)
- rows owned by a LIVE process are never claimed
- poison rows abandon at the attempts cap / stale cutoff
"""

import os
import sqlite3
import threading
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
        adapter_profile=kw.get("adapter_profile"),
    )


def _row(oid):
    with dl._connect() as conn:
        r = conn.execute(
            """SELECT state, attempts, owner_pid, content, last_error
               FROM delivery_obligations WHERE obligation_id=?""",
            (oid,),
        ).fetchone()
    return None if r is None else {
        "state": r[0],
        "attempts": r[1],
        "owner_pid": r[2],
        "content": r[3],
        "last_error": r[4],
    }


def _blocking_probe():
    """Return a blocking ledger call and an event-loop progress witness."""
    ledger_started = threading.Event()
    event_loop_progressed = threading.Event()
    blocked_event_loop = []

    def _slow_ledger_call(*args, **kwargs):
        ledger_started.set()
        # Generous timeout: a genuinely blocked loop can never set the event
        # (the witness coroutine cannot run), so a longer wait only guards
        # against loaded-CI scheduling flake, not against missing the bug.
        if not event_loop_progressed.wait(timeout=5.0):
            blocked_event_loop.append(True)

    async def _event_loop_witness():
        import asyncio

        deadline = asyncio.get_running_loop().time() + 10
        while not ledger_started.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("ledger call never started")
            await asyncio.sleep(0)
        event_loop_progressed.set()

    return _slow_ledger_call, _event_loop_witness, blocked_event_loop


def _orphan(oid):
    """Make the row look like it belongs to a dead process."""
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, "
            "owner_started_at=1 WHERE obligation_id=?",
            (oid,),
        )


class TestSchemaMigration:
    def test_adds_adapter_profile_to_existing_ledger(self):
        conn = sqlite3.connect(dl._db_path())
        try:
            conn.execute(
                """CREATE TABLE delivery_obligations (
                    obligation_id TEXT PRIMARY KEY,
                    session_key TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    thread_id TEXT,
                    content TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    owner_pid INTEGER,
                    owner_started_at INTEGER,
                    last_error TEXT
                )"""
            )
            dl._initialize_schema(conn)
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(delivery_obligations)")
            }
        finally:
            conn.close()

        assert "adapter_profile" in columns


class TestStateMachine:
    def test_record_starts_pending(self):
        _record()
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


class TestRuntimeFailedSweep:
    """A live gateway may reclaim only its own transient reconnect failures."""

    def test_claims_current_process_send_path_degraded_row(self):
        _record(platform="telegram")
        dl.mark_failed("ob-1", "send_path_degraded")

        claimed = dl.sweep_failed_for_runtime("telegram")

        assert len(claimed) == 1
        assert claimed[0]["needs_marker"] is True
        assert claimed[0]["attempts"] == 1
        assert _row("ob-1")["state"] == "attempting"

    def test_permanent_failure_is_not_claimed(self):
        _record(platform="telegram")
        dl.mark_failed("ob-1", "Forbidden: bot was blocked by the user")

        assert dl.sweep_failed_for_runtime("telegram") == []
        assert _row("ob-1")["state"] == "failed"
        assert _row("ob-1")["attempts"] == 0

    def test_claim_is_platform_scoped_and_not_reclaimed_while_attempting(self):
        _record(platform="telegram")
        dl.mark_failed("ob-1", "send_path_degraded")
        _record(
            oid="ob-2",
            session_key="agent:main:slack:channel:C2",
            platform="slack",
            chat_id="C2",
        )
        dl.mark_failed("ob-2", "send_path_degraded")

        claimed = dl.sweep_failed_for_runtime("telegram")

        assert [row["obligation_id"] for row in claimed] == ["ob-1"]
        assert dl.sweep_failed_for_runtime("telegram") == []
        assert _row("ob-2")["state"] == "failed"
        assert _row("ob-2")["attempts"] == 0

    def test_other_live_owner_is_not_claimed_or_abandoned(self, monkeypatch):
        _record(platform="telegram")
        dl.mark_failed("ob-1", "send_path_degraded")
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET owner_pid=?, "
                "owner_started_at=?, attempts=? WHERE obligation_id=?",
                (12345, 101, dl.MAX_ATTEMPTS, "ob-1"),
            )
        monkeypatch.setattr(dl, "_owner_stamp", lambda: (54321, 202))

        assert dl.sweep_failed_for_runtime("telegram") == []
        assert _row("ob-1")["state"] == "failed"
        assert _row("ob-1")["attempts"] == dl.MAX_ATTEMPTS

    def test_unowned_row_is_not_claimed(self):
        _record(platform="telegram")
        dl.mark_failed("ob-1", "send_path_degraded")
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET owner_pid=NULL, "
                "owner_started_at=NULL WHERE obligation_id=?",
                ("ob-1",),
            )

        assert dl.sweep_failed_for_runtime("telegram") == []
        assert _row("ob-1")["state"] == "failed"

    def test_missing_current_process_start_stamp_fails_closed(self, monkeypatch):
        _record(platform="telegram")
        dl.mark_failed("ob-1", "send_path_degraded")
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET owner_started_at=NULL "
                "WHERE obligation_id=?",
                ("ob-1",),
            )
        monkeypatch.setattr(dl, "_owner_stamp", lambda: (os.getpid(), None))

        assert dl.sweep_failed_for_runtime("telegram") == []
        assert _row("ob-1")["state"] == "failed"

    def test_same_pid_with_different_start_stamp_is_not_claimed(self, monkeypatch):
        _record(platform="telegram")
        dl.mark_failed("ob-1", "send_path_degraded")
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET owner_pid=?, owner_started_at=? "
                "WHERE obligation_id=?",
                (os.getpid(), 101, "ob-1"),
            )
        monkeypatch.setattr(dl, "_owner_stamp", lambda: (os.getpid(), 202))

        assert dl.sweep_failed_for_runtime("telegram") == []
        assert _row("ob-1")["state"] == "failed"

    def test_profile_scope_never_claims_another_bot_identity(self):
        _record(platform="telegram")
        dl.mark_failed("ob-1", "send_path_degraded")
        _record(
            oid="ob-2",
            session_key="agent:reviewer:telegram:dm:C2",
            platform="telegram",
            chat_id="C2",
            adapter_profile="reviewer",
        )
        dl.mark_failed("ob-2", "send_path_degraded")

        claimed = dl.sweep_failed_for_runtime("telegram", profile="reviewer")

        assert [row["obligation_id"] for row in claimed] == ["ob-2"]
        assert claimed[0]["profile"] == "reviewer"
        assert _row("ob-1")["state"] == "failed"

    def test_current_owner_row_at_attempt_cap_is_abandoned(self):
        _record(platform="telegram")
        dl.mark_failed("ob-1", "send_path_degraded")
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET attempts=? WHERE obligation_id=?",
                (dl.MAX_ATTEMPTS, "ob-1"),
            )

        assert dl.sweep_failed_for_runtime("telegram") == []
        assert _row("ob-1")["state"] == "abandoned"

    def test_delivered_row_is_never_reclaimed_by_reconnect_sweep(self):
        """Idempotency: once delivered, a reconnect sweep must not re-send.

        Strongest form: the row previously failed with the allowlisted
        transient error and is force-restamped with that ``last_error`` even
        after delivery, so the ONLY guard standing between the sweep and a
        duplicate send is the ``state='delivered'`` filter itself.
        """
        _record(platform="telegram")
        dl.mark_failed("ob-1", "send_path_degraded")

        # First reconnect legitimately claims and (successfully) redelivers.
        assert len(dl.sweep_failed_for_runtime("telegram")) == 1
        dl.mark_delivered("ob-1")
        # Simulate a mark_delivered that leaves the retryable error string
        # behind: even then, a delivered row must never be reclaimed.
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET last_error=? "
                "WHERE obligation_id=?",
                ("send_path_degraded", "ob-1"),
            )

        assert dl.sweep_failed_for_runtime("telegram") == []
        row = _row("ob-1")
        assert row is not None
        assert row["state"] == "delivered"
        assert row["attempts"] == 1

    def test_current_owner_stale_row_is_abandoned(self):
        _record(platform="telegram")
        dl.mark_failed("ob-1", "send_path_degraded")
        now = time.time()
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET created_at=? WHERE obligation_id=?",
                (now - dl.STALE_AFTER_SECONDS - 1, "ob-1"),
            )

        assert dl.sweep_failed_for_runtime("telegram", now=now) == []
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


class TestLedgerEnabled:
    def test_default_on(self):
        assert dl.ledger_enabled({}) is True
        assert dl.ledger_enabled({"gateway": {}}) is True


class TestGatewayRedeliverySweep:
    """Drive the real GatewayRunner._redeliver_pending_obligations."""

    @staticmethod
    def _runner(adapter=None):
        from gateway.config import Platform
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.adapters = {Platform.SLACK: adapter} if adapter else {}
        runner._profile_adapters = {}
        runner._active_profile_name = lambda: "default"
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
    async def test_startup_redelivery_uses_persisted_transport_owner(self):
        from gateway.config import Platform

        _record(
            session_key="agent:routed-profile:slack:channel:C1",
            adapter_profile="credential-owner",
        )
        _orphan("ob-1")
        default_adapter = self._adapter()
        owner_adapter = self._adapter()
        runner = self._runner(default_adapter)
        runner._profile_adapters = {
            "credential-owner": {Platform.SLACK: owner_adapter}
        }

        n = await runner._redeliver_pending_obligations()

        assert n == 1
        default_adapter.send.assert_not_awaited()
        owner_adapter.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_startup_does_not_claim_disconnected_transport_owner(self):
        _record(
            session_key="agent:routed-profile:slack:channel:C1",
            adapter_profile="credential-owner",
        )
        _orphan("ob-1")
        default_adapter = self._adapter()
        runner = self._runner(default_adapter)

        n = await runner._redeliver_pending_obligations()

        assert n == 0
        default_adapter.send.assert_not_awaited()
        assert _row("ob-1")["state"] == "pending"
        assert _row("ob-1")["attempts"] == 0

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
    async def test_runtime_failed_redelivery_clears_resume_before_send(self):
        from gateway.config import Platform

        _record(platform="slack")
        dl.mark_failed("ob-1", "send_path_degraded")
        adapter = self._adapter()
        runner = self._runner(adapter)

        n = await runner._redeliver_failed_obligations_for_platform(Platform.SLACK)

        assert n == 1
        runner._async_session_store.clear_resume_pending.assert_awaited_once_with(
            "agent:main:slack:channel:C1"
        )
        assert adapter.send.await_count == 1
        assert adapter.send.call_args.kwargs["content"].startswith(
            dl.RECONNECTED_MARKER
        )
        assert _row("ob-1")["state"] == "delivered"

    @pytest.mark.asyncio
    async def test_runtime_profile_redelivery_uses_matching_bot_adapter(self):
        from gateway.config import Platform

        _record(
            session_key="agent:reviewer:slack:channel:C1",
            platform="slack",
            adapter_profile="reviewer",
        )
        dl.mark_failed("ob-1", "send_path_degraded")
        default_adapter = self._adapter()
        reviewer_adapter = self._adapter()
        runner = self._runner(default_adapter)
        runner._profile_adapters = {
            "reviewer": {Platform.SLACK: reviewer_adapter}
        }

        n = await runner._redeliver_failed_obligations_for_platform(
            Platform.SLACK, profile="reviewer"
        )

        assert n == 1
        default_adapter.send.assert_not_awaited()
        reviewer_adapter.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_runtime_missing_adapter_releases_unsent_claim(self):
        from gateway.config import Platform

        _record(platform="slack")
        dl.mark_failed("ob-1", "send_path_degraded")
        runner = self._runner()

        n = await runner._redeliver_failed_obligations_for_platform(Platform.SLACK)

        assert n == 0
        assert _row("ob-1")["state"] == "failed"
        assert _row("ob-1")["attempts"] == 0
        assert _row("ob-1")["last_error"] == "send_path_degraded"

    @pytest.mark.asyncio
    async def test_runtime_clear_failure_does_not_send_or_lose_retry(self):
        from gateway.config import Platform

        _record(platform="slack")
        dl.mark_failed("ob-1", "send_path_degraded")
        adapter = self._adapter()
        runner = self._runner(adapter)
        runner._async_session_store.clear_resume_pending.side_effect = RuntimeError(
            "session store unavailable"
        )

        n = await runner._redeliver_failed_obligations_for_platform(Platform.SLACK)

        assert n == 0
        adapter.send.assert_not_awaited()
        assert _row("ob-1")["state"] == "failed"
        assert _row("ob-1")["attempts"] == 0
        assert _row("ob-1")["last_error"] == "send_path_degraded"

    @pytest.mark.parametrize(
        ("send_success", "ledger_method"),
        [(True, "mark_delivered"), (False, "mark_failed")],
    )
    @pytest.mark.asyncio
    async def test_slow_state_update_does_not_block_event_loop(
        self, send_success, ledger_method
    ):
        import asyncio

        _record()
        _orphan("ob-1")
        runner = self._runner(self._adapter(success=send_success))
        slow_update, event_loop_witness, blocked_event_loop = _blocking_probe()

        with patch.object(dl, ledger_method, side_effect=slow_update):
            await asyncio.gather(
                runner._redeliver_pending_obligations(), event_loop_witness()
            )

        assert blocked_event_loop == []

    @pytest.mark.asyncio
    async def test_clear_resume_pending_before_send_so_a_hang_cannot_also_resume(
        self,
    ):
        """A hung redelivery send must still clear resume_pending.

        Otherwise a timed-out startup-restore gate would schedule resume and
        replay a turn whose answer is already in the ledger (#91969).
        """
        import asyncio

        _record()
        _orphan("ob-1")
        hang = asyncio.Event()

        async def hanging_send(**_kwargs):
            await hang.wait()
            return MagicMock(success=True, error="")

        adapter = MagicMock()
        adapter.send = hanging_send
        runner = self._runner(adapter)
        task = asyncio.create_task(runner._redeliver_pending_obligations())

        deadline = asyncio.get_running_loop().time() + 2
        while runner._async_session_store.clear_resume_pending.await_count == 0:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("resume_pending was not cleared before send")
            await asyncio.sleep(0)

        runner._async_session_store.clear_resume_pending.assert_awaited_once_with(
            "agent:main:slack:channel:C1"
        )
        assert not task.done()

        hang.set()
        assert await task == 1


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



class TestOwnerAlivePidProbe:
    """_owner_alive's no-start-time fallback must route through
    gateway.status._pid_exists, never a raw ``os.kill(pid, 0)`` probe.

    On Windows ``os.kill(pid, 0)`` is NOT a no-op: CPython maps sig=0 to
    ``GenerateConsoleCtrlEvent(0, pid)`` (bpo-14484), so probing a LIVE pid
    whose start time psutil could not read would Ctrl+C its console group.
    Pattern per the windows-native-support reference: patch
    ``gateway.status._pid_exists``, not ``os.kill``.
    """

    def _no_start_time(self, monkeypatch):
        from gateway import status

        monkeypatch.setattr(status, "get_process_start_time", lambda pid: None)

    def test_alive_when_pid_exists(self, monkeypatch):
        from gateway import status

        self._no_start_time(monkeypatch)
        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        assert dl._owner_alive(12345, 999) is True

    def test_dead_when_pid_gone(self, monkeypatch):
        from gateway import status

        self._no_start_time(monkeypatch)
        monkeypatch.setattr(status, "_pid_exists", lambda pid: False)
        assert dl._owner_alive(12345, 999) is False

    def test_raw_os_kill_probe_never_used(self, monkeypatch):
        """Regression guard: the probe must not touch os.kill when
        gateway.status._pid_exists is importable (i.e. always in-tree)."""
        from gateway import status

        self._no_start_time(monkeypatch)
        calls = []
        monkeypatch.setattr(status, "_pid_exists", lambda pid: calls.append(pid) or True)
        monkeypatch.setattr(
            dl.os, "kill", lambda *a, **k: (_ for _ in ()).throw(AssertionError("raw os.kill probe used"))
        )
        assert dl._owner_alive(4242, 999) is True
        assert calls == [4242]

    def test_probe_exception_means_dead(self, monkeypatch):
        from gateway import status

        self._no_start_time(monkeypatch)

        def boom(pid):
            raise RuntimeError("probe blew up")

        monkeypatch.setattr(status, "_pid_exists", boom)
        assert dl._owner_alive(12345, 999) is False


class TestRecoveredMediaDelivery:
    """#99846: recovered finals must honor the same explicit media contract
    as live delivery. A persisted final containing ``MEDIA:<path>`` used to be
    replayed through ``adapter.send()`` as text only — the obligation was
    marked delivered without the attachment ever being uploaded."""

    @staticmethod
    def _media_adapter(tmp_path, *, send_success=True):
        """Adapter double whose media send methods record calls; the real
        ``send_image_file``/``send_document``/... are never exercised because
        the adapters in these tests are MagicMocks — what matters is that
        redelivery dispatches to the RIGHT method with the RIGHT path."""
        adapter = MagicMock()
        adapter.name = "slack"
        adapter.platform = "slack"
        adapter.send = AsyncMock(
            return_value=MagicMock(success=send_success, error="" if send_success else "nope")
        )
        for m in (
            "send_image_file",
            "send_document",
            "send_voice",
            "send_video",
            "send_multiple_images",
        ):
            setattr(adapter, m, AsyncMock(return_value=MagicMock(success=True, error="")))
        return adapter

    @staticmethod
    def _runner(adapter):
        from gateway.config import Platform
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.adapters = {Platform.SLACK: adapter}
        runner._profile_adapters = {}
        runner._active_profile_name = lambda: "default"
        _store = MagicMock()
        _store.clear_resume_pending = AsyncMock()
        _store._store = None
        runner.session_store = None
        runner._async_session_store = _store
        return runner

    @pytest.mark.asyncio
    async def test_recovered_image_final_calls_send_image_file(self, tmp_path):
        img = tmp_path / "proof.png"
        img.write_bytes(b"\x89PNG fake")
        _record(content=f"proof attached\nMEDIA:{img}")
        _orphan("ob-1")
        adapter = self._media_adapter(tmp_path)
        runner = self._runner(adapter)

        n = await runner._redeliver_pending_obligations()

        assert n == 1
        # The text part is delivered WITHOUT the MEDIA: directive...
        sent = adapter.send.call_args.kwargs
        assert sent["content"] == "proof attached"
        # ...and the attachment goes out through the image sender.
        adapter.send_image_file.assert_awaited_once()
        img_kwargs = adapter.send_image_file.call_args.kwargs
        assert img_kwargs["image_path"] == str(img)
        assert img_kwargs["chat_id"] == "C1"
        assert _row("ob-1")["state"] == "delivered"

    @pytest.mark.asyncio
    async def test_recovered_image_batch_uses_send_multiple_images(self, tmp_path):
        img1 = tmp_path / "a.jpg"
        img1.write_bytes(b"jpg1")
        img2 = tmp_path / "b.png"
        img2.write_bytes(b"png1")
        _record(content=f"MEDIA:{img1}\nMEDIA:{img2}")
        _orphan("ob-1")
        adapter = self._media_adapter(tmp_path)
        runner = self._runner(adapter)

        n = await runner._redeliver_pending_obligations()

        assert n == 1
        adapter.send_multiple_images.assert_awaited_once()
        batch = adapter.send_multiple_images.call_args.kwargs["images"]
        assert [u for u, _ in batch] == [f"file://{img1}", f"file://{img2}"]
        adapter.send_image_file.assert_not_awaited()
        # Text reduced to empty after extraction + no fallback send needed:
        # the batch itself carries the delivery.
        adapter.send.assert_not_awaited()
        assert _row("ob-1")["state"] == "delivered"

    @pytest.mark.asyncio
    async def test_recovered_document_routes_to_send_document(self, tmp_path):
        doc = tmp_path / "report.pdf"
        doc.write_bytes(b"%PDF fake")
        _record(content=f"report ready\nMEDIA:{doc}")
        _orphan("ob-1")
        adapter = self._media_adapter(tmp_path)
        runner = self._runner(adapter)

        await runner._redeliver_pending_obligations()

        adapter.send_document.assert_awaited_once()
        assert adapter.send_document.call_args.kwargs["file_path"] == str(doc)
        adapter.send_image_file.assert_not_awaited()
        adapter.send_voice.assert_not_awaited()
        assert _row("ob-1")["state"] == "delivered"

    @pytest.mark.asyncio
    async def test_recovered_video_routes_to_send_video(self, tmp_path):
        vid = tmp_path / "clip.mp4"
        vid.write_bytes(b"mp4 fake")
        _record(content=f"MEDIA:{vid}")
        _orphan("ob-1")
        adapter = self._media_adapter(tmp_path)
        runner = self._runner(adapter)

        await runner._redeliver_pending_obligations()

        adapter.send_video.assert_awaited_once()
        assert adapter.send_video.call_args.kwargs["video_path"] == str(vid)
        assert _row("ob-1")["state"] == "delivered"

    @pytest.mark.asyncio
    async def test_recovered_audio_routes_to_send_voice(self, tmp_path):
        audio = tmp_path / "note.mp3"
        audio.write_bytes(b"mp3 fake")
        _record(content=f"MEDIA:{audio}")
        _orphan("ob-1")
        adapter = self._media_adapter(tmp_path)
        runner = self._runner(adapter)

        await runner._redeliver_pending_obligations()

        # Non-Telegram platform: every audio ext routes through the audio sender.
        adapter.send_voice.assert_awaited_once()
        assert adapter.send_voice.call_args.kwargs["audio_path"] == str(audio)
        adapter.send_document.assert_not_awaited()
        assert _row("ob-1")["state"] == "delivered"

    @pytest.mark.asyncio
    async def test_recovered_voice_directive_sets_is_voice(self, tmp_path):
        audio = tmp_path / "note.mp3"
        audio.write_bytes(b"mp3 fake")
        _record(
            content=f"[[audio_as_voice]]\nMEDIA:{audio}",
        )
        _orphan("ob-1")
        adapter = self._media_adapter(tmp_path)
        runner = self._runner(adapter)

        await runner._redeliver_pending_obligations()

        adapter.send_voice.assert_awaited_once()
        assert adapter.send_voice.call_args.kwargs["is_voice"] is True
        assert _row("ob-1")["state"] == "delivered"

    @pytest.mark.asyncio
    async def test_recovered_unsafe_media_path_is_dropped_not_uploaded(self, tmp_path):
        # Path does not exist → validate_media_delivery_path rejects it → no
        # upload, and the directive must NOT leak into the delivered text.
        _record(content=f"see file\nMEDIA:{tmp_path / 'missing.png'}")
        _orphan("ob-1")
        adapter = self._media_adapter(tmp_path)
        runner = self._runner(adapter)

        n = await runner._redeliver_pending_obligations()

        assert n == 1
        adapter.send_image_file.assert_not_awaited()
        adapter.send_multiple_images.assert_not_awaited()
        sent = adapter.send.call_args.kwargs
        assert "MEDIA:" not in sent["content"]
        assert _row("ob-1")["state"] == "delivered"

    @pytest.mark.asyncio
    async def test_recovered_media_failure_marks_row_failed_not_delivered(self, tmp_path):
        img = tmp_path / "proof.png"
        img.write_bytes(b"\x89PNG fake")
        _record(content=f"MEDIA:{img}")
        _orphan("ob-1")
        adapter = self._media_adapter(tmp_path)
        # Text send succeeds but the attachment upload fails.
        adapter.send_image_file = AsyncMock(
            return_value=MagicMock(success=False, error="upload denied")
        )
        runner = self._runner(adapter)

        n = await runner._redeliver_pending_obligations()

        # Nothing was deliverable in full — the obligation must stay retryable.
        assert n == 0
        adapter.send_image_file.assert_awaited_once()
        assert _row("ob-1")["state"] == "failed"
        assert _row("ob-1")["last_error"]

    @pytest.mark.asyncio
    async def test_recovered_media_send_raises_marks_row_failed(self, tmp_path):
        doc = tmp_path / "report.pdf"
        doc.write_bytes(b"%PDF fake")
        _record(content=f"MEDIA:{doc}")
        _orphan("ob-1")
        adapter = self._media_adapter(tmp_path)
        adapter.send_document = AsyncMock(side_effect=RuntimeError("boom"))
        runner = self._runner(adapter)

        n = await runner._redeliver_pending_obligations()

        assert n == 0
        assert _row("ob-1")["state"] == "failed"

    @pytest.mark.asyncio
    async def test_recovered_media_failure_with_text_still_marks_failed(self, tmp_path):
        """Text + attachment where the attachment fails: the text may have
        reached the user, but the obligation is NOT fully delivered — partial
        delivery must not falsely acknowledge completion (#99846)."""
        img = tmp_path / "proof.png"
        img.write_bytes(b"\x89PNG fake")
        _record(content=f"proof attached\nMEDIA:{img}")
        _orphan("ob-1")
        adapter = self._media_adapter(tmp_path)
        adapter.send_image_file = AsyncMock(
            return_value=MagicMock(success=False, error="flood wait")
        )
        runner = self._runner(adapter)

        n = await runner._redeliver_pending_obligations()

        assert n == 0
        adapter.send.assert_awaited_once()  # text went out first
        assert _row("ob-1")["state"] == "failed"

    @pytest.mark.asyncio
    async def test_recovered_marker_prepend_survives_media_extraction(self, tmp_path):
        """needs_marker rows (attempting/failed) carry the recovered-reply
        marker — it must land on the cleaned TEXT, after extraction, so it can
        never be eaten by MEDIA parsing."""
        img = tmp_path / "proof.png"
        img.write_bytes(b"\x89PNG fake")
        _record(content=f"proof attached\nMEDIA:{img}")
        dl.mark_attempting("ob-1")
        _orphan("ob-1")
        adapter = self._media_adapter(tmp_path)
        runner = self._runner(adapter)

        n = await runner._redeliver_pending_obligations()

        assert n == 1
        sent = adapter.send.call_args.kwargs
        assert sent["content"].startswith(dl.RECOVERED_MARKER)
        assert sent["content"].endswith("proof attached")
        assert "MEDIA:" not in sent["content"]
        adapter.send_image_file.assert_awaited_once()
        assert _row("ob-1")["state"] == "delivered"

    @pytest.mark.asyncio
    async def test_recovered_as_document_routes_image_to_send_document(self, tmp_path):
        img = tmp_path / "big.png"
        img.write_bytes(b"\x89PNG fake")
        _record(content=f"[[as_document]]\nMEDIA:{img}")
        _orphan("ob-1")
        adapter = self._media_adapter(tmp_path)
        runner = self._runner(adapter)

        await runner._redeliver_pending_obligations()

        # [[as_document]] preserves bytes: image-ext files skip the photo path.
        adapter.send_document.assert_awaited_once()
        assert adapter.send_document.call_args.kwargs["file_path"] == str(img)
        adapter.send_image_file.assert_not_awaited()
        adapter.send_multiple_images.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_recovered_only_dangling_media_tag_is_not_falsely_acknowledged(self, tmp_path):
        """Content whose every deliverable part evaporates (only a MEDIA tag
        with a nonexistent path): send nothing (never echo host paths into
        chat) and do NOT mark delivered — the obligation stays retryable and
        the attempts cap retires it (#99846, mirroring the live path's refusal
        to leak paths)."""
        _record(content=f"MEDIA:{tmp_path / 'gone.png'}")
        _orphan("ob-1")
        adapter = self._media_adapter(tmp_path)
        runner = self._runner(adapter)

        n = await runner._redeliver_pending_obligations()

        assert n == 0
        adapter.send.assert_not_awaited()
        adapter.send_image_file.assert_not_awaited()
        assert _row("ob-1")["state"] == "failed"

    @pytest.mark.asyncio
    async def test_recovered_plain_text_final_unchanged(self, tmp_path):
        """No media → behavior identical to the pre-fix redelivery path."""
        _record(content="the final answer")
        _orphan("ob-1")
        adapter = self._media_adapter(tmp_path)
        runner = self._runner(adapter)

        n = await runner._redeliver_pending_obligations()

        assert n == 1
        sent = adapter.send.call_args.kwargs
        assert sent["content"] == "the final answer"
        adapter.send_image_file.assert_not_awaited()
        adapter.send_document.assert_not_awaited()
        adapter.send_voice.assert_not_awaited()
        assert _row("ob-1")["state"] == "delivered"

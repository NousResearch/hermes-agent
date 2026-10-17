"""Tests for the external drain-control marker contract + gateway state machine.

Task 2.2/2.3. Two layers:
  * drain_control.py — the presence-based marker contract (write/clear/read,
    HERMES_HOME-scoped, never-raises).
  * GatewayRunner enter/exit/watcher + the new-turn accept gate — the
    reversible state machine driven by the marker.

Mocked tests are necessary-not-sufficient here (the HARD live-validation gate,
Q-B, exercises a real `hermes gateway run`); these lock the unit contract.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import gateway.drain_control as dc
import gateway.session_runtime_workers as runtime_workers
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_contract import Principal, SessionRef, Submission
from hermes_state import SessionDB
from hermes_state_runtime import (
    RuntimeStoreError,
    admit_session_input,
    begin_runtime_epoch,
    find_session_admission,
    get_session_admission,
    settle_session_input,
)
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source
from datetime import UTC


# ---------------------------------------------------------------------------
# Marker contract (drain_control.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


class TestMarkerContract:
    def test_absent_by_default(self, home):
        assert dc.drain_requested() is False
        assert dc.read_drain_request() is None

    def test_write_then_present(self, home):
        payload = dc.write_drain_request(principal="nas")
        assert dc.drain_requested() is True
        assert payload["action"] == "drain"
        assert payload["principal"] == "nas"
        body = dc.read_drain_request()
        assert body is not None and body["principal"] == "nas"


class TestSuppressNotification:
    """The generic suppress_notification flag on the drain marker.

    Gates ONLY the gateway's home-channel shutdown broadcast (NAS auto-update
    sets it true). Default-false so legacy/operator drains behave as before.
    The reader reuses the NS-570 epoch-staleness check so an orphaned marker
    can never silence a fresh gateway.
    """

    def test_default_false(self, home):
        payload = dc.write_drain_request(principal="nas")
        assert payload["suppress_notification"] is False
        assert dc.drain_notification_suppressed() is False

    def test_flag_round_trips_true(self, home):
        payload = dc.write_drain_request(principal="nas", suppress_notification=True)
        assert payload["suppress_notification"] is True
        body = dc.read_drain_request()
        assert body is not None and body["suppress_notification"] is True
        assert dc.drain_notification_suppressed() is True


# ---------------------------------------------------------------------------
# Instantiation-epoch staleness (NS-570: orphaned marker on durable volume)
# ---------------------------------------------------------------------------


class TestInstantiationEpoch:
    def test_write_stamps_current_epoch(self, home):
        payload = dc.write_drain_request(principal="nas")
        assert payload["epoch"] == dc.current_instantiation_epoch()
        body = dc.read_drain_request()
        assert body is not None and body["epoch"] == dc.current_instantiation_epoch()


    def test_marker_from_prior_instantiation_reads_as_absent(self, home, monkeypatch):
        # THE NS-570 REGRESSION. A begin-drain marker written by a PREVIOUS
        # container/VM instantiation survives on the durable HERMES_HOME volume
        # across a machine restart. The freshly-restarted gateway (new epoch)
        # must treat it as absent, NOT re-engage drain.
        monkeypatch.setattr(dc, "current_instantiation_epoch", lambda: "epoch-OLD")
        dc.write_drain_request(principal="nas")  # stamps "epoch-OLD"
        assert dc.drain_requested() is True  # same epoch → active

        # Simulate the restart: a brand-new instantiation epoch.
        monkeypatch.setattr(dc, "current_instantiation_epoch", lambda: "epoch-NEW")
        # The marker file is still physically present on the volume…
        assert dc.drain_request_path().exists() is True
        # …but it is ignored because its epoch belongs to a prior instantiation.
        assert dc.drain_requested() is False


    def test_current_epoch_empty_when_proc_unreadable(self, monkeypatch):
        # When neither /proc identity source is readable, the epoch is "" so
        # the staleness check is disabled rather than crashing.
        from pathlib import Path as _P

        orig_read_text = _P.read_text

        def _boom(self, *a, **k):
            if str(self).startswith("/proc/"):
                raise OSError("no /proc")
            return orig_read_text(self, *a, **k)

        dc.current_instantiation_epoch.cache_clear()
        monkeypatch.setattr(_P, "read_text", _boom)
        try:
            assert dc.current_instantiation_epoch() == ""
        finally:
            dc.current_instantiation_epoch.cache_clear()


# ---------------------------------------------------------------------------
# requested_at max-age (#85433: same-epoch orphaned marker, no restart)
# ---------------------------------------------------------------------------


class TestMarkerMaxAge:

    def test_expired_marker_reads_as_absent(self, home):
        # THE #85433 REGRESSION. A drain-gated action completes WITHOUT a
        # machine restart, so the epoch still matches — but the writer never
        # cancelled the drain. The orphan must not wedge the gateway forever.
        from datetime import datetime, timedelta, timezone

        dc.write_drain_request(principal="nas", suppress_notification=True)
        body = dc.read_drain_request()
        assert body is not None
        body["requested_at"] = (
            datetime.now(UTC)
            - timedelta(seconds=dc.DRAIN_REQUEST_MAX_AGE_SECONDS + 60)
        ).isoformat()
        dc.drain_request_path().write_text(json.dumps(body), encoding="utf-8")

        # The marker file is still physically present, with the CURRENT epoch…
        assert dc.drain_request_path().exists() is True
        # …but it is ignored because it outlived any legitimate drain.
        assert dc.drain_requested() is False
        # The suppression flag of an expired orphan is likewise ignored.
        assert dc.drain_notification_suppressed() is False

    def test_marker_without_timestamp_still_honoured(self, home):
        # Leniency contract: no requested_at (legacy/corrupt body) must fail
        # toward quiescing, exactly like the epoch check.
        payload = {"action": "drain", "epoch": dc.current_instantiation_epoch()}
        dc.drain_request_path().write_text(json.dumps(payload), encoding="utf-8")
        assert dc.drain_requested() is True

    def test_unparseable_timestamp_still_honoured(self, home):
        payload = {
            "action": "drain",
            "epoch": dc.current_instantiation_epoch(),
            "requested_at": "not-a-timestamp",
        }
        dc.drain_request_path().write_text(json.dumps(payload), encoding="utf-8")
        assert dc.drain_requested() is True

    def test_naive_timestamp_treated_as_utc(self, home):
        # A writer that stamped a tz-naive ISO string must still expire.
        from datetime import datetime, timedelta, timezone

        stale_naive = (
            datetime.now(UTC)
            - timedelta(seconds=dc.DRAIN_REQUEST_MAX_AGE_SECONDS + 60)
        ).replace(tzinfo=None)
        payload = {
            "action": "drain",
            "epoch": dc.current_instantiation_epoch(),
            "requested_at": stale_naive.isoformat(),
        }
        dc.drain_request_path().write_text(json.dumps(payload), encoding="utf-8")
        assert dc.drain_requested() is False

    def test_expiry_warning_logged_once_per_marker(self, home, caplog):
        # The watcher polls every 1s; an expired orphan must warn ONCE, not
        # once per tick (~86k/day). A refreshed marker that expires again
        # warns again (new requested_at).
        import logging
        from datetime import datetime, timedelta, timezone

        def _write_expired(offset_seconds):
            dc.write_drain_request(principal="nas")
            body = dc.read_drain_request()
            assert body is not None
            body["requested_at"] = (
                datetime.now(UTC)
                - timedelta(seconds=dc.DRAIN_REQUEST_MAX_AGE_SECONDS + offset_seconds)
            ).isoformat()
            dc.drain_request_path().write_text(json.dumps(body), encoding="utf-8")

        _write_expired(60)
        with caplog.at_level(logging.WARNING, logger="gateway.drain_control"):
            assert dc.drain_requested() is False
            assert dc.drain_requested() is False  # second poll tick
            assert dc.drain_requested() is False  # third poll tick
        expired_logs = [r for r in caplog.records if "expired drain marker" in r.message]
        assert len(expired_logs) == 1

        caplog.clear()
        _write_expired(120)  # a DIFFERENT requested_at that is also expired
        with caplog.at_level(logging.WARNING, logger="gateway.drain_control"):
            assert dc.drain_requested() is False
        expired_logs = [r for r in caplog.records if "expired drain marker" in r.message]
        assert len(expired_logs) == 1

    def test_rewrite_refreshes_the_clock(self, home):
        # The sanctioned keep-alive: re-writing the marker bumps requested_at,
        # so a deliberately long drain stays honoured.
        from datetime import datetime, timedelta, timezone

        dc.write_drain_request(principal="nas")
        body = dc.read_drain_request()
        assert body is not None
        body["requested_at"] = (
            datetime.now(UTC)
            - timedelta(seconds=dc.DRAIN_REQUEST_MAX_AGE_SECONDS + 60)
        ).isoformat()
        dc.drain_request_path().write_text(json.dumps(body), encoding="utf-8")
        assert dc.drain_requested() is False  # expired…
        dc.write_drain_request(principal="nas")  # …keep-alive re-write
        assert dc.drain_requested() is True


# ---------------------------------------------------------------------------
# Gateway state machine (enter / exit / idempotency)
# ---------------------------------------------------------------------------


def _drain_runner():
    runner, adapter = make_restart_runner()
    runner._external_drain_active = False
    # Bind the real methods under test.
    runner._enter_external_drain = GatewayRunner._enter_external_drain.__get__(
        runner, GatewayRunner
    )
    runner._exit_external_drain = GatewayRunner._exit_external_drain.__get__(
        runner, GatewayRunner
    )
    return runner, adapter


class TestDrainStateMachine:




    def test_exit_during_shutdown_does_not_revert_to_running(self):
        runner, _ = _drain_runner()
        runner._enter_external_drain()
        runner._update_runtime_status.reset_mock()
        # A shutdown drain is now in progress — exit must NOT resurrect running.
        runner._draining = True
        runner._exit_external_drain()
        assert runner._external_drain_active is False
        runner._update_runtime_status.assert_not_called()


# ---------------------------------------------------------------------------
# Watcher reconciliation
# ---------------------------------------------------------------------------


class TestDrainWatcher:

    @pytest.mark.asyncio
    async def test_watcher_enters_then_exits_with_marker(self, home):
        runner, _ = _drain_runner()
        runner._drain_control_watcher = GatewayRunner._drain_control_watcher.__get__(
            runner, GatewayRunner
        )
        # Drive a few ticks manually rather than spinning the loop.
        dc.write_drain_request()
        task = asyncio.create_task(runner._drain_control_watcher(interval=0.02))
        await asyncio.sleep(0.06)
        assert runner._external_drain_active is True
        dc.clear_drain_request()
        await asyncio.sleep(0.06)
        assert runner._external_drain_active is False
        runner._running = False
        await asyncio.sleep(0.04)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# New-turn accept gate
# ---------------------------------------------------------------------------


class TestNewTurnGate:
    @pytest.mark.asyncio
    async def test_new_turn_refused_during_external_drain(self):
        runner, _ = _drain_runner()
        runner._external_drain_active = True
        event = MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=make_restart_source(),
            message_id="m1",
        )
        result = await runner._handle_message(event)
        assert result is not None
        assert "draining" in result.lower()


def _real_admission_authority(tmp_path, *, draining=False, restart_wait=False):
    """Build a real gateway authority over a temporary SQLite admission ledger."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="telegram")
    runner = SimpleNamespace(
        _draining=draining,
        _external_drain_active=False,
        _running=True,
        session_runtime_descriptor={
            "state": "draining" if draining else "ready",
            "drain_reason": "restart_wait" if restart_wait else None,
        },
        config=SimpleNamespace(multiplex_profiles=False),
        _adapter_for_source=lambda source: None,
        _handle_message=lambda event: None,
    )
    authority = SessionAuthority(
        runner,
        profile_id="p",
        instance_id="owner",
        db=db,
        epoch=begin_runtime_epoch(db, instance_id="owner"),
    )
    authority.sessions["s"] = LiveSession(
        SessionSource(platform=Platform.TELEGRAM, chat_id="c", user_id="human"),
        "route",
    )
    # The regression concerns admission durability; no execution worker should consume the row.
    authority._schedule = lambda ref: None
    return db, runner, authority


@pytest.mark.asyncio
async def test_external_drain_race_refuses_before_sqlite_admission_commit(tmp_path, monkeypatch):
    """A drain that starts after loop preflight still blocks the transaction's authorizer."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db, runner, authority = _real_admission_authority(tmp_path)
    original_execute_write = db._execute_write

    def race_execute_write(write, patience_s=None):
        # This is the scheduler race: the loop-side preflight has passed, then an external
        # drain becomes active immediately before the real SQLite callback invokes its gate.
        def drain_before_authorize(conn):
            runner._external_drain_active = True
            return write(conn)

        return original_execute_write(drain_before_authorize, patience_s=patience_s)

    db._execute_write = race_execute_write
    request = Submission("race-1", SessionRef("p", "s"), {"text": "hello"}, "queue")
    try:
        with pytest.raises(RuntimeStoreError, match="runtime_draining"):
            await authority.submit(
                Principal("human", "p", frozenset({"session:submit"}), "test"), request
            )
        assert find_session_admission(
            db, principal_id="human", session_id="s", request_id="race-1"
        ) is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_restart_wait_queue_only_admits_during_ordinary_restart(tmp_path, monkeypatch):
    """The restart after-turn wait retains its explicit queued-input exception."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db, _, authority = _real_admission_authority(tmp_path, draining=True, restart_wait=True)
    request = Submission("restart-1", SessionRef("p", "s"), {"text": "hello"}, "queue")
    try:
        receipt = await authority.submit(
            Principal("human", "p", frozenset({"session:submit"}), "test"),
            request,
            queue_while_restarting=True,
        )
        assert receipt.status == "queued"
        row = find_session_admission(
            db, principal_id="human", session_id="s", request_id="restart-1"
        )
        assert row is not None and row["status"] == "queued"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_existing_admission_claims_and_settles_during_external_drain(tmp_path, monkeypatch):
    """External drain refuses new input but does not strand an already durable FIFO head."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db, runner, authority = _real_admission_authority(tmp_path)
    request = Submission("claim-1", SessionRef("p", "s"), {"text": "hello"}, "queue")
    ref = SessionRef("p", "s")
    try:
        receipt = await authority.submit(
            Principal("human", "p", frozenset({"session:submit"}), "test"), request
        )
        assert receipt.status == "queued"
        runner._external_drain_active = True

        row, first = await authority._claim_next(ref, authority.sessions["s"])
        assert first["admission_id"] == receipt.admission_id
        assert row["status"] == "started"
        settled = settle_session_input(
            db,
            epoch=authority.epoch,
            admission_id=row["admission_id"],
            generation=row["generation"],
            outcome="completed",
            result={"final_response": "done"},
        )
        assert settled["status"] == "terminal"
        assert get_session_admission(db, admission_id=receipt.admission_id)["outcome"] == "completed"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_external_drain_overrides_restart_wait_for_new_input(tmp_path, monkeypatch):
    """An external drain still refuses new writes even during restart's queue-only wait."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db, runner, authority = _real_admission_authority(tmp_path, draining=True, restart_wait=True)
    runner._external_drain_active = True
    request = Submission("external-restart-1", SessionRef("p", "s"), {"text": "hello"}, "queue")
    try:
        with pytest.raises(RuntimeStoreError, match="runtime_draining"):
            await authority.submit(
                Principal("human", "p", frozenset({"session:submit"}), "test"),
                request,
                queue_while_restarting=True,
            )
        assert find_session_admission(
            db, principal_id="human", session_id="s", request_id="external-restart-1"
        ) is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_ordinary_stop_still_refuses_existing_claim(tmp_path, monkeypatch):
    """An ordinary owner stop still pauses a durable head rather than claiming it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db, runner, authority = _real_admission_authority(tmp_path)
    request = Submission("stop-1", SessionRef("p", "s"), {"text": "hello"}, "queue")
    ref = SessionRef("p", "s")
    try:
        receipt = await authority.submit(
            Principal("human", "p", frozenset({"session:submit"}), "test"), request
        )
        runner._draining = True
        with pytest.raises(RuntimeStoreError, match="runtime_draining"):
            await authority._claim_next(ref, authority.sessions["s"])
        assert get_session_admission(db, admission_id=receipt.admission_id)["status"] == "queued"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_empty_claim_preflight_retries_head_committed_before_external_drain(tmp_path, monkeypatch):
    """A head committed after an empty preflight is re-read and drained during external quiescence."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db, runner, authority = _real_admission_authority(tmp_path)
    ref = SessionRef("p", "s")
    live = authority.sessions["s"]
    original_tracked_write = runtime_workers.tracked_write
    injected = False
    observed_claim_results = []

    async def race_tracked_write(authority_arg, write, then=None, *, ordered=False, after=None):
        nonlocal injected
        if not injected:
            injected = True
            admit_session_input(
                db,
                epoch=authority.epoch,
                principal_id="native-race",
                session_id="s",
                request_id="empty-preflight-1",
                payload={"text": "arrived during claim"},
            )
            runner._external_drain_active = True
        result = await original_tracked_write(
            authority_arg, write, then=then, ordered=ordered, after=after
        )
        observed_claim_results.append(result)
        return result

    monkeypatch.setattr(runtime_workers, "tracked_write", race_tracked_write)

    async def execute_claimed(_ref, _live, _row):
        return "done", "completed"

    authority._execute_claimed = execute_claimed
    try:
        await authority._drain(ref)
        row = find_session_admission(
            db,
            principal_id="native-race",
            session_id="s",
            request_id="empty-preflight-1",
        )
        assert row is not None and row["status"] == "terminal"
        assert row["outcome"] == "completed"
        assert observed_claim_results and observed_claim_results[0] is False
        assert live.pause_notified is False
    finally:
        db.close()

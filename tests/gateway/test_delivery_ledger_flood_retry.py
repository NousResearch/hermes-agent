"""A flood-refused final reply is redelivered once the penalty passes.

Adapters fail a flood-controlled final send closed as ``flood_control:<seconds>`` so
the send coroutine never sleeps a long penalty (#91969). The comment there says the
delivery ledger owns the wait. It did not: ``sweep_failed_for_runtime`` only replayed
``send_path_degraded`` rows, so a flood-refused row sat in ``failed`` until the next
restart's ``sweep_recoverable``, which redelivered it hours late under the "gateway
restarted during delivery" marker. On 5 Sep 2026 a reply refused at 08:58 arrived at
12:15 that way, labelled as a possible duplicate although whether the platform had
accepted any part of it was unknowable.

Now:
- ``flood_control:*`` rows are runtime-retryable, but only once their own deadline
  (refusal time plus the platform's wait) has passed; neither an early timer nor a
  reconnect sweep spends an attempt inside the penalty window;
- the runner arms one timer per adapter identity; the slot stays occupied until the
  timer ends, only the running timer may arm its successor into it, and it re-arms
  for whatever is still waiting;
- every flood redelivery carries a marker that names the rate limit (the platform may
  have accepted earlier chunks, and the stored text's raw length cannot tell: MarkdownV2
  escaping alone can turn a one-message reply into two requests); a claimed row's stale
  refusal is cleared so an interrupted resend is seen as uncertain by the next boot; a
  claim released unsent (resume flag not clearable, adapter gone) keeps its flood error
  and so its place on the timer;
- at boot, a dead owner's not-yet-due flood row is adopted rather than resent early, and
  returned flagged so its session's resume flag is cleared (the answer is in the ledger)
  without the row being sent inside the penalty; a legacy row without a profile is
  normalised to 'default' so the timer's runtime sweep can claim it.
"""

from __future__ import annotations

import asyncio
import os
import time as _time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import delivery_ledger as dl
from gateway import run_startup
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult

T0 = 1_700_000_000.0


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    monkeypatch.setattr(dl, "_owner_stamp", lambda: (os.getpid(), 202))
    yield


@pytest.fixture
def clock(monkeypatch):
    """A controllable wall clock shared by the ledger and the runner (both call ``time.time``)."""
    now = [T0]
    monkeypatch.setattr(_time, "time", lambda: now[0])
    return now


def _record(oid, *, platform="telegram", chat_id="5230977008", content="the final answer", profile=None):
    dl.record_obligation(
        obligation_id=oid, session_key=f"agent:main:{platform}:dm:{chat_id}", platform=platform,
        chat_id=chat_id, thread_id=None, content=content, adapter_profile=profile)
    dl.mark_attempting(oid)


def _row(oid):
    with dl._connect() as conn:
        r = conn.execute(
            "SELECT state, attempts, last_error, owner_pid FROM delivery_obligations WHERE obligation_id=?", (oid,)
        ).fetchone()
    return None if r is None else {"state": r[0], "attempts": r[1], "last_error": r[2], "owner_pid": r[3]}


def _owner(oid):
    """(owner_pid, owner_started_at, adapter_profile) of a row."""
    with dl._connect() as conn:
        return conn.execute(
            "SELECT owner_pid, owner_started_at, adapter_profile FROM delivery_obligations"
            " WHERE obligation_id=?", (oid,)).fetchone()


DEAD_OWNER = (999_999, 1)


def _record_for_dead_owner(oid, error, **kwargs):
    """A flood-refused row whose last owner is gone: what a restart finds in the ledger."""
    live = dl._owner_stamp
    dl._owner_stamp = lambda: DEAD_OWNER
    try:
        _record(oid, **kwargs)
        dl.mark_failed(oid, error)
    finally:
        dl._owner_stamp = live


def _runner(adapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner._active_profile_name = lambda: "default"
    runner._running = True
    store = MagicMock()
    store.clear_resume_pending = AsyncMock()
    store._store = None
    runner.session_store = None
    runner._async_session_store = store
    return runner


def _adapter(success=True, error="", side_effect=None):
    adapter = MagicMock()
    if side_effect is not None:
        adapter.send = AsyncMock(side_effect=side_effect)
    else:
        adapter.send = AsyncMock(return_value=SendResult(success=success, error=error))
    return adapter


async def _drain_timers(runner):
    """Await every flood timer, including successors armed while awaiting."""
    while runner._flood_redelivery_tasks:
        await asyncio.gather(*list(runner._flood_redelivery_tasks.values()))


# ---------------------------------------------------------------------------
# Classification, waits, deadlines, certainty.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error, expected", [
    ("flood_control:185.0", True),
    ("FLOOD_CONTROL:9", True),
    ("send_path_degraded", False),
    ("Forbidden: bot was blocked by the user", False),
    ("", False),
    (None, False),
    # PTB's own RetryAfter wording, which is what a row carries when it was written by a send path
    # that had not been normalized, or by an older build. Unrecognised, such a row gets no timer.
    ("Flood control exceeded. Retry in 185 seconds", True),
    ("flood control exceeded. retry in 30 seconds", True),
    # The flood wording is required, not merely a delay: an unrelated error that suggests retrying
    # must not be read as a flood refusal.
    ("Bad Gateway; retry in 5 seconds", False),
])
def test_is_flood_error(error, expected):
    assert dl.is_flood_error(error) is expected


@pytest.mark.parametrize("error, expected", [
    ("flood_control:185.0", 185.0),
    ("flood_control:5820", 5820.0),        # a 97-minute penalty is kept as the platform stated it
    ("flood_control:abc", 60.0),           # unreadable -> default
    ("flood_control:0", 60.0),
    ("send_path_degraded", 60.0),
    # The platform's own wording states the delay just as precisely as the canonical prefix, so the
    # deadline must come from it rather than the generic default.
    ("Flood control exceeded. Retry in 185 seconds", 185.0),
    ("Bad Gateway; retry in 5 seconds", 60.0),
])
def test_flood_wait_seconds(error, expected):
    assert dl.flood_wait_seconds(error) == pytest.approx(expected)


@pytest.mark.parametrize("seconds, expected", [
    (185.0, 187.0),                        # the wait plus slack
    (5820, 902.0),                         # capped: the timer wakes early and re-checks, it never sends early
    (-5, 2.0),                             # already due -> just the slack
    ("abc", 62.0),
])
def test_flood_retry_delay(seconds, expected):
    assert dl.flood_retry_delay(seconds) == pytest.approx(expected)


def test_flood_not_before_is_the_refusal_time_plus_the_platform_wait():
    assert dl.flood_not_before(T0, "flood_control:185.0") == pytest.approx(T0 + 185.0)
    assert dl.flood_not_before(None, "flood_control:9") == pytest.approx(9.0)


@pytest.mark.parametrize("content", ["x" * 10, "." * 3000, "word " * 1200])
def test_every_flood_redelivery_carries_the_rate_limit_marker(clock, monkeypatch, content):
    """Raw length proves nothing about how many requests the adapter made: MarkdownV2 escaping turns 3000
    dots into 6000 UTF-16 units, two Telegram messages, and the send result does not say which chunk was
    refused. So there is no length-based certainty; the marker is unconditional, at runtime and at boot."""
    _record("ob-runtime", content=content)
    dl.mark_failed("ob-runtime", "flood_control:9")
    (runtime_row,) = dl.sweep_failed_for_runtime("telegram", now=T0 + 20)
    assert runtime_row["needs_marker"] is True and runtime_row["marker"] == dl.FLOOD_MARKER

    _record("ob-boot", content=content)
    dl.mark_failed("ob-boot", "flood_control:9")
    monkeypatch.setattr(dl, "_owner_alive", lambda pid, started: False)
    boot_rows = {row["obligation_id"]: row for row in dl.sweep_recoverable(now=T0 + 20)}
    assert boot_rows["ob-boot"]["needs_marker"] is True and boot_rows["ob-boot"]["marker"] == dl.FLOOD_MARKER


# ---------------------------------------------------------------------------
# The runtime sweep: which rows it claims, when, and how it marks them.
# ---------------------------------------------------------------------------

def test_runtime_sweep_claims_a_due_flood_row_under_the_rate_limit_marker_and_clears_the_stale_refusal(clock):
    _record("ob-flood")
    dl.mark_failed("ob-flood", "flood_control:185.0")
    _record("ob-degraded")
    dl.mark_failed("ob-degraded", "send_path_degraded")
    _record("ob-blocked")
    dl.mark_failed("ob-blocked", "Forbidden: bot was blocked by the user")

    claimed = {row["obligation_id"]: row for row in dl.sweep_failed_for_runtime("telegram", now=T0 + 200)}

    assert set(claimed) == {"ob-flood", "ob-degraded"}, "a permanent rejection must never be replayed"
    assert claimed["ob-flood"]["needs_marker"] is True
    assert claimed["ob-flood"]["marker"] == dl.FLOOD_MARKER
    assert claimed["ob-flood"]["last_error"] == "flood_control:185.0", "the pre-claim error rides along for a release"
    assert claimed["ob-degraded"]["needs_marker"] is True
    assert claimed["ob-degraded"]["marker"] == dl.RECONNECTED_MARKER
    assert claimed["ob-degraded"]["last_error"] == "send_path_degraded"
    flood = _row("ob-flood")
    assert flood["state"] == "attempting" and flood["attempts"] == 1
    assert flood["last_error"] is None, "the claim is a fresh attempt; the old refusal must not survive it"
    assert _row("ob-blocked")["state"] == "failed"


def test_runtime_sweep_leaves_a_flood_row_inside_its_wait_and_reports_its_deadline(clock):
    _record("ob-short")
    dl.mark_failed("ob-short", "flood_control:10")
    _record("ob-long")
    dl.mark_failed("ob-long", "flood_control:185")

    claimed = dl.sweep_failed_for_runtime("telegram", now=T0 + 12)

    assert [r["obligation_id"] for r in claimed] == ["ob-short"], "the 185s row is still inside its penalty"
    assert _row("ob-long") == {"state": "failed", "attempts": 0, "last_error": "flood_control:185", "owner_pid": os.getpid()}
    waiting = dl.pending_flood_retries(now=T0 + 12)
    assert waiting == [{"platform": "telegram", "profile": "default", "not_before": pytest.approx(T0 + 185)}]


def test_a_chunked_reply_refused_by_flood_control_keeps_the_marker(clock):
    _record("ob-long-text", content="word " * 1200)  # ~6000 chars: two Telegram messages
    dl.mark_failed("ob-long-text", "flood_control:9")

    (row,) = dl.sweep_failed_for_runtime("telegram", now=T0 + 20)

    assert row["needs_marker"] is True, "chunk 1 may have landed before chunk 2 was refused"
    assert row["marker"] == dl.FLOOD_MARKER, "no reconnect happened; the marker must name the rate limit"


# ---------------------------------------------------------------------------
# The boot sweep: adopt what is not due, resend what is, stay honest after a crash.
# ---------------------------------------------------------------------------

def test_boot_sweep_adopts_a_dead_owners_flood_row_inside_its_wait(clock, monkeypatch):
    _record_for_dead_owner("ob-flood", "flood_control:185.0")
    monkeypatch.setattr(dl, "_owner_alive", lambda pid, started: False)

    (adopted,) = dl.sweep_recoverable(now=T0 + 30)

    assert adopted["adopted"] is True and adopted["not_before"] == pytest.approx(T0 + 185.0), \
        "returned so the caller clears its resume flag, flagged so nothing sends it inside the penalty"
    assert adopted["attempts"] == 0 and "marker" not in adopted
    row = _row("ob-flood")
    assert row["state"] == "failed" and row["attempts"] == 0 and row["last_error"] == "flood_control:185.0"
    assert _owner("ob-flood") == (os.getpid(), 202, "default"), "ownership moved from the dead process"


def test_a_legacy_row_without_a_profile_is_normalised_so_the_timer_can_claim_it(clock, monkeypatch):
    """Rows recorded before ``adapter_profile`` existed carry NULL. The boot sweep accepts them on a
    non-multiplexed gateway, but the runtime sweep matches the profile exactly and the timer asks for
    'default', so an adopted NULL row could only ever wake the timer without being sent."""
    _record_for_dead_owner("ob-legacy", "flood_control:185.0")
    with dl._connect() as conn:
        conn.execute("UPDATE delivery_obligations SET adapter_profile=NULL WHERE obligation_id='ob-legacy'")
    monkeypatch.setattr(dl, "_owner_alive", lambda pid, started: False)

    (adopted,) = dl.sweep_recoverable(
        now=T0 + 30, deliverable_platforms={"telegram"},
        deliverable_targets={("telegram", "default"), ("telegram", None)})

    assert adopted["profile"] == "default" and _owner("ob-legacy") == (os.getpid(), 202, "default")
    assert dl.pending_flood_retries(now=T0 + 30) == [
        {"platform": "telegram", "profile": "default", "not_before": pytest.approx(T0 + 185.0)}]
    (claimed,) = dl.sweep_failed_for_runtime("telegram", now=T0 + 200, profile="default")
    assert claimed["obligation_id"] == "ob-legacy" and claimed["marker"] == dl.FLOOD_MARKER


def test_boot_sweep_resends_a_due_flood_row_under_the_rate_limit_marker_and_a_midsend_row_under_the_restart_one(clock, monkeypatch):
    _record("ob-flood")
    dl.mark_failed("ob-flood", "flood_control:185.0")
    _record("ob-midsend")  # crashed mid-await: the platform MAY have it
    monkeypatch.setattr(dl, "_owner_alive", lambda pid, started: False)

    claimed = {row["obligation_id"]: row for row in dl.sweep_recoverable(now=T0 + 300)}

    assert claimed["ob-flood"]["needs_marker"] is True
    assert claimed["ob-flood"]["marker"] == dl.FLOOD_MARKER
    assert claimed["ob-midsend"]["needs_marker"] is True
    assert "marker" not in claimed["ob-midsend"], "a crash mid-send is the runner's restart marker"
    flood = _row("ob-flood")
    assert flood["state"] == "attempting" and flood["last_error"] is None and flood["attempts"] == 1


def test_boot_sweep_keeps_the_marker_for_a_chunked_flood_refused_reply(clock, monkeypatch):
    _record("ob-long-text", content="word " * 1200)  # two Telegram messages; chunk 1 may have landed
    dl.mark_failed("ob-long-text", "flood_control:9")
    monkeypatch.setattr(dl, "_owner_alive", lambda pid, started: False)

    (row,) = dl.sweep_recoverable(now=T0 + 300)

    assert row["needs_marker"] is True
    assert row["marker"] == dl.FLOOD_MARKER, "no restart interrupted this send; the marker must name the rate limit"


def test_an_interrupted_flood_resend_gets_the_marker_on_the_next_boot(clock, monkeypatch):
    """Runtime claim, platform accepts, process dies before mark_delivered: the next boot must see an
    ordinary interrupted send (restart marker), not a flood row."""
    _record("ob-flood")
    dl.mark_failed("ob-flood", "flood_control:9")
    (claimed,) = dl.sweep_failed_for_runtime("telegram", now=T0 + 20)
    assert claimed["marker"] == dl.FLOOD_MARKER
    monkeypatch.setattr(dl, "_owner_alive", lambda pid, started: False)  # that process is gone

    (recovered,) = dl.sweep_recoverable(now=T0 + 60)

    assert recovered["needs_marker"] is True
    assert "marker" not in recovered, "the refusal was cleared by the claim: this is an ordinary restart recovery"


# ---------------------------------------------------------------------------
# The runner: timers wait the platform's figure, never block their own successor,
# never send early, and stop at shutdown.
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_sleep(monkeypatch, clock):
    """asyncio.sleep advances the shared clock instead of waiting."""
    slept: list[float] = []
    real_sleep = asyncio.sleep

    async def _sleep(delay):
        slept.append(delay)
        clock[0] += delay
        await real_sleep(0)

    monkeypatch.setattr(run_startup.asyncio, "sleep", _sleep)
    return slept


@pytest.mark.asyncio
async def test_schedule_flood_redelivery_waits_then_runs_the_runtime_sweep(fake_sleep):
    runner = _runner(_adapter())
    runner._redeliver_failed_obligations_for_platform = AsyncMock(return_value=1)

    delay = runner._schedule_flood_redelivery(Platform.TELEGRAM, profile=None, error="flood_control:185.0")
    again = runner._schedule_flood_redelivery(Platform.TELEGRAM, profile=None, error="flood_control:300")

    assert delay == pytest.approx(187.0)
    assert again is None, "a refusal due no earlier than the armed timer must not arm another"
    assert len(runner._flood_redelivery_tasks) == 1
    await _drain_timers(runner)

    assert fake_sleep == [pytest.approx(187.0)]
    runner._redeliver_failed_obligations_for_platform.assert_awaited_once_with(Platform.TELEGRAM, profile=None)
    assert runner._flood_redelivery_tasks == {}, "a finished timer frees its slot"


@pytest.mark.asyncio
async def test_schedule_flood_redelivery_does_nothing_after_shutdown(monkeypatch):
    runner = _runner(_adapter())
    runner._redeliver_failed_obligations_for_platform = AsyncMock(return_value=1)
    real_sleep = asyncio.sleep

    async def _sleep(delay):
        runner._running = False  # the gateway stopped while the timer was waiting
        await real_sleep(0)

    monkeypatch.setattr(run_startup.asyncio, "sleep", _sleep)
    runner._schedule_flood_redelivery("telegram", error="flood_control:5")
    await _drain_timers(runner)

    runner._redeliver_failed_obligations_for_platform.assert_not_awaited()


@pytest.mark.asyncio
async def test_timer_delivers_the_refused_reply_under_the_rate_limit_marker_end_to_end(fake_sleep):
    _record("ob-flood", content="Here is the plan for today.")
    dl.mark_failed("ob-flood", "flood_control:185.0")
    adapter = _adapter(success=True)
    runner = _runner(adapter)

    runner._schedule_flood_redelivery(Platform.TELEGRAM, error="flood_control:185.0")
    await _drain_timers(runner)

    sent = adapter.send.call_args.kwargs
    assert sent["content"] == dl.FLOOD_MARKER + "Here is the plan for today."
    assert "restarted" not in sent["content"] and "reconnected" not in sent["content"]
    assert _row("ob-flood")["state"] == "delivered"
    assert fake_sleep == [pytest.approx(187.0)]


@pytest.mark.asyncio
async def test_a_chunked_reply_is_redelivered_under_the_rate_limit_marker(fake_sleep):
    _record("ob-long-text", content="word " * 1200)  # two Telegram messages; chunk 1 may have landed
    dl.mark_failed("ob-long-text", "flood_control:9")
    adapter = _adapter(success=True)
    runner = _runner(adapter)

    runner._schedule_flood_redelivery(Platform.TELEGRAM, error="flood_control:9")
    await _drain_timers(runner)

    sent = adapter.send.call_args.kwargs["content"]
    assert sent.startswith(dl.FLOOD_MARKER)
    assert "reconnected" not in sent and "restarted" not in sent
    assert _row("ob-long-text")["state"] == "delivered"


@pytest.mark.asyncio
async def test_a_redelivery_refused_again_arms_a_successor_timer(fake_sleep):
    """A refusal during the sweep must not be blocked by the timer's own slot: it arms its successor."""
    _record("ob-flood")
    dl.mark_failed("ob-flood", "flood_control:10")
    adapter = _adapter(side_effect=[SendResult(success=False, error="flood_control:30.0"), SendResult(success=True)])
    runner = _runner(adapter)

    runner._schedule_flood_redelivery(Platform.TELEGRAM, error="flood_control:10")
    await _drain_timers(runner)

    assert adapter.send.await_count == 2
    assert fake_sleep == [pytest.approx(12.0), pytest.approx(32.0)], "the successor waits the NEW refusal's figure"
    assert _row("ob-flood")["state"] == "delivered"
    assert _row("ob-flood")["attempts"] == 2
    assert runner._flood_redelivery_tasks == {}


@pytest.mark.asyncio
async def test_a_capped_timer_wakes_early_but_never_sends_early(fake_sleep):
    """A 2000s penalty: the timer sleeps at most 15 min at a time, finds the row not yet due, re-arms for the
    remainder, and sends exactly once when the platform's deadline has actually passed."""
    _record("ob-flood")
    dl.mark_failed("ob-flood", "flood_control:2000")
    adapter = _adapter(success=True)
    runner = _runner(adapter)

    runner._schedule_flood_redelivery(Platform.TELEGRAM, error="flood_control:2000")
    await _drain_timers(runner)

    assert adapter.send.await_count == 1
    assert fake_sleep == [pytest.approx(902.0), pytest.approx(902.0), pytest.approx(198.0)]
    assert _row("ob-flood")["state"] == "delivered" and _row("ob-flood")["attempts"] == 1


@pytest.mark.asyncio
async def test_a_shorter_sibling_timer_does_not_send_the_longer_row_early(fake_sleep):
    _record("ob-short", content="short one")
    dl.mark_failed("ob-short", "flood_control:10")
    _record("ob-long", content="long one")
    dl.mark_failed("ob-long", "flood_control:185")
    adapter = _adapter(success=True)
    runner = _runner(adapter)

    assert runner._schedule_flood_redelivery(Platform.TELEGRAM, error="flood_control:10") == pytest.approx(12.0)
    assert runner._schedule_flood_redelivery(Platform.TELEGRAM, error="flood_control:185") is None
    await _drain_timers(runner)

    contents = [c.kwargs["content"] for c in adapter.send.await_args_list]
    assert contents == [dl.FLOOD_MARKER + "short one", dl.FLOOD_MARKER + "long one"]
    assert fake_sleep[0] == pytest.approx(12.0)
    assert fake_sleep[1] == pytest.approx(175.0), "re-armed for the longer row's REMAINING wait, plus slack"
    assert _row("ob-short")["attempts"] == 1 and _row("ob-long")["attempts"] == 1


@pytest.mark.asyncio
async def test_a_shorter_refusal_replaces_a_longer_timer_that_is_still_asleep(fake_sleep):
    """Arrival order reversed: the 185s timer is armed first. Left alone it would hold the 10s row for
    the full 185s. The shorter refusal replaces the sleeping timer and re-arms for the longer row."""
    _record("ob-long", content="long one")
    dl.mark_failed("ob-long", "flood_control:185")
    _record("ob-short", content="short one")
    dl.mark_failed("ob-short", "flood_control:10")
    adapter = _adapter(success=True)
    runner = _runner(adapter)

    assert runner._schedule_flood_redelivery(Platform.TELEGRAM, error="flood_control:185") == pytest.approx(187.0)
    assert runner._schedule_flood_redelivery(Platform.TELEGRAM, error="flood_control:10") == pytest.approx(12.0)
    assert len(runner._flood_redelivery_tasks) == 1, "one live timer per identity, the shorter one"
    await _drain_timers(runner)

    contents = [c.kwargs["content"] for c in adapter.send.await_args_list]
    assert contents == [dl.FLOOD_MARKER + "short one", dl.FLOOD_MARKER + "long one"]
    assert fake_sleep == [pytest.approx(12.0), pytest.approx(175.0)], "the 187s sleep never happened"
    assert _row("ob-short")["attempts"] == 1 and _row("ob-long")["attempts"] == 1
    assert runner._flood_redelivery_tasks == {} and runner._flood_redelivery_slots == {}


@pytest.mark.asyncio
async def test_a_row_refused_while_the_timer_sweeps_is_armed_by_that_same_timer(fake_sleep):
    """A final refused during the timer's own redelivery send lands in the ledger after the sweep
    started. The timer's synchronous post-sweep arm reads the ledger once the sends are done and
    picks it up, so the row is never left without a timer."""
    _record("ob-first")
    dl.mark_failed("ob-first", "flood_control:10")
    adapter = _adapter(success=True)
    runner = _runner(adapter)
    fired = {"done": False}

    async def _send(**kwargs):
        if not fired["done"]:
            fired["done"] = True
            _record("ob-second", content="refused mid-sweep")
            dl.mark_failed("ob-second", "flood_control:10")
            # A concurrent refusal's schedule request is declined while this timer holds the slot.
            runner._schedule_flood_redelivery(Platform.TELEGRAM, error="flood_control:10")
        return SendResult(success=True, message_id="9")

    adapter.send = AsyncMock(side_effect=_send)
    runner._schedule_flood_redelivery(Platform.TELEGRAM, error="flood_control:10")
    await _drain_timers(runner)

    assert _row("ob-first")["state"] == "delivered"
    assert _row("ob-second")["state"] == "delivered", "the mid-sweep refusal was armed by the same timer"
    assert runner._flood_redelivery_tasks == {} and runner._flood_redelivery_slots == {}


@pytest.mark.asyncio
async def test_a_timer_already_sweeping_is_not_replaced(fake_sleep):
    """Once the sleep is over the timer may be mid-send; a refusal arriving then arms nothing extra and
    the successor logic (the running timer re-arming itself) takes over."""
    runner = _runner(_adapter())
    seen = []

    async def _sweep(target, profile=None):
        async def _refusal_from_another_task():
            return runner._schedule_flood_redelivery(Platform.TELEGRAM, error="flood_control:1")

        seen.append(await asyncio.create_task(_refusal_from_another_task()))
        return 0

    runner._redeliver_failed_obligations_for_platform = _sweep
    runner._schedule_flood_redelivery(Platform.TELEGRAM, error="flood_control:60")
    await _drain_timers(runner)

    assert seen == [None], "a refusal from outside the running timer cannot cancel it mid-sweep"


@pytest.mark.asyncio
async def test_boot_redelivery_arms_a_timer_for_an_adopted_flood_row(fake_sleep, monkeypatch):
    _record_for_dead_owner("ob-flood", "flood_control:120", content="adopted at boot")
    monkeypatch.setattr(dl, "_owner_alive", lambda pid, started: False)
    adapter = _adapter(success=True)
    runner = _runner(adapter)

    n = await runner._redeliver_pending_obligations()   # the boot path: claim + redeliver

    assert n == 0, "nothing is due at boot"
    adapter.send.assert_not_awaited()
    runner._async_session_store.clear_resume_pending.assert_awaited_once_with(
        "agent:main:telegram:dm:5230977008"), "the answer is in the ledger: the turn must not be re-run"
    assert _owner("ob-flood") == (os.getpid(), 202, "default")
    assert len(runner._flood_redelivery_tasks) == 1
    await _drain_timers(runner)
    assert adapter.send.call_args.kwargs["content"] == dl.FLOOD_MARKER + "adopted at boot"
    assert _row("ob-flood")["state"] == "delivered"


@pytest.mark.asyncio
async def test_a_failed_resume_flag_clear_keeps_the_flood_row_on_the_timer(fake_sleep):
    """Review finding on the first cut: the unsent claim was released as ``send_path_degraded``, which
    dropped the row from ``pending_flood_retries`` and ended the timer, stranding the reply until a reconnect
    or restart. Released with its own flood error, it waits the platform's figure again and is then sent."""
    _record("ob-flood", content="the plan")
    dl.mark_failed("ob-flood", "flood_control:30")
    adapter = _adapter(success=True)
    runner = _runner(adapter)
    runner._async_session_store.clear_resume_pending = AsyncMock(
        side_effect=[RuntimeError("session store locked"), None])

    runner._schedule_flood_redelivery(Platform.TELEGRAM, error="flood_control:30")
    await _drain_timers(runner)

    assert adapter.send.await_count == 1
    assert _row("ob-flood") == {"state": "delivered", "attempts": 1, "last_error": None, "owner_pid": os.getpid()}, \
        "the released claim spent no attempt"
    assert fake_sleep == [pytest.approx(32.0), pytest.approx(32.0)], "re-armed for the platform's figure, not dropped"


@pytest.mark.asyncio
async def test_a_runtime_claim_whose_adapter_vanished_is_released_with_its_flood_error(clock):
    _record("ob-flood")
    dl.mark_failed("ob-flood", "flood_control:9")
    (row,) = dl.sweep_failed_for_runtime("telegram", now=T0 + 20)
    runner = _runner(_adapter())
    runner.adapters = {}  # the reconnect that armed the timer is gone again

    assert await runner._obligation_adapter(row) is None

    assert _row("ob-flood") == {"state": "failed", "attempts": 0, "last_error": "flood_control:9", "owner_pid": os.getpid()}
    assert dl.pending_flood_retries()[0]["platform"] == "telegram", "still the flood timer's business"


# ---------------------------------------------------------------------------
# The adapter hook: a flood-refused final send arms the timer; other outcomes do not.
# ---------------------------------------------------------------------------

def _telegram_adapter_with_runner():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    runner = MagicMock()
    runner._schedule_flood_redelivery = MagicMock(return_value=187.0)
    runner._redeliver_failed_obligations_for_platform = AsyncMock(return_value=0)
    adapter.gateway_runner = runner
    adapter._final_delivery_adapter = lambda source: adapter
    return adapter, runner


def _event():
    return SimpleNamespace(source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id="5230977008", thread_id=None),
                           text="what should I eat", message_id="77")


@pytest.mark.asyncio
async def test_flood_refused_final_send_arms_the_timer():
    adapter, runner = _telegram_adapter_with_runner()
    _record("ob-x")

    await adapter._finalize_delivery_obligation(
        "ob-x", SendResult(success=False, error="flood_control:185.0"), _event(), adapter)

    assert _row("ob-x")["state"] == "failed"
    runner._schedule_flood_redelivery.assert_called_once_with(
        Platform.TELEGRAM, profile=None, error="flood_control:185.0")


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [
    SendResult(success=True),
    SendResult(success=False, error="send_path_degraded"),
    SendResult(success=False, error="Forbidden: bot was blocked by the user"),
])
async def test_other_outcomes_do_not_arm_the_flood_timer(result):
    adapter, runner = _telegram_adapter_with_runner()
    _record("ob-y")

    await adapter._finalize_delivery_obligation("ob-y", result, _event(), adapter)

    runner._schedule_flood_redelivery.assert_not_called()


# ---------------------------------------------------------------------------
# The adapter normalizes every definite flood refusal, so the ledger sees one shape.
# ---------------------------------------------------------------------------

def _real_telegram_adapter():
    """A real Telegram adapter with a stub bot, for driving the send and edit flood paths."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._bot = MagicMock()
    return adapter


def test_a_persisted_row_with_the_platforms_own_wording_is_dated_from_it():
    """The boot sweep decides adopt-or-claim from the row's deadline. Read as an ordinary failure, a
    raw flood row is claimed at once and spends its one attempt inside the penalty that caused it."""
    raw = "Flood control exceeded. Retry in 185 seconds"

    assert dl.is_flood_error(raw) is True
    assert dl.flood_wait_seconds(raw) == pytest.approx(185.0)
    assert dl.flood_not_before(T0, raw) == pytest.approx(T0 + 185.0)


@pytest.mark.asyncio
async def test_a_short_flood_that_outlives_the_retries_fails_closed_canonically():
    """A wait under the inline cap is slept and retried. When the flood is still there after the last
    attempt the send used to raise, handing the caller the platform's wording instead of the
    canonical result, so no redelivery timer was armed and the reply waited for the next restart."""
    from telegram.error import RetryAfter

    adapter = _real_telegram_adapter()
    adapter._send_chunk_markdown_or_plain = AsyncMock(side_effect=RetryAfter(1))

    result = await adapter._send_chunk_with_retries(
        "5230977008", "the final answer", 0, None, None, None, False,
        adapter._telegram_error_types())

    assert isinstance(result, SendResult)
    assert result.success is False
    assert result.error == "flood_control:1.0"
    assert dl.is_flood_error(result.error) is True
    assert result.retry_after == pytest.approx(1.0)
    assert adapter._send_chunk_markdown_or_plain.await_count == 3


@pytest.mark.asyncio
async def test_an_edit_still_flooded_after_the_inline_wait_fails_closed_canonically():
    """The edit path sleeps a short wait once and retries. A retry that is refused again, typically
    for far longer, must report the new delay canonically rather than as raw text."""
    from telegram.error import RetryAfter

    adapter = _real_telegram_adapter()
    adapter._edit_text = AsyncMock(side_effect=[RetryAfter(1), RetryAfter(185)])

    result = await adapter.edit_message("5230977008", "900", "the final answer")

    assert result.success is False
    assert result.error == "flood_control:185.0"
    assert dl.flood_wait_seconds(result.error) == pytest.approx(185.0)

"""A flood-refused final reply is redelivered once the penalty passes, plainly when that is safe.

Adapters fail a flood-controlled final send closed as ``flood_control:<seconds>`` so
the send coroutine never sleeps a long penalty (#91969). The comment there says the
delivery ledger owns the wait. It did not: ``sweep_failed_for_runtime`` only replayed
``send_path_degraded`` rows, so a flood-refused row sat in ``failed`` until the next
restart's ``sweep_recoverable``, which redelivered it hours late under the "gateway
restarted during delivery" marker. On 5 Sep 2026 a reply refused at 08:58 arrived at
12:15 that way, labelled as a possible duplicate although the platform had never
accepted it.

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
- at boot, a dead owner's not-yet-due flood row is adopted rather than resent early.
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
])
def test_is_flood_error(error, expected):
    assert dl.is_flood_error(error) is expected


@pytest.mark.parametrize("error, expected", [
    ("flood_control:185.0", 185.0),
    ("flood_control:5820", 5820.0),        # a 97-minute penalty is kept as the platform stated it
    ("flood_control:abc", 60.0),           # unreadable -> default
    ("flood_control:0", 60.0),
    ("send_path_degraded", 60.0),
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
    _record("ob-flood")
    dl.mark_failed("ob-flood", "flood_control:185.0")
    monkeypatch.setattr(dl, "_owner_alive", lambda pid, started: False)

    claimed = dl.sweep_recoverable(now=T0 + 30)

    assert claimed == [], "not due yet: resending now would burn an attempt inside the penalty"
    row = _row("ob-flood")
    assert row["state"] == "failed" and row["attempts"] == 0 and row["owner_pid"] == os.getpid()
    assert dl.pending_flood_retries(now=T0 + 30)[0]["not_before"] == pytest.approx(T0 + 185)


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
    again = runner._schedule_flood_redelivery(Platform.TELEGRAM, profile=None, error="flood_control:30")

    assert delay == pytest.approx(187.0)
    assert again is None, "a second refusal while the timer is armed must not arm another"
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
    """The bug Codex caught in the first cut: the running timer must not block its own re-arm."""
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
async def test_boot_redelivery_arms_a_timer_for_an_adopted_flood_row(fake_sleep, monkeypatch):
    _record("ob-flood", content="adopted at boot")
    dl.mark_failed("ob-flood", "flood_control:120")
    monkeypatch.setattr(dl, "_owner_alive", lambda pid, started: False)
    adapter = _adapter(success=True)
    runner = _runner(adapter)

    n = await runner._redeliver_pending_obligations()   # the boot path: claim + redeliver

    assert n == 0, "nothing is due at boot"
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

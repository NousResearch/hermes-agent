"""End-to-end validation of the channel claim lock against the 2026-10-23 incident (t_24b1959a).

Replays the incident pattern on channel 8grzznya3trpbfxjwf5goe778h with a real
GatewayRunner + MattermostAdapter + SessionAuthority over real SQLite; only the
model transport is stubbed (deterministic answers, fixed delays). The six checks
from the card:

1. exactly one active session per (platform, channel) at all times, single
   `Working` heartbeat stream, no divergent iteration counters;
2. exactly one answer per inbound message;
3. a thread reply routes only to the session owning the thread root;
4. queued events drain when the active session ends;
5. lease expiry after a gateway kill releases the claim and a waiting event
   admits cleanly (LEASE patched 300 s -> 2 s to exercise the same code path);
6. cross-instance: a second gateway stands down and posts nothing.

The harness is deliberately free of claim-surface imports so the same file
runs RED on pre-claim-lock trees (merge-base 4d04303818) for evidence.
"""

from __future__ import annotations

import asyncio
import os
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import pytest_asyncio

from tests.gateway._claim_e2e_support import (
    CHANNEL,
    INCIDENT_ROOT,
    SLOW_TOKEN,
    CaptureHandler,
    answer_for,
    channel_claim_rows,
    compression_block_lines,
    db_rows,
    install_journal,
    mm_ws_event,
    patch_adapter_recording,
    patch_model_transport,
    remove_journal,
    session_keys_for_channel,
    state_db_path,
    write_home_config,
)
from tests.waiting import async_wait_until

REPO = Path(__file__).resolve().parents[2]


class World:
    """One gateway instance (pipe A) with a full real stack and recorded surface."""

    def __init__(self, monkeypatch, tmp_path):
        self.tmp = Path(tmp_path)
        self.calls = {"async": 0, "sync": 0}
        self.max_channel_sessions_seen = 0
        self.chan = CHANNEL

    async def setup(self, monkeypatch, *, lease_seconds: int | None = None):
        # env first: per-call reads pick these up regardless of import order
        monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "1")
        monkeypatch.setenv("CUSTOM_API_KEY", "stub-key")
        monkeypatch.setenv("CUSTOM_BASE_URL", "http://127.0.0.1:9/v1")
        monkeypatch.setenv("MATTERMOST_ALLOWED_USERS", "gweslien")
        # Queue semantics for mid-turn inbounds: the redirect path (default
        # `interrupt`) folds a follow-up into the live turn, which cannot show the
        # "one answer per inbound message + queued drain" contract the incident
        # replay asserts. The claim-lock stand-down parks the event instead.
        monkeypatch.setenv("HERMES_GATEWAY_BUSY_INPUT_MODE", "queue")

        write_home_config(state_db_path().parent)

        from hermes_constants import get_hermes_home

        from gateway.config import GatewayConfig, Platform, PlatformConfig
        from gateway.run import GatewayRunner
        from gateway.session_authority import initialize_session_authority
        from plugins.platforms.mattermost.adapter import MattermostAdapter

        from tests.gateway._claim_e2e_support import SendLog

        if lease_seconds is not None:
            import hermes_state_runtime as rt

            _orig_acquire = rt.acquire_channel_claim
            _orig_renew = rt.renew_channel_claim
            _orig_transfer = rt.transfer_channel_claim

            def _acquire(db, *a, **kw):
                kw["lease_seconds"] = lease_seconds
                return _orig_acquire(db, *a, **kw)

            def _renew(db, *a, **kw):
                kw["lease_seconds"] = lease_seconds
                return _orig_renew(db, *a, **kw)

            def _transfer(db, *a, **kw):
                kw["lease_seconds"] = lease_seconds
                return _orig_transfer(db, *a, **kw)

            monkeypatch.setattr(rt, "acquire_channel_claim", _acquire)
            monkeypatch.setattr(rt, "renew_channel_claim", _renew)
            monkeypatch.setattr(rt, "transfer_channel_claim", _transfer)

        self.journal: CaptureHandler = install_journal()

        # The replay pins enforce: the harness regression-tests the acting gate (the
        # monitor rollout phase is covered by the test_channel_claims.py unit tests).
        from gateway.config import ClaimLockConfig

        self.runner = GatewayRunner(GatewayConfig(group_sessions_per_user=False,
            claim_lock=ClaimLockConfig(mode="enforce")))
        self.adapter = MattermostAdapter(PlatformConfig(
            enabled=True, token="fixture-token", typing_indicator=False,
            extra={"reply_mode": "thread"}))
        self.adapter.gateway_runner = self.runner
        self.runner.adapters = {Platform.MATTERMOST: self.adapter}

        self.authority = await initialize_session_authority(
            self.runner, profile_id="default", instance_id="instA")

        self.runner._wire_adapter_handlers(self.adapter, busy_text_mode="queue")

        self.sendlog = SendLog()
        patch_adapter_recording(monkeypatch, self.adapter, self.sendlog)
        self.calls = patch_model_transport(monkeypatch)

    def record_channel_sessions(self) -> None:
        self.max_channel_sessions_seen = max(
            self.max_channel_sessions_seen, len(session_keys_for_channel(self.chan)))

    async def deliver(self, event: dict) -> asyncio.Task:
        """Full inbound pipeline as an overlappable task (dedup, authority, claim gate, turn)."""
        task = asyncio.create_task(self.adapter._handle_ws_event(event))
        await asyncio.sleep(0)
        return task

    async def wait_turn_started(self, budget: float = 10.0) -> None:
        await async_wait_until(lambda: self.calls["async"] + self.calls["sync"] >= 1, timeout=budget)

    def trace(self, label: str) -> None:
        import json as _json

        snap = {
            "at": label,
            "calls": dict(self.calls),
            "n_sends": len(self.sendlog.sends),
            "n_edits": len(self.sendlog.edits),
            "sends": [(s["message_id"], s["content"][:70]) for s in self.sendlog.sends],
            "edits": [(e["message_id"], e["content"][:70]) for e in self.sendlog.edits],
        }
        if os.environ.get("HERMES_E2E_TRACE"):
            print("[trace] " + _json.dumps(snap))

    async def wait_answered(self, text: str, budget: float = 15.0) -> None:
        want = answer_for(text)

        def _present():
            return any(want in s["content"] for s in self.sendlog.sends) or any(
                want in e["content"] for e in self.sendlog.edits)

        await async_wait_until(_present, timeout=budget)

    async def wait_session_exists(self, budget: float = 10.0) -> None:
        await async_wait_until(lambda: len(session_keys_for_channel(self.chan)) >= 1, timeout=budget)

    def stand_downs(self) -> list[str]:
        return [ln for ln in self.journal.lines("stand-down", "channel claim")] + \
               [ln for ln in self.journal.lines("Channel claim stand-down")] + \
               [ln for ln in self.journal.lines("claim held by gateway instance")]

    def silent_stand_downs(self) -> list[str]:
        return self.journal.lines("channel_claimed_elsewhere") + \
            self.journal.lines("channel_claim_inflight")


@pytest_asyncio.fixture
async def world(monkeypatch, tmp_path):
    w = World(monkeypatch, tmp_path)
    await w.setup(monkeypatch)
    yield w
    remove_journal(w.journal)


# ── Check 1 + 2 + (drain half of 4): double delivery during a live turn ───────
@pytest.mark.asyncio
async def test_replay_double_delivery_single_answer(world, tmp_path):
    p1_text = "What is the status of the B200 nodes?" + SLOW_TOKEN
    p1 = mm_ws_event("post-p1", p1_text)

    task1 = await world.deliver(p1)
    await world.wait_session_exists()
    await world.wait_turn_started()
    world.trace("turn-started")
    world.record_channel_sessions()

    # redelivery of the SAME post id while the turn is live (platform double-fire)
    redelivery_task = await world.deliver(mm_ws_event("post-p1", p1_text))
    await redelivery_task
    world.trace("redelivery-done")

    # a second message while the channel has one live session (must queue/merge)
    p2_text = "Follow-up: and the think nodes?"
    task2 = await world.deliver(mm_ws_event("post-p2", p2_text))
    world.record_channel_sessions()

    await task1
    await world.wait_answered(p1_text)
    world.trace("p1-answered")
    world.record_channel_sessions()

    await task2
    await world.wait_answered(p2_text)
    world.trace("p2-answered")
    world.record_channel_sessions()
    await asyncio.sleep(0.5)  # health: allow HX013 -- absence window: elapsed time with no second answer IS the duplicate-detection assertion

    world.record_channel_sessions()
    assert world.max_channel_sessions_seen == 1, \
        f"channel sessions seen: {session_keys_for_channel()}"

    a1 = world.sendlog.answers_for(p1_text)
    assert len(a1) == 1, (
        f"answers for p1: {len(a1)} (expected exactly 1); "
        f"sends={[(s['message_id'], s['content'][:60]) for s in world.sendlog.sends]}; "
        f"edits={[(e['message_id'], e['content'][:60]) for e in world.sendlog.edits]}")
    a2 = world.sendlog.answers_for(p2_text)
    assert len(a2) == 1, (
        f"answers for p2: {len(a2)} (expected exactly 1); "
        f"sends={[(s['message_id'], s['content'][:60]) for s in world.sendlog.sends]}; "
        f"edits={[(e['message_id'], e['content'][:60]) for e in world.sendlog.edits]}")

    admissions = db_rows("session_admissions")
    parked = [r for r in admissions if str(r.get("state", "")).lower() in ("parked", "waiting")]
    assert not parked, f"admissions still parked: {parked}"

    # the redelivery must NOT have produced a stand-down bubble or a second stream
    assert not world.stand_downs(), f"unexpected stand-downs: {world.stand_downs()}"


# ── Check 3: thread reply routes to the thread-root session ───────────────────
@pytest.mark.asyncio
async def test_replay_thread_reply_routes_to_thread_root(world):
    p1_text = "Which vendor supplies the GPU sleds?" + SLOW_TOKEN
    p1_id = "post-t1"
    task1 = await world.deliver(mm_ws_event(p1_id, p1_text))
    await world.wait_session_exists()
    await world.wait_turn_started()

    # thread reply arrives while the root turn is live: must fold into owner lane
    p2_text = "What about the latency figures?"
    reply_task = await world.deliver(mm_ws_event("post-t2", p2_text, root_id=p1_id))
    world.record_channel_sessions()

    await task1
    await world.wait_answered(p1_text)
    await reply_task
    await world.wait_answered(p2_text, budget=30.0)
    world.record_channel_sessions()

    assert world.max_channel_sessions_seen == 1, \
        f"thread reply split the channel: {session_keys_for_channel()}"

    # incident-flavored foreign thread root must still key to the channel lane
    p3_text = "Side question from another thread"
    task3 = await world.deliver(mm_ws_event("post-t3", p3_text, root_id=INCIDENT_ROOT))
    await task3
    await world.wait_answered(p3_text)
    world.record_channel_sessions()

    assert world.max_channel_sessions_seen == 1

    for text in (p1_text, p2_text, p3_text):
        answers = world.sendlog.answers_for(text)
        assert len(answers) == 1, f"answers for {text[:30]!r}: {len(answers)}"


# ── Check 4: queued event drains when the active session turn ends ────────────
@pytest.mark.asyncio
async def test_queued_event_drains_after_turn_end(world):
    p1_text = "Run the quarterly capacity drill" + SLOW_TOKEN
    task1 = await world.deliver(mm_ws_event("post-q1", p1_text))
    await world.wait_session_exists()
    await world.wait_turn_started()

    p2_text = "Drain check: reply queued behind the live turn"
    task2 = await world.deliver(mm_ws_event("post-q2", p2_text))
    world.record_channel_sessions()

    await task1
    await world.wait_answered(p1_text)
    world.trace("p1-answered")
    world.record_channel_sessions()

    await task2
    await world.wait_answered(p2_text)
    world.record_channel_sessions()
    await asyncio.sleep(0.5)  # health: allow HX013 -- absence window: elapsed time after the queued answer proves no stray second turn surfaced

    assert world.max_channel_sessions_seen == 1
    assert len(world.sendlog.answers_for(p2_text)) == 1
    assert len(world.sendlog.answers_for(p1_text)) == 1

    admissions = db_rows("session_admissions")
    parked = [r for r in admissions if str(r.get("state", "")).lower() in ("parked", "waiting")]
    assert not parked, f"admissions did not drain: {parked}"


# ── Check 1 (heartbeat purity) + compression-config duplication guard ───────
@pytest.mark.asyncio
async def test_single_working_stream_and_single_compression_block(world, caplog):
    # pytest's caplog survives the plugin loader's logging reconfiguration
    # (a fresh CaptureHandler on world.journal does not), so the
    # compression-config check below reads REAL captured records.
    import logging as _logging

    caplog.set_level(_logging.DEBUG)
    p1_text = "Produce the long risk analysis" + SLOW_TOKEN
    task1 = await world.deliver(mm_ws_event("post-h1", p1_text))
    await world.wait_session_exists()
    await world.wait_turn_started()
    world.record_channel_sessions()
    await task1
    await world.wait_answered(p1_text)
    world.record_channel_sessions()

    hearts = world.sendlog.heartbeat_edits()
    assert hearts, "no heartbeat ticks captured (notify interval 1 s over slow turn)"
    edited_ids = {h["message_id"] for h in hearts}
    assert len(edited_ids) == 1, f"divergent heartbeat streams: {edited_ids}"

    # iteration counters must not diverge: monotone non-decreasing across ticks
    iters: list[int] = []
    for h in hearts:
        for tok in h["content"].replace(",", " ").split():
            if tok.isdigit():
                iters.append(int(tok))
                break
    if iters:
        assert iters == sorted(iters), f"heartbeat iteration counters regressed: {iters}"

    assert len(world.sendlog.answers_for(p1_text)) == 1

    caplog_records = [(r.name, r.levelname, r.getMessage()) for r in caplog.records]
    blocks = compression_block_lines(caplog_records)
    # The per-session compression-config emission is conditional (it fires on the
    # aux small-context anomaly); the incident symptom was the SAME block twice
    # for one message because two sessions booted. 0 or 1 block is clean.
    assert len(blocks) <= 1, f"compression-config blocks duplicated: {blocks}"
    assert world.max_channel_sessions_seen == 1


# ── Check 5: lease expiry after a simulated gateway kill ─────────────────────
@pytest.mark.asyncio
async def test_lease_expiry_releases_claim_and_waiting_event_admits(world, monkeypatch):
    # NOTE: production lease is 300 s; the identical expiry path runs with a 2 s
    # lease (partial-patch; the claim gate late-imports these module functions).
    w = world
    await _rebuild_world_with_lease(w, monkeypatch, lease_seconds=2)

    p1_text = "Answer this then go silent" + SLOW_TOKEN
    task1 = await w.deliver(mm_ws_event("post-x1", p1_text))
    await w.wait_session_exists()
    await w.wait_turn_started()
    w.record_channel_sessions()

    # simulated gateway kill: cancel the runner mid-turn; no renew will happen
    task1.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task1

    # claim still held for up to the lease window (holder named, not released)
    await async_wait_until(lambda: bool(channel_claim_rows()), timeout=2)
    rows = channel_claim_rows()
    assert rows, "no channel claim row after kill"
    lease_expire = float(rows[0]["lease_expire"])

    # let the lease expire (kill semantics: nothing renews), then a waiting event
    await async_wait_until(lambda: time.time() >= lease_expire + 0.3, timeout=6)
    p2_text = "Waiting event after the crash window"
    task2 = await w.deliver(mm_ws_event("post-x2", p2_text))
    await task2
    await w.wait_answered(p2_text)
    await asyncio.sleep(0.5)  # health: allow HX013 -- absence window: elapsed time with no stand-down proves the crashed claim is fully superseded

    # claim re-acquired (release-on-expiry + fresh admission), single lane, one answer
    rows_after = channel_claim_rows()
    assert rows_after, "claim row missing after expiry re-acquire"
    assert rows_after[0]["lease_expire"] >= lease_expire, "lease not renewed by new admission"
    claims_now = {r["session_id"] for r in rows_after}
    assert len(claims_now) == 1, f"claim split across sessions: {claims_now}"

    assert w.max_channel_sessions_seen == 1, f"sessions during/after kill: {session_keys_for_channel()}"
    assert len(w.sendlog.answers_for(p2_text)) == 1
    assert len(w.sendlog.answers_for(p1_text)) <= 1
    assert not w.stand_downs(), "stand-down fired for a waiting event after expiry"


async def _rebuild_world_with_lease(w, monkeypatch, *, lease_seconds: int) -> None:
    """Re-create the world with the short-lease partials active."""
    remove_journal(w.journal)
    await w.setup(monkeypatch, lease_seconds=lease_seconds)


# ── Check 6: second gateway on the same channel stands down (cross-instance) ──
@pytest.mark.asyncio
async def test_cross_instance_second_gateway_stands_down(world, monkeypatch, tmp_path):
    relay_root = tmp_path / "relay-root"
    relay_root.mkdir()
    monkeypatch.setenv("GATEWAY_RELAY_INSTANCE_ID", "gatewayA-workstation")

    # pipe A must write its relay claim into the shared routing root
    world.runner.session_store._routing_home = relay_root

    p1_text = "Claim the channel for gateway A" + SLOW_TOKEN
    task1 = await world.deliver(mm_ws_event("post-c1", p1_text))
    await world.wait_session_exists()
    await world.wait_turn_started()
    world.record_channel_sessions()
    await task1
    await world.wait_answered(p1_text)
    world.record_channel_sessions()

    # A now holds channel_claims for the dm lane in the shared relay root
    claim_files = [p for p in relay_root.rglob("*") if p.is_file()]
    assert claim_files, "gateway A wrote no relay claim under the shared routing root"

    job = {
        "home": str(tmp_path / "homeB"),
        "relay_root": str(relay_root),
        "instance_id": "gatewayB-mac",
        "event": mm_ws_event("post-c2", "Second instance asks the same DM"),
    }
    job_path = tmp_path / "jobB.json"
    job_path.write_text(json.dumps(job))

    driver = Path(__file__).resolve().parent / "_pipe_b_driver.py"
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(driver), str(job_path),
        cwd=str(REPO),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await asyncio.wait_for(proc.communicate(), timeout=120)
    (tmp_path / "pipeB.err").write_text(err.decode(errors="replace"))
    (tmp_path / "pipeB.out").write_text(out.decode(errors="replace"))
    assert proc.returncode == 0, f"pipe B driver failed: {err.decode()[-4000:]}"
    result = json.loads(out.decode().strip().splitlines()[-1])

    # the stand-down: B posts nothing, its journal names the foreign holder
    assert result["sends"] == [], f"gateway B posted on someone else's channel: {result['sends']}"
    assert result["session_keys"] == [], "gateway B admitted a session on a held channel"
    standdown_lines = [ln for ln in result["log"] if
                       ("standing down" in ln and "claim" in ln)
                       or "stand-down" in ln
                       or "channel_claim_standdown" in ln]
    assert standdown_lines, "no stand-down line in gateway B journal:\n" + "\n".join(result["log"][-40:])
    assert any("gatewayA" in ln for ln in standdown_lines) or any("instA" in ln for ln in result["log"]), \
        "stand-down does not name the holding gateway instance"

    # A remains the sole voice: one session, one answer for its own message
    assert world.max_channel_sessions_seen == 1
    assert len(world.sendlog.answers_for(p1_text)) == 1

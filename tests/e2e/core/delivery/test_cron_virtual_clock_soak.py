"""C13 — virtual-clock soak of the REAL cron ticker loop (~30 virtual days per scenario).

Real: ``InProcessCronScheduler.start`` (the gateway's ticker loop: startup recovery, tick, heartbeat
bookkeeping), ``cron.scheduler.tick`` / due scan / pending slots / fire claims + heartbeat /
executions ledger / delivery routing / ``mark_job_run``, a second OS process sharing HERMES_HOME
(its ticker races the in-process one for every tick; the tick lock admits one). Fire-claim
contention between two OS processes is ``test_two_replicas_contend_for_every_fire``.
Fake: the agent (``cron.scheduler.run_job``) and the platform wire (``_send_to_platform``), which
record what they were handed. The gateway housekeeping thread's drain of the durable delivery
queue is performed by the driver after every tick settles (``H.drain_deliveries``).
Time: one file-backed virtual clock (see ``_cron_clock``).

Oracle (independent of cron/jobs.py): per job, the expected fire set is computed from croniter
over the job's configured zone (hermes ``timezone`` if set, else the process zone — "server
local"), stepping only through the tick instants the harness actually drove, with the documented
policies:
* a wall time that does not exist (spring-forward gap) fires once, shifted by the gap; a repeated
  wall time (fall-back fold) fires once, on its first occurrence;
* a missed recurring occurrence fires ONCE on the first tick after it (``catch_up_missed``
  default true), whether within grace (late) or beyond it (catch-up), then re-anchors on "now";
* intervals re-anchor on the fire instant; a one-shot beyond ``ONESHOT_GRACE_SECONDS`` never
  fires; manual runs (direct or trigger) never consume or re-stamp the next scheduled occurrence.
After every tick: the actual fires == expected fires, every stored ``next_run_at`` equals the
model's pending occurrence and is strictly after virtual now, and the ticker reported no error.
A run held past the fire-claim TTL: after every virtual 30 s its claim was refreshed by the run's
heartbeat, and a contender's ``claim_job_for_fire`` loses to it.
At the end: no execution is non-terminal, no queued send is left undrained; every row's status
and delivery outcome (a ``queued`` handoff resolved through the queue's verdict) matches what the
fake runner/sink actually saw (delivered => the sink got exactly that execution's message).
"""

from __future__ import annotations

import bisect
import dataclasses
import math
import os
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pytest

croniter_mod = pytest.importorskip("croniter")

from tests.e2e.core.delivery import _cron_clock as H  # noqa: E402
from tests.e2e.core.delivery._pending_fixes import gap_open  # noqa: E402

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="flock/SIGKILL multi-process soak is Linux-only")

REPO_ROOT = Path(__file__).resolve().parents[4]
GRID_SECONDS = 6 * 3600  # sparse idle ticks; due/pre-due ticks are added per occurrence
HOLD_STEPS = 30  # long run: 30 x 30 s = 15 virtual minutes (> 5 min fire-claim TTL)
UTC = timezone.utc


@dataclasses.dataclass(frozen=True)
class Scenario:
    id: str
    hermes_tz: Optional[str]
    process_tz: str
    start: date  # DST day is always start + 9
    child: bool = False  # a second ticker process + SIGKILL/restart mid-run
    days: int = 30


SCENARIOS = [
    # nothing configured, process in UTC: server-local == UTC
    Scenario("unset_utc_spring", None, "UTC", date(2026, 2, 27), child=True),
    # configured zone with no DST under a foreign DST process zone (process DST night inside)
    Scenario("shanghai_on_la_fall", "Asia/Shanghai", "America/Los_Angeles", date(2026, 10, 23)),
    # configured US zone across spring-forward, fixed +08:00 process zone
    Scenario("newyork_on_shanghai_spring", "America/New_York", "Asia/Shanghai", date(2026, 2, 27)),
    # configured US zone across fall-back, fixed +08:00 process zone, multi-process faults
    Scenario("newyork_on_shanghai_fall", "America/New_York", "Asia/Shanghai",
             date(2026, 10, 23), child=True),
]
# Scenario id -> (fix PR, gap). While the PR's probe still reproduces the defect (see
# _pending_fixes) ONLY the schedule oracle's two checks (fired set, stored next_run_at) may deviate:
# each deviation is recorded, the model adopts the stored slot, and the soak runs on to its last
# virtual day with every other check live; the cell XFAILs at the end only if a deviation was seen
# (and fails if none was: the probe and the soak disagree). Once the fix is in the tree the
# scenario is a plain test again.
GAPS = {
    # #119969: with NO timezone configured the next occurrence keeps the base time's fixed UTC
    # offset, so a DST process zone fires 09:00 at 10:00 local after spring-forward
    "unset_on_newyork_spring": (119970, "#119969 no-tz DST fixed offset (fixed by #119970)"),
}
SCENARIOS.append(Scenario("unset_on_newyork_spring", None, "America/New_York", date(2026, 2, 27),
                          days=12))


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat()


def _ceil_minute(ts: float) -> float:
    return math.ceil(ts / 60.0) * 60.0


# --- independent schedule model --------------------------------------------------------------

class CronOccurrences:
    """All instants of a cron expr in ``zone`` over a window: wall times from croniter (naive),
    each mapped with fold=0 (gap -> shifted forward, fold -> first occurrence), deduped."""

    def __init__(self, expr: str, zone, start_ts: float, end_ts: float):
        wall = datetime.fromtimestamp(start_ts, zone).replace(tzinfo=None) - timedelta(days=2)
        stop = datetime.fromtimestamp(end_ts, zone).replace(tzinfo=None) + timedelta(days=2)
        it = croniter_mod.croniter(expr, wall)
        instants = set()
        while True:
            w = it.get_next(datetime)
            if w > stop:
                break
            instants.add(w.replace(tzinfo=zone, fold=0).timestamp())
        self.instants = sorted(instants)

    def first_after(self, ts: float) -> float:
        return self.instants[bisect.bisect_right(self.instants, ts)]


@dataclasses.dataclass
class ModelJob:
    name: str
    kind: str  # cron | interval | once
    pending: Optional[float]
    occ: Optional[CronOccurrences] = None
    interval: float = 0.0
    times: Optional[int] = None
    fired: int = 0
    trigger: bool = False
    done: bool = False

    def advance_from(self, t: float) -> None:
        if self.times is not None and self.fired >= self.times:
            self.done, self.pending = True, None
        elif self.kind == "cron":
            self.pending = self.occ.first_after(t)
        elif self.kind == "interval":
            self.pending = t + self.interval
        else:
            self.done, self.pending = True, None


class Model:
    def __init__(self):
        self.jobs: Dict[str, ModelJob] = {}
        self.fires: List[tuple] = []  # (name, instant_iso | None, fired_at)

    def active(self):
        return [j for j in self.jobs.values() if not j.done]

    def tick(self, t: float) -> None:
        for job in self.active():
            if job.trigger:
                job.trigger = False
                job.fired += 1
                self.fires.append((job.name, None, t))
                job.advance_from(t)
                continue
            if job.pending is None or job.pending > t:
                continue
            if job.kind == "once" and t - job.pending > 120:  # ONESHOT_GRACE_SECONDS
                job.done, job.pending = True, None
                continue
            job.fired += 1
            self.fires.append((job.name, _iso(job.pending), t))
            job.advance_from(t)

    def manual(self, name: str, t: float) -> None:
        job = self.jobs[name]
        job.fired += 1
        self.fires.append((name, None, t))
        if job.kind == "interval":
            job.pending = t + job.interval
        if job.times is not None and job.fired >= job.times:
            job.done, job.pending = True, None


# --- the soak driver -------------------------------------------------------------------------

@dataclasses.dataclass(order=True)
class Event:
    ts: float
    seq: int
    kind: str = dataclasses.field(compare=False)
    arg: object = dataclasses.field(compare=False, default=None)


class Soak:
    def __init__(self, sc: Scenario, home: Path, control: H.Control, env: Dict[str, str]):
        self.sc, self.home, self.control = sc, home, control
        self.zone = ZoneInfo(sc.hermes_tz or sc.process_tz)
        self.clock = H.VirtualClock(control.clock_file)
        self.start_ts = self.local(0, 0, 0, 30)
        self.end_ts = self.local(sc.days, 0, 0, 0)
        self.model = Model()
        self.host = H.SchedulerHost()
        self.child = H.ChildHost(control, env, REPO_ROOT) if sc.child else None
        self.parent_up = False
        self.ids: Dict[str, str] = {}
        self.seen_runs = 0
        self.held: set = set()
        self.killed_execs: set = set()
        self.ticks = 0
        self.events: List[Event] = []
        pr = GAPS.get(sc.id, (None,))[0]
        self.tolerate_gap = pr is not None and gap_open(pr)
        self.gap_hits: List[str] = []

    # helpers
    def local(self, day: int, hh: int, mm: int, ss: int = 0) -> float:
        d = self.sc.start + timedelta(days=day)
        return datetime(d.year, d.month, d.day, hh, mm, ss, tzinfo=self.zone).timestamp()

    def at(self, ts: float, kind: str, arg=None) -> None:
        self.events.append(Event(ts, len(self.events), kind, arg))

    def now(self) -> float:
        return self.clock.now_ts()

    def up_hosts(self):
        hosts = [self.host] if self.parent_up else []
        if self.child is not None and self.child.alive:
            hosts.append(self.child)
        return hosts

    def describe(self, t: float) -> str:
        return f"[{self.sc.id} @ {datetime.fromtimestamp(t, self.zone).isoformat()}]"

    # job creation through the real store API
    def create(self, name: str, schedule: str, *, repeat=None, chat="1000"):
        from cron.jobs import create_job

        job = create_job(prompt=f"soak {name}", schedule=schedule, name=name, repeat=repeat,
                         deliver=f"telegram:{chat}")
        self.ids[name] = job["id"]
        t = self.now()
        from cron.jobs import parse_schedule

        parsed = parse_schedule(schedule)
        if parsed["kind"] == "cron":
            mj = ModelJob(name, "cron", None, occ=CronOccurrences(
                parsed["expr"], self.zone, self.start_ts, self.end_ts), times=repeat)
            mj.pending = mj.occ.first_after(t)
        elif parsed["kind"] == "interval":
            mj = ModelJob(name, "interval", t + parsed["minutes"] * 60,
                          interval=parsed["minutes"] * 60, times=repeat)
        else:
            mj = ModelJob(name, "once", datetime.fromisoformat(parsed["run_at"]).timestamp(),
                          times=1)
        self.model.jobs[name] = mj

    # stepping
    def next_tick(self, now: float) -> float:
        if not self.up_hosts():
            return math.inf
        cands = [(math.floor(now / GRID_SECONDS) + 1) * GRID_SECONDS]
        for job in self.model.active():
            if job.trigger:
                cands.append(now + 30)
            if job.pending is not None:
                due = _ceil_minute(job.pending)
                cands += [c for c in (due - 60, due) if c > now]
        return min(c for c in cands if c > now)

    def settle(self) -> None:
        import cron.scheduler as sched

        held_ids = {self.ids[n] for n in self.held}

        def quiet():
            if set(sched.get_running_job_ids()) - held_ids:  # cheap in-process check first
                return False
            return not [r for r in H.non_terminal(self.home) if r["job_id"] not in held_ids]

        H.wait_until(quiet, f"scheduler quiescence {self.describe(self.now())}")
        H.drain_deliveries()  # the gateway's housekeeping drain, once the runs have booked their sends

    def tick(self, hosts=None, entering: Optional[str] = None) -> None:
        t = self.now()
        H.tick_together(self.up_hosts() if hosts is None else hosts)
        self.ticks += 1
        if entering is not None:  # a held run: wait until it has really started
            self._entered_exec(entering)
        self.settle()
        self.model.tick(t)
        self.check(t)

    def check(self, t: float) -> None:
        from cron.jobs import get_ticker_last_error

        where = self.describe(t)
        starts = [r for r in H.read_jsonl(self.control.runs) if r["event"] == "start"]
        actual = sorted((r["name"], r["instant"], r["fired_at"]) for r in starts[self.seen_runs:])
        self.seen_runs = len(starts)
        expected = sorted(self.model.fires)
        self.model.fires = []
        if actual != expected:
            self.gap_deviation(f"{where} fired set mismatch: actual={actual} expected={expected}")
        err = get_ticker_last_error()
        assert not err, f"{where} ticker recorded an error: {err}"
        by_id = {j["id"]: j for j in H.store_jobs(self.home)}
        for name, mj in self.model.jobs.items():
            stored = by_id.get(self.ids.get(name))
            if mj.done or stored is None:
                continue
            nra = stored.get("next_run_at")
            assert nra, f"{where} {name}: active job lost next_run_at: {stored}"
            nra_ts = datetime.fromisoformat(nra).timestamp()
            assert nra_ts > t, f"{where} {name}: next_run_at {nra} not after virtual now"
            if nra_ts != mj.pending:
                self.gap_deviation(f"{where} {name}: next_run_at {nra} != expected {_iso(mj.pending)}")
                mj.pending = nra_ts  # follow the store so the rest of the soak stays meaningful

    def gap_deviation(self, msg: str) -> None:
        """A schedule-oracle deviation: a failure, unless it is this scenario's open known gap."""
        if not self.tolerate_gap:
            raise AssertionError(msg)
        self.gap_hits.append(msg)

    # fault handlers
    def handle(self, ev: Event) -> None:
        getattr(self, f"ev_{ev.kind}")(ev.arg)

    def ev_create(self, arg):
        self.create(*arg[:2], **arg[2])

    def ev_behavior(self, arg):
        self.control.set_behaviors(arg)

    def ev_manual(self, name):
        from cron.jobs import get_job
        from tools.cronjob_tools import _execute_job_now

        t = self.now()
        result = _execute_job_now(get_job(self.ids[name]))
        assert result.get("claimed") is not False, f"{self.describe(t)} manual run refused: {result}"
        self.settle()
        self.model.manual(name, t)
        self.check(t)

    def ev_trigger(self, name):
        from cron.jobs import trigger_job

        trigger_job(self.ids[name])
        self.model.jobs[name].trigger = True

    def ev_down(self, _arg):
        if self.parent_up:
            self.host.stop()
            self.parent_up = False

    def ev_up(self, _arg):
        self.host.start()  # boot: startup recovery + immediate tick
        self.parent_up = True
        self.ticks += 1
        t = self.now()
        rows = {r["id"]: r for r in H.ledger_rows(self.home)}
        for exec_id in self.killed_execs:
            assert rows[exec_id]["status"] not in ("claimed", "running"), (
                f"{self.describe(t)} restart left the SIGKILLed run non-terminal: {rows[exec_id]}")
        self.settle()
        self.model.tick(t)
        self.check(t)

    def ev_child_up(self, _arg):
        t = self.now()
        self.child.start()  # child's boot tick at t
        H.tick_together([self.host] if self.parent_up else [])
        self.ticks += 1
        self.settle()
        self.model.tick(t)
        self.check(t)

    def _entered_exec(self, name: str) -> str:
        entered = self.control.gates / f"{name}.entered"

        def exec_id():  # written atomically by the fake agent; never read a half-written file
            return entered.exists() and entered.read_text(encoding="utf-8").strip()

        return H.wait_until(exec_id, f"{name} run to start")

    def ev_long_run(self, name):
        """A run that outlives the fire-claim TTL while the ticker keeps ticking."""
        job_id = self.ids[name]
        hold = self.control.hold_file(name)
        hold.write_text("1", encoding="utf-8")
        self.held.add(name)
        self.tick(entering=name)
        exec_id = self._entered_exec(name)
        t0 = self.now()
        from cron.jobs import claim_job_for_fire

        for k in range(1, HOLD_STEPS + 1):
            self.clock.set(t0 + 30 * k)
            want = t0 + 30 * k - 1

            def fresh():
                job = next(j for j in H.store_jobs(self.home) if j["id"] == job_id)
                claim = job.get("fire_claim") or {}
                return claim.get("at") and datetime.fromisoformat(claim["at"]).timestamp() >= want

            # The run's heartbeat thread must keep the lease fresh at virtual cadence; a broken
            # heartbeat fails here, not silently later.
            H.wait_until(fresh, f"{self.describe(self.now())} {name}: fire-claim heartbeat refresh")
            # A contender (another replica's fire, a manual run) must meet a LIVE claim, also
            # after the claim's first stamp is older than the TTL.
            assert claim_job_for_fire(job_id) is False, (
                f"{self.describe(self.now())} {name}: a contender took the fire claim "
                f"{30 * k}s into a live run (lease not kept fresh)")
            self.tick()
        hold.unlink()
        self.held.discard(name)
        self.settle()
        self.long_run_exec = exec_id

    def ev_solo_child(self, _arg):
        """Only the child process ticks from here (the in-process gateway is down)."""
        self.ev_down(None)

    def ev_child_kill_run(self, name):
        """The child fires ``name``, and is SIGKILLed while the run holds a live claim."""
        hold = self.control.hold_file(name)
        hold.write_text("1", encoding="utf-8")
        self.held.add(name)
        t = self.now()
        H.tick_together([self.child])
        self.ticks += 1
        exec_id = self._entered_exec(name)
        self.model.tick(t)
        self.child.kill()
        self.killed_execs.add(exec_id)
        rows = {r["id"]: r for r in H.ledger_rows(self.home)}
        assert rows[exec_id]["status"] == "running", rows[exec_id]
        hold.unlink()
        (self.control.gates / f"{name}.entered").unlink()
        self.held.discard(name)
        self.check(t)

    # main
    def schedule_faults(self) -> None:
        L = self.local
        self.at(L(2, 12, 17, 30), "manual", "brief-0900")
        self.at(L(3, 8, 59, 30), "manual", "brief-0900")  # inside the 60 s pre-due skew
        self.at(L(4, 8, 0), "behavior", {"brief-0900": {"delivery": "fail"}})
        self.at(L(4, 10, 0), "behavior", {})
        self.at(L(5, 2, 0), "behavior", {"gap-0230": {"agent": "fail"}})
        self.at(L(5, 3, 0), "behavior", {})
        self.at(L(6, 8, 0), "behavior", {"brief-0900": {"delivery": "slow"}})
        self.at(L(6, 10, 0), "behavior", {})
        self.at(L(8, 0, 0, 10), "create", ("every-2h", "every 2h", {"repeat": 48, "chat": "1003"}))
        self.at(L(8, 20, 10, 10), "trigger", "every-2h")
        self.at(L(9, 0, 5), "create", ("quarter", "*/15 * * * *", {"repeat": 24, "chat": "1004"}))
        self.at(L(10, 9, 0), "long_run", "brief-0900")
        once_b = datetime.fromtimestamp(L(11, 10, 0), self.zone).isoformat()
        self.at(L(11, 6, 0), "create", ("oneshot-missed", once_b, {"chat": "1006"}))
        self.at(L(11, 7, 0, 5), "down")  # 5 h outage: 09:00 beyond 2 h grace -> one catch-up
        self.at(L(11, 12, 0, 20), "up")
        if self.sc.days <= 13:
            return
        self.at(L(13, 7, 30, 5), "down")  # 09:00 missed by 45 min -> fires late, once
        self.at(L(13, 9, 45, 20), "up")
        if self.child is not None:
            self.at(L(15, 0, 0), "child_up")  # both tickers tick every instant; the tick lock admits one
            self.at(L(17, 2, 0, 5), "solo_child")
            self.at(L(17, 2, 30), "child_kill_run", "gap-0230")
            self.at(L(17, 3, 10, 20), "up")  # gateway restart: startup recovery
        self.events.sort()

    def run(self) -> None:
        self.clock.set(self.start_ts)
        self.create("brief-0900", "0 9 * * *", chat="1001")
        self.create("gap-0230", "30 2 * * *", chat="1002")
        once_a = datetime.fromtimestamp(self.local(5, 12, 34), self.zone).isoformat()
        self.create("oneshot", once_a, chat="1005")
        self.schedule_faults()
        self.events.sort()
        self.ev_up(None)
        while True:
            now = self.now()
            nxt_tick = self.next_tick(now)
            nxt_ev = self.events[0].ts if self.events else math.inf
            target = min(nxt_tick, nxt_ev)
            if target >= self.end_ts:
                break
            self.clock.set(max(target, now))
            if nxt_ev <= nxt_tick:
                self.handle(self.events.pop(0))
            else:
                self.tick()
        self.clock.set(self.end_ts)
        self.settle()
        if self.child is not None:
            self.child.stop()
        self.ev_down(None)

    # final truth checks
    def final_checks(self) -> Dict[str, int]:
        rows = {r["id"]: r for r in H.ledger_rows(self.home)}
        runs = H.read_jsonl(self.control.runs)
        starts = {r["exec"]: r for r in runs if r["event"] == "start"}
        ends = {r["exec"] for r in runs if r["event"] == "end"}
        sink = H.read_jsonl(self.control.sink)
        deliveries = H.delivery_status(self.home)
        where = f"[{self.sc.id} final]"
        stuck = [r for r in rows.values() if r["status"] in ("claimed", "running")]
        assert not stuck, f"{where} non-terminal executions left: {stuck}"
        undrained = {e: s for e, s in deliveries.items() if s in ("pending", "delivering")}
        assert not undrained, f"{where} queued sends never drained: {undrained}"
        assert set(starts) - ends == self.killed_execs, f"{where} runs without an end"
        assert len(starts) == len(set(starts)), f"{where} duplicate execution ids"
        for exec_id, start in starts.items():
            row = rows.get(exec_id)
            assert row is not None, f"{where} run {exec_id} ({start['name']}) has no ledger row"
            sent = [s for s in sink if s["exec"] == exec_id]
            ok_sent = [s for s in sent if s["ok"]]
            outcome = H.effective_outcome(row, deliveries)
            label = f"{where} {start['name']} exec={exec_id} status={row['status']} " \
                    f"outcome={row['delivery_outcome']}/{outcome} sink={sent} error={row.get('error')!r}"
            if exec_id in self.killed_execs:
                assert row["status"] == "unknown", label
                assert not sent, label
                continue
            assert len(sent) <= 1, f"{label}: delivered more than once"
            if outcome == "delivered":
                assert len(ok_sent) == 1, f"{label}: 'delivered' but the sink never got it"
            if ok_sent:
                assert outcome == "delivered", f"{label}: sink got it, ledger says not"
            if sent and not ok_sent:
                assert outcome == "failed", f"{label}: failed send not recorded"
            assert "nterrupt" not in str(row.get("error") or ""), f"{label}: marked interrupted"
            agent_failed = "provider outage" in str(row.get("error") or "")
            if agent_failed:
                assert row["status"] == "failed", label
            else:
                assert row["status"] == "completed", label
                assert ok_sent or sent, f"{label}: successful run never reached the platform"
                assert not ok_sent or ok_sent[0]["exec_in_text"], f"{label}: wrong body delivered"
        for exec_id, row in rows.items():
            if exec_id not in starts:  # attempts that never ran must say so truthfully
                assert row["status"] in ("failed", "unknown", "skipped"), f"{where} ghost row {row}"
        if hasattr(self, "long_run_exec"):
            row = rows[self.long_run_exec]
            assert (row["status"], H.effective_outcome(row, deliveries)) == ("completed", "delivered"), (
                f"{where} long run lost its lease: {row}")
        for job in H.store_jobs(self.home):
            assert not job.get("pending_slot"), f"{where} leftover pending_slot {job}"
        return {"ticks": self.ticks, "executions": len(rows), "sink": len(sink)}


@pytest.fixture
def soak_env(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.endswith("_API_KEY") or key in ("INVOCATION_ID", "HERMES_TIMEZONE"):
            monkeypatch.delenv(key, raising=False)
    home_root = tmp_path / "home"
    hermes_home = home_root / ".hermes"
    hermes_home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home_root))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    original_tz = os.environ.get("TZ")
    yield tmp_path, hermes_home, monkeypatch
    if original_tz is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = original_tz
    time.tzset()
    import hermes_time

    hermes_time.reset_cache()


def _run_scenario(sc: Scenario, soak_env) -> Dict[str, int]:
    tmp_path, hermes_home, monkeypatch = soak_env
    monkeypatch.setenv("TZ", sc.process_tz)
    time.tzset()
    lines = ["platforms:", "  telegram:", "    enabled: true", "    token: fake-soak-token"]
    if sc.hermes_tz:
        lines.insert(0, f"timezone: {sc.hermes_tz}")
    (hermes_home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    control = H.Control(tmp_path / "control").ensure()
    import cron.jobs as jobs

    # Liveness markers fsync twice per loop iteration (the dominant wall cost at virtual
    # cadence); they are diagnostics, not scheduling, and are covered by tests/cron.
    monkeypatch.setattr(jobs, "record_ticker_heartbeat", lambda **_kw: None)
    H.install(H.VirtualClock(control.clock_file), control, monkeypatch.setattr)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env["PYTHONPATH"] = str(REPO_ROOT)
    soak = Soak(sc, hermes_home, control, env)
    try:
        soak.run()
    finally:
        # A failure mid-hold must not leak a parked run (and its writes) into the next scenario.
        for name in list(soak.held):
            control.hold_file(name).unlink(missing_ok=True)
        import cron.scheduler as sched

        H.wait_until(lambda: not sched.get_running_job_ids(), f"{sc.id}: held runs to drain")
        if soak.child is not None:
            soak.child.kill()
        soak.ev_down(None)
    return soak.final_checks(), soak.gap_hits


@pytest.mark.parametrize("sc", [pytest.param(s, id=s.id) for s in SCENARIOS])
def test_cron_virtual_clock_soak(sc, soak_env):
    stats, gap_hits = _run_scenario(sc, soak_env)
    print(f"C13 {sc.id}: {stats}")
    assert sc.days < 30 or stats["executions"] > 60
    if gap_hits:  # only reachable while GAPS[sc.id]'s probe says the defect reproduces
        pytest.xfail(f"{GAPS[sc.id][1]}: {len(gap_hits)} deviation(s), first: {gap_hits[0]}")
    assert sc.id not in GAPS or not gap_open(GAPS[sc.id][0]), (
        f"#{GAPS[sc.id][0]} probe says the defect reproduces, but the soak never deviated")


# --- two replicas, one store: the fire claim decides every fire ---------------------------------

REPLICA_ROUNDS = 12


def test_two_replicas_contend_for_every_fire(soak_env):
    """Two external-provider replicas (this process + a second OS process) receive the same fire
    for every due occurrence and call the real ``fire_due`` at once (a barrier parks each inside
    ``claim_job_for_fire`` until both are there). The winner's run is held open, so the loser
    always meets a live claim. Per round: exactly one replica claims, exactly one run starts for
    that slot and is delivered once, the loser's attempt is recorded as not acquired, and the
    store re-arms to a later slot, so no occurrence is fired twice or skipped."""
    tmp_path, hermes_home, monkeypatch = soak_env
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    (hermes_home / "config.yaml").write_text(
        "platforms:\n  telegram:\n    enabled: true\n    token: fake-soak-token\n", encoding="utf-8")
    control = H.Control(tmp_path / "control").ensure()
    import cron.jobs as jobs

    monkeypatch.setattr(jobs, "record_ticker_heartbeat", lambda **_kw: None)
    clock = H.VirtualClock(control.clock_file)
    H.install(clock, control, monkeypatch.setattr)
    H.install_claim_barrier(control, monkeypatch.setattr)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env["PYTHONPATH"] = str(REPO_ROOT)
    clock.set(datetime(2026, 3, 1, 0, 0, 30, tzinfo=UTC).timestamp())
    names = {}
    for name, schedule in (("hourly", "0 * * * *"), ("every-45m", "every 45m")):
        names[name] = jobs.create_job(prompt=f"replica {name}", schedule=schedule, name=name,
                                      deliver="telegram:2000")["id"]
    provider = H.replica_scheduler()
    peer = H.ReplicaHost(control, env, REPO_ROOT)
    peer.start()
    fired = []
    try:
        for rnd in range(1, REPLICA_ROUNDS + 1):
            slot, name = min((datetime.fromisoformat(jobs.get_job(i)["next_run_at"]).timestamp(), n)
                             for n, i in names.items())  # the next due occurrence of either job
            job_id = names[name]
            assert slot > clock.now_ts(), f"round {rnd} {name}: slot {_iso(slot)} already passed"
            clock.set(slot + 5)  # the fire arrives 5 s after its slot at both replicas
            hold = control.hold_file(name)
            hold.write_text("1", encoding="utf-8")
            mine: Dict[str, bool] = {}

            def fire_here(job_id=job_id, rnd=rnd):
                mine["won"] = H.fire_as_replica(provider, job_id, rnd)

            t = threading.Thread(target=fire_here)
            t.start()
            peer.request(rnd, job_id)

            def both_claimed():
                return [c for c in H.read_jsonl(control.claims) if c["round"] == rnd][1:]

            H.wait_until(both_claimed, f"round {rnd}: both replicas' claim outcome")
            entered = control.gates / f"{name}.entered"
            H.wait_until(entered.exists, f"round {rnd}: the winner's run to start")
            hold.unlink()
            t.join(timeout=H.DEADLINE_SECONDS)
            assert not t.is_alive(), f"round {rnd}: in-process replica never finished"
            outcomes = sorted([mine["won"], peer.result(rnd)])
            entered.unlink()
            H.drain_deliveries()  # the gateway's housekeeping drain of the winner's queued send
            claims = [c for c in H.read_jsonl(control.claims) if c["round"] == rnd]
            assert sorted(c["pid"] for c in claims) == sorted([os.getpid(), peer.pid]), claims
            assert sorted(c["won"] for c in claims) == [False, True], f"round {rnd}: {claims}"
            assert outcomes == [False, True], f"round {rnd}: fire_due results {outcomes}"
            starts = [r for r in H.read_jsonl(control.runs) if r["event"] == "start"]
            fired.append((name, slot, next(c["pid"] for c in claims if c["won"])))
            assert len(starts) == rnd, f"round {rnd} {name}: {len(starts)} runs started for {rnd} fires"
            assert datetime.fromisoformat(starts[-1]["instant"]).timestamp() == slot, starts[-1]
            nxt = datetime.fromisoformat(jobs.get_job(job_id)["next_run_at"]).timestamp()
            assert nxt > clock.now_ts(), f"round {rnd} {name}: not re-armed past now ({_iso(nxt)})"
    finally:
        peer.stop()
    rows = H.ledger_rows(hermes_home)
    deliveries = H.delivery_status(hermes_home)
    sink = H.read_jsonl(control.sink)
    starts = {r["exec"] for r in H.read_jsonl(control.runs) if r["event"] == "start"}
    won = [r for r in rows if r["id"] in starts]
    lost = [r for r in rows if r["id"] not in starts]
    assert len(won) == len(lost) == REPLICA_ROUNDS, (len(won), len(lost), rows)
    assert all((r["status"], H.effective_outcome(r, deliveries)) == ("completed", "delivered")
               for r in won), (won, deliveries)
    assert all(r["status"] == "failed" and "not acquired" in (r["error"] or "") for r in lost), lost
    assert sorted(s["exec"] for s in sink if s["ok"]) == sorted(starts), sink
    print(f"C13 replicas: winners by pid {[(n, pid) for n, _, pid in fired]}")

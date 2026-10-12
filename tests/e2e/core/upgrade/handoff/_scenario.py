"""One user's machine across one real ``hermes update``, observed once per column.

A manual gateway (with its cron ticker and embedded kanban dispatcher) and a manual dashboard run
in one shared-PID-namespace sandbox. A kanban worker is in flight and a one-shot cron job comes due
mid-update. The scenario runs once per column and records what the user would see; each property
is its own test over that record, gated on its own issue. The expensive part (stage, boot, update,
settle) is paid once per column, not once per property.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import re
import secrets
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.e2e.core._pending_fixes import known_failure, known_gate
from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade.handoff import _handoff as X
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall

CRON_DUE_S = 65  # after the update starts: the checkout holds the new code, the old gateway still ticks
_TASK_RE = re.compile(r"work kanban task (t_[0-9a-f]+)")
_PORT_IN_USE = re.compile(r"address already in use|port \d+ is (already )?in use|errno 98", re.IGNORECASE)
REPLY = "handoff reply"
_IMPORT_ERR = re.compile(r"ModuleNotFoundError|ImportError|No module named|cannot import name")

# Open issues behind a property, per column (a fix PR deletes its entry). Each pattern matches only the
# verdict text its bug produces; any other failure of the same property propagates.
# A dashboard started from a shell that runs under a systemd service (every GitHub-hosted job runs in
# hosted-compute-agent.service) sits in that unit's cgroup; the update restarts the unit instead of
# respawning the dashboard. Fixed at HEAD by #124940; the N-1 column keeps this gate until a release
# carrying the fix is N-1.
_FOREIGN_UNIT_GATE = (r"the update restarted the systemd unit \S+ the dashboard was started under",
                      "gated on #124938: the update restarts the systemd unit whose cgroup a manual dashboard "
                      "was started in instead of respawning the dashboard")
DASHBOARD_GATES = {"n1": [_FOREIGN_UNIT_GATE]}
CRON_GATES = {
    "n1": (r"fired into the update swap window and failed importing",
           "gated on #113293: a cron job due during an update fires into the swap window, fails, and loses its slot"),
}

# ``hermes update``'s own line for each systemd unit it restarted (or tried to) after stopping a dashboard.
_UNIT_RESTART = re.compile(r"✓ restarted systemd service (\S+\.service)|⚠ (\S+\.service): systemctl restart returned")


@contextlib.contextmanager
def _gated(entries):
    """``known_failure`` over several open issues: the one whose pattern the failure matches xfails it."""
    with contextlib.ExitStack() as stack:
        for pattern, reason in entries:
            stack.enter_context(known_failure(pattern, reason))
        yield


class Model:
    """The fake provider's brain. Chat turns get ``REPLY``; the cron job's prompt is counted; each
    kanban worker completes its own card (``kanban_complete`` counted per card), and the first card's
    worker blocks on its first call until ``release`` so it spans the whole update."""

    def __init__(self, cron_marker: str) -> None:
        self.cron_marker = cron_marker
        self.cron_calls: list[float] = []
        self.completes: Counter[str] = Counter()
        self.hold: str | None = None
        self.holding = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, rec: dict):
        msgs = rec["body"].get("messages", [])
        text = json.dumps(msgs)
        tid = next((m.group(1) for m in (_TASK_RE.search(str(x.get("content"))) for x in msgs) if m), None)
        if tid is not None:
            if msgs[-1].get("role") == "tool":
                return Text("card closed")
            with self._lock:
                if self.hold is None and not self.release.is_set():
                    self.hold = tid
            if tid == self.hold and not self.release.is_set():
                self.holding.set()
                self.release.wait(timeout=1500)
            with self._lock:
                self.completes[tid] += 1
            return ToolCall("kanban_complete", {"summary": f"done {tid}"})
        if self.cron_marker in text:
            with self._lock:
                self.cron_calls.append(time.time())
            return Text(f"cron ran {self.cron_marker}")
        return Text(REPLY)


# -- small views over the sandbox -------------------------------------------------------------------


def _dashboard_health(port: int) -> dict | None:
    status, body = X.http("GET", f"http://127.0.0.1:{port}/api/health", key=None, timeout=5)
    return body if status == 200 and isinstance(body, dict) else None


_LISTENER_PY = r"""
import os, sys
port = int(sys.argv[1]); inodes = set()
for table in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        rows = open(table).read().splitlines()[1:]
    except OSError:
        continue
    for row in rows:
        f = row.split()
        if f[3] == "0A" and int(f[1].rsplit(":", 1)[1], 16) == port:
            inodes.add(f[9])
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            link = os.readlink(f"/proc/{pid}/fd/{fd}")
            if link.startswith("socket:[") and link[8:-1] in inodes:
                print(pid); sys.exit(0)
    except OSError:
        pass
"""


def listener_pid(inst: X.Install, port: int) -> int | None:
    """The sandbox PID owning the LISTEN socket on ``port`` (``/proc/net/tcp`` inode -> ``/proc/<pid>/fd``)."""
    cp = inst.host.run([sys.executable, "-c", _LISTENER_PY, str(port)], timeout=60, quiet=True)
    return int(cp.stdout.strip()) if cp.stdout.strip().isdigit() else None


def dashboards(inst: X.Install) -> list[dict]:
    """Top-level ``hermes dashboard`` processes (a dashboard's own helpers are not a second one)."""
    out = [p for p in inst.host.procs()
           if p["state"] not in ("Z", "X") and "dashboard" in p["cmdline"] and "update" not in p["cmdline"]
           and any("hermes" in a for a in p["cmdline"])]
    pids = {p["pid"] for p in out}
    return [p for p in out if p["ppid"] not in pids]


def _kanban(inst: X.Install, sql: str, args: tuple = ()) -> list[dict]:
    db = inst.hermes_home / "kanban.db"
    if not db.exists():
        return []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, args).fetchall()]
    finally:
        con.close()


def card_status(inst: X.Install, tid: str) -> str:
    rows = _kanban(inst, "SELECT status FROM tasks WHERE id = ?", (tid,))
    return rows[0]["status"] if rows else ""


def create_card(inst: X.Install, title: str) -> str:
    cp = inst.cli("kanban", "create", title, "--assignee", "default", "--json", timeout=120)
    assert cp.returncode == 0, "kanban create failed\n" + H.describe(cp)
    return json.loads(cp.stdout[cp.stdout.index("{"):])["id"]


def cron_jobs(inst: X.Install) -> list[dict]:
    data = X.read_json(inst.hermes_home / "cron" / "jobs.json")
    jobs = data.get("jobs", data) if isinstance(data, dict) else data
    return [j for j in (jobs or []) if isinstance(j, dict)]


def cron_outputs(inst: X.Install, job_id: str) -> list[str]:
    d = inst.hermes_home / "cron" / "output" / job_id
    return sorted(p.name for p in d.iterdir()) if d.is_dir() else []


def worker_logs(inst: X.Install) -> str:
    return "\n".join(f"--- {p.relative_to(inst.hermes_home)} ---\n{p.read_text(errors='replace')[-2500:]}"
                     for p in sorted(inst.hermes_home.rglob("t_*.log")))


def cron_failure(inst: X.Install, name: str, job: dict | None) -> str:
    """Why the job's run failed, as the scheduler logged it ('' when nothing failed)."""
    if job and job.get("last_error"):
        return str(job["last_error"])
    pat = re.compile(rf"Job '{re.escape(name)}' failed: (.+)")
    for log in (inst.root / "gateway.log", inst.hermes_home / "logs" / "gateway.log",
                inst.hermes_home / "logs" / "errors.log"):
        m = pat.search(log.read_text(errors="replace")) if log.exists() else None
        if m:
            return m.group(1).strip()
    return ""


def _run_segments(log: str) -> list[str]:
    """A card's worker log, one segment per worker run (each starts with its ``Query:`` line)."""
    return [s for s in re.split(r"(?m)^(?=Query: work kanban task )", log) if s.strip()]


def _overlap(runs: list[dict]) -> list[tuple[int, int]]:
    """Pairs of runs of ONE card that were live at the same time: a duplicate worker."""
    out = []
    for i, a in enumerate(runs):
        for b in runs[i + 1:]:
            a_end, b_end = a["ended_at"] or float("inf"), b["ended_at"] or float("inf")
            if a["started_at"] < b_end and b["started_at"] < a_end:
                out.append((a["id"], b["id"]))
    return out


class WorkerSampler:
    """Samples the sandbox process table once a second while the block runs and records, per card,
    the most kanban workers (``hermes ... chat -q "work kanban task <id>"``) alive at the same moment:
    two at once is a duplicate spawn, whatever the board says afterwards."""

    def __init__(self, inst: X.Install, interval: float = 1.0) -> None:
        self.inst, self.interval = inst, interval
        self.peak: Counter[str] = Counter()
        self.evidence: dict[str, list] = {}
        self._stop = threading.Event()

    def _sample(self) -> None:
        live = [p for p in self.inst.host.procs() if p["state"] not in ("Z", "X")]
        by_card: dict[str, list[dict]] = {}
        for p in live:
            m = _TASK_RE.search(" ".join(p["cmdline"]))
            if m:
                by_card.setdefault(m.group(1), []).append(p)
        for tid, procs in by_card.items():
            pids = {p["pid"] for p in procs}
            roots = sorted(p["pid"] for p in procs if p["ppid"] not in pids)
            if len(roots) > self.peak[tid]:
                self.peak[tid] = len(roots)
                self.evidence[tid] = [round(time.time()), roots]

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:  # a probe racing sandbox teardown; the next sample decides
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "WorkerSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=70)


def _quiet_wait(pred, *, timeout: float, what: str, interval: float = 1.0):
    try:
        return X.wait_for(pred, timeout=timeout, what=what, interval=interval)
    except AssertionError:
        return None


# -- the scenario ---------------------------------------------------------------------------------


def run(column: str, root: Path) -> SimpleNamespace:
    """Boot the fleet, update, let everything settle, and record what the user would see."""
    o = SimpleNamespace(column=column)
    model = Model(f"CRON-HANDOFF-{secrets.token_hex(4)}")
    with FakeLLMServer(model) as srv, X.cell(
            column, root, srv.base_url, extra={"kanban": {"dispatch_interval_seconds": 3}}) as inst:
        before = X.start_gateway(inst)
        o.old_pid = before["pid"]
        with X.FileWatch(inst.hermes_home / "gateway.pid") as watch:
            assert X.pid_file_pid(inst) == o.old_pid, f"premise: gateway.pid names the gateway: {watch.render()}"
            status, body = X.chat(inst.port, "hello before the update")
            assert status == 200 and X.reply_text(body) == REPLY, f"premise: a turn works before: {body}"

            o.dash_port = X.free_port()
            inst.spawn("dashboard", "dashboard", "--no-open", "--host", "127.0.0.1", "--port", str(o.dash_port))
            try:  # the first ``hermes dashboard`` builds the web UI, as it does for the user
                X.wait_for(lambda: _dashboard_health(o.dash_port), timeout=600, what="the dashboard /api/health")
            except AssertionError as exc:
                raise AssertionError(f"premise: the dashboard never came up: {exc}\n{inst.diagnostics()}") from None
            o.dash_old = listener_pid(inst, o.dash_port)
            assert o.dash_old, "premise: nothing listens on the dashboard port"

            sampler = WorkerSampler(inst).__enter__()
            o.card1 = create_card(inst, "in flight across the update")
            assert model.holding.wait(timeout=240) and model.hold == o.card1, (
                f"premise: the gateway's dispatcher never ran a worker for {o.card1}\n"
                f"{worker_logs(inst)}\n{inst.diagnostics()}")

            o.boots = len(X.gateway_starts(inst))
            o.target = X.publish_target(inst)
            due = dt.datetime.now(dt.timezone.utc).replace(microsecond=0) + dt.timedelta(seconds=CRON_DUE_S + 5)
            cp = inst.cli("cron", "create", due.isoformat(), f"Reply with the word done. {model.cron_marker}",
                          "--name", model.cron_marker)
            assert cp.returncode == 0, "premise: cron create failed\n" + H.describe(cp)
            job = next((j for j in cron_jobs(inst) if j.get("name") == model.cron_marker), None)
            assert job and job.get("id"), f"premise: the job is not in jobs.json: {cron_jobs(inst)}"
            o.job_id = job["id"]
            assert not model.cron_calls, "premise: the job ran before it was due"

            o.up = inst.update()
            o.sha_after = inst.sha()

            o.ident = X.settle_gateway(inst, o.old_pid)
            model.release.set()
            o.chat_after = X.chat(inst.port, "hello after the update") if o.ident else (0, "")
            o.card2 = create_card(inst, "created after the update")

            gone_since: list[float] = []

            def dash_back():
                pid = listener_pid(inst, o.dash_port)
                if pid and pid != o.dash_old and _dashboard_health(o.dash_port):
                    return pid
                # No dashboard process at all for 20s: the respawn died (or never happened); stop waiting.
                if dashboards(inst):
                    gone_since.clear()
                elif not gone_since:
                    gone_since.append(time.monotonic())
                return -1 if gone_since and time.monotonic() - gone_since[0] > 20 else None
            back = _quiet_wait(dash_back, timeout=240, what="a new dashboard on its port")
            o.dash_new = back if back and back > 0 else None

            if o.ident:  # the cron ticker and the kanban dispatcher live in the gateway
                _quiet_wait(lambda: dt.datetime.now(dt.timezone.utc) > due + dt.timedelta(seconds=5), timeout=600,
                            what="the cron due time")
                _quiet_wait(lambda: model.cron_calls, timeout=240, what="the cron job's provider call")
                _quiet_wait(lambda: card_status(inst, o.card1) == "done" and card_status(inst, o.card2) == "done",
                            timeout=300, what="both kanban cards to finish")
                # A re-fire or a duplicate worker shows up within a full ticker interval on the new gateway.
                settle_until = max([time.time(), *model.cron_calls]) + 65
                time.sleep(max(0.0, settle_until - time.time()))

            sampler.__exit__()
            o.worker_peak, o.worker_evidence = dict(sampler.peak), sampler.evidence
            o.gateways = X.gateway_pids(inst)
            o.status = inst.cli("gateway", "status", timeout=120)
            o.cron_list = inst.cli("cron", "list", "--all", timeout=120)
            time.sleep(2)  # let the pid-file watch see what the status probes did
        o.pid_history = watch.history
        o.relaunch = X.relaunch_verdict(inst, o.old_pid, o.boots, o.ident, o.up)
        o.pid_file = X.pid_file_verdict(inst, o.ident["pid"], watch.history) if o.ident else "no gateway"
        o.dash_listener = listener_pid(inst, o.dash_port)
        o.dash_roots = dashboards(inst)
        restart_log = inst.hermes_home / "logs" / "dashboard-restart.log"
        o.dash_restarts = restart_log.read_text(errors="replace") if restart_log.exists() else ""
        o.cron_calls = list(model.cron_calls)
        o.cron_outputs = cron_outputs(inst, o.job_id)
        o.cron_job_after = next((j for j in cron_jobs(inst) if j.get("id") == o.job_id), None)
        o.cron_error = cron_failure(inst, model.cron_marker, o.cron_job_after)
        o.completes = dict(model.completes)
        o.cards = {t: card_status(inst, t) for t in (o.card1, o.card2)}
        o.runs = {t: _kanban(inst, "SELECT id, outcome, started_at, ended_at FROM task_runs WHERE task_id = ? "
                                   "ORDER BY id", (t,)) for t in (o.card1, o.card2)}
        o.worker_logs = worker_logs(inst)
        card_log = {t: next((p.read_text(errors="replace") for p in inst.hermes_home.rglob(f"{t}.log")), "")
                    for t in (o.card1, o.card2)}
        # Workers spawned after the update: every run of the new card, and every re-run of the in-flight one.
        o.post_update_worker_logs = "\n".join(_run_segments(card_log[o.card2]) + _run_segments(card_log[o.card1])[1:])
        inst.logs.append(restart_log)
        # bwrap keeps the caller's cgroup: the sandbox's processes (the dashboard too) live in ours.
        o.cgroup = Path("/proc/self/cgroup").read_text(errors="replace").strip() if sys.platform == "linux" else ""
        o.diag = (inst.diagnostics(o.up) + f"\n--- cgroup of the sandbox ---\n{o.cgroup}"
                  f"\n--- gateway.pid history ---\n{watch.render()}"
                  f"\n--- cron job ---\n{json.dumps(o.cron_job_after, indent=1)[-2000:]}"
                  f"\n--- kanban ---\ncards={o.cards} provider kanban_complete calls={o.completes}\n"
                  f"peak live workers per card={o.worker_peak} (at, pids)={o.worker_evidence}\nruns={json.dumps(o.runs)}"
                  f"\n--- worker logs ---\n{o.worker_logs[-5000:]}")
    return o


# -- the properties -------------------------------------------------------------------------------


class HandoffProperties:
    """Every property of the hand-off, over one recorded scenario per column (``column`` on the
    subclass). Each test is independent: a failure in one subsystem never hides another's."""

    column: str = ""

    @pytest.fixture(scope="class")
    def fleet(self):
        if self.column == "n1" and not X.refs().base:
            pytest.skip("no release tag reachable (shallow checkout)")
        base = os.environ.get("TMPDIR") or tempfile.gettempdir()
        root = Path(tempfile.mkdtemp(prefix="ho", dir=base))  # short: AF_UNIX paths under $HERMES_HOME
        try:
            yield run(self.column, root)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def _relaunched(self, o) -> None:
        with known_gate(X.RELAUNCH_GATES, o.column):
            assert not o.relaunch, f"{o.relaunch}\n{o.diag}"

    def test_update_moves_the_checkout(self, fleet):
        assert fleet.sha_after == fleet.target, f"the update did not move the checkout\n{fleet.diag}"

    def test_gateway_is_relaunched_once_on_the_new_commit(self, fleet):
        o = fleet
        self._relaunched(o)
        assert [p["pid"] for p in o.gateways] == [o.ident["pid"]], f"expected exactly one gateway: {o.gateways}\n{o.diag}"
        status, body = o.chat_after
        assert status == 200 and X.reply_text(body) == REPLY, f"no turn after the update: {body}\n{o.diag}"

    def test_update_exit_status_matches_what_came_back(self, fleet):
        """Nothing is reported as success when it failed, and a failure exit names something real."""
        o = fleet
        out = o.up.stdout + o.up.stderr
        assert X.TRACEBACK not in out, f"the update crashed\n{o.diag}"
        down = [name for name, ok in (("the gateway", o.ident), ("the dashboard", o.dash_new)) if not ok]
        if o.up.returncode == 0:
            assert not down, f"the update exited 0 though {' and '.join(down)} did not come back\n{o.diag}"
        else:
            assert down, (f"the update exited {o.up.returncode} though the gateway and the dashboard came back"
                          f"\n{o.diag}")

    def test_gateway_pid_file_and_status_find_the_relaunched_gateway(self, fleet):
        o = fleet
        self._relaunched(o)
        assert not o.pid_file, f"{o.pid_file}\n{o.diag}"
        assert f"Gateway is running (PID: {o.ident['pid']})" in o.status.stdout, (
            f"`hermes gateway status` cannot find the relaunched gateway:\n{H.describe(o.status)}\n{o.diag}")

    def test_dashboard_is_back_on_its_port(self, fleet):
        o = fleet
        out = o.up.stdout + o.up.stderr
        assert "TypeError" not in out, f"the update tripped over the dashboard\n{o.diag}"
        with _gated(DASHBOARD_GATES.get(o.column, ())):
            assert o.dash_new is not None, f"{dashboard_verdict(o)}\n{o.diag}"
        assert len(o.dash_roots) == 1, f"expected exactly one dashboard: {o.dash_roots}\n{o.diag}"
        assert o.dash_listener == o.dash_new, f"the dashboard on port {o.dash_port} churned\n{o.diag}"
        assert not _PORT_IN_USE.search(o.dash_restarts), f"the respawned dashboard hit port-in-use\n{o.diag}"

    def test_cron_job_due_mid_update_runs_exactly_once(self, fleet):
        o = fleet
        self._relaunched(o)
        with known_gate(CRON_GATES, o.column):
            assert len(o.cron_calls) == 1, f"{cron_verdict(o)}\n{o.diag}"
        assert len(o.cron_outputs) == 1, f"run records: {o.cron_outputs}\n{o.diag}"
        assert o.cron_list.returncode == 0 and X.TRACEBACK not in o.cron_list.stdout + o.cron_list.stderr, (
            f"the new code cannot read the job back:\n{H.describe(o.cron_list)}\n{o.diag}")
        if o.cron_job_after is not None:  # a spent one-shot may be dropped; a kept one records its run
            job = o.cron_job_after
            assert job.get("last_run_at") and job.get("last_status") == "ok", f"bookkeeping lost: {job}\n{o.diag}"

    def test_kanban_inflight_worker_finishes_once_and_new_workers_import(self, fleet):
        o = fleet
        self._relaunched(o)
        m = _IMPORT_ERR.search(o.post_update_worker_logs)
        assert m is None, f"a kanban worker spawned after the update failed to import ({m and m.group(0)})\n{o.diag}"
        assert o.cards[o.card2] == "done", f"the card created after the update never finished\n{o.diag}"
        assert o.cards[o.card1] == "done", f"the in-flight card never finished\n{o.diag}"
        for t in (o.card1, o.card2):
            n = o.worker_peak.get(t, 0)
            assert n <= 1, f"card {t} had {n} live workers at once {o.worker_evidence.get(t)}\n{o.diag}"
            dup = _overlap(o.runs[t])
            assert not dup, f"card {t} had two live workers at once (runs {dup})\n{o.diag}"
            done = [r for r in o.runs[t] if r["outcome"] == "completed"]
            assert len(done) == 1, f"card {t} has {len(done)} completed runs\n{o.diag}"
        # The in-flight worker finishes, or the update interrupts it and it is re-queued exactly once.
        assert len(o.runs[o.card1]) <= 2, f"the in-flight card ran {len(o.runs[o.card1])} times\n{o.diag}"
        assert len(o.runs[o.card2]) == 1, f"the new card ran {len(o.runs[o.card2])} times\n{o.diag}"


def dashboard_verdict(o) -> str:
    """Why no new dashboard serves its port, in words a gate can key on."""
    if re.search(r"^SyntaxError", o.dash_restarts, re.MULTILINE) and "restarted:" in o.up.stdout:
        return (f"the respawned dashboard died parsing its launcher: the update replayed the pre-update argv and "
                f"Python read a shell script (SyntaxError in logs/dashboard-restart.log); port {o.dash_port} is dark")
    # The sandbox runs no Hermes unit, so any unit the dashboard stop restarts is the one the test runner
    # (and so the hand-started dashboard) happens to live in.
    unit = next((a or b for a, b in _UNIT_RESTART.findall(o.up.stdout)), None)
    if unit and not o.dash_restarts:
        return (f"the update restarted the systemd unit {unit} the dashboard was started under (cgroup "
                f"{o.cgroup}) instead of respawning the dashboard; port {o.dash_port} is dark")
    return f"no new dashboard serves port {o.dash_port} (old pid {o.dash_old}, listener now {o.dash_listener})"


def cron_verdict(o) -> str:
    """How many times the job due mid-update reached the provider, and why not when it never did."""
    n = len(o.cron_calls)
    if n == 0 and _IMPORT_ERR.search(o.cron_error or ""):
        return (f"the job due mid-update fired into the update swap window and failed importing "
                f"({o.cron_error[:240]}); it never reached the provider")
    return f"the job due mid-update ran {n} times" + (f" (last error: {o.cron_error[:240]})" if o.cron_error else "")

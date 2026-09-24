"""Auto-decompose billing bounds, proven against a REAL ``hermes gateway run`` embedded dispatcher.

The gateway (no messaging platforms -> a kanban/cron-only process) ticks the embedded dispatcher
every second; each tick auto-decomposes triage cards through the auxiliary model, which is the
recording fake provider (decomposer calls carry no ``tools`` so they land in ``aux_requests()``).
Ticks are counted without sleeps: a fresh "probe" triage card is created only after the previous
one left triage, so K probes promoted == at least K dispatcher ticks elapsed.

Verdicts come from the provider's request log (what was billed) and ``kanban.db`` rows (tasks,
task_links, status) -- never log wording.

* green control: a valid 3-child graph bills exactly one aux call, creates exactly 3 child rows
  linked under the root (plus the declared sibling edge) and moves the root out of triage;
* a card whose decomposition always fails (malformed reply / HTTP 5xx) must not be re-billed on
  every tick forever (#118872, #118603);
* a 500-child decompose reply must not become 500 child rows (#118607).
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import pytest

from tests.e2e.core.kanban._helpers import PY, Board, wait_until
from tests.fakes.fake_llm_provider import Error, FakeLLMServer, Response, Text

# The gateway child is isolated (scratch HOME/HERMES_HOME, own lock dir) and reaped by PID below.
pytestmark = [
    pytest.mark.spawns_gateway_lookalike,
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups for the gateway child"),
]

KNOWN = {
    "malformed": "#118872 a triage card whose decompose reply is unusable is re-billed every tick forever",
    "http_500": "#118603 a triage card whose decompose call 5xxs is retried every dispatcher tick forever",
    "test_huge_decompose_reply_is_bounded": "#118607 a 500-child decompose reply creates 500 child rows",
}

# Dispatcher knobs through documented config: tick every second, never spawn workers (we only
# measure the decompose phase), no per-tick cap starvation, one HTTP request per aux attempt.
GATEWAY_CONFIG = (
    "kanban:\n"
    "  dispatch_interval_seconds: 1\n"
    "  max_spawn: 0\n"
    "  auto_decompose: true\n"
    "  auto_decompose_per_tick: 10\n"
    "auxiliary:\n"
    "  transient_retries: 0\n"
)
PROBE_TICKS = 6          # K: dispatcher ticks proven elapsed while the doomed card sits in triage
MAX_FAILED_ATTEMPTS = 3  # generous bound on aux bills for one never-decomposable card over K ticks
MAX_CHILDREN = 64        # generous bound over the prompt's advisory 2-6 children
TASK_ID_RE = re.compile(r"Task id: (t_[0-9a-f]+)")


class RebilledEveryTick(AssertionError):
    """The doomed card was billed on (nearly) every tick; the only failure the known xfail covers."""


class UnboundedFanout(AssertionError):
    """A decompose reply fanned out past the child bound; the only failure the known xfail covers."""


def _known(name: str, exc: type[BaseException]) -> list:
    return [pytest.mark.xfail(strict=True, raises=exc, reason=KNOWN[name])] if name in KNOWN else []


# fake aux model ----------------------------------------------------------------------------------
Reply = Callable[[dict], Response]


def _single(_rec: dict) -> Text:
    return Text(json.dumps({"fanout": False, "rationale": "one unit", "title": "probe promoted",
                            "body": "do the one thing"}))


def _graph(n: int, edges: dict[int, list[int]] | None = None) -> Reply:
    tasks = [{"title": f"child {i}", "body": f"spec {i}", "assignee": None,
              "parents": (edges or {}).get(i, [])} for i in range(n)]
    payload = json.dumps({"fanout": True, "rationale": "split", "tasks": tasks})
    return lambda _rec: Text(payload, chunk_chars=4096)


FAILURES: dict[str, Reply] = {
    "malformed": lambda _rec: Text("Sorry, I can't produce a task graph for this one."),
    "http_500": lambda _rec: Error(500, "upstream exploded"),
}


class AuxModel:
    """Answers each decompose call by the task id in its prompt; records which card was billed."""

    def __init__(self) -> None:
        self.replies: dict[str, Reply] = {}

    @staticmethod
    def task_id(body: dict) -> str | None:
        m = TASK_ID_RE.search(json.dumps(body.get("messages", [])))
        return m.group(1) if m else None

    def __call__(self, rec: dict) -> Response:
        return self.replies.get(self.task_id(rec["body"]) or "", _single)(rec)


def billed_for(srv: FakeLLMServer, tid: str) -> int:
    return sum(1 for body in srv.aux_requests() if AuxModel.task_id(body) == tid)


# board readers -----------------------------------------------------------------------------------
def children_of(b: Board, root: str) -> list[str]:
    """Decomposed children: the root is linked under every child (task_links child_id == root)."""
    return [r["parent_id"] for r in b._q("SELECT parent_id FROM task_links WHERE child_id = ?", (root,))]


def task_count(b: Board) -> int:
    return b._q("SELECT COUNT(*) AS n FROM tasks")[0]["n"]


# real gateway ------------------------------------------------------------------------------------
@contextmanager
def gateway(b: Board) -> Iterator[subprocess.Popen]:
    env = b.env()
    # The child's HOME is itself the scratch board root, so ~/.hermes IS the temp home here; the
    # inherited live-system guard would refuse it (same as the delivery/tenancy gateway children).
    env.update({"HERMES_GATEWAY_LOCK_DIR": str(b.root / "gw-locks"), "PYTHONUNBUFFERED": "1",
                "HERMES_STATE_DB_GUARD_BYPASS": "1"})
    log_path = b.root / "gateway.log"
    with open(log_path, "wb") as log:
        proc = subprocess.Popen([PY, "-m", "hermes_cli.main", "gateway", "run"], cwd=str(b.root),
                                env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        yield proc
    finally:
        for sig, grace in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):  # windows-footgun: ok — POSIX-gated (skips on win32)
            if proc.poll() is not None:
                break
            try:
                os.killpg(proc.pid, sig)  # windows-footgun: ok — POSIX-gated (skips on win32)
            except ProcessLookupError:
                break
            try:
                proc.wait(grace)
            except subprocess.TimeoutExpired:
                continue


def gateway_tail(b: Board) -> str:
    p = b.root / "gateway.log"
    return p.read_text(encoding="utf-8", errors="replace")[-3000:] if p.exists() else "(no gateway log)"


def prove_ticks(b: Board, proc: subprocess.Popen, k: int) -> None:
    """Create probe cards one at a time; each must be auto-decomposed (leave triage) by the gateway
    before the next is created, so returning means >= k further dispatcher ticks ran."""
    for i in range(k):
        probe = b.create(f"tick probe {i}", "--triage")
        wait_until(lambda: proc.poll() is not None or b.task(probe)["status"] != "triage", 90,
                   f"probe {i} auto-decomposed by the gateway dispatcher")
        assert proc.poll() is None, f"gateway exited rc={proc.returncode}\n{gateway_tail(b)}"


# tests -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("failure", [pytest.param(name, marks=_known(name, RebilledEveryTick))
                                     for name in FAILURES])
def test_failing_triage_card_is_not_rebilled_every_tick(tmp_path: Path, failure: str) -> None:
    model = AuxModel()
    with FakeLLMServer(aux=model) as srv:
        b = Board(tmp_path, srv.base_url, extra_config=GATEWAY_CONFIG)
        doomed = b.create("an idea the decomposer can never split", "--triage")
        model.replies[doomed] = FAILURES[failure]
        with gateway(b) as proc:
            wait_until(lambda: proc.poll() is not None or billed_for(srv, doomed) >= 1, 120,
                       f"first auto-decompose attempt on the doomed card\n{gateway_tail(b)}")
            assert proc.poll() is None, f"gateway exited rc={proc.returncode}\n{gateway_tail(b)}"
            prove_ticks(b, proc, PROBE_TICKS)
            bills = billed_for(srv, doomed)
        # Harness invariants (stay red regardless of the known bug): the failure produced no graph.
        assert children_of(b, doomed) == [], b.diag(doomed)
        assert b.events(doomed, "decomposed") == [], b.diag(doomed)
        if bills > MAX_FAILED_ATTEMPTS:
            raise RebilledEveryTick(
                f"doomed card billed {bills} aux decompose calls over >= {PROBE_TICKS + 1} dispatcher "
                f"ticks (bound {MAX_FAILED_ATTEMPTS}); status={b.task(doomed)['status']}")


def test_valid_three_child_graph_bills_once_and_links_children(tmp_path: Path) -> None:
    """Green control through the same gateway harness: one bill, three linked children, root out of
    triage, the declared sibling dependency persisted, and no re-bill on later ticks."""
    model = AuxModel()
    with FakeLLMServer(aux=model) as srv:
        b = Board(tmp_path, srv.base_url, extra_config=GATEWAY_CONFIG)
        root = b.create("ship the feature", "--triage")
        model.replies[root] = _graph(3, edges={2: [0]})
        with gateway(b) as proc:
            wait_until(lambda: proc.poll() is not None or b.task(root)["status"] != "triage", 120,
                       f"root auto-decomposed\n{gateway_tail(b)}")
            assert proc.poll() is None, f"gateway exited rc={proc.returncode}\n{gateway_tail(b)}"
            prove_ticks(b, proc, 2)
        kids = children_of(b, root)
        assert len(kids) == 3, b.diag(root)
        assert billed_for(srv, root) == 1, "a decomposed root must never be billed again"
        assert b.task(root)["status"] not in ("triage", "archived"), b.diag(root)
        assert task_count(b) == 1 + 3 + 2  # root + children + the two tick probes
        rows = {b.task(k)["title"]: k for k in kids}
        assert sorted(rows) == ["child 0", "child 1", "child 2"]
        edge = b._q("SELECT 1 FROM task_links WHERE parent_id = ? AND child_id = ?",
                    (rows["child 0"], rows["child 2"]))
        assert edge, "declared dependency child 0 -> child 2 was not persisted"
        assert [e["payload"]["child_ids"] for e in b.events(root, "decomposed")] == [
            [rows[f"child {i}"] for i in range(3)]]


@pytest.mark.xfail(strict=True, raises=UnboundedFanout,
                   reason=KNOWN["test_huge_decompose_reply_is_bounded"])
def test_huge_decompose_reply_is_bounded(tmp_path: Path) -> None:
    """One decompose call answering 500 children must be rejected or capped, not fanned out.
    Driven through the real ``hermes kanban decompose`` CLI (same decompose_task path)."""
    model = AuxModel()
    with FakeLLMServer(aux=model) as srv:
        b = Board(tmp_path, srv.base_url)
        root = b.create("tiny chore", "--triage")
        model.replies[root] = _graph(500)
        b.cli("decompose", root, "--json", timeout=240, check=False)
        assert billed_for(srv, root) == 1, "the decompose CLI must make exactly one aux call"
        kids = children_of(b, root)
        # A rejection leaves the root in triage with zero children; a cap leaves <= the bound.
        assert kids or b.task(root)["status"] == "triage", b.diag(root)
        assert task_count(b) == 1 + len(kids), "child rows exist that are not linked under the root"
        if len(kids) > MAX_CHILDREN:
            raise UnboundedFanout(f"500-child reply created {len(kids)} child rows (bound {MAX_CHILDREN})")

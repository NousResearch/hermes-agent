"""A rate-limited implementer run must stay a neutral, once-billed requeue — and must not strand the
card's later review handoff (#119070).

Real processes end to end: every tick is a real ``hermes kanban dispatch`` process, every worker a
real ``hermes chat -q`` process spawned by it, talking to the recording fake provider (the only
fake — it stands in for the vendor HTTP API). The fake plays two roles, told apart by what the
dispatcher put in the worker's system prompt: a review-lane worker is started with the bundled
``sdlc-review`` skill preloaded, an implementer is not.

Flow under test (``HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS=0``, ``agent.api_max_retries: 1``):

1. tick: implementer spawned; the provider answers HTTP 429; the worker exits 75.
2. tick: the dead worker is reaped as ``rate_limited`` (no failure tick) and respawned; the retry
   calls ``kanban_request_review`` -> the card moves to ``review``.
3. tick: the review lane must spawn the reviewer, which approves (``kanban_complete``) -> ``done``.

Verdicts come from ``kanban.db`` rows (tasks / task_runs / task_events), the dispatcher's JSON tick
result, and the request stream the fake provider recorded — never from log wording.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.core.kanban._helpers import Board
from tests.fakes.fake_llm_provider import Error, FakeLLMServer, Text, ToolCall

pytestmark = [
    pytest.mark.skipif(sys.platform != "linux", reason="worker liveness reads /proc"),
    # Teardown SIGKILLs the dispatcher-spawned workers (our grandchildren, outside the pytest subtree).
    pytest.mark.live_system_guard_bypass,
]

# The review lane preloads the bundled review skill; the dispatcher's own contract (kanban.md:
# "spawn the assigned profile with the bundled sdlc-review skill").
REVIEW_SKILL = "sdlc-review"
RATE_LIMIT_EXIT_CODE = 75  # KANBAN_RATE_LIMIT_EXIT_CODE — documented worker exit contract
MAX_TICKS = 6  # a healthy rate-limited card is done on tick 3; the rest prove "forever"

# Scenario -> reason. A listed scenario is a strict xfail that only matches its dedicated exception.
KNOWN: dict[str, str] = {
    "rate_limited_then_review": "#119070 stale rate-limit stamp parks the review handoff as blocker_auth",
}


class ReviewerNeverSpawned(AssertionError):
    """The card reached ``review`` but the review lane never started a reviewer (#119070)."""


def _known(name: str):
    if name not in KNOWN:
        return ()
    return (pytest.mark.xfail(strict=True, raises=ReviewerNeverSpawned, reason=KNOWN[name]),)


# fake provider -----------------------------------------------------------------------------------


def _system_text(body: dict) -> str:
    parts = []
    for m in body.get("messages", []):
        if m.get("role") == "system":
            c = m.get("content")
            parts.append(c if isinstance(c, str) else " ".join(p.get("text", "") for p in c or []))
    return "\n".join(parts)


@dataclass
class Attempt:
    role: str  # "impl" | "review"
    requests: int = 0
    answers: list[str] = field(default_factory=list)


class TwoRoleModel:
    """Scripted vendor: implementer attempts 1..``rate_limited_attempts`` get HTTP 429, later ones
    hand off with ``kanban_request_review``; a reviewer approves with ``kanban_complete``. Every
    request is attributed to the worker attempt (fresh session) that sent it."""

    def __init__(self, rate_limited_attempts: int) -> None:
        self.rate_limited_attempts = rate_limited_attempts
        self.attempts: list[Attempt] = []
        self._lock = threading.Lock()

    def __call__(self, rec: dict[str, Any]) -> Any:
        body = rec["body"]
        msgs = body.get("messages", [])
        role = "review" if f'"{REVIEW_SKILL}"' in _system_text(body) else "impl"
        with self._lock:
            fresh = not any(m.get("role") in ("assistant", "tool") for m in msgs)
            if fresh or not self.attempts or self.attempts[-1].role != role:
                self.attempts.append(Attempt(role))
            att = self.attempts[-1]
            att.requests += 1
            resp = self._answer(role, att, msgs, fresh)
            att.answers.append(type(resp).__name__ if not isinstance(resp, ToolCall) else resp.name)
            return resp

    def _answer(self, role: str, att: Attempt, msgs: list, fresh: bool) -> Any:
        if not fresh and msgs and msgs[-1].get("role") == "tool":
            return Text("done")
        if role == "review":
            return ToolCall("kanban_complete", {"summary": "review: approved"})
        impl_no = sum(1 for a in self.attempts if a.role == "impl")
        if impl_no <= self.rate_limited_attempts:
            return Error(status=429, message="Rate limit exceeded: quota for this key", retry_after=0)
        return ToolCall("kanban_request_review", {"summary": "implementation ready for review"})

    def by_role(self, role: str) -> list[Attempt]:
        return [a for a in self.attempts if a.role == role]


# board driving -----------------------------------------------------------------------------------


@dataclass
class Flow:
    board: Board
    tid: str
    model: TwoRoleModel
    ticks: list[dict]

    def run_outcomes(self) -> list[str]:
        return [r["outcome"] for r in self.board.runs(self.tid)]

    def diag(self) -> str:
        att = [(a.role, a.requests, a.answers) for a in self.model.attempts]
        return f"ticks={self.ticks}\nattempts={att}\n{self.board.diag(self.tid)}"


def drive(board: Board, tid: str, model: TwoRoleModel, max_ticks: int = MAX_TICKS) -> list[dict]:
    """Run real dispatcher ticks; after each spawn wait for THAT worker process to exit so the next
    tick sees its final state (a dead worker is reaped on the tick after it dies)."""
    ticks: list[dict] = []
    for n in range(1, max_ticks + 1):
        res = board.dispatch()
        spawned = [s["task_id"] for s in res.get("spawned", [])]
        if tid in spawned:
            pid = board.events(tid, "spawned")[-1]["payload"]["pid"]
            board.wait_worker_exit(tid, int(pid))
        t = board.task(tid)
        ticks.append({
            "tick": n, "spawned": spawned.count(tid), "status": t["status"],
            "consecutive_failures": t["consecutive_failures"],
            "guarded": [g.get("reason") for g in res.get("respawn_guarded", []) if g.get("task_id") == tid],
        })
        if t["status"] == "done":
            break
    return ticks


def _flow(root: Path, rate_limited_attempts: int) -> Any:
    model = TwoRoleModel(rate_limited_attempts)
    with FakeLLMServer(model) as srv:
        board = Board(root, srv.base_url)
        tid = board.create(f"rate-limit review flow ({rate_limited_attempts} x 429)")
        try:
            ticks = drive(board, tid, model)
        finally:
            board.kill_workers()
        return Flow(board, tid, model, ticks)


@pytest.fixture(scope="module")
def rate_limited_flow(tmp_path_factory: pytest.TempPathFactory) -> Flow:
    """One card, one 429 on the first implementer attempt; shared by the billing and #119070 tests."""
    return _flow(tmp_path_factory.mktemp("kanban-rl"), rate_limited_attempts=1)


# tests -------------------------------------------------------------------------------------------


def test_rate_limited_attempt_is_billed_once_and_requeued_without_a_failure(rate_limited_flow: Flow) -> None:
    f = rate_limited_flow
    b, tid, diag = f.board, f.tid, f.diag()
    runs = b.runs(tid)
    assert len(runs) >= 2, diag
    # Billing contract of the 429 attempt: booked neutral, never counted against the breaker.
    assert runs[0]["outcome"] == "rate_limited", diag
    assert runs[1]["outcome"] == "review_requested", diag
    rl_events = b.events(tid, "rate_limited")
    assert len(rl_events) == 1 and rl_events[0]["payload"]["exit_code"] == RATE_LIMIT_EXIT_CODE, diag
    assert rl_events[0]["run_id"] == runs[0]["id"], diag
    assert all(t["consecutive_failures"] == 0 for t in f.ticks), diag
    assert b.task(tid)["consecutive_failures"] == 0, diag
    for kind in ("crashed", "gave_up", "blocked", "protocol_violation"):
        assert not b.events(tid, kind), f"{kind} event on a rate-limited card\n{diag}"
    # Cooldown is 0: the reap and the respawn happen on the SAME tick right after the 429 worker died.
    assert [t["spawned"] for t in f.ticks[:2]] == [1, 1], diag
    # Provider-side billing: the 429 attempt is ONE request (api_max_retries: 1, no retry loop and
    # no fallback re-send); the retry is exactly the handoff tool call plus its closing turn.
    impl = f.model.by_role("impl")
    assert [(a.requests, a.answers) for a in impl] == [
        (1, ["Error"]), (2, ["kanban_request_review", "Text"]),
    ], diag
    assert len(runs) - 2 == len(f.model.by_role("review")), diag


def test_clean_handoff_spawns_the_reviewer_on_the_next_tick(tmp_path: Path) -> None:
    """Control: without a rate-limit stamp the same handoff reaches a reviewer that finishes the card
    — proves the harness can see a reviewer spawn, so the #119070 verdict is not a harness blind spot."""
    f = _flow(tmp_path, rate_limited_attempts=0)
    b, tid, diag = f.board, f.tid, f.diag()
    assert b.task(tid)["status"] == "done", diag
    assert f.run_outcomes() == ["review_requested", "completed"], diag
    assert [t["spawned"] for t in f.ticks] == [1, 1], diag
    assert not any(t["guarded"] for t in f.ticks), diag
    assert [(a.role, a.requests) for a in f.model.attempts] == [("impl", 2), ("review", 2)], diag
    assert b.task(tid)["consecutive_failures"] == 0, diag


@pytest.mark.parametrize("scenario", [pytest.param("rate_limited_then_review", marks=_known("rate_limited_then_review"))])
def test_rate_limited_then_review_handoff_reaches_the_reviewer(rate_limited_flow: Flow, scenario: str) -> None:
    f = rate_limited_flow
    b, tid, diag = f.board, f.tid, f.diag()
    # Precondition (harness, stays red if broken): the card really got to the review lane via a
    # rate-limited run followed by a successful handoff.
    assert f.run_outcomes()[:2] == ["rate_limited", "review_requested"], diag
    reviewers = f.model.by_role("review")
    guarded = [g for t in f.ticks for g in t["guarded"]]
    if not reviewers or b.task(tid)["status"] != "done":
        raise ReviewerNeverSpawned(
            f"{scenario}: status={b.task(tid)['status']} after {len(f.ticks)} ticks, "
            f"reviewer attempts={len(reviewers)}, respawn_guarded={guarded}\n{diag}")
    # Once fixed, the whole contract must hold, not just "something spawned".
    assert f.run_outcomes() == ["rate_limited", "review_requested", "completed"], diag
    assert [(a.requests, a.answers) for a in reviewers] == [(2, ["kanban_complete", "Text"])], diag
    assert "blocker_auth" not in guarded, diag
    assert not b.events(tid, "gave_up") and b.task(tid)["consecutive_failures"] == 0, diag

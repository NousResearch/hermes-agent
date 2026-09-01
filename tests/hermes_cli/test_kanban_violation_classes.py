"""Executable tests for the three-way clean-exit classification (t_8fc16a73).

Three unrelated faults used to produce the identical error string
"worker exited cleanly (rc=0) without calling kanban_complete or kanban_block
— protocol violation", which made any one of them disproportionately expensive
to diagnose. They are now split:

  * ``provider_error``  — fast provider spawn-bounce (403/429/5xx billing wall).
    Released WITHOUT consuming the failure budget or the protocol-violation
    streak; handed to the circuit breaker (fleet pause).
  * ``workspace_error`` — workspace / worktree / git contention. The card is
    BLOCKED with ``capability`` kind instead of retried into the same broken
    workspace.
  * ``no_checkpoint``   — the genuine case: a model that finished but never
    checkpointed. Keeps today's bounded retry.

Each test asserts the distinct error text, the ``violation_class`` machine
field, and the retry-budget effect (no failure-consumption regression).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import hermes_cli.kanban_db as kb


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (same as circuit_home)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _dead_worker(conn, task_id, pid, *, runtime):
    """Claim ``task_id`` to a dead pid and stamp the reap registry so
    detect_crashed_workers sees a 1s clean exit (WEXITSTATUS 0)."""
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, task_id, claimer=f"{host}:w{pid}")
    started_at = int(time.time()) - runtime
    conn.execute(
        "UPDATE tasks SET worker_pid=?, consecutive_failures=0, "
        "started_at=? WHERE id=?",
        (pid, started_at, task_id),
    )
    conn.commit()
    kb._record_worker_exit(pid, 0 << 8)
    _recent = kb._recent_worker_exits.get(int(pid))
    raw = _recent[0] if _recent else (0 << 8)
    kb._recent_worker_exits[int(pid)] = (raw, float(started_at + runtime))
    return started_at


def _write_log(isolated_home, task_id, text: str):
    log_dir = isolated_home / "kanban" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{task_id}.log").write_text(text, encoding="utf-8")


PROVIDER_LOG = (
    "ERROR openrouter: HTTP 403 Forbidden — provider returned monthly key "
    "limit reached (billing quota)\n"
)
WORKSPACE_LOG = (
    "workspace: .../hermes-agent is not inside a git repo and does not "
    "point at a git repo root\n"
)


def _trigger(isolated_home, monkeypatch, *, log_text):
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="cls", assignee="a")
        _dead_worker(conn, tid, pid=50500, runtime=1)
        if log_text is not None:
            _write_log(isolated_home, tid, log_text)
        crashed = kb.detect_crashed_workers(conn)
        run = kb.latest_run(conn, tid)
        task = kb.get_task(conn, tid)
        circuit = kb._circuit_status(conn).get("state")
        streak = kb._protocol_violation_streak(conn, tid)
    return tid, crashed, run, task, circuit, streak


# ---------------------------------------------------------------------------
# provider_error
# ---------------------------------------------------------------------------

def test_provider_error_class_releases_without_failure_and_trips_breaker(
    isolated_home, monkeypatch,
):
    """A fast+clean exit with a provider 4xx/5xx in the log is classified
    ``provider_error``: distinct text + ``violation_class`` field, released to
    source (breaker open) WITHOUT consuming the failure budget or the
    protocol-violation streak."""
    _tid, crashed, run, task, circuit, streak = _trigger(
        isolated_home, monkeypatch, log_text=PROVIDER_LOG,
    )
    assert run is not None and task is not None
    assert run.metadata.get("violation_class") == "provider_error"
    assert run.metadata.get("provider_outage") is True
    assert "provider" in (run.error or "").lower()
    # Retry-budget effect: released as a non-failure to the circuit breaker.
    assert crashed == [], "a provider wall is NOT a worker crash"
    assert task.consecutive_failures == 0, \
        "provider_error must not consume the failure budget"
    assert task.status == "ready", "released back to source for the breaker"
    assert circuit == "open", "the fleet-wide breaker must trip for a provider wall"
    assert streak == 0, "provider_error must not extend the violation streak"


# ---------------------------------------------------------------------------
# workspace_error
# ---------------------------------------------------------------------------

def test_workspace_error_blocks_capability_without_retrying(isolated_home, monkeypatch):
    """A clean exit caused by workspace/worktree contention is classified
    ``workspace_error`` and BLOCKS the card with a ``capability`` kind instead
    of retrying into the same broken workspace."""
    tid, crashed, run, task, circuit, streak = _trigger(
        isolated_home, monkeypatch, log_text=WORKSPACE_LOG,
    )
    assert run is not None and task is not None
    assert run.metadata.get("violation_class") == "workspace_error"
    assert run.outcome == "blocked"
    assert "workspace" in (run.error or "").lower()
    assert crashed == [], "a blocked card is not accounted as a crash"
    assert task.status == "blocked"
    assert task.block_kind == "capability"
    # MINOR (Rodge round-1): the workspace block must stamp last_failure_error
    # so the assessor sees a reason string, like the other failure branches.
    assert task.last_failure_error and "workspace" in task.last_failure_error.lower(), \
        f"assessor must see a reason string, got {task.last_failure_error!r}"
    assert task.consecutive_failures == 0, \
        "workspace_error must not consume the failure budget"
    assert circuit == "closed", \
        "workspace contention is a card fault, not a fleet outage"
    assert streak == 0, "workspace_error must not extend the violation streak"


# ---------------------------------------------------------------------------
# no_checkpoint
# ---------------------------------------------------------------------------

def test_no_checkpoint_keeps_bounded_retry(isolated_home, monkeypatch):
    """A clean exit with no provider/workspace error is the genuine
    ``no_checkpoint``: distinct text + field, keeping today's bounded
    protocol-violation retry (below the violation limit the card is re-queued,
    and the unified failure counter is not ticked)."""
    tid, crashed, run, task, circuit, streak = _trigger(
        isolated_home, monkeypatch, log_text=None,
    )
    assert run is not None and task is not None
    assert run.metadata.get("violation_class") == "no_checkpoint"
    assert "no_checkpoint" in (run.error or "")
    assert tid in crashed, "a genuine no_checkpoint is a bounded-retry crash"
    assert task.status == "ready", "below-budget violation re-queues, not blocks"
    assert task.consecutive_failures == 0, \
        "a below-budget violation must not tick the unified failure counter"
    assert circuit == "closed", "no_checkpoint is not a fleet outage"
    assert streak == 1, "a single genuine violation starts the bounded streak"


def test_benign_workspace_mention_is_no_checkpoint_not_workspace_error(
    isolated_home, monkeypatch,
):
    """Regression (Rodge round-1, t_8fc16a73): the bare noun ``workspace``
    appears in benign model reasoning (measured 126/166 real worker logs), so a
    genuine no_checkpoint whose finished-model log tail merely mentions
    ``workspace`` must NOT be misclassified ``workspace_error`` (which would
    block the card as capability instead of the bounded retry it needs). Only a
    real contention signature (e.g. ``not inside a git repo``) trips
    workspace_error."""
    tid, crashed, run, task, circuit, streak = _trigger(
        isolated_home, monkeypatch,
        log_text=(
            "I am working in the workspace at /Users/werolloperator/Projects. "
            "The worktree is checked out. The task is complete — I forgot to "
            "call kanban_complete."
        ),
    )
    assert run is not None and task is not None
    assert run.metadata.get("violation_class") == "no_checkpoint", (
        f"benign 'workspace'/'worktree' mention must stay no_checkpoint, "
        f"got {run.metadata.get('violation_class')!r}"
    )
    assert tid in crashed, "a benign log tail is still a bounded-retry crash"
    assert task.status == "ready", "must re-queue for bounded retry, not block"
    assert task.consecutive_failures == 0
    assert circuit == "closed"
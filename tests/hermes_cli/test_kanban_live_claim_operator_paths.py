"""Tests: the operator can always recover, and every run-closing write is fenced.

Round-2 review of PR #151601 (t_1f3b0108) found two gaps in the first fix.

1. ``reclaim_task`` learned to defer on a worker that survived termination,
   and ``hermes kanban reclaim`` got ``--force``. The HTTP surfaces did not:
   ``POST /tasks/{id}/reclaim``, ``POST /runs/{run}/terminate`` and
   ``reassign --reclaim`` all turned that deferral into "not in a claimable
   state (not running, or unknown id)" for a card that IS running and IS
   known, with no force route. The observed field condition (SIGTERM raises
   OSError, so SIGKILL is never reached: every one of the 42 reclaim events on
   t_f16a7cec recorded ``sigkill: false``) does not self-clear, so the
   operator needs that lever.

2. The live-claim fence covered ``complete_task``, ``request_review`` and
   ``block_task``, but ``schedule_task`` and ``request_changes`` also end the
   live run from an ``expected_run_id=None`` caller. ``scheduled`` is terminal
   by this codebase's own definition (``tools/kanban_tools.py`` gates it in
   goal_mode "same as blocked"), so a revoked worker parking the card is
   inside the card's acceptance criterion.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.discard(str(kb.kanban_db_path(board="default").resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


@pytest.fixture
def live_worker():
    procs: list[subprocess.Popen] = []

    def _spawn() -> int:
        proc = subprocess.Popen(["sleep", "120"])
        procs.append(proc)
        return proc.pid

    yield _spawn
    for proc in procs:
        proc.terminate()
        proc.wait()


def _unkillable(_pid, _sig):
    """The observed field condition: SIGTERM raises, the worker keeps running."""
    raise PermissionError(1, "Operation not permitted")


def _wedged(conn, live_worker, title="wedged"):
    """A running card whose host-local worker survives termination."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title=title, assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    kbd._set_worker_pid(conn, tid, live_worker())
    return tid


def _revoked_predecessor(conn, live_worker, title="succ"):
    """Worker A's claim is revoked; successor B is claimed and live.

    A's claim carries no worker pid, so the reclaim that revokes it has nothing
    to terminate and needs no ``force`` -- this setup runs against main too.
    """
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title=title, assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    stale_run = kb.get_task(conn, tid).current_run_id
    kb.reclaim_task(conn, tid, reason="revoked")
    kb.claim_task(conn, tid, claimer=f"{host}:B")
    kbd._set_worker_pid(conn, tid, live_worker())
    return tid, stale_run


def _survives_termination(pid, lock, **_kw):
    """Termination report matching the field condition on t_f16a7cec.

    All 42 reclaim events there recorded ``host_local: true,
    termination_attempted: true, terminated: false, sigkill: false`` -- the
    SIGTERM raised OSError, so SIGKILL was never reached. Used where the call
    site has no ``signal_fn`` hook (the HTTP endpoints, reassign).
    """
    return {"prev_pid": pid, "host_local": True, "termination_attempted": True,
            "terminated": False, "sigkill": False}


# ---------------------------------------------------------------------------
# 1. The deferral is reported as itself, and every surface can force past it
# ---------------------------------------------------------------------------


def test_reclaim_reports_the_deferral_distinctly(conn, live_worker):
    """A deferred reclaim must not look like "unknown id, or not running"."""
    tid = _wedged(conn, live_worker)

    ok, code = kb.reclaim_task(
        conn, tid, reason="operator", signal_fn=_unkillable, with_reason=True)

    assert ok is False
    assert code == kb.RECLAIM_WORKER_ALIVE
    assert kb.get_task(conn, tid).status == "running"

    # An absent card is the OTHER failure, and must stay distinguishable.
    assert kb.reclaim_task(conn, "t_nope", with_reason=True) == (
        False, kb.RECLAIM_NOT_CLAIMABLE)


def test_reclaim_endpoint_explains_a_live_worker_and_takes_force(conn, live_worker, monkeypatch):
    """POST /tasks/{id}/reclaim: honest 409, and a force route out of it."""
    from fastapi import HTTPException

    from plugins.kanban.dashboard import plugin_api

    tid = _wedged(conn, live_worker)
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", _survives_termination)

    with pytest.raises(HTTPException) as exc:
        plugin_api.reclaim_task_endpoint(tid, plugin_api.ReclaimBody(reason="ops"), board=None)
    detail = str(exc.value.detail)
    assert "survived termination" in detail
    assert "not running, or unknown id" not in detail, (
        "the card IS running and IS known; this message sends the operator "
        "looking for a different problem")
    assert kb.get_task(conn, tid).status == "running"

    body = plugin_api.reclaim_task_endpoint(
        tid, plugin_api.ReclaimBody(reason="ops", force=True), board=None)
    assert body["ok"] is True
    assert kb.get_task(conn, tid).status == "ready"


def test_terminate_run_endpoint_takes_force(conn, live_worker, monkeypatch):
    """POST /runs/{run}/terminate is the same lever and needs the same route."""
    from fastapi import HTTPException

    from plugins.kanban.dashboard import plugin_api

    tid = _wedged(conn, live_worker)
    run_id = kb.get_task(conn, tid).current_run_id
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", _survives_termination)

    with pytest.raises(HTTPException) as exc:
        plugin_api.terminate_run_endpoint(
            run_id, plugin_api.TerminateRunBody(reason="ops"), board=None)
    detail = str(exc.value.detail)
    assert "survived termination" in detail
    assert "no longer in a reclaimable state" not in detail

    body = plugin_api.terminate_run_endpoint(
        run_id, plugin_api.TerminateRunBody(reason="ops", force=True), board=None)
    assert body["ok"] is True
    assert kb.get_task(conn, tid).status == "ready"


def test_reassign_reclaim_first_reports_the_deferral(conn, live_worker, monkeypatch):
    """`reassign --reclaim` must not tell the operator to pass --reclaim."""
    tid = _wedged(conn, live_worker)
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", _survives_termination)

    ok, code = kb.reassign_task(
        conn, tid, "other", reclaim_first=True, reason="model broken", with_reason=True)
    assert ok is False
    assert code == kb.RECLAIM_WORKER_ALIVE
    task = kb.get_task(conn, tid)
    assert task.status == "running" and task.assignee == "w"

    ok2, _ = kb.reassign_task(
        conn, tid, "other", reclaim_first=True, force=True, with_reason=True)
    assert ok2 is True
    assert kb.get_task(conn, tid).assignee == "other"


def test_cli_reclaim_and_reassign_messages_name_the_live_worker(conn, live_worker, monkeypatch, capsys):
    """The CLI prints the deferral, not the generic "not running" line."""
    import argparse

    from hermes_cli import kanban_parser

    def _parse(argv):
        root = argparse.ArgumentParser()
        kanban_parser.build_parser(root.add_subparsers(dest="cmd"))
        return root.parse_args(["kanban"] + argv)

    tid = _wedged(conn, live_worker)
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", _survives_termination)

    assert kc._cmd_reclaim(_parse(["reclaim", tid])) != 0
    err = capsys.readouterr().err
    assert "survived termination" in err and "--force" in err

    assert kc._cmd_reassign(_parse(["reassign", tid, "other", "--reclaim"])) != 0
    err = capsys.readouterr().err
    assert "survived termination" in err
    assert "pass --reclaim" not in err, "told to do the thing they just did"

    assert kc._cmd_reassign(_parse(["reassign", tid, "other", "--reclaim", "--force"])) == 0


# ---------------------------------------------------------------------------
# 2. schedule_task and request_changes get the same fence as their siblings
# ---------------------------------------------------------------------------


def test_schedule_task_refuses_a_live_claim_without_ownership(conn, live_worker):
    """A revoked worker must not park the card its successor is running."""
    tid, _stale = _revoked_predecessor(conn, live_worker, "sched")

    with pytest.raises(kb.LiveClaimError):
        kb.schedule_task(conn, tid, reason="revoked worker parks the card")
    task = kb.get_task(conn, tid)
    assert task.status == "running"
    assert task.current_run_id is not None


def test_schedule_task_allows_the_owning_run_and_force(conn, live_worker):
    """The fence must not break the worker it protects, or the operator."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="own", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    kbd._set_worker_pid(conn, tid, live_worker())
    run_id = kb.get_task(conn, tid).current_run_id

    assert kb.schedule_task(conn, tid, reason="next week", expected_run_id=run_id) is True
    assert kb.get_task(conn, tid).status == "scheduled"

    tid2 = _wedged(conn, live_worker, "forced")
    assert kb.schedule_task(conn, tid2, reason="human", force=True) is True
    assert kb.get_task(conn, tid2).status == "scheduled"


def test_schedule_task_unaffected_without_a_live_worker(conn):
    """The ordinary CLI/library path keeps working untouched."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")

    assert kb.schedule_task(conn, tid, reason="later") is True
    assert kb.get_task(conn, tid).status == "scheduled"


def _review_card_with_revoked_reviewer(conn, live_worker):
    """Implementer hands off; reviewer A is revoked; reviewer B is live."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="rc", assignee="impl")
    kb.claim_task(conn, tid, claimer=f"{host}:I")
    impl_run = kb.get_task(conn, tid).current_run_id
    assert kb.request_review(conn, tid, summary="impl done", expected_run_id=impl_run)
    # A review-phase card is claimed with claim_review_task, not claim_task.
    assert kb.claim_review_task(conn, tid, claimer=f"{host}:RA") is not None
    kb.reclaim_task(conn, tid, reason="revoked")
    assert kb.claim_review_task(conn, tid, claimer=f"{host}:RB") is not None
    kbd._set_worker_pid(conn, tid, live_worker())
    return tid, kb.get_task(conn, tid).current_run_id


def test_request_changes_refuses_a_live_claim_without_ownership(conn, live_worker):
    """A revoked reviewer must not close the successor reviewer's run."""
    tid, live_run = _review_card_with_revoked_reviewer(conn, live_worker)

    ok, detail = kb.request_changes(conn, tid, reason="revoked reviewer verdict")

    assert ok is False
    assert "live claim" in (detail or "")
    task = kb.get_task(conn, tid)
    assert task.status == "running"
    assert task.current_run_id == live_run


def test_request_changes_allows_the_owning_run(conn, live_worker):
    """The live reviewer's own verdict still lands."""
    tid, live_run = _review_card_with_revoked_reviewer(conn, live_worker)

    ok, implementer = kb.request_changes(
        conn, tid, reason="three issues", expected_run_id=live_run)

    assert ok is True
    assert implementer == "impl"
    assert kb.get_task(conn, tid).status in ("ready", "todo")


def test_request_changes_force_overrides_a_live_claim(conn, live_worker):
    """The operator override exists on this verb too."""
    tid, _live_run = _review_card_with_revoked_reviewer(conn, live_worker)

    ok, implementer = kb.request_changes(conn, tid, reason="human verdict", force=True)

    assert ok is True
    assert implementer == "impl"

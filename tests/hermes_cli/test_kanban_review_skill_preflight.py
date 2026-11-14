"""A review claim must fail LOUDLY at claim time when a forced review skill is unresolvable.

A ``review`` claim force-loads the review skill (``sdlc-review``). If that
skill cannot be resolved in the assignee's profile the worker only discovers
it *after* the claim, inside the spawned session
(``gateway/session_kanban.py`` raises ``ValueError('kanban_skills_unavailable')``),
which looks like a crash loop: ``worker_result exit_code: 1`` with an empty
``last_output``, a ``crashed`` event, a retry into the same lane, and finally
``archived`` (live: t_8ccc6fa5 and t_7ae729ee, runs 368/369 and 356/357).

The guard these tests pin down resolves every forced skill BEFORE the claim:

* unresolvable -> the card is NOT claimed, a ``review_lane_unavailable`` event
  names the missing skill, an ERROR is logged naming the skill and the tick,
  no ``crashed`` event is written and no attempt is burned;
* all resolvable -> the claim proceeds exactly as before (no behaviour change).

The skill lookup is the real one (``agent.skill_commands``) against an
isolated ``HERMES_HOME``; the third test monkeypatches the resolver so a skill
that exists on disk still resolves as missing, proving the guard trusts the
resolver (disabled / ambiguous skills) instead of guessing from a directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

REVIEW_SKILL = "sdlc-review"


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB and an empty skills root."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _write_skill(home: Path, name: str) -> Path:
    skill_dir = home / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Review kanban handoffs and route verified outcomes.\n---\n\nReview body.\n",
        encoding="utf-8",
    )
    return skill_dir


def _review_card(conn, assignee: str = "default") -> str:
    """Create a card and walk it to ``status='review'`` like a real worker would."""
    tid = kb.create_task(conn, title="impl a feature", assignee=assignee)
    assert kb.claim_task(conn, tid)
    claimed = kb.get_task(conn, tid)
    assert claimed is not None
    run_id = claimed.current_run_id
    assert run_id is not None
    assert kb.request_review(conn, tid, summary="done", reviewer=assignee, expected_run_id=run_id)
    reviewing = kb.get_task(conn, tid)
    assert reviewing is not None and reviewing.status == "review"
    return tid


def _events(conn, tid: str) -> list[tuple[str, dict | None]]:
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (tid,)
    ).fetchall()
    return [(r["kind"], json.loads(r["payload"]) if r["payload"] else None) for r in rows]


def _spawner():
    """Records ``(task_id, skills the worker would be force-loaded with)``."""
    calls: list[tuple[str, list[str]]] = []

    def spawn(task, workspace, board=None):
        calls.append((task.id, list(task.skills or [])))
        return 4242

    return spawn, calls


def _log_text(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(r.getMessage() for r in caplog.records)


def test_review_claim_refused_when_forced_skill_is_missing(kanban_home: Path, caplog) -> None:
    """RED: on an unguarded dispatcher the card is claimed and spawns a worker
    that dies in ~5s with ``kanban_skills_unavailable``. With the guard the
    claim never happens and the reason is on the card."""
    with kbc.connect() as conn:
        tid = _review_card(conn)
        before = _events(conn, tid)
        spawn, calls = _spawner()
        with caplog.at_level(logging.ERROR):
            res = kbd.dispatch_once(conn, dry_run=False, spawn_fn=spawn)
        new = _events(conn, tid)[len(before):]
        task = kb.get_task(conn, tid)

    new_kinds = [k for k, _ in new]

    # No worker was started, so no attempt can be burned.
    assert calls == []
    assert (tid, [REVIEW_SKILL]) in res.review_lane_unavailable
    # The card is untouched and still needs a human.
    assert task is not None
    assert task.status == "review"
    assert task.current_run_id is None
    # Loud, greppable signal on the card instead of a crash loop.
    assert "review_lane_unavailable" in new_kinds, new_kinds
    assert "crashed" not in new_kinds, new_kinds
    assert "claimed" not in new_kinds, new_kinds
    payload = [p for k, p in new if k == "review_lane_unavailable"][-1]
    assert payload is not None
    assert payload["missing_skills"] == [REVIEW_SKILL]
    assert payload["message"].startswith("review lane is unable to start: missing skill")
    assert payload["lane"] == "review"
    # ... and in the dispatcher log, naming the skill and the tick.
    logs = _log_text(caplog)
    assert f"review lane is unable to start: missing skill {REVIEW_SKILL}" in logs, logs
    assert tid in logs, logs
    assert "tick=" in logs, logs


def test_review_claim_refused_when_resolver_reports_missing(kanban_home, caplog, monkeypatch) -> None:
    """The guard must believe the RESOLVER, not the filesystem: a disabled or
    ambiguous skill exists on disk yet force-loads as missing, and that is
    exactly the case that dies post-claim."""
    _write_skill(kanban_home, REVIEW_SKILL)
    from agent import skill_commands as skc

    monkeypatch.setattr(
        skc,
        "build_preloaded_skills_prompt",
        lambda identifiers, *a, **kw: ("", [], [str(i) for i in identifiers]),
    )
    with kbc.connect() as conn:
        tid = _review_card(conn)
        before = _events(conn, tid)
        spawn, calls = _spawner()
        with caplog.at_level(logging.ERROR):
            res = kbd.dispatch_once(conn, dry_run=False, spawn_fn=spawn)
        new_kinds = [k for k, _ in _events(conn, tid)[len(before):]]
        task = kb.get_task(conn, tid)

    assert calls == []
    assert (tid, [REVIEW_SKILL]) in res.review_lane_unavailable
    assert task is not None
    assert task.status == "review" and task.current_run_id is None
    assert "review_lane_unavailable" in new_kinds, new_kinds
    assert "crashed" not in new_kinds, new_kinds
    assert f"review lane is unable to start: missing skill {REVIEW_SKILL}" in _log_text(caplog)


def test_review_claim_proceeds_when_forced_skill_resolves(kanban_home: Path) -> None:
    """No behaviour change on the happy path: the card is claimed into the
    review lane, the forced skill rides the claim, and no refusal is recorded."""
    _write_skill(kanban_home, REVIEW_SKILL)
    with kbc.connect() as conn:
        tid = _review_card(conn)
        before = _events(conn, tid)
        spawn, calls = _spawner()
        res = kbd.dispatch_once(conn, dry_run=False, spawn_fn=spawn)
        new = _events(conn, tid)[len(before):]
        task = kb.get_task(conn, tid)

    new_kinds = [k for k, _ in new]
    assert [c[0] for c in calls] == [tid]
    # ``getattr`` so this no-behaviour-change test also runs (and passes) on an
    # unguarded dispatcher: the point is that the happy path is untouched.
    assert getattr(res, "review_lane_unavailable", []) == []
    assert task is not None
    assert task.status == "running"
    assert task.current_run_id is not None
    assert "review_lane_unavailable" not in new_kinds, new_kinds
    # The forced skill is still force-loaded onto the spawned worker (the guard
    # preflights the requirement; it does not replace it). The lane appends it to
    # ``claimed.skills``, which is what spawn_fn receives.
    assert REVIEW_SKILL in calls[0][1], calls[0]

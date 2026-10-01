"""A review-lane worker must never be spawned with a skill it cannot load.

``hermes ... --skills X`` aborts the worker during CLI init when every requested
skill is missing or operator-disabled, and the review lane force-loads
``sdlc-review``. A profile that disables it therefore produced a worker that died
before its first turn — no terminal board call, no exit trailer — and the card
was re-claimed into the review lane until the protocol-violation budget
auto-blocked it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    return home


class _Task:
    id = "t_x"
    assignee = "default"

    def __init__(self, skills):
        self.skills = list(skills)


def test_unresolvable_worker_skills_reports_a_disabled_skill(monkeypatch):
    from agent import skill_commands

    monkeypatch.setattr(
        skill_commands,
        "build_preloaded_skills_prompt",
        lambda names, task_id=None, excluded_loaded_names=None: ("", [], ["sdlc-review"]),
    )
    assert kbd._unresolvable_worker_skills(_Task(["sdlc-review"])) == {"sdlc-review"}


def test_unresolvable_worker_skills_fails_open(monkeypatch):
    from agent import skill_commands

    def boom(*_a, **_k):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(skill_commands, "build_preloaded_skills_prompt", boom)
    assert kbd._unresolvable_worker_skills(_Task(["sdlc-review"])) == set()


def test_worker_log_reads_are_scoped_to_the_current_run(kanban_home):
    """The log is append-mode across re-runs: a run that wrote no trailer of its
    own must not inherit the previous run's rc or final words."""
    log = kb.worker_log_path("t_scope")
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"the model said something\n\n{KANBAN_WORKER_EXIT_TRAILER}0\n")
        f.write(f"{kbd._RUN_START_MARKER}7\n")
        f.write("Query: work kanban task t_scope\nError: Unknown skill(s): sdlc-review\n")

    assert kbd._worker_log_exit_code("t_scope") is None
    out = kbd._worker_final_output("t_scope")
    assert "the model said something" not in out
    assert "Unknown skill(s)" in out

    # A run that DID reach its epilogue still reads its own trailer.
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"{kbd._RUN_START_MARKER}8\n")
        f.write(f"done\n\n{KANBAN_WORKER_EXIT_TRAILER}0\n")
    assert kbd._worker_log_exit_code("t_scope") == 0


# ---------------------------------------------------------------------------
# Dispatch level: the review lane's deferral (and its boundary)
# ---------------------------------------------------------------------------


def _dispatch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda _name: True)
    monkeypatch.setattr(
        cfgmod,
        "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )


def _review_card(conn, *, assignee: str = "reviewer", skills=None) -> str:
    """A card parked in the ``review`` column, ready for the review lane."""
    tid = kb.create_task(
        conn, title="domain review", assignee=assignee, skills=skills or [],
    )
    implementation = kb.claim_task(conn, tid)
    assert implementation is not None
    assert kb.request_review(
        conn, tid, summary="ready", expected_run_id=implementation.current_run_id,
    )
    return tid


def test_review_dispatch_defers_when_the_forced_skill_cannot_load(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline behavior: no spawn, infrastructure ``spawn_failed``, claim
    released, card back in ``review``, reason on the board, retry budget intact.
    """
    from hermes_cli import kanban_db_connect as kbc

    _dispatch_env(monkeypatch)
    monkeypatch.setattr(
        kbd, "_unresolvable_worker_skills", lambda _task, board=None: {"sdlc-review"},
    )
    spawned: list[str] = []

    with kbc.connect() as conn:
        tid = _review_card(conn)
        result = kbd.dispatch_once(conn, spawn_fn=lambda task, _ws: spawned.append(task.id))
        task = kb.get_task(conn, tid)
        run = conn.execute(
            "SELECT outcome, metadata, error FROM task_runs WHERE task_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        events = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (tid,),
        ).fetchall()

    assert spawned == []
    assert tid not in [row[0] for row in result.spawned]
    assert result.auto_blocked == []
    # Claim released, card back in review, nothing charged to its budget.
    assert task.status == "review"
    assert task.claim_lock is None
    assert task.consecutive_failures == 0
    assert "sdlc-review" in (task.last_failure_error or "")
    # Recorded as host infrastructure, not a card failure.
    assert run["outcome"] == "spawn_failed"
    assert json.loads(run["metadata"] or "{}").get("infrastructure") is True
    assert "spawn_failed" in [event["kind"] for event in events]


def test_review_dispatch_spawns_when_only_a_pinned_skill_is_unresolvable(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale pinned name alongside a loadable forced skill is NOT fatal: the
    CLI warns and continues, so the lane must still spawn the reviewer."""
    from hermes_cli import kanban_db_connect as kbc

    _dispatch_env(monkeypatch)
    monkeypatch.setattr(
        kbd,
        "_unresolvable_worker_skills",
        lambda _task, board=None: {"domain-specific-review"},
    )
    captured: list[list[str]] = []

    with kbc.connect() as conn:
        tid = _review_card(conn, skills=["domain-specific-review"])
        result = kbd.dispatch_once(
            conn, spawn_fn=lambda task, _ws: captured.append(list(task.skills or [])) or None,
        )
        task = kb.get_task(conn, tid)

    assert tid in [row[0] for row in result.spawned]
    assert captured == [["domain-specific-review", "sdlc-review"]]
    assert task.status == "running"
    assert task.consecutive_failures == 0
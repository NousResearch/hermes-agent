"""A review-lane worker must never be spawned with a skill it cannot load.

``hermes ... --skills X`` aborts the worker during CLI init when every requested
skill is missing or operator-disabled, and the review lane force-loads
``sdlc-review``. A profile that disables it therefore produced a worker that died
before its first turn — no terminal board call, no exit trailer — and the card
was re-claimed into the review lane until the protocol-violation budget
auto-blocked it.
"""

from __future__ import annotations

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
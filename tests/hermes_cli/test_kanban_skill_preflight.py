"""The preload identifier list and its resolution, checkable without spawning a worker.

Criterion these tests pin: a profile that cannot resolve *any* preloaded identifier answers the
same question a worker would ask, and the answer names the identifiers. The list itself is not
guessed — :func:`hermes_cli.kanban_db.worker_skill_identifiers` is the one definition the
dispatcher and the gateway also use (task-pinned skills plus ``sdlc-review`` on the review lane),
so this file asserts the list *and* the resolution separately.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_skill_preflight as preflight
from hermes_cli.kanban_db_connect import connect


@pytest.fixture
def fleet_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A root with one provisioned profile (``probe``) and one ghost profile name."""
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "probe"
    skill = profile / "skills" / "demo"
    skill.mkdir(parents=True)
    (profile / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (skill / "SKILL.md").write_text(
        "---\nname: demo\ndescription: a demo skill\n---\n\nDemo body.\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_worker_skill_identifiers_is_the_single_definition(fleet_home):
    assert kb.worker_skill_identifiers(None) == []
    assert kb.worker_skill_identifiers([], review_lane=True) == ["sdlc-review"]
    assert kb.worker_skill_identifiers(["pinned"], review_lane=False) == ["pinned"]
    # Order is stable and the review skill never doubles up on a task that pinned it.
    assert kb.worker_skill_identifiers(["pinned", "sdlc-review"], review_lane=True) == [
        "pinned", "sdlc-review"]


def test_resolution_runs_against_the_assignee_profile_home(fleet_home):
    assert preflight.resolve_for_profile("probe", ["demo"]) == (["demo"], [])
    loaded, missing = preflight.resolve_for_profile("probe", ["demo", "absent"])
    assert loaded == ["demo"] and missing == ["absent"]


def test_only_a_total_miss_is_unsatisfiable(fleet_home):
    # One resolved identifier is enough for a worker turn to start: a partial miss is a warning.
    assert preflight.unsatisfiable_preload("probe", ["demo", "absent"]) is None
    assert preflight.unsatisfiable_preload("probe", []) is None
    assert preflight.unsatisfiable_preload("probe", ["absent"]) == ["absent"]
    # An unresolvable profile home is not evidence of an unprovisioned profile: fail open and let
    # the worker's own guard decide.
    assert preflight.unsatisfiable_preload("ghost", ["absent"]) is None


def test_failure_reason_names_the_profile_and_every_missing_skill():
    reason = preflight.preload_failure_reason("portfolio-lead", ["sdlc-review", "other"])
    assert preflight.PREFLIGHT_FAILURE_MARKER in reason
    assert "portfolio-lead" in reason and "sdlc-review" in reason and "other" in reason


def test_sweep_reports_an_unprovisioned_assignee_instead_of_skipping_it(fleet_home, capsys, monkeypatch):
    # ``preload``/``argparse`` builds its help strings through gettext, which on Windows loads
    # ``messages.mo`` from the interpreter's own share/locale tree — inside the real Hermes home
    # when the borrowed test interpreter lives there, tripping tests/home_io_guard.py. The lookup
    # is harmless to the code under test, so neutralise the translator rather than the guard.
    import argparse

    monkeypatch.setattr(argparse, "_", lambda message: message, raising=False)

    with connect() as conn:
        kb.create_task(conn, title="review me", assignee="probe")
        kb.create_task(conn, title="ghost", assignee="ghost-profile")
        report = preflight.sweep_assignees(conn)

    by_profile = {entry["profile"]: entry for entry in report}
    assert by_profile["probe"]["state"] == "unsatisfiable"
    assert by_profile["probe"]["missing"] == ["sdlc-review"]
    assert by_profile["ghost-profile"]["state"] == "unknown"

    assert preflight.main(["--json"]) == 1
    printed = json.loads(capsys.readouterr().out)
    assert {entry["profile"] for entry in printed} == {"probe", "ghost-profile"}

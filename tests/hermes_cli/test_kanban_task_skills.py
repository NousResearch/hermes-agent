"""Task-skills normalization at card creation: joined strings and toolset names
never reach a card (a slash-joined pseudo-skill persisted to ``tasks.skills``
aborts the dispatched worker at boot, on every retry)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_skills import _normalize_task_skills


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.mark.parametrize("joined, refusal", [
    ("github-pr-workflow,github-code-review", "cannot contain comma"),
    ("github-pr-workflow/github-code-review", "cannot contain '/'"),
    ("skills/github-code-review", "cannot contain '/'"),
    ("github-code-review/", "cannot contain '/'"),
])
def test_refuses_joined_strings(joined, refusal):
    """A joined multi-name string must never land in one argv slot: it persists to
    the card verbatim and the dispatched worker then fails to resolve it at boot."""
    with pytest.raises(ValueError, match=refusal):
        _normalize_task_skills([joined])


def test_slash_refusal_names_the_fix():
    """The '/' refusal must be actionable — name separate names, not a path form."""
    with pytest.raises(ValueError, match="pass separate names"):
        _normalize_task_skills(["github-pr-workflow/github-code-review"])


def test_rejects_toolset_names():
    with pytest.raises(ValueError, match="are toolset names, not skill name"):
        _normalize_task_skills(["web", "terminal"])


def test_strips_and_dedupes():
    assert _normalize_task_skills([" a ", "", "b", "a"]) == ["a", "b"]
    assert _normalize_task_skills(None) is None


def test_create_task_refuses_slash_joined_skills_before_persist(kanban_home):
    """The refusal must fire at the persistence boundary: no card may be created
    carrying a slash-joined pseudo-skill for a worker to crash on later."""
    with kbc.connect() as conn:
        with pytest.raises(ValueError, match="cannot contain '/'"):
            kb.create_task(conn, title="poisoned card", assignee="worker-a",
                           skills=["github-pr-workflow/github-code-review"])
        assert kb.list_tasks(conn) == []


# --- assignee-profile resolution: a card's skills must load where the card runs ------
#
# Profiles are islands: ``--skill`` names are validated against the TARGET assignee's
# skill trees at create/assign/reassign time. A name that only exists in the creating
# profile previously persisted fine and aborted the dispatched worker at boot
# (``ValueError: kanban_skills_unavailable``) on every retry, with no output.

def _plant_skill(skills_dir: Path, name: str) -> None:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test skill {name}\n---\n\n# {name}\n",
        encoding="utf-8",
    )


def _live_profile(home: Path, name: str) -> Path:
    """Profile dir carrying an identity marker (config.yaml) — live for dispatch."""
    profile = home / "profiles" / name
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "config.yaml").write_text("{}\n", encoding="utf-8")
    return profile


def test_create_refuses_skill_missing_in_assignee_profile(kanban_home):
    """A well-formed name that resolves nowhere in the assignee's trees is refused at
    the persistence boundary, naming both the skill and the profile."""
    analyst = _live_profile(kanban_home, "polybot-analyst")
    _plant_skill(analyst / "skills", "kanban-worker")
    with kbc.connect() as conn:
        kb.create_task(conn, title="healthy card", assignee="polybot-analyst",
                       skills=["kanban-worker"])
        with pytest.raises(
            ValueError,
            match=r"measurement-instrument-operations.*assignee profile 'polybot-analyst'",
        ):
            kb.create_task(conn, title="probe", assignee="polybot-analyst",
                           skills=["measurement-instrument-operations"])
        assert len(kb.list_tasks(conn)) == 1


def test_assign_refuses_card_skills_the_new_profile_cannot_load(kanban_home):
    """Reassignment moves the card onto another profile's island: its skills must
    resolve THERE, at the moment of the handoff, not at the next dispatch."""
    analyst = _live_profile(kanban_home, "polybot-analyst")
    _plant_skill(analyst / "skills", "kanban-orchestrator")
    director = _live_profile(kanban_home, "polybot-director")
    _plant_skill(director / "skills", "measurement-instrument-operations")
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="deferred card",
                                 skills=["measurement-instrument-operations"])
        with pytest.raises(
            ValueError,
            match=r"measurement-instrument-operations.*assignee profile 'polybot-analyst'",
        ):
            kb.assign_task(conn, task_id, "polybot-analyst")
        deferred = kb.get_task(conn, task_id)
        assert deferred is not None and deferred.assignee is None
        assert kb.assign_task(conn, task_id, "polybot-director") is True


def test_create_refuses_skill_installed_only_in_creating_profile(kanban_home):
    """The exact incident shape: the skill exists in the creating (default) profile
    only, so it must not pass validation for an assignee that lacks it."""
    _plant_skill(kanban_home / "skills", "measurement-instrument-operations")
    _live_profile(kanban_home, "polybot-analyst")
    with kbc.connect() as conn:
        with pytest.raises(
            ValueError,
            match=r"measurement-instrument-operations.*profile 'polybot-analyst'",
        ):
            kb.create_task(conn, title="cross-profile card", assignee="polybot-analyst",
                           skills=["measurement-instrument-operations"])
        assert kb.list_tasks(conn) == []


def test_create_accepts_skill_from_assignee_external_dirs(kanban_home):
    """The guard searches the trees the WORKER searches: ``skills.external_dirs``
    from the assignee's own config count, so shared trees neither false-reject nor
    get bypassed."""
    analyst = _live_profile(kanban_home, "polybot-analyst")
    shared = kanban_home.parent / "shared-skills"
    _plant_skill(shared, "measurement-instrument-operations")
    (analyst / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {shared}\n", encoding="utf-8")
    with kbc.connect() as conn:
        kb.create_task(conn, title="external skill card", assignee="polybot-analyst",
                       skills=["measurement-instrument-operations"])
        assert len(kb.list_tasks(conn)) == 1


def test_create_skips_validation_when_assignee_is_not_live(kanban_home):
    """An assignee profile that is not live on this root cannot be verified up front
    (the card is undispatchable anyway and the dispatcher records unknown_assignee),
    so validation fails open instead of blocking authoring."""
    with kbc.connect() as conn:
        kb.create_task(conn, title="ghost card", assignee="ghost",
                       skills=["measurement-instrument-operations"])
        assert len(kb.list_tasks(conn)) == 1


def test_create_skips_plugin_qualified_names(kanban_home):
    """A plugin-qualified name resolves from the worker's plugin registry, not these
    trees, so it is not existence-checked here."""
    _live_profile(kanban_home, "polybot-analyst")
    with kbc.connect() as conn:
        kb.create_task(conn, title="plugin skill card", assignee="polybot-analyst",
                       skills=["acme:helper"])
        assert len(kb.list_tasks(conn)) == 1


def test_reassign_refusal_leaves_claim_and_assignee_untouched(kanban_home):
    """``reassign --reclaim`` must validate BEFORE releasing the live claim, or a
    refused handoff leaves the card unclaimed and unrunnable."""
    analyst = _live_profile(kanban_home, "polybot-analyst")
    _plant_skill(analyst / "skills", "kanban-worker")
    _live_profile(kanban_home, "polybot-reviewer")
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="claimed card", assignee="polybot-analyst",
                                 skills=["kanban-worker"])
        assert kb.claim_task(conn, task_id, claimer="worker-lock") is not None
        with pytest.raises(
            ValueError,
            match=r"kanban-worker.*assignee profile 'polybot-reviewer'",
        ):
            kb.reassign_task(conn, task_id, "polybot-reviewer", reclaim_first=True)
        row = conn.execute(
            "SELECT assignee, claim_lock, status FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        assert row["assignee"] == "polybot-analyst"
        assert row["claim_lock"] == "worker-lock"
        assert row["status"] == "running"


def test_cli_create_exits_nonzero_naming_skill_and_assignee(kanban_home, capsys):
    """The operator-facing surface: non-zero exit and a message naming the skill and
    the assignee profile, with no card written."""
    _live_profile(kanban_home, "polybot-analyst")
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    kc.build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args([
        "kanban", "create", "probe",
        "--skill", "measurement-instrument-operations",
        "--assignee", "polybot-analyst",
    ])
    assert kc.kanban_command(args) == 1
    err = capsys.readouterr().err
    assert "measurement-instrument-operations" in err
    assert "polybot-analyst" in err

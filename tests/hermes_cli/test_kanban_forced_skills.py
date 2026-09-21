"""Forced skills on a Kanban card are validated against the profile that will run it.

A card's ``--skill`` / ``skills=[...]`` names are handed to the worker as
``hermes --skills <name>``. Nothing downstream treats an unresolvable name as an
error: ``build_preloaded_skills_prompt`` reports it *missing* and the worker runs
anyway, without the specialist context the card exists to pin. The failure is
therefore invisible until someone reads the run log, and it costs a full worker
attempt every time.

These are contract tests for the shared pre-queue validation: it runs for every
creation surface (CLI and model tool alike), it resolves names against the
ASSIGNEE's profile rather than the creator's, and it rejects a bad list whole.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def install_skill(home: Path, category: str, name: str, description: str = "a test skill") -> Path:
    """Write ``<home>/skills/<category>/<name>/SKILL.md`` the way the scanner expects."""
    skill_dir = home / "skills" / category / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\nDo the thing.\n",
        encoding="utf-8",
    )
    return skill_dir


@pytest.fixture()
def board(tmp_path, monkeypatch):
    """Isolated Hermes home + kanban board with one profile ('alpha') on disk.

    ``alpha`` has exactly one skill installed, under the ``writing`` category.
    """
    home = tmp_path / ".hermes"
    (home / "profiles" / "alpha").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "default")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    install_skill(home / "profiles" / "alpha", "writing", "translation")

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return kb


def _create(kb, **kwargs):
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        return kb.create_task(conn, title="t", assignee="alpha", **kwargs)


def _task_count(kb) -> int:
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        return int(conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"])


# --- the validation itself ---------------------------------------------------

def test_installed_skill_is_accepted(board):
    """The baseline: a name the assignee profile really has must still create."""
    task_id = _create(board, skills=["translation"])
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        assert board.get_task(conn, task_id).skills == ["translation"]


def test_category_name_is_not_a_skill(board):
    """``writing`` is the CATEGORY directory holding ``translation``; force-loading
    it resolves nothing, so it must be refused and named as a category."""
    with pytest.raises(ValueError) as excinfo:
        _create(board, skills=["writing"])
    message = str(excinfo.value)
    assert "writing" in message
    assert "categor" in message.lower(), message
    # The remedy is the skill inside it, so the error must point at it.
    assert "translation" in message, message
    assert _task_count(board) == 0


def test_mixed_list_is_rejected_atomically(board):
    """One bad name in a list of three leaves NO card behind: a partially-honoured
    forced-skill list is the silent failure this validation exists to stop."""
    with pytest.raises(ValueError) as excinfo:
        _create(board, skills=["translation", "not-a-real-skill", "writing"])
    message = str(excinfo.value)
    assert "not-a-real-skill" in message
    assert "writing" in message, "every invalid name is reported, not just the first"
    assert _task_count(board) == 0


def test_no_valid_names_reports_all_of_them(board):
    """When nothing in the list resolves, the operator gets the whole list back —
    reporting one name at a time turns a typo sweep into N round trips."""
    with pytest.raises(ValueError) as excinfo:
        _create(board, skills=["alpha-bogus", "beta-bogus"])
    message = str(excinfo.value)
    assert "alpha-bogus" in message and "beta-bogus" in message
    assert _task_count(board) == 0


def test_close_name_gets_a_suggestion(board):
    """A one-character typo must surface the intended skill; without it the
    operator's next move is to go read the profile's skills directory by hand."""
    with pytest.raises(ValueError) as excinfo:
        _create(board, skills=["translatio"])
    assert "translation" in str(excinfo.value)


def test_disabled_skill_is_refused_because_the_worker_will_not_load_it(board, tmp_path):
    """Installed is not the same as loadable.

    ``build_preloaded_skills_prompt`` loads forced skills with
    ``disabled_as_missing=True``, so an operator-disabled name reaches the worker as
    *missing* — the exact silent failure this validation exists to stop. Asserted
    against that loader rather than against a string: the two must agree on what a
    forced name resolves to, whichever way either side is rewritten.
    """
    import yaml

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    profile_home = tmp_path / ".hermes" / "profiles" / "alpha"
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump({"skills": {"disabled": ["translation"]}}), encoding="utf-8")

    token = set_hermes_home_override(str(profile_home))
    try:
        from agent.skill_commands import build_preloaded_skills_prompt
        _text, loaded, missing = build_preloaded_skills_prompt(["translation"])
    finally:
        reset_hermes_home_override(token)
    assert missing == ["translation"] and loaded == [], (loaded, missing)

    with pytest.raises(ValueError) as excinfo:
        _create(board, skills=["translation"])
    assert "translation" in str(excinfo.value)
    assert _task_count(board) == 0


def test_validation_resolves_against_the_assignee_not_the_creator(board, tmp_path):
    """The creating profile's skills say nothing about the assignee's.

    ``beta`` has ``research-digest`` installed and ``alpha`` does not, so the same
    card is valid for one assignee and invalid for the other.
    """
    home = tmp_path / ".hermes"
    (home / "profiles" / "beta").mkdir(parents=True)
    install_skill(home / "profiles" / "beta", "research", "research-digest")
    # Installed on the CREATOR (default profile) too — must not launder the name.
    install_skill(home, "research", "research-digest")

    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        beta_id = board.create_task(conn, title="ok", assignee="beta", skills=["research-digest"])
        assert board.get_task(conn, beta_id).skills == ["research-digest"]
        with pytest.raises(ValueError) as excinfo:
            board.create_task(conn, title="no", assignee="alpha", skills=["research-digest"])
    assert "alpha" in str(excinfo.value)


def test_unresolvable_profile_does_not_block_creation(board):
    """Fail OPEN when the assignee has no profile directory to enumerate.

    Control-plane lanes (``orion-cc`` and friends) are legitimate assignees with no
    Hermes profile on disk; refusing their cards would be a regression, and the
    dispatcher's backstop still guards the spawn.
    """
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        task_id = board.create_task(
            conn, title="lane", assignee="orion-cc", skills=["whatever-they-use"])
        assert board.get_task(conn, task_id).skills == ["whatever-they-use"]


# --- project skill context ---------------------------------------------------

def test_project_local_skill_counts_when_the_repo_is_trusted(board, tmp_path):
    """A card anchored to a project may force-load that project's own skills.

    Project-local skills (``<repo>/.hermes/skills``) only load from a repo listed in
    ``skills.trusted_project_dirs``; the validator must apply the same rule against
    the card's project repo instead of the validator's own cwd.
    """
    import yaml

    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    install_skill(repo / ".hermes", "house", "house-style")

    home = tmp_path / ".hermes"
    profile_home = home / "profiles" / "alpha"
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump({"skills": {"trusted_project_dirs": [str(repo)]}}), encoding="utf-8")

    from hermes_cli import kanban_db_skills as kbs
    # Without project context the name is unknown ...
    assert kbs.check_forced_skills(["house-style"], assignee="alpha")
    # ... and with it, it resolves.
    assert kbs.check_forced_skills(["house-style"], assignee="alpha", project_path=str(repo)) == []


def test_untrusted_project_skill_is_still_rejected(board, tmp_path):
    """No trust entry = the worker will not load it either, so the card is bad."""
    repo = tmp_path / "untrusted"
    (repo / ".git").mkdir(parents=True)
    install_skill(repo / ".hermes", "house", "sneaky-style")

    from hermes_cli import kanban_db_skills as kbs
    problems = kbs.check_forced_skills(["sneaky-style"], assignee="alpha", project_path=str(repo))
    assert [p.name for p in problems] == ["sneaky-style"]


# --- both creation surfaces share it -----------------------------------------

def test_model_tool_creation_is_validated(board, monkeypatch):
    """``kanban_create`` (the orchestrator's fan-out tool) rejects the same names."""
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from tools import kanban_tools as kt
    out = json.loads(kt._handle_create(
        {"title": "fan out", "assignee": "alpha", "skills": ["translatio"]}))
    assert "task_id" not in out, out
    assert "translation" in str(out.get("error") or ""), out
    assert _task_count(board) == 0


def test_cli_creation_is_validated(board, capsys):
    """``hermes kanban create --skill <typo>`` exits non-zero instead of tracebacking."""
    import argparse

    from hermes_cli import kanban

    args = argparse.Namespace(
        title="cli card", body=None, assignee="alpha", parent=[], workspace=None,
        branch=None, project=None, tenant=None, priority=0, triage=False,
        idempotency_key=None, max_runtime=None, created_by="user", skills=["translatio"],
        max_retries=None, model_override=None, provider_override=None,
        completion_contract=None, goal_mode=False, goal_max_turns=None,
        initial_status="running", json=False,
    )
    assert kanban._cmd_create(args) != 0
    assert "translation" in capsys.readouterr().err
    assert _task_count(board) == 0

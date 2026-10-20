"""The curator must not archive a skill that an unfinished kanban card will force-load.

These run the real lookup end to end: real boards under the shared kanban root, a named
profile home for the curator, and no stub over ``_kanban_referenced_skills``.
"""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    home = root / "profiles" / "implementer"
    (home / "skills").mkdir(parents=True)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("HERMES_KANBAN_HOME", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)

    import tools.skill_usage as usage
    importlib.reload(usage)
    import agent.curator as curator
    importlib.reload(curator)
    monkeypatch.setattr(curator, "_load_config", lambda: {})
    monkeypatch.setattr(usage, "_prune_builtins_enabled", lambda: False)

    import hermes_cli.kanban_db as kb
    from hermes_cli.kanban_db_connect import connect_closing
    yield {"home": home, "curator": curator, "usage": usage, "kb": kb, "connect": connect_closing}


def _idle_skill(env, name: str, days: int = 200) -> None:
    d = env["home"] / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: x\n---\n", encoding="utf-8")
    u = env["usage"]
    ts = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    data = u.load_usage()
    data[name] = u._empty_record()
    data[name].update(created_by="agent", created_at=ts, last_used_at=ts, last_activity_at=ts, use_count=1)
    u.save_usage(data)


def _card(env, *, board: str, assignee: str, skills: list[str]) -> str:
    kb = env["kb"]
    kb.create_board(board)
    with env["connect"](board=board) as conn:
        return kb.create_task(conn, title="t", assignee=assignee, skills=skills, board=board)


def _state(env, name: str) -> str:
    return env["usage"].load_usage()[name]["state"]


def test_skill_named_by_open_card_for_this_profile_survives(profile_env):
    _idle_skill(profile_env, "ui-redesign")
    _card(profile_env, board="site", assignee="implementer", skills=["ui-redesign"])

    counts = profile_env["curator"].apply_automatic_transitions()

    assert counts["archived"] == 0
    assert _state(profile_env, "ui-redesign") == profile_env["usage"].STATE_ACTIVE


def test_card_for_another_profile_does_not_protect(profile_env):
    _idle_skill(profile_env, "ui-redesign")
    _card(profile_env, board="site", assignee="reviewer", skills=["ui-redesign"])

    profile_env["curator"].apply_automatic_transitions()

    assert _state(profile_env, "ui-redesign") == profile_env["usage"].STATE_ARCHIVED


def test_finished_card_releases_protection(profile_env):
    _idle_skill(profile_env, "ui-redesign")
    kb = profile_env["kb"]
    task_id = _card(profile_env, board="site", assignee="implementer", skills=["ui-redesign"])
    with profile_env["connect"](board="site") as conn:
        kb.archive_task(conn, task_id)

    profile_env["curator"].apply_automatic_transitions()

    assert _state(profile_env, "ui-redesign") == profile_env["usage"].STATE_ARCHIVED


def test_candidate_list_flags_kanban_referenced_skills(profile_env):
    _idle_skill(profile_env, "ui-redesign")
    _idle_skill(profile_env, "unrelated")
    _card(profile_env, board="site", assignee="implementer", skills=["ui-redesign"])

    lines = profile_env["curator"]._render_candidate_list().splitlines()

    assert any(l.startswith("- ui-redesign ") and "kanban=yes" in l for l in lines)
    assert any(l.startswith("- unrelated ") and "kanban=no" in l for l in lines)


def test_unreadable_board_never_breaks_the_curator(profile_env):
    _idle_skill(profile_env, "ui-redesign")
    kb = profile_env["kb"]
    kb.create_board("broken")
    Path(kb.kanban_db_path("broken")).write_bytes(b"not a sqlite database")

    counts = profile_env["curator"].apply_automatic_transitions()

    assert counts["archived"] == 1


def test_protection_follows_the_active_profile_home(profile_env, tmp_path, monkeypatch):
    implementer = profile_env["home"]
    reviewer = tmp_path / ".hermes" / "profiles" / "reviewer"
    (reviewer / "skills").mkdir(parents=True)
    (reviewer / "config.yaml").write_text("{}\n", encoding="utf-8")
    _card(profile_env, board="site", assignee="implementer", skills=["ui-redesign"])
    curator, usage = profile_env["curator"], profile_env["usage"]

    _idle_skill(profile_env, "ui-redesign")
    assert curator.apply_automatic_transitions()["archived"] == 0

    monkeypatch.setenv("HERMES_HOME", str(reviewer))
    profile_env["home"] = reviewer
    _idle_skill(profile_env, "ui-redesign")
    curator.apply_automatic_transitions()
    assert usage.load_usage()["ui-redesign"]["state"] == usage.STATE_ARCHIVED

    monkeypatch.setenv("HERMES_HOME", str(implementer))
    profile_env["home"] = implementer
    curator.apply_automatic_transitions()
    assert usage.load_usage()["ui-redesign"]["state"] == usage.STATE_ACTIVE

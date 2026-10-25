"""Review dispatch must not force-load ``sdlc-review`` where the reviewer profile disables it.

``hermes --skills X`` treats a disabled skill as missing and exits 1 ("Unknown skill(s)"), so a
review spawn that force-loaded a disabled ``sdlc-review`` was booked crashed -> crashed -> gave_up
on a card whose implementation had already been handed off.
"""

from __future__ import annotations

from pathlib import Path

import pytest

@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _review_card(conn, spawn):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    tid = kb.create_task(conn, title="review me", assignee="reviewer", skills=["domain-review"])
    impl = kb.claim_task(conn, tid)
    assert kb.request_review(conn, tid, summary="done", expected_run_id=impl.current_run_id)
    return tid, kbd.dispatch_once(conn, spawn_fn=spawn)


def _setup(kanban_home: Path, monkeypatch: pytest.MonkeyPatch, config_yaml: str) -> Path:
    import agent.skill_utils as su
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod
    import hermes_constants as hc

    monkeypatch.setattr(su, "get_config_path", hc.get_config_path)
    monkeypatch.setattr(su, "get_skills_dir", hc.get_skills_dir)
    monkeypatch.setattr(su, "get_subprocess_home", hc.get_subprocess_home)

    reviewer_home = kanban_home / "profiles" / "reviewer"
    reviewer_home.mkdir(parents=True)
    (reviewer_home / "config.yaml").write_text(config_yaml, encoding="utf-8")
    skill_dir = reviewer_home / "skills" / "domain-review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: domain-review\ndescription: Fixture review\n---\nReview guidance.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(profmod, "resolve_profile_env", lambda name: str(reviewer_home))
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {"kanban": {"review_dispatch": True}})
    return reviewer_home


def test_disabled_sdlc_review_is_not_force_loaded(kanban_home, monkeypatch):
    _setup(kanban_home, monkeypatch, "skills:\n  disabled:\n    - sdlc-review\n")
    captured: list[list[str]] = []

    def spawn(task, workspace):
        captured.append(list(task.skills or []))
        return None

    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect() as conn:
        tid, res = _review_card(conn, spawn)

    assert tid in [t[0] for t in res.spawned]
    assert captured == [["domain-review"]]


def test_enabled_sdlc_review_is_still_force_loaded(kanban_home, monkeypatch):
    _setup(kanban_home, monkeypatch, "skills:\n  disabled: []\n")
    captured: list[list[str]] = []

    def spawn(task, workspace):
        captured.append(list(task.skills or []))
        return None

    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect() as conn:
        _review_card(conn, spawn)

    assert captured == [["domain-review", "sdlc-review"]]


def test_skill_disabled_helper_fails_open(kanban_home, monkeypatch):
    from hermes_cli import kanban_db_dispatch as kbd

    assert kbd._skill_disabled_for_assignee("sdlc-review", None) is False
    import hermes_cli.profiles as profmod

    def boom(name):
        raise RuntimeError("no such profile")

    monkeypatch.setattr(profmod, "resolve_profile_env", boom)
    assert kbd._skill_disabled_for_assignee("sdlc-review", "reviewer") is False

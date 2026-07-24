"""Regression tests for #44072: ``kanban create --skill`` must validate
skill names at dispatch time.

Workers preload ``--skill`` values at CLI startup and crash with
``ValueError: Unknown skill(s): <name>`` on unknown names — burning the
task's retry budget and completing with an empty result. ``kanban create``
now rejects tasks whose force-loaded skills won't resolve for the worker's
profile home, before the task is enqueued.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _install_skill(root: Path, rel_dir: str, frontmatter_name: str | None = None) -> None:
    skill_dir = root / "skills" / rel_dir
    skill_dir.mkdir(parents=True, exist_ok=True)
    name = frontmatter_name or rel_dir.rsplit("/", 1)[-1]
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test skill\n---\n\n# {name}\n",
        encoding="utf-8",
    )


def _task_count() -> int:
    with kb.connect() as conn:
        return len(kb.list_tasks(conn))


# ---------------------------------------------------------------------------
# unresolvable_task_skills (unit)
# ---------------------------------------------------------------------------

def test_unknown_skill_reported(kanban_home):
    _install_skill(kanban_home, "devops/real-skill")
    missing = kb.unresolvable_task_skills(
        ["real-skill", "no-such-skill"], str(kanban_home)
    )
    assert missing == ["no-such-skill"]


def test_relative_path_form_resolves(kanban_home):
    _install_skill(kanban_home, "devops/nested-skill")
    assert kb.unresolvable_task_skills(
        ["devops/nested-skill"], str(kanban_home)
    ) == []


def test_frontmatter_name_alias_resolves(kanban_home):
    # Dir is "short-alias" but the frontmatter name is what skills_list
    # exposes — skill_view accepts it, so the validator must too.
    _install_skill(kanban_home, "cat/short-alias", frontmatter_name="long-public-name")
    assert kb.unresolvable_task_skills(
        ["long-public-name"], str(kanban_home)
    ) == []


def test_legacy_flat_md_resolves(kanban_home):
    flat = kanban_home / "skills" / "misc"
    flat.mkdir(parents=True)
    (flat / "oldstyle.md").write_text("# oldstyle\n", encoding="utf-8")
    assert kb.unresolvable_task_skills(["oldstyle"], str(kanban_home)) == []


def test_kanban_worker_and_plugin_names_get_benefit_of_doubt(kanban_home):
    # kanban-worker is gated by the dispatcher itself (dropped when absent,
    # never fatal); plugin registries are per-home state we can't inspect
    # cross-profile — neither may be rejected here.
    (kanban_home / "skills").mkdir()
    assert kb.unresolvable_task_skills(
        ["kanban-worker", "someplugin:some-skill"], str(kanban_home)
    ) == []


def test_external_dirs_from_config_are_searched(kanban_home, tmp_path):
    ext = tmp_path / "external-skills"
    (ext / "shipped-elsewhere").mkdir(parents=True)
    (ext / "shipped-elsewhere" / "SKILL.md").write_text(
        "---\nname: shipped-elsewhere\n---\n", encoding="utf-8"
    )
    (kanban_home / "skills").mkdir()
    (kanban_home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {ext.as_posix()}\n", encoding="utf-8"
    )
    assert kb.unresolvable_task_skills(
        ["shipped-elsewhere"], str(kanban_home)
    ) == []


def test_missing_skills_root_rejects_concrete_names(kanban_home):
    # No <home>/skills at all → preloading any skill is fatal at worker
    # startup ("Skills directory does not exist yet").
    assert kb.unresolvable_task_skills(["anything"], str(kanban_home)) == ["anything"]


# ---------------------------------------------------------------------------
# kanban create wiring (end-to-end via run_slash)
# ---------------------------------------------------------------------------

def test_create_rejects_unknown_skill(kanban_home):
    _install_skill(kanban_home, "devops/real-skill")
    out = kc.run_slash("create 'doomed task' --skill some-unknown-skill")
    assert "unknown skill(s)" in out.lower()
    assert "some-unknown-skill" in out
    assert "Created" not in out
    assert _task_count() == 0


def test_create_accepts_installed_skill(kanban_home):
    _install_skill(kanban_home, "devops/real-skill")
    out = kc.run_slash("create 'good task' --skill real-skill")
    assert "Created" in out
    with kb.connect() as conn:
        task = kb.list_tasks(conn)[0]
    assert list(task.skills) == ["real-skill"]


def test_create_validates_against_assignee_profile_home(kanban_home):
    # Skill installed ONLY in the assignee profile's home — the root home
    # doesn't have it. Create must validate against the profile the worker
    # will actually run under, not the creator's home.
    profile_home = kanban_home / "profiles" / "coder"
    _install_skill(profile_home, "devops/profile-only-skill")
    out = kc.run_slash(
        "create 'profile task' --assignee coder --skill profile-only-skill"
    )
    assert "Created" in out

    # And the inverse: the skill is in the creator's root home but NOT in
    # the assignee's profile home — exactly the failure in #44072.
    _install_skill(kanban_home, "devops/root-only-skill")
    out = kc.run_slash(
        "create 'doomed profile task' --assignee coder --skill root-only-skill"
    )
    assert "unknown skill(s)" in out.lower()
    assert "root-only-skill" in out


def test_create_without_skills_skips_validation(kanban_home):
    # No --skill → no validation, even with no skills dir present.
    out = kc.run_slash("create 'plain task' --assignee alice")
    assert "Created" in out

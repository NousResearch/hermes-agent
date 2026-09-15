"""Skill-pin resolution for kanban cards: write-time flag + worker fail-soft.

A card's ``skills`` column is force-loaded into the worker via ``--skills`` and
resolved in the ASSIGNEE's home. A pin that matches nothing on disk used to
abort the run at agent init (``Unknown skill(s): seo, marketing``), so the card
crashed on every dispatch. These tests pin both halves of the fix:

* :func:`unresolved_skill_pins` flags a dead pin when the card is written;
* the worker path drops it with a ``skill_pin_unresolved`` task event instead of
  raising (see ``tests/hermes_cli/test_cli_preloaded_skills.py``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_skills as ks


def _write_skill(root: Path, rel: str, name: str) -> Path:
    skill_dir = root / rel
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Fixture skill.\n---\n\nBody.\n", encoding="utf-8")
    return skill_dir


@pytest.fixture
def skills_root():
    """The active home's skills dir (isolated per test by the hermetic fixture)."""
    from hermes_constants import get_skills_dir

    root = get_skills_dir()
    root.mkdir(parents=True, exist_ok=True)
    ks.clear_skill_catalog_cache()
    yield root
    ks.clear_skill_catalog_cache()


def test_pin_resolution_accepts_name_path_and_frontmatter(skills_root):
    _write_skill(skills_root, "creative/ascii-art", "ASCII Art")
    _write_skill(skills_root, "flat-skill", "Flat Skill")
    ks.clear_skill_catalog_cache()

    assert ks.unresolved_skill_pins(["ascii-art"]) == []          # leaf basename
    assert ks.unresolved_skill_pins(["creative/ascii-art"]) == []  # namespaced path
    assert ks.unresolved_skill_pins(["ASCII Art"]) == []           # frontmatter name
    assert ks.unresolved_skill_pins(["Flat Skill"]) == []          # slugged below
    assert ks.unresolved_skill_pins(["flat-skill"]) == []


def test_category_label_is_flagged_not_silently_accepted(skills_root):
    """`research`/`creative` are directories, not skills — the audit's finding."""
    _write_skill(skills_root, "creative/ascii-art", "ASCII Art")
    ks.clear_skill_catalog_cache()

    assert ks.unresolved_skill_pins(["creative"]) == ["creative"]
    assert ks.unresolved_skill_pins(["research"]) == ["research"]


def test_mixed_pins_report_only_the_dead_names(skills_root):
    _write_skill(skills_root, "creative/ascii-art", "ASCII Art")
    ks.clear_skill_catalog_cache()

    assert ks.unresolved_skill_pins(["ascii-art", "seo", "marketing"]) == ["seo", "marketing"]
    assert ks.unresolved_skill_pins([]) == []
    assert ks.unresolved_skill_pins(None) == []


def test_catalog_sees_a_skill_installed_after_the_first_lookup(skills_root):
    """The per-root mtime cache key must not freeze a stale catalog."""
    ks.clear_skill_catalog_cache()
    assert ks.unresolved_skill_pins(["brand-new-skill"]) == ["brand-new-skill"]

    _write_skill(skills_root, "brand-new-skill", "brand-new-skill")

    assert ks.unresolved_skill_pins(["brand-new-skill"]) == []


def test_create_task_flags_a_dead_pin_on_the_card(tmp_path):
    db_path = tmp_path / "kanban.db"
    kbc.init_db(db_path)
    with kbc.connect(db_path) as conn:
        tid = kb.create_task(
            conn, title="dead pin", assignee="worker", skills=["definitely-not-a-skill"])

    with kbc.connect(db_path) as conn:
        events = [e for e in kb.list_events(conn, tid) if e.kind == ks.SKILL_PIN_UNRESOLVED]

    assert len(events) == 1
    assert events[0].payload["skills"] == ["definitely-not-a-skill"]
    assert events[0].payload["phase"] == "create"


def test_create_task_stays_silent_for_resolvable_pins(tmp_path, skills_root):
    _write_skill(skills_root, "real-skill", "real-skill")
    ks.clear_skill_catalog_cache()

    db_path = tmp_path / "kanban.db"
    kbc.init_db(db_path)
    with kbc.connect(db_path) as conn:
        tid = kb.create_task(conn, title="good pin", assignee="worker", skills=["real-skill"])

    with kbc.connect(db_path) as conn:
        events = [e for e in kb.list_events(conn, tid) if e.kind == ks.SKILL_PIN_UNRESOLVED]

    assert events == []


def _create_args(**overrides) -> argparse.Namespace:
    base = dict(
        title="pinned card", body=None, assignee="worker", created_by=None, workspace=None,
        branch=None, max_runtime=None, max_retries=None, project=None, tenant=None, priority=0,
        parent=[], triage=False, idempotency_key=None, skills=[], model_override=None,
        provider_override=None, goal_mode=False, goal_max_turns=None, completion_contract=None,
        initial_status="running", json=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cli_create_warns_on_a_dead_pin(tmp_path, monkeypatch, capsys):
    """`hermes kanban create --skill <dead>` must warn, not silently accept."""
    from hermes_cli import kanban as kcli

    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    rc = kcli._cmd_create(_create_args(skills=["seo"]))

    assert rc == 0
    captured = capsys.readouterr()
    assert "seo" in captured.err
    assert "resolve to no installed skill" in captured.err

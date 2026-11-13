"""Curator inactivity archival must not orphan a live skill's ``related_skills`` link.

Usage only bumps when a skill is loaded, so a skill that another skill declares as related can
age out while its dependent is still served; archiving it leaves a dangling link in the
dependent. The curator protects such skills the same way it protects cron-referenced ones.
"""
import importlib
from datetime import datetime, timedelta, UTC
from pathlib import Path

import pytest


@pytest.fixture
def curator_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    import tools.skill_usage as usage
    importlib.reload(usage)
    from agent import curator
    importlib.reload(curator)
    monkeypatch.setattr(curator, "_run_llm_review", lambda prompt: "llm-stub")
    monkeypatch.setattr(curator, "_load_config", dict)
    monkeypatch.setattr(curator, "_cron_referenced_skills", set)
    monkeypatch.setattr(usage, "_prune_builtins_enabled", lambda: False)
    return home, usage, curator


def _write_skill(root: Path, name: str, frontmatter_extra: str = "") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: x\n{frontmatter_extra}---\n\n# {name}\n", encoding="utf-8",
    )
    return d


def _age_out(usage, name: str, days: int = 200, *, created_by: str | None = "agent") -> None:
    ts = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    data = usage.load_usage()
    data[name] = usage._empty_record()
    data[name].update(created_by=created_by, created_at=ts, last_used_at=ts, last_activity_at=ts, use_count=1)
    usage.save_usage(data)


_HERMES_META = "metadata:\n  hermes:\n    related_skills: [helper]\n"


@pytest.mark.parametrize("extra", [_HERMES_META, "related_skills: [helper]\n", "related_skills: helper\n"],
                         ids=["metadata-hermes-list", "top-level-list", "top-level-scalar"])
def test_skill_named_in_live_related_skills_survives_inactivity(curator_env, extra):
    home, usage, curator = curator_env
    skills = home / "skills"
    _write_skill(skills, "helper")
    _age_out(usage, "helper")
    _write_skill(skills / "category", "dependent", extra)

    counts = curator.apply_automatic_transitions()

    assert counts["archived"] == 0
    assert usage.load_usage()["helper"]["state"] == usage.STATE_ACTIVE
    assert (skills / "helper" / "SKILL.md").is_file()


def test_bundled_skill_named_in_live_related_skills_survives_under_prune_builtins(curator_env, monkeypatch):
    home, usage, curator = curator_env
    skills = home / "skills"
    _write_skill(skills, "gif-search")
    (skills / ".bundled_manifest").write_text("gif-search:abc\n", encoding="utf-8")
    monkeypatch.setattr(usage, "_prune_builtins_enabled", lambda: True)
    _age_out(usage, "gif-search", created_by=None)
    _write_skill(skills / "media", "spotify", "related_skills: [gif-search]\n")

    counts = curator.apply_automatic_transitions()

    assert counts["archived"] == 0
    assert usage.load_usage()["gif-search"]["state"] == usage.STATE_ACTIVE


def test_reference_from_an_archived_skill_does_not_protect(curator_env):
    home, usage, curator = curator_env
    skills = home / "skills"
    _write_skill(skills, "helper")
    _age_out(usage, "helper")
    _write_skill(skills / ".archive", "old-dependent", "related_skills: [helper]\n")

    counts = curator.apply_automatic_transitions()

    assert counts["archived"] == 1
    assert usage.load_usage()["helper"]["state"] == usage.STATE_ARCHIVED


def test_self_reference_does_not_protect(curator_env):
    home, usage, curator = curator_env
    skills = home / "skills"
    _write_skill(skills, "helper", "related_skills: [helper]\n")
    _age_out(usage, "helper")

    counts = curator.apply_automatic_transitions()

    assert counts["archived"] == 1
    assert usage.load_usage()["helper"]["state"] == usage.STATE_ARCHIVED


def test_reference_from_a_disabled_skill_does_not_protect(curator_env, monkeypatch):
    home, usage, curator = curator_env
    skills = home / "skills"
    _write_skill(skills, "helper")
    _age_out(usage, "helper")
    _write_skill(skills, "switched-off", "related_skills: [helper]\n")
    import agent.skill_utils as su
    monkeypatch.setattr(su, "get_disabled_skill_names", lambda platform=None: {"switched-off"})

    counts = curator.apply_automatic_transitions()

    assert counts["archived"] == 1
    assert usage.load_usage()["helper"]["state"] == usage.STATE_ARCHIVED


@pytest.mark.parametrize("where,text", [
    ("body", "Use the `helper` skill for this step.\n"),
    ("body", 'Load it with skill_view("helper") first.\n'),
    ("reference", "See the `helper` skill.\n"),
], ids=["body-backtick", "body-skill-view", "references-file"])
def test_skill_named_as_a_skill_token_by_a_live_skill_survives_inactivity(curator_env, where, text):
    home, usage, curator = curator_env
    skills = home / "skills"
    _write_skill(skills, "helper")
    _age_out(usage, "helper")
    dependent = _write_skill(skills / "category", "dependent")
    if where == "body":
        with (dependent / "SKILL.md").open("a", encoding="utf-8") as fh:
            fh.write(text)
    else:
        (dependent / "references").mkdir()
        (dependent / "references" / "notes.md").write_text(text, encoding="utf-8")

    counts = curator.apply_automatic_transitions()

    assert counts["archived"] == 0
    assert usage.load_usage()["helper"]["state"] == usage.STATE_ACTIVE


def test_unreferenced_idle_skill_still_archives_beside_referenced_ones(curator_env):
    """The guard must not neuter pruning: only the named dependency is spared."""
    home, usage, curator = curator_env
    skills = home / "skills"
    for name in ("helper", "orphan"):
        _write_skill(skills, name)
        _age_out(usage, name)
    _write_skill(skills, "dependent", "related_skills: [helper]\n")
    with (skills / "dependent" / "SKILL.md").open("a", encoding="utf-8") as fh:
        fh.write("Mentions orphan in prose and `orphan-tool` as a command, but never as a skill.\n")

    counts = curator.apply_automatic_transitions()

    assert counts["archived"] == 1
    states = usage.load_usage()
    assert states["helper"]["state"] == usage.STATE_ACTIVE
    assert states["orphan"]["state"] == usage.STATE_ARCHIVED


def test_pinned_skills_are_untouched_whether_or_not_referenced(curator_env):
    home, usage, curator = curator_env
    skills = home / "skills"
    for name in ("pinned-referenced", "pinned-alone"):
        _write_skill(skills, name)
        _age_out(usage, name)
        usage.set_pinned(name, True)
    _write_skill(skills, "dependent", "related_skills: [pinned-referenced]\n")

    counts = curator.apply_automatic_transitions()

    assert counts["archived"] == 0 and counts["marked_stale"] == 0
    states = usage.load_usage()
    for name in ("pinned-referenced", "pinned-alone"):
        assert states[name]["pinned"] is True
        assert states[name]["state"] == usage.STATE_ACTIVE

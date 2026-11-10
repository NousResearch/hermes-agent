"""The budgeted skills index: over budget, the least-used stale skills go names-only and stay loadable."""
import json
import os
import time
from datetime import UTC, datetime, timedelta

from agent import prompt_builder as pb
from agent.skills_index_delta import index_entries

TOOLS = {"skills_list", "skill_view", "skill_manage"}


def _iso(days_ago):
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()


def _library(home, skills):
    """*skills*: name -> (use_count, last_used days ago | None, created days ago | None for "no usage record")."""
    usage = {}
    for name, (uses, used_ago, created_ago) in skills.items():
        path = home / "skills" / "devops" / name / "SKILL.md"
        path.parent.mkdir(parents=True)
        path.write_text(f"---\nname: {name}\ndescription: What {name} is for, spelled out at some length.\n---\nBody.\n",
                        encoding="utf-8")
        if created_ago is not None:
            usage[name] = {"use_count": uses, "last_used_at": _iso(used_ago) if used_ago is not None else None,
                           "created_at": _iso(created_ago), "last_patched_at": None}
    (home / "skills" / ".usage.json").write_text(json.dumps(usage), encoding="utf-8")
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)


def _bare(index):
    return {name for name, row in index_entries(index).items() if ": " not in row}


def test_over_budget_scrubs_only_stale_least_used_skills_and_keeps_every_name(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _library(tmp_path, {
        "stale-never-used": (0, None, 90), "stale-used-once": (1, 60, 90), "stale-used-often": (40, 30, 90),
        "used-yesterday": (1, 1, 90), "created-yesterday": (0, None, 1), "old-favourite": (99, 20, 300),
    })
    old = time.time() - 90 * 86400
    os.utime(tmp_path / "skills" / "devops" / "created-yesterday" / "SKILL.md", (old, old))  # the record decides

    full = pb.build_skills_system_prompt(available_tools=TOOLS)
    assert pb.build_skills_system_prompt(available_tools=TOOLS, index_budget_tokens=100_000) == full
    assert not _bare(full)

    tiered = pb.build_skills_system_prompt(available_tools=TOOLS, index_budget_tokens=1)
    assert set(index_entries(tiered)) == set(index_entries(full))
    # Half of six, fewest uses first; the recently used and the recently created are never candidates.
    assert _bare(tiered) == {"stale-never-used", "stale-used-once", "stale-used-often"}
    # A budget that one description covers takes only the lowest-ranked skill: the set is a prefix of one order.
    from agent.model_metadata import estimate_tokens_rough
    assert _bare(pb.build_skills_system_prompt(
        available_tools=TOOLS, index_budget_tokens=estimate_tokens_rough(full) - 1)) == {"stale-never-used"}


def test_newly_installed_skill_without_a_usage_record_keeps_its_description(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _library(tmp_path, {"installed-today": (0, None, None), "installed-last-year": (0, None, None),
                        "stale-a": (3, 40, 90), "stale-b": (5, 40, 90)})
    old = time.time() - 365 * 86400
    os.utime(tmp_path / "skills" / "devops" / "installed-last-year" / "SKILL.md", (old, old))

    bare = _bare(pb.build_skills_system_prompt(available_tools=TOOLS, index_budget_tokens=1))

    assert "installed-today" not in bare
    assert bare == {"installed-last-year", "stale-a"}

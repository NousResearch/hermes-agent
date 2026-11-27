from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "skill_usage_brief.py"

spec = importlib.util.spec_from_file_location("skill_usage_brief", SCRIPT_PATH)
assert spec is not None
assert spec.loader is not None
skill_usage_brief = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = skill_usage_brief
spec.loader.exec_module(skill_usage_brief)


def write_skill(root: Path, relative: str, *, name: str, description: str = "Useful skill", created_by: str | None = None) -> None:
    frontmatter = ["---", f"name: {name}", f"description: {description}"]
    if created_by is not None:
        frontmatter.append(f"created_by: {created_by}")
    frontmatter.extend(["---", "", "# Body", "", "Do work."])
    path = root / relative / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(frontmatter), encoding="utf-8")


def write_usage(root: Path, data: dict) -> Path:
    path = root / ".usage.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_classifies_stale_unused_and_missing_usage(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    write_skill(skills, "productivity/stale", name="stale")
    write_skill(skills, "productivity/unused", name="unused")
    write_skill(skills, "productivity/missing", name="missing")
    usage_file = write_usage(
        skills,
        {
            "stale": {
                "state": "active",
                "pinned": False,
                "use_count": 3,
                "view_count": 4,
                "last_used_at": "2026-05-01T00:00:00+00:00",
            },
            "unused": {
                "state": "active",
                "pinned": False,
                "use_count": 0,
                "view_count": 0,
                "last_used_at": None,
                "created_at": "2026-05-15T00:00:00+00:00",
            },
        },
    )

    brief = skill_usage_brief.build_brief(
        skills_dir=skills,
        usage_file=usage_file,
        now=skill_usage_brief.parse_datetime("2026-07-01T00:00:00+00:00"),
        stale_days=30,
        unused_grace_days=14,
        include_missing_usage=True,
    )

    reasons = {item.name: item.reason for item in brief.items}
    assert reasons == {
        "stale": "stale",
        "unused": "never_used",
        "missing": "missing_usage",
    }


def test_suppresses_missing_usage_by_default(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    write_skill(skills, "missing", name="missing")
    usage_file = write_usage(skills, {})

    brief = skill_usage_brief.build_brief(
        skills_dir=skills,
        usage_file=usage_file,
        now=skill_usage_brief.parse_datetime("2026-07-01T00:00:00+00:00"),
    )

    assert brief.items == []


def test_suppresses_pinned_and_archived_skills(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    write_skill(skills, "pinned", name="pinned")
    write_skill(skills, "archived", name="archived")
    usage_file = write_usage(
        skills,
        {
            "pinned": {"state": "active", "pinned": True, "use_count": 0, "last_used_at": None},
            "archived": {"state": "archived", "pinned": False, "use_count": 0, "last_used_at": None},
        },
    )

    brief = skill_usage_brief.build_brief(
        skills_dir=skills,
        usage_file=usage_file,
        now=skill_usage_brief.parse_datetime("2026-07-01T00:00:00+00:00"),
    )

    assert brief.items == []


def test_json_output_uses_stable_relative_paths(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    write_skill(skills, "category/old", name="old", description="Old skill")
    usage_file = write_usage(
        skills,
        {
            "old": {
                "state": "active",
                "pinned": False,
                "use_count": 2,
                "last_used_at": "2026-05-01T00:00:00+00:00",
            }
        },
    )

    brief = skill_usage_brief.build_brief(
        skills_dir=skills,
        usage_file=usage_file,
        now=skill_usage_brief.parse_datetime("2026-07-01T00:00:00+00:00"),
    )
    payload = json.loads(skill_usage_brief.render_json(brief))

    assert payload["summary"] == {"total_items": 1, "stale": 1, "never_used": 0, "missing_usage": 0}
    assert payload["items"][0]["path"] == "category/old/SKILL.md"
    assert payload["items"][0]["description"] == "Old skill"


def test_markdown_mentions_actions_without_skill_body(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    write_skill(skills, "old", name="old", description="Old skill")
    usage_file = write_usage(
        skills,
        {
            "old": {
                "state": "active",
                "pinned": False,
                "use_count": 1,
                "last_used_at": "2026-05-01T00:00:00+00:00",
            }
        },
    )

    brief = skill_usage_brief.build_brief(
        skills_dir=skills,
        usage_file=usage_file,
        now=skill_usage_brief.parse_datetime("2026-07-01T00:00:00+00:00"),
    )
    markdown = skill_usage_brief.render_markdown(brief)

    assert "Skill usage brief" in markdown
    assert "old" in markdown
    assert "hermes curator" in markdown
    assert "Do work." not in markdown


def test_silent_if_empty_is_exact_marker(tmp_path: Path, capsys) -> None:
    skills = tmp_path / "skills"
    write_skill(skills, "fresh", name="fresh")
    usage_file = write_usage(
        skills,
        {
            "fresh": {
                "state": "active",
                "pinned": False,
                "use_count": 1,
                "last_used_at": "2026-06-30T00:00:00+00:00",
            }
        },
    )

    exit_code = skill_usage_brief.main(
        [
            "--skills-dir",
            str(skills),
            "--usage-file",
            str(usage_file),
            "--now",
            "2026-07-01T00:00:00+00:00",
            "--silent-if-empty",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == "[SILENT]\n"

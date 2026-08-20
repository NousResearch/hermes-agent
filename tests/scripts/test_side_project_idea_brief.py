import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "side_project_idea_brief.py"


def load_module():
    spec = importlib.util.spec_from_file_location("side_project_idea_brief", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_ideas_accepts_json_object_and_filters_done_archived(tmp_path):
    mod = load_module()
    backlog = tmp_path / "ideas.json"
    backlog.write_text(
        json.dumps(
            {
                "ideas": [
                    {
                        "title": "AI skill radar",
                        "lane": "information aggregation",
                        "problem": "Useful AI tools are scattered across GitHub and X.",
                        "action": "Collect 10 repos and score what Joe should install.",
                        "impact": 5,
                        "effort": 2,
                        "confidence": 4,
                        "source": "manual backlog",
                    },
                    {"title": "Old task", "done": True},
                    {"title": "Archived task", "status": "archived"},
                ]
            }
        ),
        encoding="utf-8",
    )

    ideas = mod.load_ideas(backlog)

    assert [idea.title for idea in ideas] == ["AI skill radar"]
    assert ideas[0].impact == 5
    assert ideas[0].effort == 2
    assert ideas[0].source == "manual backlog"


def test_rank_ideas_prefers_high_impact_low_effort_relevant_lanes():
    mod = load_module()
    ideas = [
        mod.Idea(
            title="Generic photo album cleanup",
            lane="life ops",
            problem="Photos are messy.",
            impact=3,
            effort=4,
            confidence=3,
        ),
        mod.Idea(
            title="AI dev community radar",
            lane="information aggregation",
            problem="Joe needs useful AI/dev tooling signals without doomscrolling.",
            impact=5,
            effort=2,
            confidence=4,
            source="GitHub trending seed",
        ),
    ]

    ranked = mod.rank_ideas(ideas, today="2026-05-29")

    assert ranked[0].title == "AI dev community radar"
    assert ranked[0].score > ranked[1].score
    assert "information aggregation" in ranked[0].reasons
    assert "source/evidence present" in ranked[0].reasons


def test_render_brief_uses_joe_style_traditional_chinese_sections():
    mod = load_module()
    ranked = mod.rank_ideas(
        [
            mod.Idea(
                title="Personal reminder sweeper",
                lane="personal reminders",
                problem="Household recurring tasks are easy to forget.",
                action="Create a YAML reminder seed file and review monthly.",
                impact=4,
                effort=2,
                confidence=4,
                source="Joe operating manual",
                notes="Keep local-only; do not send reminders automatically.",
            )
        ],
        today="2026-05-29",
    )

    rendered = mod.render_brief(ranked, today="2026-05-29", top=1)

    assert rendered.startswith("## TL;DR")
    assert "Fact / verified" in rendered
    assert "Hypothesis" in rendered
    assert "Action for Joe" in rendered
    assert "Source: Joe operating manual" in rendered
    assert "Personal reminder sweeper" in rendered
    assert "本地優先" in rendered


def test_render_brief_returns_exact_silent_when_empty_and_requested():
    mod = load_module()

    assert mod.render_brief([], today="2026-05-29", silent_if_empty=True) == "[SILENT]"


def test_cli_outputs_brief_for_json_input(tmp_path):
    backlog = tmp_path / "ideas.json"
    backlog.write_text(
        json.dumps(
            [
                {
                    "title": "Investment thesis checklist",
                    "lane": "investment framework",
                    "problem": "Thesis notes lack a repeatable risk/review cadence.",
                    "action": "Draft a one-page checklist for new positions.",
                    "impact": 5,
                    "effort": 2,
                    "confidence": 4,
                    "source": "manual seed",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(backlog), "--today", "2026-05-29", "--top", "1"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Investment thesis checklist" in result.stdout
    assert "## TL;DR" in result.stdout
    assert "Action for Joe" in result.stdout

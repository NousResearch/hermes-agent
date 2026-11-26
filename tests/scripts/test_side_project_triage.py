import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "side_project_triage.py"

spec = importlib.util.spec_from_file_location("side_project_triage", SCRIPT_PATH)
assert spec is not None
assert spec.loader is not None
side_project_triage = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = side_project_triage
spec.loader.exec_module(side_project_triage)


def test_parse_ideas_extracts_metadata_priority_and_urls(tmp_path):
    ideas = tmp_path / "ideas.md"
    ideas.write_text(
        """
# Side Projects

## Life ops
- [high] Fridge cadence helper — effort: low; leverage: medium; reuse: Apple Reminders; next: draft recurring reminder
- AI tools radar | effort: medium | leverage: high | audience: Joe | https://github.com/example/radar
""".strip()
        + "\n",
        encoding="utf-8",
    )

    entries = side_project_triage.parse_ideas(ideas)

    assert [entry.title for entry in entries] == ["Fridge cadence helper", "AI tools radar"]
    assert entries[0].section == "Life ops"
    assert entries[0].priority == "high"
    assert entries[0].effort == "low"
    assert entries[0].leverage == "medium"
    assert entries[0].reuse == "Apple Reminders"
    assert entries[0].next_action == "draft recurring reminder"
    assert entries[1].urls == ["https://github.com/example/radar"]


def test_ranked_brief_favors_low_effort_high_leverage_reuse(tmp_path):
    ideas = tmp_path / "ideas.md"
    ideas.write_text(
        """
# Ideas
- Giant app rewrite — effort: high; leverage: medium; next: write RFC
- [medium] Existing-tool dashboard — effort: low; leverage: high; reuse: Notion; next: build tiny template
- Blocked experiment — effort: low; leverage: high; status: blocked; next: wait for API access
""".strip()
        + "\n",
        encoding="utf-8",
    )

    entries = side_project_triage.parse_ideas(ideas)
    ranked = side_project_triage.rank_ideas(entries)
    brief = side_project_triage.build_markdown_brief(entries)

    assert [entry.title for entry in ranked] == ["Existing-tool dashboard", "Blocked experiment", "Giant app rewrite"]
    assert "## Top picks" in brief
    assert brief.index("Existing-tool dashboard") < brief.index("Giant app rewrite")
    assert "## Parking lot" in brief
    assert "Blocked experiment" in brief


def test_parse_ideas_skips_guidance_sections(tmp_path):
    ideas = tmp_path / "ideas.md"
    ideas.write_text(
        """
# Ideas

## Product candidates
- Real candidate — effort: low; leverage: high

## Output template
- Title — effort: low; leverage: high; next: do something

## Safety notes
- Never send messages automatically

## Scoring contract
- This instruction should not become an idea
""".strip()
        + "\n",
        encoding="utf-8",
    )

    entries = side_project_triage.parse_ideas(ideas)

    assert [entry.title for entry in entries] == ["Real candidate"]


def test_brief_reports_missing_next_actions(tmp_path):
    ideas = tmp_path / "ideas.md"
    ideas.write_text(
        """
# Ideas
- Ready idea — effort: low; leverage: high; next: ship a script
- Vague idea — effort: low; leverage: high
""".strip()
        + "\n",
        encoding="utf-8",
    )

    brief = side_project_triage.build_markdown_brief(side_project_triage.parse_ideas(ideas))

    assert "## Missing next actions" in brief
    assert "Vague idea" in brief
    assert "Ready idea" not in brief.split("## Missing next actions", maxsplit=1)[1]


def test_cli_outputs_json(tmp_path):
    ideas = tmp_path / "ideas.md"
    ideas.write_text("- Quick win — effort: low; leverage: high; next: test it\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(ideas), "--format", "json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["count"] == 1
    assert payload["ideas"][0]["title"] == "Quick win"
    assert payload["ideas"][0]["score"] > 0

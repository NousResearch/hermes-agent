"""Contracts for focused workflow packages shipped with starter bots."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BOT_ROOT = REPO_ROOT / "optional-skills" / "bots"
BOTS = {
    "research-analyst": "research-brief.md",
    "inbox-triage": "triage-tracker.csv",
    "competitor-watch": "watchlist.csv",
    "sales-call-coach": "scorecard.md",
    "design-critic": "critique.md",
    "recruiting-coordinator": "interview-loop-tracker.csv",
    "credit-card-optimizer": "card-inventory.csv",
    "meeting-recap-deck": "recap-outline.md",
}
REQUIRED_SECTIONS = (
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
)


def _parse_skill(bot: str) -> tuple[dict, str, str]:
    text = (BOT_ROOT / bot / "SKILL.md").read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    assert match, f"{bot}: missing YAML frontmatter"
    return yaml.safe_load(match.group(1)), match.group(2), text


@pytest.mark.parametrize("bot", BOTS)
def test_skill_frontmatter_and_sections(bot: str):
    fm, body, text = _parse_skill(bot)
    assert fm["name"] == bot
    assert fm["version"] == "0.1.0"
    assert fm["author"] == "Mark (unsupportedpastels), Hermes Agent"
    assert fm["license"] == "MIT"
    assert fm["platforms"] == ["linux", "macos", "windows"]
    assert len(fm["description"]) <= 60
    assert fm["description"].endswith(".")
    assert fm["metadata"]["hermes"]["category"] == "bots"
    assert fm["metadata"]["hermes"]["tags"]
    assert "related_skills" in fm["metadata"]["hermes"]
    for section in REQUIRED_SECTIONS:
        assert section in body, f"{bot}: missing {section}"
    assert "/home/" not in text


@pytest.mark.parametrize("bot,template", BOTS.items())
def test_sample_rubric_and_template_assets_are_referenced(bot: str, template: str):
    _, _, text = _parse_skill(bot)
    skill_dir = BOT_ROOT / bot
    required = (
        "references/sample-input.md",
        "references/expected-output-rubric.md",
        f"templates/{template}",
    )
    for relative in required:
        path = skill_dir / relative
        assert path.is_file(), f"{bot}: missing {relative}"
        assert relative in text, f"{bot}: SKILL.md does not reference {relative}"
    sample = (skill_dir / "references" / "sample-input.md").read_text(encoding="utf-8")
    assert "SYNTHETIC FIXTURE" in sample
    assert "not real" in sample.lower()
    rubric = (skill_dir / "references" / "expected-output-rubric.md").read_text(
        encoding="utf-8"
    )
    assert "## Pass criteria" in rubric
    assert "expected real result" not in rubric.lower()
    template_text = (skill_dir / "templates" / template).read_text(encoding="utf-8")
    assert "{{" in template_text, f"{bot}: template must contain blank placeholders"


@pytest.mark.parametrize("bot", BOTS)
def test_procedure_steps_have_completion_criteria(bot: str):
    _, body, _ = _parse_skill(bot)
    steps = re.findall(
        r"^### \d+\..*?(?=^### \d+\.|^## )", body, re.MULTILINE | re.DOTALL
    )
    assert len(steps) >= 4, f"{bot}: workflow needs at least four checkable steps"
    for step in steps:
        assert "Done when" in step


@pytest.mark.parametrize("bot", BOTS)
def test_related_skills_resolve(bot: str):
    fm, _, _ = _parse_skill(bot)
    for skill in fm["metadata"]["hermes"]["related_skills"]:
        hits = list(REPO_ROOT.glob(f"skills/**/{skill}/SKILL.md")) + list(
            REPO_ROOT.glob(f"optional-skills/**/{skill}/SKILL.md")
        )
        assert hits, f"{bot}: unresolved related skill {skill}"


def test_workflow_specific_safety_and_honesty_contracts():
    bodies = {bot: _parse_skill(bot)[1].lower() for bot in BOTS}
    assert "fixture-only" in bodies["research-analyst"]
    assert "web_search" in bodies["research-analyst"]
    assert "fixture-only" in bodies["competitor-watch"]
    assert "web_extract" in bodies["competitor-watch"]
    assert "untrusted data" in bodies["inbox-triage"]
    assert "do not send" in bodies["inbox-triage"]
    assert "transcript" in bodies["sales-call-coach"]
    assert "do not promise transcription" in bodies["sales-call-coach"]
    assert "vision_analyze" in bodies["design-critic"]
    assert "text-described fallback" in bodies["design-critic"]
    assert "do not rank candidates" in bodies["recruiting-coordinator"]
    assert "do not book or send" in bodies["recruiting-coordinator"]
    assert "nickname" in bodies["credit-card-optimizer"]
    assert "never request" in bodies["credit-card-optimizer"]
    assert "current official issuer" in bodies["credit-card-optimizer"]
    assert "plain recap" in bodies["meeting-recap-deck"]
    assert "powerpoint" in bodies["meeting-recap-deck"]
    for body in bodies.values():
        assert "explicit activation" in body
        assert "automatically create" in body


def test_credit_card_helper_calculates_and_preserves_input(tmp_path: Path):
    script = BOT_ROOT / "credit-card-optimizer" / "scripts" / "compare_cards.py"
    source = tmp_path / "cards.csv"
    source.write_text(
        "nickname,reward_rate_percent,cap_remaining,credit_remaining\n"
        "Everyday,2,1000,0\n"
        "Travel,3,20,5\n",
        encoding="utf-8",
    )
    before = source.read_bytes()
    result = subprocess.run(
        [sys.executable, str(script), "--cards", str(source), "--amount", "100"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = json.loads(result.stdout)
    assert [row["nickname"] for row in rows] == ["Travel", "Everyday"]
    assert rows[0]["eligible_spend"] == "20.00"
    assert rows[0]["estimated_value"] == "5.60"
    assert rows[1]["estimated_value"] == "2.00"
    assert source.read_bytes() == before


def test_csv_templates_are_parseable():
    for bot, template in BOTS.items():
        if not template.endswith(".csv"):
            continue
        path = BOT_ROOT / bot / "templates" / template
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert rows and len(rows[0]) >= 3

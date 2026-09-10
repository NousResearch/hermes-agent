"""Tests for the offline side-project idea triage helper."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "side_project_idea_triage.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("side_project_idea_triage", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


triage = _load_module()


def test_parse_ideas_ignores_instructions_and_template_sections():
    markdown = """
# Brain dump

Use this file for side project ideas. The bullets below are guidance, not data.
- Impact: 5
- First step: Do not parse this

## Template
- Area: copy this
- Impact: 5
- Effort: 1
- First step: Fill this later

## Investor signal digest
- Area: information aggregation
- Impact: 5
- Effort: 2
- Confidence: 4
- Urgency: 5
- First step: Parse watchlist links and produce a daily brief
- Notes: Compounds BD top-of-funnel quality

### Fridge chore reminder
- Area: personal systems
- Impact: 3
- Effort: 1
- Confidence: 5
- First step: Emit due/overdue household task summary
"""

    ideas = triage.parse_ideas(markdown)

    assert [idea.title for idea in ideas] == [
        "Investor signal digest",
        "Fridge chore reminder",
    ]
    assert ideas[0].area == "information aggregation"
    assert ideas[0].impact == 5
    assert ideas[0].first_step == "Parse watchlist links and produce a daily brief"


def test_ranking_rewards_leverage_and_penalizes_blockers_and_vagueness():
    markdown = """
## Maybe build a huge super app someday
- Area: side projects
- Impact: 5
- Effort: 5
- Confidence: 2
- First step:
- Notes: maybe research more someday

## Investor signal digest
- Area: information aggregation
- Impact: 5
- Effort: 2
- Confidence: 4
- Urgency: 5
- First step: Parse watchlist links and produce a daily brief

## Bank integration autoposter
- Area: finance
- Impact: 5
- Effort: 1
- Confidence: 5
- First step: Wire credentials and send messages
- Blocked: true
"""

    ranked = triage.rank_ideas(triage.parse_ideas(markdown))

    assert ranked[0].title == "Investor signal digest"
    assert ranked[-1].title == "Maybe build a huge super app someday"
    blocked = next(idea for idea in ranked if idea.title == "Bank integration autoposter")
    assert blocked.blocked is True
    assert "blocked" in blocked.score_explanation.lower()


def test_markdown_output_contains_ranked_table_and_defer_section():
    ideas = triage.rank_ideas(
        triage.parse_ideas(
            """
## Quick food logger
- Area: health
- Impact: 4
- Effort: 2
- Confidence: 4
- First step: Convert meal notes to weekly macro deltas

## OAuth bank sync
- Area: finance
- Impact: 5
- Effort: 1
- Confidence: 4
- First step: Connect live account
- Blocked: true
"""
        )
    )

    output = triage.render_markdown(ideas)

    assert output.startswith("# Side Project Idea Triage")
    assert "| 1 | Quick food logger | health |" in output
    assert "## Top recommendation" in output
    assert "## Watch / defer" in output
    assert "OAuth bank sync" in output


def test_cli_json_output(tmp_path, capsys):
    sample = tmp_path / "ideas.md"
    sample.write_text(
        """
## Skill watcher
- Area: dev tools
- Impact: 4
- Effort: 2
- Confidence: 4
- First step: Scan local skill catalog and flag stale entries
""".strip(),
        encoding="utf-8",
    )

    exit_code = triage.main([str(sample), "--format", "json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ideas"][0]["title"] == "Skill watcher"
    assert payload["ideas"][0]["score"] > 0


def test_cli_no_ideas_exits_nonzero(tmp_path, capsys):
    sample = tmp_path / "ideas.md"
    sample.write_text("# Notes only\n- Impact: 5\n", encoding="utf-8")

    exit_code = triage.main([str(sample)])

    assert exit_code == 2
    assert "No side-project ideas found" in capsys.readouterr().err

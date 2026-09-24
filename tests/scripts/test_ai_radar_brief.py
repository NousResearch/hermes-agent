"""Tests for the Joe-style AI radar briefing helper."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_ai_radar_brief_under_test",
        Path(__file__).resolve().parents[2] / "scripts" / "ai_radar_brief.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_rss_xml_extracts_core_fields():
    module = _load_module()
    xml = """<?xml version="1.0"?>
    <rss><channel><title>Builder Feed</title>
      <item>
        <title>New agentic coding workflow for GitHub PR reviews</title>
        <link>https://example.com/agent-pr</link>
        <pubDate>Tue, 26 May 2026 01:02:03 GMT</pubDate>
        <description>Teams use AI agents to review pull requests.</description>
      </item>
    </channel></rss>
    """

    items = module.parse_rss_xml(xml, source="Builder Feed")

    assert len(items) == 1
    assert items[0].title == "New agentic coding workflow for GitHub PR reviews"
    assert items[0].url == "https://example.com/agent-pr"
    assert items[0].source == "Builder Feed"
    assert "2026" in items[0].published
    assert "AI agents" in items[0].summary


def test_rank_items_prioritizes_joe_relevant_themes():
    module = _load_module()
    relevant = module.RadarItem(
        title="Neobank raises Series A to launch stablecoin payment rails",
        url="https://example.com/neobank-stablecoin",
        source="Funding Feed",
        summary="A fintech team is building payment and yield products for merchants.",
    )
    unrelated = module.RadarItem(
        title="New mechanical keyboard colorway launches",
        url="https://example.com/keyboard",
        source="Gadget Feed",
        summary="A keyboard vendor announced a limited run.",
    )

    ranked = module.rank_items([unrelated, relevant])

    assert ranked[0].url == "https://example.com/neobank-stablecoin"
    assert ranked[0].score > ranked[1].score
    assert {"payments", "fintech", "funding"} <= set(ranked[0].matched_themes)


def test_render_brief_uses_joe_morning_report_shape():
    module = _load_module()
    item = module.RadarItem(
        title="Open-source AI agent adds workflow memory",
        url="https://example.com/agent-memory",
        source="GitHub Trending",
        summary="The project stores reusable skills and improves coding workflows.",
        published="2026-05-26",
    )

    brief = module.render_brief([item], limit=1)

    assert brief.startswith("## TL;DR")
    assert "### 1. Open-source AI agent adds workflow memory" in brief
    assert "**Fact / verified:**" in brief
    assert "**Hypothesis:**" in brief
    assert "**Action for Joe:**" in brief
    assert "https://example.com/agent-memory" in brief
    assert "GitHub Trending" in brief


def test_render_brief_can_suppress_empty_output():
    module = _load_module()

    assert module.render_brief([], silent_if_empty=True) == "[SILENT]"
    assert "沒有新的 radar item" in module.render_brief([], silent_if_empty=False)


def test_load_items_from_json_and_yaml_files(tmp_path):
    module = _load_module()
    json_path = tmp_path / "items.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "title": "AI dev tool improves issue triage",
                    "url": "https://example.com/triage",
                    "source": "Example",
                    "summary": "Uses agents to score issues and PRs.",
                    "published": "2026-05-26",
                }
            ]
        ),
        encoding="utf-8",
    )
    yaml_path = tmp_path / "items.yaml"
    yaml_path.write_text(
        "- title: Stablecoin checkout startup raises seed\n"
        "  url: https://example.com/checkout\n"
        "  source: Funding Feed\n"
        "  summary: Payments and merchant acquiring workflow.\n",
        encoding="utf-8",
    )

    json_items = module.load_items_file(json_path)
    yaml_items = module.load_items_file(yaml_path)

    assert json_items[0].title == "AI dev tool improves issue triage"
    assert yaml_items[0].url == "https://example.com/checkout"

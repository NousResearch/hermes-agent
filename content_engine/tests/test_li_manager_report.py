"""Contract tests for the LinkedIn manager HTML report renderer."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ce_dir = Path(__file__).resolve().parent.parent
if str(ce_dir) not in sys.path:
    sys.path.insert(0, str(ce_dir))

import li_manager as li
import li_manager_report as report


def _insight() -> li.LiArtifact:
    return li.LiArtifact(
        id="li_pm_insig_test",
        lane=li.LANE_DAILY_PACKAGE,
        kind=li.KIND_PM_INSIGHT,
        body="A short, opinion-led PM Insight summary.",
        source_url="https://sahil.blog/post/context-vs-model",
        pack=li.ArgumentPack(
            claim="A claim",
            evidence="A source",
            mechanism="A mechanism",
            position="A position",
            context={"source_text": "The original full-post excerpt.", "author": "sahil"},
        ),
    )


def _post() -> li.LiArtifact:
    return li.LiArtifact(
        id="li_ai_pm_pos_test",
        lane=li.LANE_DAILY_PACKAGE,
        kind=li.KIND_AI_PM_POST,
        body="An independent, evidence-backed AI/PM post.",
        pack=li.ArgumentPack(
            claim="A claim",
            evidence="A source",
            mechanism="A mechanism",
            position="A position",
            context={"source_text": "The evidence source.", "source_url": "https://example.com/src"},
        ),
    )


def test_report_contains_original_and_recommendation(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "REPORT_DIR", tmp_path)
    path = report.render_report(
        [_insight(), _post()], lane="daily", title="Daily LinkedIn package"
    )
    text = path.read_text()
    assert "The original full-post excerpt." in text
    assert "The evidence source." in text
    assert "A short, opinion-led PM Insight summary." in text
    assert "An independent, evidence-backed AI/PM post." in text
    assert "Open original" in text
    assert "Why this angle" in text
    assert text.count('<article class="card"') == 2


def test_report_shows_pm_insight_direct_link(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "REPORT_DIR", tmp_path)
    path = report.render_report(
        [_insight()], lane="daily", title="Daily LinkedIn package"
    )
    text = path.read_text()
    assert "https://sahil.blog/post/context-vs-model" in text
    assert "PM Insight summary" in text


def test_report_escapes_source_content(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "REPORT_DIR", tmp_path)
    artifact = _insight()
    artifact.pack.context["source_text"] = '<script>alert("x")</script>'
    text = report.render_report(
        [artifact], lane="daily", title="Review"
    ).read_text()
    assert "<script>" not in text
    assert "&lt;script&gt;" in text

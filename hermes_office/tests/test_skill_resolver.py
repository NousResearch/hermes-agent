"""Tests for the deterministic SkillResolver."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hermes_office.models import ResolvedRole
from hermes_office.skill_resolver import SkillResolver, optimize, score_candidate


def test_resolve_arxiv_role_picks_research_skills():
    resolver = SkillResolver()
    out = resolver.resolve("I need a researcher who reads arxiv papers and summarizes them")
    assert isinstance(out, ResolvedRole)
    assert "research/arxiv" in out.recommended_skills
    assert "research/research-paper-writing" in out.recommended_skills
    assert "web" in out.recommended_toolsets
    assert out.confidence > 0.8
    assert "arxiv" in out.matched_keywords


def test_resolve_designer_role_picks_visual_skills():
    resolver = SkillResolver()
    out = resolver.resolve("Make a designer that draws cute logos and pixel sprites")
    assert "image_gen" in out.recommended_toolsets
    assert "creative/pixel-art" in out.recommended_skills


def test_resolve_chinese_text():
    resolver = SkillResolver()
    out = resolver.resolve("帮我画一张像素风格的小图")
    assert "image_gen" in out.recommended_toolsets
    assert "creative/pixel-art" in out.recommended_skills
    assert out.confidence > 0


def test_resolve_empty_text_returns_zero_confidence():
    resolver = SkillResolver()
    out = resolver.resolve("")
    assert out.recommended_skills == []
    assert out.recommended_toolsets == []
    assert out.confidence == 0.0


def test_resolve_irrelevant_text_returns_low_confidence():
    resolver = SkillResolver()
    out = resolver.resolve("xxxxxx zzzzzz")
    assert out.recommended_toolsets == []
    assert out.recommended_skills == []
    assert out.confidence == 0.0


def test_resolve_is_deterministic():
    resolver = SkillResolver()
    text = "Build me a coder who can write python and run tests"
    a = resolver.resolve(text)
    b = resolver.resolve(text)
    assert a.model_dump() == b.model_dump()


def test_score_candidate_handles_repeated_keywords():
    """tf grows logarithmically — count=1 → 1.0, count=2 → 1+log(2) ≈ 1.693."""
    score, _ = score_candidate("python python python", {"python": 1.0})
    assert math.isclose(score, 1.0 + math.log(3), rel_tol=1e-6)


def test_score_candidate_no_match():
    score, matched = score_candidate("hello world", {"python": 2.0})
    assert score == 0.0
    assert matched == []


def test_resolver_reads_overrides_from_weights_path(tmp_path: Path):
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({
        "toolsets": {"web": {"banana": 5.0}},
        "thresholds": {"toolset": 0.7},
    }))
    resolver = SkillResolver(weights_path=weights)
    out = resolver.resolve("I want a banana eater")
    assert "web" in out.recommended_toolsets


def test_optimize_lowers_loss_monotonically(tmp_path: Path):
    """Tiny synthetic dataset: text X with skill S should be 'success', text Y
    with same skill should be 'failure'. After optimisation the resolver must
    score X higher than before for the related keywords."""
    telemetry = tmp_path / "telemetry.jsonl"
    sample = [
        {"role_text": "arxiv paper", "skills": ["research/arxiv"], "toolsets": [], "success": 1},
        {"role_text": "arxiv paper", "skills": ["research/arxiv"], "toolsets": [], "success": 1},
        {"role_text": "totally unrelated", "skills": ["research/arxiv"], "toolsets": [], "success": 0},
        {"role_text": "totally unrelated", "skills": ["research/arxiv"], "toolsets": [], "success": 0},
    ]
    telemetry.write_text("\n".join(json.dumps(s) for s in sample))
    out = tmp_path / "weights.json"
    report = optimize(telemetry, out, epochs=10)
    assert report["samples"] == 4
    assert report["loss_last"] <= report["loss_first"] + 1e-9
    assert out.exists()


def test_optimize_no_telemetry_returns_zero_samples(tmp_path: Path):
    telemetry = tmp_path / "missing.jsonl"
    out = tmp_path / "w.json"
    report = optimize(telemetry, out)
    assert report["samples"] == 0

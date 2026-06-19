"""Tests for the expression heuristic and viseme estimator ports."""

from __future__ import annotations

from plugins.teams_voice import expression as expr
from plugins.teams_voice import viseme_estimate as viz


def test_expression_priority_and_defaults():
    assert expr.infer_emotion("") == expr.NEUTRAL
    assert expr.infer_emotion("just a plain sentence") == expr.NEUTRAL
    assert expr.infer_emotion("That's perfect!!") == expr.SURPRISED  # !! beats happy word
    assert expr.infer_emotion("That is great and wonderful") == expr.HAPPY
    assert expr.infer_emotion("Unfortunately the deal failed") == expr.SAD
    assert expr.infer_emotion("wow that is something") == expr.SURPRISED


def test_estimate_visemes_even_spread_and_collapse():
    marks = viz.estimate_visemes("aa", 100)
    # two 'a' chars collapse to a single mark (same viseme id)
    assert len(marks) == 1
    assert marks[0].viseme_id == viz.VISEME_AA
    assert marks[0].t_ms == 0


def test_estimate_visemes_empty_inputs():
    assert viz.estimate_visemes("", 100) == []
    assert viz.estimate_visemes("hello", 0) == []


def test_visemes_from_alignment_sorted_payload():
    marks = viz.visemes_from_alignment([("m", 30), ("a", 0), ("p", 60)])
    payload = viz.marks_to_payload(marks)
    # sorted by time; first is the 'a'->open at t=0
    assert payload[0]["tMs"] == 0
    assert all("tMs" in m and "visemeId" in m for m in payload)

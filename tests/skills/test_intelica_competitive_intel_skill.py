"""Tests for the intelica-competitive-intel optional skill.

Hermetic: stdlib + pytest only, no live network calls, no httpx. Exercises
the shipped scripts/validate_analysis.py helper directly.
"""
import importlib.util
import pathlib

import pytest

_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "optional-skills" / "research" / "intelica-competitive-intel"
    / "scripts" / "validate_analysis.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_analysis", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def validator():
    return _load_module()


def test_valid_monitor_analysis_passes(validator):
    ok, reason = validator.validate_analysis({
        "intelica_moat_index": 0.78,
        "decision_recommendation": {"action": "monitor", "confidence_score": 0.82},
        "confidence": "low",
        "detected_competitors": ["A", "B"],
    })
    assert ok, reason


def test_out_of_range_imi_fails(validator):
    ok, reason = validator.validate_analysis({
        "intelica_moat_index": 1.5,
        "decision_recommendation": {"action": "monitor"},
    })
    assert not ok
    assert "intelica_moat_index" in reason


def test_invalid_action_fails(validator):
    ok, reason = validator.validate_analysis({
        "intelica_moat_index": 0.5,
        "decision_recommendation": {"action": "buy_now"},
    })
    assert not ok
    assert "action" in reason


def test_missing_decision_recommendation_fails(validator):
    ok, reason = validator.validate_analysis({"intelica_moat_index": 0.5})
    assert not ok
    assert "decision_recommendation" in reason


def test_competitors_must_be_list(validator):
    ok, reason = validator.validate_analysis({
        "intelica_moat_index": 0.5,
        "decision_recommendation": {"action": "enter"},
        "detected_competitors": "not a list",
    })
    assert not ok
    assert "detected_competitors" in reason

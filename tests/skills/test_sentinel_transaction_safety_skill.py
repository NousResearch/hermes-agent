"""Tests for the sentinel-transaction-safety optional skill.

Hermetic: stdlib + pytest only, no live network calls, no httpx. Exercises
the shipped scripts/validate_verdict.py helper directly, per the skill-test
requirement to test a real artifact rather than only mocked constants.
"""
import importlib.util
import pathlib

import pytest

_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "optional-skills" / "security" / "sentinel-transaction-safety"
    / "scripts" / "validate_verdict.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_verdict", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def validator():
    return _load_module()


def test_valid_safe_verdict_passes(validator):
    ok, reason = validator.validate_verdict({
        "verdict": "SAFE", "sentinelScore": 94, "grade": "AAA",
    })
    assert ok, reason


def test_valid_unsafe_verdict_passes(validator):
    ok, reason = validator.validate_verdict({
        "verdict": "UNSAFE", "sentinelScore": 4, "grade": "D",
    })
    assert ok, reason


def test_missing_verdict_field_fails(validator):
    ok, reason = validator.validate_verdict({"sentinelScore": 50})
    assert not ok
    assert "verdict" in reason


def test_out_of_range_score_fails(validator):
    ok, reason = validator.validate_verdict({"verdict": "SAFE", "sentinelScore": 150})
    assert not ok
    assert "sentinelScore" in reason


def test_invalid_grade_fails(validator):
    ok, reason = validator.validate_verdict({
        "verdict": "SAFE", "sentinelScore": 90, "grade": "Z",
    })
    assert not ok
    assert "grade" in reason


def test_non_dict_input_fails(validator):
    ok, reason = validator.validate_verdict("not a dict")
    assert not ok

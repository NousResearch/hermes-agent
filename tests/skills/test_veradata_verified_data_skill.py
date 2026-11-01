"""Tests for the veradata-verified-data optional skill.

Hermetic: stdlib + pytest only, no live network calls, no httpx. Exercises
the shipped scripts/validate_rates.py helper directly.
"""
import importlib.util
import pathlib

import pytest

_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "optional-skills" / "research" / "veradata-verified-data"
    / "scripts" / "validate_rates.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_rates", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def validator():
    return _load_module()


def test_valid_co_rates_passes(validator):
    ok, reason = validator.validate_rates({
        "country": "CO", "usd_cop": 3248.87, "trm_official": 3248.87,
    })
    assert ok, reason


def test_expected_country_mismatch_fails(validator):
    ok, reason = validator.validate_rates(
        {"country": "MX", "usd_mxn": 18.2}, expected_country="CO",
    )
    assert not ok
    assert "expected country" in reason


def test_unknown_country_fails(validator):
    ok, reason = validator.validate_rates({"country": "US", "usd": 1.0})
    assert not ok
    assert "country" in reason


def test_no_numeric_signal_fails(validator):
    ok, reason = validator.validate_rates({"country": "CO", "source": "text only"})
    assert not ok
    assert "numeric rate signal" in reason

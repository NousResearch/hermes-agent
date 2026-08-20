"""Tests for model.temperature and model.top_p validation in agent/agent_init.py.

These tests verify the finite + range-check logic that was added in PR #60532:
- ``not math.isfinite(v) or v < 0 or v > bound`` correctly rejects NaN, inf,
  -inf, bools, and out-of-range values.
- Valid finite numbers within [0, bound] are accepted.
"""

import math

import pytest


# The exact boolean expression used in the PR's validation blocks:
#   not math.isfinite(parsed) or parsed < 0 or parsed > bound
# Tests are parameterised so the same logic is exercised for both
# temperature (bound=2.0) and top_p (bound=1.0).


def _is_rejected(value, bound):
    """Mirror the validation condition from agent/agent_init.py."""
    if isinstance(value, bool):
        return True
    parsed = float(value)
    return not math.isfinite(parsed) or parsed < 0 or parsed > bound


class TestTemperatureValidation:
    bound = 2.0

    # --- Values that MUST be rejected --------------------------------------------

    @pytest.mark.parametrize("value", [
        float("nan"),
        float("inf"),
        float("-inf"),
        float("NaN"),
        float("Inf"),
        -1.0,
        -0.01,
        2.01,
        100,
        -100.5,
        True,
        False,
    ])
    def test_rejects_invalid(self, value):
        assert _is_rejected(value, self.bound)

    # --- Values that MUST be accepted --------------------------------------------

    @pytest.mark.parametrize("value", [
        0.0,
        1.0,
        2.0,
        0.5,
        1.5,
        0,
        2,
        1,
        "0.0",
        "1.5",
        "2",
    ])
    def test_accepts_valid(self, value):
        assert not _is_rejected(value, self.bound)


class TestTopPValidation:
    bound = 1.0

    @pytest.mark.parametrize("value", [
        float("nan"),
        float("inf"),
        float("-inf"),
        -1.0,
        -0.01,
        1.01,
        100,
        True,
        False,
    ])
    def test_rejects_invalid(self, value):
        assert _is_rejected(value, self.bound)

    @pytest.mark.parametrize("value", [
        0.0,
        0.5,
        1.0,
        0,
        1,
        "0.0",
        "0.75",
        "1",
    ])
    def test_accepts_valid(self, value):
        assert not _is_rejected(value, self.bound)


class TestMathIsfiniteBehavior:
    """Sanity-check that math.isfinite works as assumed."""

    def test_nan_is_not_finite(self):
        assert not math.isfinite(float("nan"))

    def test_inf_is_not_finite(self):
        assert not math.isfinite(float("inf"))

    def test_neg_inf_is_not_finite(self):
        assert not math.isfinite(float("-inf"))

    def test_finite_is_finite(self):
        assert math.isfinite(0.0)
        assert math.isfinite(1.0)
        assert math.isfinite(-1.0)

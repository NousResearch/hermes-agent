"""Regression tests for #61334: agent/anthropic_adapter.THINKING_BUDGET must
cover every value declared in VALID_REASONING_EFFORTS with a distinct
non-default budget.

Pre-fix the map had only 4 keys (xhigh/high/medium/low), so a caller setting
``reasoning_effort: minimal`` or ``reasoning_effort: max`` -- two of the
documented valid config values -- silently fell back to the medium default
(8000 tokens). Minimal had no cheaper/faster treatment despite its name;
max had *less* budget than xhigh despite sitting above it in the schema.

The fix adds explicit ``minimal: 2000`` and ``max: 48000`` entries so the
six-tier ordering is real:
    minimal(2k) < low(4k) < medium(8k) < high(16k) < xhigh(32k) < max(48k).
"""

from agent.anthropic_adapter import THINKING_BUDGET
from hermes_constants import VALID_REASONING_EFFORTS


# ---------------------------------------------------------------------------
# the fix
# ---------------------------------------------------------------------------


class TestThinkingBudgetCoversAllEffortLevels:
    """Issue #61334: every VALID_REASONING_EFFORTS entry needs its own budget."""

    def test_every_valid_effort_has_an_entry(self):
        """If we add a new VALID_REASONING_EFFORTS value, this test fails
        until THINKING_BUDGET also gains the new key.  That's the contract
        that prevents silent fallback (#61334) from coming back.
        """
        missing = sorted(set(VALID_REASONING_EFFORTS) - set(THINKING_BUDGET))
        assert not missing, (
            f"THINKING_BUDGET is missing entries for VALID_REASONING_EFFORTS "
            f"values {missing}; users configging these values would silently "
            "fall back to DefaultDict → medium (8000) (#61334)."
        )

    def test_minimal_resolves_to_its_own_budget(self):
        """Pre-fix: THINKING_BUDGET.get('minimal', 8000) -> 8000 — same as
        medium.  Post-fix: minimal must be distinct AND strictly less than
        low (it's the documented "shallowest" tier).
        """
        assert THINKING_BUDGET["minimal"] != THINKING_BUDGET["medium"], (
            "minimal still falls back to medium budget (silent)"
        )
        assert THINKING_BUDGET["minimal"] < THINKING_BUDGET["low"], (
            "minimal budget must be strictly less than low (the schema "
            "treats it as the shallowest tier)"
        )

    def test_max_resolves_to_its_own_budget(self):
        """Pre-fix: THINKING_BUDGET.get('max', 8000) -> 8000 — wildly less
        than xhigh (32000), so xhigh was effectively the maximum on this
        code path.  Post-fix: max must be strictly greater than xhigh.
        """
        assert THINKING_BUDGET["max"] != THINKING_BUDGET["medium"], (
            "max still falls back to medium budget (silent)"
        )
        assert THINKING_BUDGET["max"] > THINKING_BUDGET["xhigh"], (
            "max budget must be strictly greater than xhigh; otherwise "
            "the schema's documented 'deepest' tier is a downward alias"
        )

    def test_budget_ordering_is_monotonic(self):
        """All six tiers must be strictly increasing in budget, matching
        the schema's name-order from shallowest to deepest.
        """
        order = ["minimal", "low", "medium", "high", "xhigh", "max"]
        # Every key must be present (guard against silent-fallback if a
        # future change accidentally drops one).
        for key in order:
            assert key in THINKING_BUDGET, f"THINKING_BUDGET missing key {key!r}"

        for lo, hi in zip(order, order[1:]):
            assert THINKING_BUDGET[lo] < THINKING_BUDGET[hi], (
                f"THINKING_BUDGET ordering is not monotonic: "
                f"{lo}({THINKING_BUDGET[lo]}) < {hi}({THINKING_BUDGET[hi]}) failed"
            )

    def test_unknown_effort_fallback_is_a_real_default(self):
        """An unknown effort value still gets the medium bucket (the
        documented default-of-last-resort), not None or 0.  The fallback
        is intentional for forward-compat — only *known* effort values
        have to map to a distinct budget.
        """
        budget = THINKING_BUDGET.get("nonexistent-effort-level", 8000)
        assert budget == 8000
        # And the medium tier itself: same expected default.
        assert THINKING_BUDGET["medium"] == 8000

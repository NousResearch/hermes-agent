"""Tests for THINKING_BUDGET covering all valid effort levels."""

from agent.anthropic_adapter import THINKING_BUDGET
from hermes_constants import VALID_REASONING_EFFORTS


class TestThinkingBudget:
    """Tests for THINKING_BUDGET covering all valid effort levels."""
    
    def test_thinking_budget_has_all_valid_levels(self):
        """THINKING_BUDGET must include all VALID_REASONING_EFFORTS."""
        
        for effort in VALID_REASONING_EFFORTS:
            assert effort in THINKING_BUDGET, (
                f"VALID_REASONING_EFFORTS includes '{effort}' but "
                f"THINKING_BUDGET is missing it. Non-adaptive models will "
                f"fall back to the default (8000) instead of using the "
                f"correct budget."
            )
    
    def test_thinking_budget_values_are_ordered(self):
        """THINKING_BUDGET values should increase with effort level."""
        
        # Expected ordering: minimal < low < medium < high < xhigh < max
        expected_order = [
            ("minimal", 4000),
            ("low", 4000),
            ("medium", 8000),
            ("high", 16000),
            ("xhigh", 32000),
            ("max", 64000),
        ]
        
        for effort, expected_value in expected_order:
            actual_value = THINKING_BUDGET.get(effort)
            assert actual_value == expected_value, (
                f"THINKING_BUDGET['{effort}'] = {actual_value}, "
                f"expected {expected_value}"
            )
    
    def test_minimal_and_max_dont_fallback_to_default(self):
        """minimal and max should resolve to correct budgets, not default 8000."""
        
        # Before fix: THINKING_BUDGET.get("minimal", 8000) -> 8000 (wrong)
        # After fix: THINKING_BUDGET.get("minimal", 8000) -> 4000 (correct)
        assert THINKING_BUDGET.get("minimal") == 4000, (
            "minimal effort should map to 4000 tokens, not fall back to 8000"
        )
        
        # Before fix: THINKING_BUDGET.get("max", 8000) -> 8000 (wrong)
        # After fix: THINKING_BUDGET.get("max", 8000) -> 64000 (correct)
        assert THINKING_BUDGET.get("max") == 64000, (
            "max effort should map to 64000 tokens, not fall back to 8000"
        )
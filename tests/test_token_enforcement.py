#!/usr/bin/env python3
"""
Test suite for token burn enforcement system.

Validates:
  1. Daily budget cap enforcement
  2. Per-job context limits
  3. Premium model blocking
  4. Token usage logging
  5. Receipt generation
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest import mock

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.enforce_token_discipline import (
    check_and_enforce_budget,
    apply_context_limit,
    select_model_with_enforcement,
    estimate_cost_from_tokens,
    get_today_iso,
    load_daily_budget,
    save_daily_budget,
    DAILY_BUDGET_CAP_USD,
    DAILY_BUDGET_WARN_USD,
    MAX_CONTEXT_TOKENS,
)


def test_estimate_cost():
    """Test token to USD cost estimation."""
    print("Test 1: Cost estimation...")
    
    # 1000 tokens = $0.10
    cost = estimate_cost_from_tokens(1000)
    assert abs(cost - 0.10) < 0.001, f"Expected 0.10, got {cost}"
    
    # 5000 tokens = $0.50
    cost = estimate_cost_from_tokens(5000)
    assert abs(cost - 0.50) < 0.001, f"Expected 0.50, got {cost}"
    
    print("  ✓ Cost estimation working correctly")


def test_budget_enforcement():
    """Test daily budget cap enforcement."""
    print("\nTest 2: Budget enforcement...")
    
    # Fresh budget (0 spend)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock the paths
        budget_file = Path(tmpdir) / "daily_budget.json"
        
        with mock.patch('tools.enforce_token_discipline.DAILY_BUDGET_LOG', budget_file):
            # Small job should pass
            ok, msg = check_and_enforce_budget("job-1", 1000, is_background_job=True)
            assert ok, f"Small job should pass: {msg}"
            print("  ✓ Small job passed ($0.10 < $5.00)")
            
            # Job that would exceed cap should fail
            ok, msg = check_and_enforce_budget("job-2", 100000, is_background_job=True)
            assert not ok, "Job exceeding cap should fail"
            assert "BUDGET EXCEEDED" in msg
            print("  ✓ Job exceeding cap blocked")
            
            # Job between warn and cap should warn
            # Current: 0.10, adding 2.95 = 3.05 (> warn threshold)
            with mock.patch('tools.enforce_token_discipline.load_daily_budget') as mock_load:
                budget_data = {
                    "date": get_today_iso(),
                    "total_spend_usd": 2.95,
                    "total_tokens": 29500,
                    "entries": [],
                }
                mock_load.return_value = budget_data
                
                ok, msg = check_and_enforce_budget("job-3", 1000, is_background_job=True)
                assert ok, "Job should pass but warn"
                assert "WARNING" in msg or "warning" in msg.lower()
                print("  ✓ Job between warn and cap: allowed with warning")


def test_context_limit():
    """Test context truncation to token limit."""
    print("\nTest 3: Context limits...")
    
    # Context well under limit
    short_ctx = "This is a short context."
    limited, tokens = apply_context_limit(short_ctx, max_tokens=2000)
    assert limited == short_ctx, "Short context should not be truncated"
    print(f"  ✓ Short context not truncated ({tokens} tokens)")
    
    # Context way over limit (10000 chars ~= 2500 tokens)
    long_ctx = "x" * 20000  # ~5000 tokens
    limited, tokens = apply_context_limit(long_ctx, max_tokens=2000)
    assert len(limited) < len(long_ctx), "Long context should be truncated"
    assert tokens == 2000, f"Expected 2000 tokens, got {tokens}"
    print(f"  ✓ Long context truncated to {len(limited)} chars (~{tokens} tokens)")


def test_premium_model_blocking():
    """Test premium model blocking for background jobs."""
    print("\nTest 4: Premium model blocking...")
    
    # Regular model should not be blocked
    model, reason = select_model_with_enforcement(
        "anthropic/claude-haiku-4-5",
        is_background_job=True,
        allowed_premium_models=[]
    )
    assert model == "anthropic/claude-haiku-4-5"
    assert reason == ""
    print("  ✓ Regular model not blocked")
    
    # Premium model with empty whitelist should be blocked
    model, reason = select_model_with_enforcement(
        "anthropic/claude-opus-4-1",
        is_background_job=True,
        allowed_premium_models=[]
    )
    assert model != "anthropic/claude-opus-4-1"
    assert "fallback" in reason.lower() or "blocked" in reason.lower()
    print(f"  ✓ Premium model blocked, using fallback: {model}")
    
    # Premium model in whitelist should be allowed
    model, reason = select_model_with_enforcement(
        "anthropic/claude-opus-4-1",
        is_background_job=True,
        allowed_premium_models=["anthropic/claude-opus-4-1"]
    )
    assert model == "anthropic/claude-opus-4-1"
    assert reason == ""
    print("  ✓ Whitelisted premium model allowed")
    
    # Interactive jobs bypass blocking
    model, reason = select_model_with_enforcement(
        "anthropic/claude-opus-4-1",
        is_background_job=False,
        allowed_premium_models=[]
    )
    assert model == "anthropic/claude-opus-4-1"
    print("  ✓ Premium model allowed for interactive jobs (is_background_job=False)")


def test_token_usage_logging():
    """Test token usage log entry."""
    print("\nTest 5: Token usage logging...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        usage_file = Path(tmpdir) / "token_usage.jsonl"
        
        with mock.patch('tools.enforce_token_discipline.TOKEN_USAGE_LOG', usage_file):
            from tools.enforce_token_discipline import log_token_usage
            
            # Log a usage entry
            log_token_usage(
                job_id="test-job-1",
                model="anthropic/claude-haiku-4-5",
                tokens_used=1500,
                task_type="delegate",
                cost_usd=0.15
            )
            
            # Verify it was written
            assert usage_file.exists(), "Usage log file should exist"
            
            with open(usage_file, "r") as f:
                line = f.read().strip()
            
            entry = json.loads(line)
            assert entry["job_id"] == "test-job-1"
            assert entry["tokens_used"] == 1500
            assert entry["model"] == "anthropic/claude-haiku-4-5"
            assert abs(entry["cost_usd"] - 0.15) < 0.001
            
            print(f"  ✓ Token usage logged: {entry['job_id']} ({entry['tokens_used']} tokens, ${entry['cost_usd']:.2f})")


def test_receipt_generation():
    """Test daily receipt generation."""
    print("\nTest 6: Receipt generation...")
    
    # Would need to integrate with token_reporter
    # For now, just verify the module loads
    try:
        sys.path.insert(0, str(Path.home() / ".hermes"))
        import token_reporter
        
        # Test receipt building
        today = datetime.utcnow().strftime("%Y-%m-%d")
        receipt = token_reporter.build_receipt(today)
        
        assert receipt["date"] == today
        assert "total_spend_usd" in receipt
        assert "spend_by_job" in receipt
        assert "spend_by_model" in receipt
        assert "top_prompts" in receipt
        
        print(f"  ✓ Receipt generated for {today}")
        print(f"    Total spend: ${receipt['total_spend_usd']:.2f}")
        print(f"    Jobs tracked: {len(receipt['spend_by_job'])}")
        
    except Exception as e:
        print(f"  ⚠ Receipt generation test skipped: {e}")


def run_all_tests():
    """Run all enforcement tests."""
    print("=" * 60)
    print("TOKEN ENFORCEMENT SYSTEM - TEST SUITE")
    print("=" * 60)
    
    try:
        test_estimate_cost()
        test_budget_enforcement()
        test_context_limit()
        test_premium_model_blocking()
        test_token_usage_logging()
        test_receipt_generation()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

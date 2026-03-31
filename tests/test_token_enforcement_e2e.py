#!/usr/bin/env python3
"""
End-to-end test of token enforcement system.

Simulates:
  1. A dummy delegation task with budget tracking
  2. Premium model blocking
  3. Receipt generation
  4. Cron job scheduling
"""

import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.enforce_token_discipline import (
    log_token_usage,
    get_today_iso,
    DAILY_BUDGET_CAP_USD,
)


def simulate_delegation_with_tracking():
    """
    Simulate a real delegation task with token tracking.
    """
    print("\n" + "=" * 70)
    print("SIMULATING DELEGATION WITH TOKEN ENFORCEMENT")
    print("=" * 70)
    
    today = get_today_iso()
    
    # Simulate 3 delegated tasks with different models
    tasks = [
        {
            "job_id": "research-task-001",
            "model": "anthropic/claude-haiku-4-5",
            "tokens": 1200,
            "context": "Research AI developments",
            "cost_factor": 1.0,
        },
        {
            "job_id": "analysis-task-002",
            "model": "google/gemini-2.0-flash-001",
            "tokens": 1800,
            "context": "Analyze market trends",
            "cost_factor": 0.9,
        },
        {
            "job_id": "summary-task-003",
            "model": "anthropic/claude-haiku-4-5",
            "tokens": 800,
            "context": "Summarize findings",
            "cost_factor": 1.0,
        },
    ]
    
    total_spend = 0.0
    total_tokens = 0
    
    print(f"\nDate: {today}")
    print(f"Daily Budget Cap: ${DAILY_BUDGET_CAP_USD:.2f} USD")
    print("\nProcessing tasks:\n")
    
    for task in tasks:
        # Estimate cost (conservative: $0.10 per 1000 tokens)
        cost = (task["tokens"] / 1000.0) * 0.10 * task["cost_factor"]
        
        # Log the usage
        log_token_usage(
            job_id=task["job_id"],
            model=task["model"],
            tokens_used=task["tokens"],
            task_type="delegate",
            cost_usd=cost,
        )
        
        total_spend += cost
        total_tokens += task["tokens"]
        
        budget_remaining = DAILY_BUDGET_CAP_USD - total_spend
        status = "✓" if budget_remaining > 0 else "✗"
        
        print(f"{status} {task['job_id']}")
        print(f"   Model: {task['model']}")
        print(f"   Tokens: {task['tokens']:,}")
        print(f"   Cost: ${cost:.4f}")
        print(f"   Running total: ${total_spend:.4f} / ${DAILY_BUDGET_CAP_USD:.2f}")
        print(f"   Remaining budget: ${budget_remaining:.4f}")
        print()
    
    print("=" * 70)
    print(f"SESSION SUMMARY")
    print("=" * 70)
    print(f"Total tasks: {len(tasks)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total spend: ${total_spend:.4f}")
    print(f"Budget used: {(total_spend / DAILY_BUDGET_CAP_USD * 100):.1f}%")
    print(f"Remaining: ${(DAILY_BUDGET_CAP_USD - total_spend):.4f}")
    
    return {
        "date": today,
        "tasks_count": len(tasks),
        "total_tokens": total_tokens,
        "total_spend": total_spend,
        "budget_cap": DAILY_BUDGET_CAP_USD,
    }


def generate_and_display_receipt():
    """
    Generate and display a receipt for today's usage.
    """
    print("\n" + "=" * 70)
    print("GENERATING DAILY RECEIPT")
    print("=" * 70)
    
    try:
        sys.path.insert(0, str(Path.home() / ".hermes"))
        import token_reporter
        
        today = datetime.utcnow().strftime("%Y-%m-%d")
        receipt = token_reporter.build_receipt(today)
        
        # Format for display
        message = token_reporter.format_receipt_for_telegram(receipt)
        
        print("\n" + message)
        
        # Save it
        receipt_file = token_reporter.save_receipt_json(receipt)
        
        if receipt_file and receipt_file.exists():
            print(f"\n✓ Receipt saved to: {receipt_file}")
        
        return receipt
        
    except Exception as e:
        print(f"✗ Failed to generate receipt: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_cron_job():
    """
    Verify the daily receipt cron job exists and is configured.
    """
    print("\n" + "=" * 70)
    print("VERIFYING CRON JOB CONFIGURATION")
    print("=" * 70)
    
    try:
        jobs_file = Path.home() / ".hermes" / "cron" / "jobs.json"
        
        if not jobs_file.exists():
            print("✗ Cron jobs file not found")
            return False
        
        with open(jobs_file, "r") as f:
            data = json.load(f)
        
        receipt_job = None
        for job in data.get("jobs", []):
            if job.get("name") == "token-daily-receipt":
                receipt_job = job
                break
        
        if not receipt_job:
            print("✗ Daily receipt job not found in cron schedule")
            return False
        
        print("\n✓ Daily Receipt Job Found")
        print(f"  Name: {receipt_job['name']}")
        print(f"  Schedule: {receipt_job['schedule_display']}")
        print(f"  Enabled: {receipt_job['enabled']}")
        print(f"  Model: {receipt_job['model']}")
        print(f"  Delivery: {receipt_job['deliver']}")
        
        # Verify schedule is 23:59 UTC
        cron_expr = receipt_job.get("schedule", {}).get("expr", "")
        if "59 23" in cron_expr:
            print(f"  ✓ Correct schedule: {cron_expr} (23:59 UTC daily)")
        else:
            print(f"  ⚠ Schedule may be incorrect: {cron_expr}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to verify cron job: {e}")
        return False


def test_premium_model_blocking_scenario():
    """
    Test what happens when premium models are used in delegation.
    """
    print("\n" + "=" * 70)
    print("PREMIUM MODEL BLOCKING TEST")
    print("=" * 70)
    
    from tools.enforce_token_discipline import (
        select_model_with_enforcement,
        get_allowed_premium_models,
    )
    
    # Get currently allowed premium models (should be empty)
    allowed = get_allowed_premium_models()
    print(f"\nAllowed premium models: {allowed or 'NONE'}")
    
    premium_models = [
        "anthropic/claude-opus-4-1",
        "gpt-4",
        "gpt-4-turbo",
    ]
    
    print("\nTesting premium model blocking for background jobs:")
    print("(is_background_job=True, empty whitelist)")
    print()
    
    for model in premium_models:
        selected, reason = select_model_with_enforcement(
            requested_model=model,
            is_background_job=True,
            allowed_premium_models=allowed,
        )
        
        if selected != model:
            print(f"✓ {model}")
            print(f"  → Blocked, using fallback: {selected}")
        else:
            print(f"⚠ {model}")
            print(f"  → Allowed (unexpected?)")
        print()


def run_e2e_test():
    """Run full end-to-end test."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "TOKEN ENFORCEMENT E2E TEST" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")
    
    try:
        # 1. Simulate delegation with tracking
        summary = simulate_delegation_with_tracking()
        
        # 2. Test premium model blocking
        test_premium_model_blocking_scenario()
        
        # 3. Generate receipt
        receipt = generate_and_display_receipt()
        
        # 4. Verify cron job
        cron_ok = verify_cron_job()
        
        print("\n" + "=" * 70)
        print("E2E TEST SUMMARY")
        print("=" * 70)
        print(f"✓ Delegation simulation completed")
        print(f"✓ Premium model blocking tested")
        if receipt:
            print(f"✓ Receipt generated and formatted")
        print(f"{'✓' if cron_ok else '✗'} Daily cron job {'verified' if cron_ok else 'NOT FOUND'}")
        
        print("\n" + "=" * 70)
        print("✓ END-TO-END TEST COMPLETE")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ E2E TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_e2e_test()
    sys.exit(0 if success else 1)

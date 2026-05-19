"""
batch_multi_wallet_executor.py - Multi-Wallet Batch Mint Executor
===================================================================
Functions:
- Execute mint plan across multiple wallets simultaneously
- Configurable concurrency
- Randomized delay between wallets
- Retry failed wallets
- CSV wallet import
- Export execution report

SAFETY:
- All individual mints still require approval
- DRY_RUN=true by default
- Private keys NEVER exposed
- Only for user's OWN burner wallets

Usage:
    python -m custom_tools.batch_multi_wallet_executor \\
        --contract 0xAddress --wallets burner1,burner2,burner3
    python -m custom_tools.batch_multi_wallet_executor \\
        --contract 0xAddress --wallet-csv wallets.csv
"""

import os
import sys
import json
import csv
import time
import random
import concurrent.futures
from datetime import datetime
from pathlib import Path

from custom_tools.check_wallet import get_web3, validate_address
from custom_tools.wallet_manager import list_wallets
from custom_tools.mint_planner import build_mint_transaction
from custom_tools.approval_queue import add_to_queue, approve
from custom_tools.mint_executor import execute_approved_tx



# Configuration
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
DEFAULT_CONCURRENCY = int(os.getenv("BATCH_CONCURRENCY", "3"))
DEFAULT_MIN_DELAY = float(os.getenv("BATCH_MIN_DELAY", "1.0"))
DEFAULT_MAX_DELAY = float(os.getenv("BATCH_MAX_DELAY", "5.0"))
MAX_RETRIES = int(os.getenv("BATCH_MAX_RETRIES", "2"))


def create_batch_plan(
    contract_address: str,
    wallet_labels: list,
    chain: str = "ethereum",
    quantity: int = 1,
    mint_function: str = None,
    mint_price_wei: int = None,
) -> list:
    """
    Create mint plans for multiple wallets.
    
    Args:
        contract_address: NFT contract address
        wallet_labels: List of wallet labels
        chain: Chain name
        quantity: Mint quantity per wallet
        mint_function: Override mint function
        mint_price_wei: Override price
    
    Returns:
        List of transaction previews
    """
    plans = []
    
    print(f"\n{'='*60}")
    print(f"  BATCH MINT PLANNER")
    print(f"  Contract: {contract_address}")
    print(f"  Wallets: {len(wallet_labels)}")
    print(f"  Quantity per wallet: {quantity}")
    print(f"  Chain: {chain}")
    print(f"{'='*60}\n")
    
    for i, label in enumerate(wallet_labels):
        print(f"  [{i+1}/{len(wallet_labels)}] Planning for wallet: {label}")
        try:
            preview = build_mint_transaction(
                contract_address,
                label,
                chain=chain,
                quantity=quantity,
                mint_function=mint_function,
                mint_price_wei=mint_price_wei,
            )
            preview["batch_index"] = i
            plans.append(preview)
            print(f"    -> OK: {preview['total_cost']}")
        except Exception as e:
            plans.append({
                "from_wallet": label,
                "batch_index": i,
                "error": str(e),
                "status": "plan_failed",
            })
            print(f"    -> FAILED: {e}")
    
    return plans



def execute_batch(
    plans: list,
    concurrency: int = DEFAULT_CONCURRENCY,
    min_delay: float = DEFAULT_MIN_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    auto_approve: bool = False,
    retries: int = MAX_RETRIES,
) -> list:
    """
    Execute batch mint across multiple wallets.
    
    Args:
        plans: List of transaction previews from create_batch_plan
        concurrency: Max parallel executions
        min_delay: Minimum random delay between executions (seconds)
        max_delay: Maximum random delay between executions (seconds)
        auto_approve: If True, auto-approve (use with caution)
        retries: Retry count per wallet
    
    Returns:
        List of execution results
    """
    if DRY_RUN:
        print("\n  [DRY_RUN=true] Batch execution in simulation mode")
        print("  Transactions will NOT be sent.\n")
    
    results = []
    valid_plans = [p for p in plans if "error" not in p]
    
    print(f"\n  Executing batch: {len(valid_plans)} wallets")
    print(f"  Concurrency: {concurrency}")
    print(f"  Delay: {min_delay}-{max_delay}s (randomized)")
    print(f"  Auto-approve: {auto_approve}")
    print()
    
    # Add all to approval queue
    queue_entries = []
    for plan in valid_plans:
        entry_id = add_to_queue(plan)
        queue_entries.append({"id": entry_id, "plan": plan})
        
        if auto_approve:
            approve(entry_id, approved_by="batch_auto")
    
    if not auto_approve:
        print("\n  All entries added to queue as PENDING.")
        print("  Approve them manually or via Telegram before execution.")
        print("  Use: python -m custom_tools.approval_queue approve --id <ID>")
        return [{"id": e["id"], "status": "pending"} for e in queue_entries]
    
    # Execute with concurrency control
    def _execute_single(entry):
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
        return execute_approved_tx(entry["id"], retries=retries)
    
    # Sequential execution with delay (safer than thread pool for web3)
    if concurrency <= 1:
        for entry in queue_entries:
            result = _execute_single(entry)
            results.append(result)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_execute_single, entry): entry
                for entry in queue_entries
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    entry = futures[future]
                    results.append({
                        "id": entry["id"],
                        "status": "failed",
                        "error": str(e),
                    })
    
    return results



def export_execution_report(results: list, filepath: str = None) -> str:
    """Export execution report to JSON/CSV."""
    if not filepath:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filepath = f".data/batch_report_{timestamp}.json"
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_wallets": len(results),
        "successful": sum(1 for r in results if r.get("status") == "sent"),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
        "dry_run": sum(1 for r in results if r.get("status") == "dry_run"),
        "pending": sum(1 for r in results if r.get("status") == "pending"),
        "results": results,
    }
    
    if filepath.endswith(".csv"):
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "status", "tx_hash", "error"])
            for r in results:
                writer.writerow([
                    r.get("id", ""),
                    r.get("status", ""),
                    r.get("tx_hash", ""),
                    r.get("error", ""),
                ])
    else:
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)
    
    print(f"  Report saved: {filepath}")
    return filepath


def load_wallets_from_csv(csv_path: str) -> list:
    """Load wallet labels from CSV file. Format: label (one per line or first column)."""
    labels = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if row and not row[0].startswith("#"):
                labels.append(row[0].strip())
    return labels


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch multi-wallet mint executor")
    parser.add_argument("--contract", required=True, help="NFT contract address")
    parser.add_argument("--wallets", help="Comma-separated wallet labels")
    parser.add_argument("--wallet-csv", help="CSV file with wallet labels")
    parser.add_argument("--chain", default="ethereum", help="Chain name")
    parser.add_argument("--quantity", type=int, default=1, help="Quantity per wallet")
    parser.add_argument("--function", help="Override mint function name")
    parser.add_argument("--price-wei", type=int, help="Override mint price")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve (dangerous)")
    parser.add_argument("--report", help="Export report path")
    
    args = parser.parse_args()
    
    # Get wallet labels
    if args.wallets:
        wallet_labels = [w.strip() for w in args.wallets.split(",")]
    elif args.wallet_csv:
        wallet_labels = load_wallets_from_csv(args.wallet_csv)
    else:
        print("Error: Provide --wallets or --wallet-csv", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Create plans
        plans = create_batch_plan(
            args.contract,
            wallet_labels,
            chain=args.chain,
            quantity=args.quantity,
            mint_function=args.function,
            mint_price_wei=args.price_wei,
        )
        
        # Execute batch
        results = execute_batch(
            plans,
            concurrency=args.concurrency,
            auto_approve=args.auto_approve,
        )
        
        # Export report
        if args.report:
            export_execution_report(results, args.report)
        else:
            export_execution_report(results)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

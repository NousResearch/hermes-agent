"""
batch_multi_wallet_executor.py - Multi-Wallet Batch Mint Executor
===================================================================
Functions:
- Read wallet labels from CSV
- Create mint plans for each wallet
- Add each plan to approval_queue as pending
- Configurable concurrency
- Random delay between executions
- Retry failed wallets
- Export execution report (JSON/CSV)
- DRY_RUN=true default

SAFETY:
- All individual mints go through approval_queue as PENDING
- DRY_RUN=true by default - no auto-send
- Private keys NEVER exposed
- Only for user's OWN burner wallets
- Never bypass allowlist/captcha/signature/anti-bot

Usage:
    python -m custom_tools.batch_multi_wallet_executor \
        --contract 0xAddress --wallets burner1,burner2,burner3
    python -m custom_tools.batch_multi_wallet_executor \
        --contract 0xAddress --wallet-csv wallets.csv
    python -m custom_tools.batch_multi_wallet_executor \
        --contract 0xAddress --wallet-csv wallets.csv --auto-approve
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
from custom_tools.approval_queue import add_to_queue, approve, get_entry
from custom_tools.mint_executor import execute_approved_tx


# Configuration
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
DEFAULT_CONCURRENCY = int(os.getenv("BATCH_CONCURRENCY", "3"))
DEFAULT_MIN_DELAY = float(os.getenv("BATCH_MIN_DELAY", "1.0"))
DEFAULT_MAX_DELAY = float(os.getenv("BATCH_MAX_DELAY", "5.0"))
MAX_RETRIES = int(os.getenv("BATCH_MAX_RETRIES", "2"))


def load_wallets_from_csv(csv_path: str) -> list:
    """
    Load wallet labels from CSV file.
    Format: one label per line (first column), lines starting with # are skipped.
    """
    labels = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip() and not row[0].strip().startswith("#"):
                labels.append(row[0].strip())
    return labels


def create_batch_plan(
    contract_address: str,
    wallet_labels: list,
    chain: str = "ethereum",
    quantity: int = 1,
    mint_function: str = None,
    mint_price_wei: int = None,
) -> list:
    """
    Create mint plans for multiple wallets and add each to approval_queue.

    Args:
        contract_address: NFT contract address
        wallet_labels: List of wallet labels
        chain: Chain name
        quantity: Mint quantity per wallet
        mint_function: Override mint function
        mint_price_wei: Override price

    Returns:
        List of dicts with wallet_label, approval_id, status
    """
    results = []

    print(f"\n{'='*60}")
    print(f"  BATCH MINT PLANNER")
    print(f"  Contract: {contract_address}")
    print(f"  Chain: {chain}")
    print(f"  Wallets: {len(wallet_labels)}")
    print(f"  Quantity per wallet: {quantity}")
    print(f"  DRY_RUN: {DRY_RUN}")
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
            # Add to approval queue as PENDING
            approval_id = add_to_queue(preview)
            results.append({
                "wallet_label": label,
                "approval_id": approval_id,
                "status": "queued_pending",
                "total_cost": preview.get("total_cost", "N/A"),
            })
            print(f"    -> Queued: approval ID #{approval_id} [{preview.get('total_cost', '')}]")

        except Exception as e:
            results.append({
                "wallet_label": label,
                "approval_id": None,
                "status": "plan_failed",
                "error": str(e),
            })
            print(f"    -> FAILED: {e}")

    # Summary
    queued = sum(1 for r in results if r["status"] == "queued_pending")
    failed = sum(1 for r in results if r["status"] == "plan_failed")
    print(f"\n  Planning complete: {queued} queued, {failed} failed")

    return results


def execute_batch(
    approval_ids: list,
    concurrency: int = DEFAULT_CONCURRENCY,
    min_delay: float = DEFAULT_MIN_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    retries: int = MAX_RETRIES,
) -> list:
    """
    Execute approved entries from the batch.

    Args:
        approval_ids: List of approval queue IDs to execute
        concurrency: Max parallel executions
        min_delay: Min random delay between executions (seconds)
        max_delay: Max random delay between executions (seconds)
        retries: Retry count per wallet

    Returns:
        List of execution results
    """
    if DRY_RUN:
        print("\n  [DRY_RUN=true] Batch execution in simulation mode")
        print("  Transactions will NOT be sent.\n")

    results = []

    # Filter to only approved entries
    executable = []
    for aid in approval_ids:
        try:
            entry = get_entry(aid)
            if entry["status"] == "approved":
                executable.append(aid)
            else:
                results.append({
                    "id": aid,
                    "status": "skipped",
                    "reason": f"Status is '{entry['status']}', not approved",
                })
        except Exception as e:
            results.append({"id": aid, "status": "error", "reason": str(e)})

    if not executable:
        print("  No approved entries to execute.")
        return results

    print(f"\n  Executing {len(executable)} approved entries")
    print(f"  Concurrency: {concurrency}")
    print(f"  Delay: {min_delay}-{max_delay}s (randomized)\n")

    def _execute_single(entry_id):
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
        return execute_approved_tx(entry_id, retries=retries)

    if concurrency <= 1:
        # Sequential
        for aid in executable:
            print(f"  --- Executing #{aid} ---")
            result = _execute_single(aid)
            results.append(result)
    else:
        # Concurrent
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_map = {
                executor.submit(_execute_single, aid): aid
                for aid in executable
            }
            for future in concurrent.futures.as_completed(future_map):
                aid = future_map[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({"id": aid, "status": "failed", "error": str(e)})

    # Summary
    sent = sum(1 for r in results if r.get("status") == "sent")
    failed = sum(1 for r in results if r.get("status") == "failed")
    dry = sum(1 for r in results if r.get("status") == "dry_run")
    skipped = sum(1 for r in results if r.get("status") == "skipped")

    print(f"\n  {'='*40}")
    print(f"  BATCH EXECUTION SUMMARY")
    print(f"  Sent: {sent} | Failed: {failed} | Dry Run: {dry} | Skipped: {skipped}")
    print(f"  {'='*40}\n")

    return results


def export_report(results: list, filepath: str = None) -> str:
    """Export execution report to JSON or CSV."""
    if not filepath:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filepath = f".data/batch_report_{timestamp}.json"

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "total": len(results),
        "summary": {
            "sent": sum(1 for r in results if r.get("status") == "sent"),
            "failed": sum(1 for r in results if r.get("status") == "failed"),
            "dry_run": sum(1 for r in results if r.get("status") == "dry_run"),
            "skipped": sum(1 for r in results if r.get("status") == "skipped"),
            "queued": sum(1 for r in results if r.get("status") == "queued_pending"),
            "plan_failed": sum(1 for r in results if r.get("status") == "plan_failed"),
        },
        "results": results,
    }

    if filepath.endswith(".csv"):
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["wallet_label", "approval_id", "status", "tx_hash", "error"])
            for r in results:
                writer.writerow([
                    r.get("wallet_label", r.get("id", "")),
                    r.get("approval_id", r.get("id", "")),
                    r.get("status", ""),
                    r.get("tx_hash", ""),
                    r.get("error", ""),
                ])
    else:
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

    print(f"  Report saved: {filepath}")
    return filepath


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch multi-wallet mint executor")
    parser.add_argument("--contract", required=True, help="NFT contract address")
    parser.add_argument("--wallets", help="Comma-separated wallet labels")
    parser.add_argument("--wallet-csv", help="CSV file with wallet labels (one per line)")
    parser.add_argument("--chain", default="ethereum", help="Chain name")
    parser.add_argument("--quantity", type=int, default=1, help="Quantity per wallet")
    parser.add_argument("--function", help="Override mint function name")
    parser.add_argument("--price-wei", type=int, help="Override mint price in wei")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Max parallel executions")
    parser.add_argument("--min-delay", type=float, default=DEFAULT_MIN_DELAY, help="Min delay between executions")
    parser.add_argument("--max-delay", type=float, default=DEFAULT_MAX_DELAY, help="Max delay between executions")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve all entries (use with caution)")
    parser.add_argument("--execute", action="store_true", help="Execute after approval (requires --auto-approve)")
    parser.add_argument("--report", help="Export report path (.json or .csv)")

    args = parser.parse_args()

    # Get wallet labels
    if args.wallets:
        wallet_labels = [w.strip() for w in args.wallets.split(",")]
    elif args.wallet_csv:
        wallet_labels = load_wallets_from_csv(args.wallet_csv)
    else:
        print("Error: Provide --wallets or --wallet-csv", file=sys.stderr)
        sys.exit(1)

    if not wallet_labels:
        print("Error: No wallet labels found", file=sys.stderr)
        sys.exit(1)

    try:
        # Step 1: Create plans and queue them
        plan_results = create_batch_plan(
            args.contract,
            wallet_labels,
            chain=args.chain,
            quantity=args.quantity,
            mint_function=args.function,
            mint_price_wei=args.price_wei,
        )

        # Step 2: Auto-approve if requested
        approval_ids = [r["approval_id"] for r in plan_results if r["approval_id"] is not None]

        if args.auto_approve and approval_ids:
            print(f"\n  Auto-approving {len(approval_ids)} entries...")
            for aid in approval_ids:
                try:
                    approve(aid, approved_by="batch_auto")
                    print(f"    -> #{aid} approved")
                except Exception as e:
                    print(f"    -> #{aid} approve failed: {e}")

        # Step 3: Execute if requested
        if args.execute and args.auto_approve and approval_ids:
            exec_results = execute_batch(
                approval_ids,
                concurrency=args.concurrency,
                min_delay=args.min_delay,
                max_delay=args.max_delay,
            )
            all_results = plan_results + exec_results
        else:
            all_results = plan_results
            if not args.auto_approve:
                print("\n  All entries queued as PENDING.")
                print("  Approve via Telegram or CLI:")
                print("    python -m custom_tools.approval_queue approve --id <ID>")
                print("  Or use --auto-approve to approve immediately.")

        # Step 4: Export report
        report_path = args.report or None
        if report_path or len(all_results) > 0:
            export_report(all_results, filepath=report_path)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

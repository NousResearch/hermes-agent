"""
mint_executor.py - Approved Transaction Executor
==================================================
Functions:
- Execute ONLY approved transactions from approval_queue
- Sign transaction (key never logged)
- Send raw transaction
- Wait for receipt
- Save tx result
- Retry failed transactions

SAFETY:
- ONLY executes transactions with status='approved'
- Private keys NEVER printed or logged
- DRY_RUN check before execution
- Full audit trail in approval_queue

Usage:
    python -m custom_tools.mint_executor --id 1
    python -m custom_tools.mint_executor --all
"""

import os
import sys
import json
import time
from web3 import Web3

from custom_tools.check_wallet import get_web3
from custom_tools.wallet_manager import get_wallet_key
from custom_tools.approval_queue import (
    get_entry,
    get_approved_pending_execution,
    mark_sent,
    mark_failed,
)



# Safety
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
MAX_RETRIES = int(os.getenv("MINT_MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("MINT_RETRY_DELAY", "5.0"))


def execute_approved_tx(entry_id: int, retries: int = MAX_RETRIES) -> dict:
    """
    Execute a single approved transaction.
    
    Args:
        entry_id: Approval queue entry ID
        retries: Number of retry attempts
    
    Returns:
        dict with execution result
    """
    # Get entry from queue
    entry = get_entry(entry_id)
    
    # SAFETY CHECK: Only execute approved transactions
    if entry["status"] != "approved":
        raise ValueError(
            f"Entry #{entry_id} status is '{entry['status']}'. "
            f"Only 'approved' transactions can be executed."
        )
    
    # SAFETY CHECK: DRY_RUN
    if DRY_RUN:
        print(f"\n  [DRY_RUN] Would execute entry #{entry_id}")
        print(f"  Contract: {entry['contract_address']}")
        print(f"  Wallet: {entry['wallet_label']}")
        print(f"  Function: {entry['mint_function']}")
        print(f"  -> Set DRY_RUN=false to execute for real")
        return {
            "id": entry_id,
            "status": "dry_run",
            "message": "DRY_RUN mode - transaction not sent",
        }
    
    chain = entry["chain"]
    w3 = get_web3(chain)
    
    # Parse tx data
    tx_data = json.loads(entry["tx_data"]) if isinstance(entry["tx_data"], str) else entry["tx_data"]
    
    # Ensure proper types
    if "value" in tx_data:
        tx_data["value"] = int(tx_data["value"])
    if "gas" in tx_data:
        tx_data["gas"] = int(tx_data["gas"])
    if "nonce" in tx_data:
        # Refresh nonce to avoid conflicts
        from_addr = Web3.to_checksum_address(entry["from_address"])
        tx_data["nonce"] = w3.eth.get_transaction_count(from_addr)


    # Add gas price if not set
    if "gasPrice" not in tx_data and "maxFeePerGas" not in tx_data:
        tx_data["gasPrice"] = w3.eth.gas_price
    
    # Execute with retries
    last_error = None
    for attempt in range(retries):
        try:
            # Get private key (NEVER log this)
            private_key = get_wallet_key(entry["wallet_label"])
            
            # Sign transaction
            signed_tx = w3.eth.account.sign_transaction(tx_data, private_key)
            
            # Clear key from memory
            private_key = None
            
            # Send raw transaction
            tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            tx_hash_hex = tx_hash.hex()
            
            print(f"  TX Sent: {tx_hash_hex}")
            print(f"  Waiting for receipt...")
            
            # Wait for receipt
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            # Update queue status
            mark_sent(entry_id, tx_hash_hex)
            
            result = {
                "id": entry_id,
                "status": "sent",
                "tx_hash": tx_hash_hex,
                "block_number": receipt["blockNumber"],
                "gas_used": receipt["gasUsed"],
                "success": receipt["status"] == 1,
            }
            
            if receipt["status"] != 1:
                mark_failed(entry_id, "Transaction reverted")
                result["status"] = "reverted"
            
            print(f"  Result: {'SUCCESS' if receipt['status'] == 1 else 'REVERTED'}")
            print(f"  Block: {receipt['blockNumber']}")
            print(f"  Gas Used: {receipt['gasUsed']}")
            
            return result
            
        except Exception as e:
            last_error = str(e)
            print(f"  Attempt {attempt + 1}/{retries} failed: {last_error}")
            
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
    
    # All retries failed
    mark_failed(entry_id, last_error)
    return {
        "id": entry_id,
        "status": "failed",
        "error": last_error,
        "retries_exhausted": True,
    }



def execute_all_approved() -> list:
    """Execute all approved transactions in queue order."""
    approved = get_approved_pending_execution()
    
    if not approved:
        print("  No approved transactions pending execution.")
        return []
    
    print(f"\n  Found {len(approved)} approved transaction(s)")
    results = []
    
    for entry in approved:
        print(f"\n  --- Executing #{entry['id']} ---")
        result = execute_approved_tx(entry["id"])
        results.append(result)
    
    # Summary
    sent = sum(1 for r in results if r["status"] == "sent")
    failed = sum(1 for r in results if r["status"] == "failed")
    dry = sum(1 for r in results if r["status"] == "dry_run")
    
    print(f"\n  === EXECUTION SUMMARY ===")
    print(f"  Sent: {sent} | Failed: {failed} | Dry Run: {dry}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Execute approved mint transactions")
    parser.add_argument("--id", type=int, help="Execute specific entry ID")
    parser.add_argument("--all", action="store_true", help="Execute all approved")
    
    args = parser.parse_args()
    
    if DRY_RUN:
        print("\n  WARNING: DRY_RUN=true - transactions will NOT be sent")
        print("  Set DRY_RUN=false in environment to execute for real\n")
    
    try:
        if args.id:
            result = execute_approved_tx(args.id)
            print(json.dumps(result, indent=2, default=str))
        elif args.all:
            results = execute_all_approved()
            print(json.dumps(results, indent=2, default=str))
        else:
            parser.print_help()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

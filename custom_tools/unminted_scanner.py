"""
unminted_scanner.py - Unminted Token ID Scanner
=================================================
Functions:
- Scan token ID range for unminted tokens
- ownerOf success = minted, revert/error = unminted
- Realtime progress printing
- Save results to CSV and JSON
- Retry logic with configurable delay
- Optimized for low-RAM VPS

Usage:
    python -m custom_tools.unminted_scanner 0xContract 1 1000
    python -m custom_tools.unminted_scanner 0xContract 1 1000 --chain base --delay 0.1
    python -m custom_tools.unminted_scanner 0xContract 1 5000 --output results.csv
"""

import os
import sys
import json
import csv
import time
from datetime import datetime
from web3 import Web3
from web3.exceptions import ContractLogicError

from custom_tools.check_wallet import get_web3, validate_address


# Minimal ABI for ownerOf
OWNER_OF_ABI = [
    {
        "inputs": [{"name": "tokenId", "type": "uint256"}],
        "name": "ownerOf",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# Default config
DEFAULT_DELAY = 0.05  # 50ms between calls
DEFAULT_RETRY = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_BATCH_SIZE = 50  # Process in small batches for memory efficiency


def scan_unminted_tokens(
    contract_address: str,
    start_id: int,
    end_id: int,
    chain: str = "ethereum",
    delay: float = DEFAULT_DELAY,
    retries: int = DEFAULT_RETRY,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_callback=None,
) -> dict:
    """
    Scan a range of token IDs to find unminted ones.
    
    Args:
        contract_address: NFT contract address
        start_id: Start of token ID range (inclusive)
        end_id: End of token ID range (inclusive)
        chain: Chain name
        delay: Delay between RPC calls (seconds)
        retries: Number of retries per failed call
        retry_delay: Delay between retries (seconds)
        batch_size: Process in batches (memory optimization)
        progress_callback: Optional callback(current, total, result)
    
    Returns:
        dict with minted, unminted, errors lists and stats
    """
    checksummed = validate_address(contract_address)
    w3 = get_web3(chain)
    
    if not w3.is_connected():
        raise ConnectionError(f"Cannot connect to {chain} RPC")
    
    contract = w3.eth.contract(address=checksummed, abi=OWNER_OF_ABI)
    
    minted = []
    unminted = []
    errors = []
    total = end_id - start_id + 1
    scanned = 0
    
    print(f"\n{'='*60}")
    print(f"  UNMINTED TOKEN SCANNER")
    print(f"  Contract: {checksummed}")
    print(f"  Chain: {chain}")
    print(f"  Range: {start_id} - {end_id} ({total} tokens)")
    print(f"  Delay: {delay}s | Retries: {retries}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for token_id in range(start_id, end_id + 1):
        success = False
        
        for attempt in range(retries):
            try:
                owner = contract.functions.ownerOf(token_id).call()
                minted.append({"token_id": token_id, "owner": owner})
                success = True
                break
            except (ContractLogicError, Exception) as e:
                error_msg = str(e).lower()
                # Check if it's genuinely unminted vs RPC error
                unminted_indicators = [
                    "nonexistent token",
                    "invalid token",
                    "owner query for nonexistent",
                    "erc721: invalid token id",
                    "token does not exist",
                ]
                
                if any(ind in error_msg for ind in unminted_indicators):
                    unminted.append({"token_id": token_id})
                    success = True
                    break
                else:
                    # Possible RPC error, retry
                    if attempt < retries - 1:
                        time.sleep(retry_delay)
                    else:
                        errors.append({"token_id": token_id, "error": str(e)})
        
        scanned += 1
        
        # Progress reporting
        if scanned % 10 == 0 or scanned == total:
            elapsed = time.time() - start_time
            rate = scanned / elapsed if elapsed > 0 else 0
            eta = (total - scanned) / rate if rate > 0 else 0
            
            progress_line = (
                f"  [{scanned}/{total}] "
                f"Minted: {len(minted)} | Unminted: {len(unminted)} | "
                f"Errors: {len(errors)} | "
                f"Rate: {rate:.1f}/s | ETA: {eta:.0f}s"
            )
            print(f"\r{progress_line}", end="", flush=True)
            
            if progress_callback:
                progress_callback(scanned, total, {
                    "minted": len(minted),
                    "unminted": len(unminted),
                    "errors": len(errors),
                })
        
        # Delay between calls
        if delay > 0:
            time.sleep(delay)
        
        # Memory optimization: flush batches
        if scanned % batch_size == 0:
            pass  # Results are already in lists, Python handles GC
    
    elapsed_total = time.time() - start_time
    
    print(f"\n\n{'='*60}")
    print(f"  SCAN COMPLETE")
    print(f"  Total scanned: {scanned}")
    print(f"  Minted: {len(minted)}")
    print(f"  Unminted: {len(unminted)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Time: {elapsed_total:.1f}s")
    print(f"{'='*60}\n")
    
    return {
        "contract": checksummed,
        "chain": chain,
        "range_start": start_id,
        "range_end": end_id,
        "total_scanned": scanned,
        "minted_count": len(minted),
        "unminted_count": len(unminted),
        "error_count": len(errors),
        "minted": minted,
        "unminted": unminted,
        "errors": errors,
        "scan_time_seconds": round(elapsed_total, 2),
        "timestamp": datetime.utcnow().isoformat(),
    }


def save_results_csv(results: dict, filepath: str):
    """Save unminted tokens to CSV file."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token_id", "status", "owner"])
        
        for item in results["minted"]:
            writer.writerow([item["token_id"], "minted", item["owner"]])
        
        for item in results["unminted"]:
            writer.writerow([item["token_id"], "unminted", ""])
        
        for item in results["errors"]:
            writer.writerow([item["token_id"], "error", item.get("error", "")])
    
    print(f"  CSV saved: {filepath}")


def save_results_json(results: dict, filepath: str):
    """Save scan results to JSON file."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"  JSON saved: {filepath}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scan for unminted token IDs")
    parser.add_argument("address", help="NFT contract address")
    parser.add_argument("start_id", type=int, help="Start token ID")
    parser.add_argument("end_id", type=int, help="End token ID")
    parser.add_argument("--chain", default="ethereum", help="Chain name")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Delay between calls (seconds)")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRY, help="Number of retries")
    parser.add_argument("--output", help="Output file path (auto-detects csv/json from extension)")
    parser.add_argument("--csv", help="Save CSV to this path")
    parser.add_argument("--json-out", help="Save JSON to this path")
    
    args = parser.parse_args()
    
    try:
        results = scan_unminted_tokens(
            args.address,
            args.start_id,
            args.end_id,
            chain=args.chain,
            delay=args.delay,
            retries=args.retries,
        )
        
        # Save outputs
        if args.output:
            if args.output.endswith(".csv"):
                save_results_csv(results, args.output)
            else:
                save_results_json(results, args.output)
        
        if args.csv:
            save_results_csv(results, args.csv)
        
        if args.json_out:
            save_results_json(results, args.json_out)
        
        # Print unminted summary
        if results["unminted"]:
            print(f"\n  Unminted Token IDs:")
            ids = [str(item["token_id"]) for item in results["unminted"][:50]]
            print(f"  {', '.join(ids)}")
            if len(results["unminted"]) > 50:
                print(f"  ... and {len(results['unminted']) - 50} more")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

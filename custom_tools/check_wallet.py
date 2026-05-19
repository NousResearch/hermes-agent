"""
check_wallet.py - ETH Wallet Balance Checker
=============================================
Functions:
- Check ETH balance of any wallet
- Checksum address validation
- Support multiple chain RPCs
- Read RPC from environment variables

Usage:
    python -m custom_tools.check_wallet 0xYourAddress
    python -m custom_tools.check_wallet 0xYourAddress --chain base
"""

import os
import sys
import json
from web3 import Web3
from web3.exceptions import InvalidAddress


# Chain RPC configuration from environment
CHAIN_RPC_MAP = {
    "ethereum": os.getenv("ETH_RPC_URL", ""),
    "eth": os.getenv("ETH_RPC_URL", ""),
    "base": os.getenv("BASE_RPC_URL", ""),
    "arbitrum": os.getenv("ARB_RPC_URL", ""),
    "arb": os.getenv("ARB_RPC_URL", ""),
    "polygon": os.getenv("POLYGON_RPC_URL", ""),
    "matic": os.getenv("POLYGON_RPC_URL", ""),
}


def get_web3(chain: str = "ethereum") -> Web3:
    """Get Web3 instance for specified chain."""
    rpc_url = CHAIN_RPC_MAP.get(chain.lower(), "")
    if not rpc_url:
        raise ValueError(
            f"No RPC URL configured for chain '{chain}'. "
            f"Set the appropriate environment variable (ETH_RPC_URL, BASE_RPC_URL, etc.)"
        )
    return Web3(Web3.HTTPProvider(rpc_url))


def validate_address(address: str) -> str:
    """Validate and return checksummed address."""
    if not Web3.is_address(address):
        raise InvalidAddress(f"Invalid Ethereum address: {address}")
    return Web3.to_checksum_address(address)


def check_balance(address: str, chain: str = "ethereum") -> dict:
    """
    Check ETH balance of a wallet address.
    
    Args:
        address: Ethereum address (with or without checksum)
        chain: Chain name (ethereum, base, arbitrum, polygon)
    
    Returns:
        dict with address, balance_wei, balance_eth, chain info
    """
    checksummed = validate_address(address)
    w3 = get_web3(chain)
    
    if not w3.is_connected():
        raise ConnectionError(f"Cannot connect to {chain} RPC")
    
    balance_wei = w3.eth.get_balance(checksummed)
    balance_eth = Web3.from_wei(balance_wei, "ether")
    
    return {
        "address": checksummed,
        "chain": chain,
        "balance_wei": str(balance_wei),
        "balance_eth": str(balance_eth),
        "connected": True,
        "block_number": w3.eth.block_number,
    }


def check_multiple_chains(address: str) -> list:
    """Check balance across all configured chains."""
    checksummed = validate_address(address)
    results = []
    
    checked_chains = set()
    for chain_name, rpc_url in CHAIN_RPC_MAP.items():
        if not rpc_url or chain_name in checked_chains:
            continue
        # Skip aliases
        if chain_name in ("eth", "arb", "matic"):
            continue
        checked_chains.add(chain_name)
        
        try:
            result = check_balance(checksummed, chain_name)
            results.append(result)
        except Exception as e:
            results.append({
                "address": checksummed,
                "chain": chain_name,
                "error": str(e),
                "connected": False,
            })
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check ETH wallet balance")
    parser.add_argument("address", help="Ethereum wallet address")
    parser.add_argument("--chain", default="ethereum", help="Chain name (ethereum, base, arbitrum, polygon)")
    parser.add_argument("--all-chains", action="store_true", help="Check all configured chains")
    
    args = parser.parse_args()
    
    try:
        if args.all_chains:
            results = check_multiple_chains(args.address)
            print(json.dumps(results, indent=2))
        else:
            result = check_balance(args.address, args.chain)
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

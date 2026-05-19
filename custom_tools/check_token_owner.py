"""
check_token_owner.py - Token Ownership Checker
================================================
Functions:
- ownerOf(tokenId) check
- Detect minted vs unminted tokens
- Batch check multiple token IDs

Usage:
    python -m custom_tools.check_token_owner 0xContract 1
    python -m custom_tools.check_token_owner 0xContract 1 --chain base
    python -m custom_tools.check_token_owner 0xContract 1-10
"""

import sys
import json
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


def check_token_owner(contract_address: str, token_id: int, chain: str = "ethereum") -> dict:
    """
    Check owner of a specific token ID.
    
    Args:
        contract_address: NFT contract address
        token_id: Token ID to check
        chain: Chain name
    
    Returns:
        dict with token ownership info
    """
    checksummed = validate_address(contract_address)
    w3 = get_web3(chain)
    
    contract = w3.eth.contract(address=checksummed, abi=OWNER_OF_ABI)
    
    try:
        owner = contract.functions.ownerOf(token_id).call()
        return {
            "contract": checksummed,
            "token_id": token_id,
            "owner": owner,
            "is_minted": True,
            "chain": chain,
        }
    except (ContractLogicError, Exception) as e:
        error_msg = str(e)
        # Common revert reasons indicating unminted
        unminted_indicators = [
            "nonexistent token",
            "invalid token",
            "owner query for nonexistent",
            "ERC721: invalid token ID",
            "token does not exist",
        ]
        is_unminted = any(indicator.lower() in error_msg.lower() for indicator in unminted_indicators)
        
        return {
            "contract": checksummed,
            "token_id": token_id,
            "owner": None,
            "is_minted": False if is_unminted else None,
            "chain": chain,
            "error": error_msg if not is_unminted else "Token not minted",
        }


def batch_check_owners(contract_address: str, token_ids: list, chain: str = "ethereum") -> list:
    """Check ownership of multiple token IDs."""
    results = []
    for tid in token_ids:
        result = check_token_owner(contract_address, tid, chain)
        results.append(result)
    return results


def parse_token_range(range_str: str) -> list:
    """Parse token ID range string like '1-10' or '1,2,3'."""
    if "-" in range_str and "," not in range_str:
        parts = range_str.split("-")
        start, end = int(parts[0]), int(parts[1])
        return list(range(start, end + 1))
    elif "," in range_str:
        return [int(x.strip()) for x in range_str.split(",")]
    else:
        return [int(range_str)]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check token ownership")
    parser.add_argument("address", help="NFT contract address")
    parser.add_argument("token_ids", help="Token ID or range (e.g., 1, 1-10, 1,2,3)")
    parser.add_argument("--chain", default="ethereum", help="Chain name")
    
    args = parser.parse_args()
    
    try:
        token_ids = parse_token_range(args.token_ids)
        
        if len(token_ids) == 1:
            result = check_token_owner(args.address, token_ids[0], args.chain)
            print(json.dumps(result, indent=2, default=str))
        else:
            results = batch_check_owners(args.address, token_ids, args.chain)
            print(json.dumps(results, indent=2, default=str))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

"""
nft_contract_check.py - ERC721 Contract Info Checker
=====================================================
Functions:
- Check if contract is ERC721
- Get name, symbol, totalSupply
- Safe fallback handling for non-standard contracts

Usage:
    python -m custom_tools.nft_contract_check 0xContractAddress
    python -m custom_tools.nft_contract_check 0xContractAddress --chain base
"""

import os
import sys
import json
from web3 import Web3
from web3.exceptions import ContractLogicError

from custom_tools.check_wallet import get_web3, validate_address


# Minimal ERC721 ABI for basic checks
ERC721_MINIMAL_ABI = [
    {
        "inputs": [],
        "name": "name",
        "outputs": [{"type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "symbol",
        "outputs": [{"type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "totalSupply",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"type": "bytes4"}],
        "name": "supportsInterface",
        "outputs": [{"type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "owner",
        "outputs": [{"type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "maxSupply",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# ERC721 interface ID
ERC721_INTERFACE_ID = "0x80ac58cd"
# ERC1155 interface ID
ERC1155_INTERFACE_ID = "0xd9b67a26"


def safe_call(contract, method_name: str, *args):
    """Safely call a contract method with fallback."""
    try:
        method = getattr(contract.functions, method_name)
        return method(*args).call()
    except (ContractLogicError, Exception):
        return None


def check_nft_contract(contract_address: str, chain: str = "ethereum") -> dict:
    """
    Check ERC721 contract information.
    
    Args:
        contract_address: Contract address to check
        chain: Chain name
    
    Returns:
        dict with contract info (name, symbol, totalSupply, etc.)
    """
    checksummed = validate_address(contract_address)
    w3 = get_web3(chain)
    
    if not w3.is_connected():
        raise ConnectionError(f"Cannot connect to {chain} RPC")
    
    # Check if address is a contract
    code = w3.eth.get_code(checksummed)
    if code == b"" or code == b"0x":
        return {
            "address": checksummed,
            "chain": chain,
            "is_contract": False,
            "error": "Address is not a contract (EOA)",
        }
    
    contract = w3.eth.contract(address=checksummed, abi=ERC721_MINIMAL_ABI)
    
    # Check ERC721 support via ERC165
    is_erc721 = safe_call(contract, "supportsInterface", bytes.fromhex(ERC721_INTERFACE_ID[2:]))
    is_erc1155 = safe_call(contract, "supportsInterface", bytes.fromhex(ERC1155_INTERFACE_ID[2:]))
    
    # Get basic info
    name = safe_call(contract, "name")
    symbol = safe_call(contract, "symbol")
    total_supply = safe_call(contract, "totalSupply")
    owner = safe_call(contract, "owner")
    max_supply = safe_call(contract, "maxSupply")
    
    result = {
        "address": checksummed,
        "chain": chain,
        "is_contract": True,
        "is_erc721": bool(is_erc721),
        "is_erc1155": bool(is_erc1155),
        "name": name if name else "Unknown",
        "symbol": symbol if symbol else "Unknown",
        "total_supply": total_supply if total_supply is not None else "N/A",
        "max_supply": max_supply if max_supply is not None else "N/A",
        "owner": owner if owner else "N/A",
    }
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check ERC721 NFT contract info")
    parser.add_argument("address", help="Contract address")
    parser.add_argument("--chain", default="ethereum", help="Chain name")
    
    args = parser.parse_args()
    
    try:
        result = check_nft_contract(args.address, args.chain)
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

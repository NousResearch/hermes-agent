"""
contract_analyzer.py - Smart Contract ABI Analyzer
====================================================
Functions:
- Load ABI from Etherscan (if available)
- Fallback to minimal ABI
- Detect common mint functions: mint(), mint(uint256), publicMint(), claim(), safeMint()
- Detect mint price variables: mintPrice, publicPrice, price, cost

Usage:
    python -m custom_tools.contract_analyzer 0xContractAddress
    python -m custom_tools.contract_analyzer 0xContractAddress --chain ethereum
"""

import os
import sys
import json
import requests
from web3 import Web3
from web3.exceptions import ContractLogicError

from custom_tools.check_wallet import get_web3, validate_address


# Etherscan API endpoints per chain
ETHERSCAN_APIS = {
    "ethereum": "https://api.etherscan.io/api",
    "base": "https://api.basescan.org/api",
    "arbitrum": "https://api.arbiscan.io/api",
    "polygon": "https://api.polygonscan.com/api",
}

# Common mint function signatures to detect
KNOWN_MINT_FUNCTIONS = [
    "mint()",
    "mint(uint256)",
    "mint(address,uint256)",
    "mint(uint256,bytes32[])",
    "publicMint()",
    "publicMint(uint256)",
    "claim()",
    "claim(uint256)",
    "safeMint(address)",
    "safeMint(address,uint256)",
    "whitelistMint(uint256,bytes32[])",
    "presaleMint(uint256,bytes32[])",
    "freeMint()",
    "freeMint(uint256)",
]

# Common price variable names
KNOWN_PRICE_VARIABLES = [
    "mintPrice",
    "publicPrice",
    "price",
    "cost",
    "MINT_PRICE",
    "PUBLIC_PRICE",
    "pricePerToken",
    "tokenPrice",
    "salePrice",
]

# Minimal fallback ABI for price checking
PRICE_CHECK_ABI = [
    {
        "inputs": [],
        "name": "mintPrice",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "publicPrice",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "price",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "cost",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]


def fetch_abi_from_etherscan(contract_address: str, chain: str = "ethereum") -> list | None:
    """
    Fetch verified ABI from Etherscan-like explorer.
    
    Args:
        contract_address: Contract address
        chain: Chain name
    
    Returns:
        ABI list or None if not available
    """
    api_key = os.getenv("ETHERSCAN_API_KEY", "")
    api_url = ETHERSCAN_APIS.get(chain)
    
    if not api_url:
        return None
    
    params = {
        "module": "contract",
        "action": "getabi",
        "address": contract_address,
        "apikey": api_key,
    }
    
    try:
        resp = requests.get(api_url, params=params, timeout=10)
        data = resp.json()
        
        if data.get("status") == "1" and data.get("result"):
            abi = json.loads(data["result"])
            return abi
    except Exception:
        pass
    
    return None


def detect_mint_functions(abi: list) -> list:
    """
    Detect mint-related functions from ABI.
    
    Returns list of detected mint functions with details.
    """
    mint_functions = []
    
    for item in abi:
        if item.get("type") != "function":
            continue
        
        name = item.get("name", "")
        
        # Check if function name matches known mint patterns
        is_mint = False
        if "mint" in name.lower() or "claim" in name.lower():
            is_mint = True
        
        if is_mint:
            inputs = item.get("inputs", [])
            input_types = [inp.get("type", "") for inp in inputs]
            state_mutability = item.get("stateMutability", "")
            
            # Build function signature
            sig = f"{name}({','.join(input_types)})"
            
            mint_functions.append({
                "name": name,
                "signature": sig,
                "inputs": inputs,
                "stateMutability": state_mutability,
                "payable": state_mutability in ("payable", "nonpayable"),
                "is_payable": state_mutability == "payable",
            })
    
    return mint_functions


def detect_price_variables(abi: list, contract_address: str, chain: str = "ethereum") -> list:
    """
    Detect and read mint price variables from contract.
    
    Returns list of detected price variables with values.
    """
    w3 = get_web3(chain)
    checksummed = validate_address(contract_address)
    
    price_vars = []
    
    # Look in ABI for view functions that might return price
    view_functions = []
    for item in abi:
        if item.get("type") != "function":
            continue
        if item.get("stateMutability") not in ("view", "pure"):
            continue
        
        name = item.get("name", "")
        outputs = item.get("outputs", [])
        inputs = item.get("inputs", [])
        
        # Check if it's a known price variable (no inputs, returns uint256)
        if (
            name.lower() in [p.lower() for p in KNOWN_PRICE_VARIABLES]
            and len(inputs) == 0
            and len(outputs) == 1
            and outputs[0].get("type") == "uint256"
        ):
            view_functions.append(item)
    
    # Try to call each detected price function
    for func in view_functions:
        name = func["name"]
        try:
            minimal_abi = [func]
            contract = w3.eth.contract(address=checksummed, abi=minimal_abi)
            value = getattr(contract.functions, name)().call()
            price_vars.append({
                "name": name,
                "value_wei": str(value),
                "value_eth": str(Web3.from_wei(value, "ether")),
            })
        except Exception:
            pass
    
    # If no ABI available, try with fallback ABI
    if not price_vars:
        contract = w3.eth.contract(address=checksummed, abi=PRICE_CHECK_ABI)
        for price_name in ["mintPrice", "publicPrice", "price", "cost"]:
            try:
                value = getattr(contract.functions, price_name)().call()
                price_vars.append({
                    "name": price_name,
                    "value_wei": str(value),
                    "value_eth": str(Web3.from_wei(value, "ether")),
                })
            except Exception:
                pass
    
    return price_vars


def analyze_contract(contract_address: str, chain: str = "ethereum") -> dict:
    """
    Full contract analysis: ABI, mint functions, price variables.
    
    Args:
        contract_address: Contract address
        chain: Chain name
    
    Returns:
        dict with full analysis results
    """
    checksummed = validate_address(contract_address)
    
    print(f"\n{'='*60}")
    print(f"  CONTRACT ANALYZER")
    print(f"  Contract: {checksummed}")
    print(f"  Chain: {chain}")
    print(f"{'='*60}\n")
    
    # Step 1: Try to fetch ABI from Etherscan
    print("  [1/3] Fetching ABI from explorer...")
    abi = fetch_abi_from_etherscan(checksummed, chain)
    abi_source = "etherscan" if abi else "fallback"
    
    if not abi:
        print("  -> ABI not verified, using fallback detection")
        abi = PRICE_CHECK_ABI  # Use minimal fallback
    else:
        print(f"  -> ABI loaded ({len(abi)} entries)")
    
    # Step 2: Detect mint functions
    print("  [2/3] Detecting mint functions...")
    mint_functions = detect_mint_functions(abi)
    if mint_functions:
        for mf in mint_functions:
            payable_str = " [PAYABLE]" if mf["is_payable"] else ""
            print(f"    -> {mf['signature']}{payable_str}")
    else:
        print("    -> No mint functions detected in ABI")
    
    # Step 3: Detect price variables
    print("  [3/3] Detecting mint price...")
    price_vars = detect_price_variables(abi, checksummed, chain)
    if price_vars:
        for pv in price_vars:
            print(f"    -> {pv['name']}: {pv['value_eth']} ETH")
    else:
        print("    -> No price variables detected (may be free mint or custom logic)")
    
    print(f"\n{'='*60}\n")
    
    return {
        "contract": checksummed,
        "chain": chain,
        "abi_source": abi_source,
        "abi_entries": len(abi),
        "mint_functions": mint_functions,
        "price_variables": price_vars,
        "detected_mint_count": len(mint_functions),
        "detected_price_count": len(price_vars),
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze NFT contract for mint functions")
    parser.add_argument("address", help="Contract address")
    parser.add_argument("--chain", default="ethereum", help="Chain name")
    parser.add_argument("--json-out", help="Save analysis to JSON file")
    
    args = parser.parse_args()
    
    try:
        result = analyze_contract(args.address, args.chain)
        
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Analysis saved to: {args.json_out}")
        else:
            print(json.dumps(result, indent=2))
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

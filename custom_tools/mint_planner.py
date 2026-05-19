"""
mint_planner.py - Mint Transaction Planner
============================================
Functions:
- Prepare mint transaction
- Estimate gas
- Estimate value (mint price)
- Detect mint function from contract
- Build tx preview
- Risk warning display
- Dry run support (DEFAULT: DRY_RUN=true)

SAFETY:
- NEVER auto-sends transactions
- Always shows full preview before execution
- All transactions require approval via approval_queue

Usage:
    python -m custom_tools.mint_planner 0xContract --wallet burner1
    python -m custom_tools.mint_planner 0xContract --wallet burner1 --quantity 2
"""

import os
import sys
import json
from web3 import Web3

from custom_tools.check_wallet import get_web3, validate_address
from custom_tools.contract_analyzer import (
    analyze_contract,
    fetch_abi_from_etherscan,
    PRICE_CHECK_ABI,
)
from custom_tools.wallet_manager import get_wallet_key, list_wallets



# Safety defaults
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"


def build_mint_transaction(
    contract_address: str,
    wallet_label: str,
    chain: str = "ethereum",
    quantity: int = 1,
    mint_function: str = None,
    mint_price_wei: int = None,
    gas_limit: int = None,
) -> dict:
    """
    Build a mint transaction preview.
    
    Args:
        contract_address: NFT contract address
        wallet_label: Label of wallet to mint from
        chain: Chain name
        quantity: Number of tokens to mint
        mint_function: Override mint function name
        mint_price_wei: Override mint price in wei
        gas_limit: Override gas limit
    
    Returns:
        dict with full transaction preview
    """
    checksummed = validate_address(contract_address)
    w3 = get_web3(chain)
    
    # Get wallet address (NOT the private key for preview)
    from custom_tools.wallet_manager import WALLETS_DIR
    from pathlib import Path
    wallet_file = WALLETS_DIR / f"{wallet_label}.json"
    if not wallet_file.exists():
        raise FileNotFoundError(f"Wallet '{wallet_label}' not found")
    
    with open(wallet_file) as f:
        wallet_data = json.load(f)
    from_address = Web3.to_checksum_address(wallet_data["address"])


    # Analyze contract if needed
    analysis = analyze_contract(checksummed, chain)
    
    # Determine mint function
    if not mint_function:
        if analysis["mint_functions"]:
            # Prefer payable mint functions
            payable_fns = [f for f in analysis["mint_functions"] if f["is_payable"]]
            if payable_fns:
                mint_function = payable_fns[0]["name"]
            else:
                mint_function = analysis["mint_functions"][0]["name"]
        else:
            mint_function = "mint"
    
    # Determine mint price
    if mint_price_wei is None:
        if analysis["price_variables"]:
            mint_price_wei = int(analysis["price_variables"][0]["value_wei"])
        else:
            mint_price_wei = 0
    
    total_value = mint_price_wei * quantity
    
    # Build ABI for the mint function
    abi = fetch_abi_from_etherscan(checksummed, chain) or []
    
    # Find the specific function ABI
    func_abi = None
    for item in abi:
        if item.get("type") == "function" and item.get("name") == mint_function:
            func_abi = item
            break


    # Build transaction data
    if func_abi:
        contract = w3.eth.contract(address=checksummed, abi=[func_abi])
        func = getattr(contract.functions, mint_function)
        
        # Determine args based on function inputs
        inputs = func_abi.get("inputs", [])
        if len(inputs) == 0:
            tx_data = func().build_transaction({
                "from": from_address,
                "value": total_value,
                "nonce": w3.eth.get_transaction_count(from_address),
            })
        elif len(inputs) == 1 and inputs[0]["type"] == "uint256":
            tx_data = func(quantity).build_transaction({
                "from": from_address,
                "value": total_value,
                "nonce": w3.eth.get_transaction_count(from_address),
            })
        else:
            # Generic: try with quantity as first uint256 arg
            tx_data = func(quantity).build_transaction({
                "from": from_address,
                "value": total_value,
                "nonce": w3.eth.get_transaction_count(from_address),
            })
    else:
        # Fallback: encode mint(uint256) manually
        mint_sig = Web3.keccak(text=f"{mint_function}(uint256)")[:4]
        encoded_qty = quantity.to_bytes(32, "big")
        data = mint_sig + encoded_qty
        
        tx_data = {
            "from": from_address,
            "to": checksummed,
            "value": total_value,
            "data": "0x" + data.hex(),
            "nonce": w3.eth.get_transaction_count(from_address),
        }


    # Estimate gas
    try:
        estimated_gas = w3.eth.estimate_gas(tx_data)
    except Exception as e:
        estimated_gas = gas_limit or 200000
    
    if gas_limit:
        tx_data["gas"] = gas_limit
    else:
        tx_data["gas"] = int(estimated_gas * 1.2)  # 20% buffer
    
    # Get gas price
    try:
        gas_price = w3.eth.gas_price
    except Exception:
        gas_price = Web3.to_wei(30, "gwei")
    
    gas_cost_wei = tx_data["gas"] * gas_price
    total_cost_wei = total_value + gas_cost_wei
    
    # Check wallet balance
    balance = w3.eth.get_balance(from_address)
    sufficient_balance = balance >= total_cost_wei
    
    # Build preview
    preview = {
        "status": "DRY_RUN" if DRY_RUN else "READY",
        "contract": checksummed,
        "chain": chain,
        "from_wallet": wallet_label,
        "from_address": from_address,
        "mint_function": mint_function,
        "quantity": quantity,
        "mint_price_per_token": str(Web3.from_wei(mint_price_wei, "ether")) + " ETH",
        "total_value": str(Web3.from_wei(total_value, "ether")) + " ETH",
        "estimated_gas": tx_data["gas"],
        "gas_price_gwei": str(Web3.from_wei(gas_price, "gwei")),
        "gas_cost": str(Web3.from_wei(gas_cost_wei, "ether")) + " ETH",
        "total_cost": str(Web3.from_wei(total_cost_wei, "ether")) + " ETH",
        "wallet_balance": str(Web3.from_wei(balance, "ether")) + " ETH",
        "sufficient_balance": sufficient_balance,
        "nonce": tx_data.get("nonce", 0),
        "tx_data": tx_data,
    }
    
    return preview



def display_tx_preview(preview: dict):
    """Display formatted transaction preview with risk warnings."""
    print(f"\n{'='*60}")
    print(f"  MINT TRANSACTION PREVIEW")
    print(f"  Status: {preview['status']}")
    print(f"{'='*60}")
    print(f"")
    print(f"  Contract:     {preview['contract']}")
    print(f"  Chain:        {preview['chain']}")
    print(f"  Function:     {preview['mint_function']}")
    print(f"  Quantity:     {preview['quantity']}")
    print(f"")
    print(f"  --- COSTS ---")
    print(f"  Mint Price:   {preview['mint_price_per_token']} x {preview['quantity']}")
    print(f"  Total Value:  {preview['total_value']}")
    print(f"  Gas Limit:    {preview['estimated_gas']}")
    print(f"  Gas Price:    {preview['gas_price_gwei']} gwei")
    print(f"  Gas Cost:     {preview['gas_cost']}")
    print(f"  TOTAL COST:   {preview['total_cost']}")
    print(f"")
    print(f"  --- WALLET ---")
    print(f"  Wallet:       {preview['from_wallet']}")
    print(f"  Address:      {preview['from_address']}")
    print(f"  Balance:      {preview['wallet_balance']}")
    print(f"  Sufficient:   {'YES' if preview['sufficient_balance'] else 'NO - INSUFFICIENT FUNDS'}")
    print(f"")
    print(f"  --- RISK WARNINGS ---")
    
    warnings = []
    if not preview["sufficient_balance"]:
        warnings.append("CRITICAL: Insufficient wallet balance!")
    if preview["quantity"] > 5:
        warnings.append("HIGH: Large quantity mint - verify contract limits")
    if "0" in preview["total_value"] and preview["total_value"].startswith("0"):
        warnings.append("INFO: Free mint detected - verify legitimacy")
    if preview["status"] == "DRY_RUN":
        warnings.append("INFO: DRY_RUN mode - transaction will NOT be sent")
    
    warnings.append("ALWAYS: Verify contract is legitimate before minting")
    warnings.append("ALWAYS: Never mint from a contract you haven't reviewed")
    
    for w in warnings:
        print(f"    ! {w}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plan a mint transaction")
    parser.add_argument("address", help="NFT contract address")
    parser.add_argument("--wallet", required=True, help="Wallet label")
    parser.add_argument("--chain", default="ethereum", help="Chain name")
    parser.add_argument("--quantity", type=int, default=1, help="Mint quantity")
    parser.add_argument("--function", help="Override mint function name")
    parser.add_argument("--price-wei", type=int, help="Override price in wei")
    
    args = parser.parse_args()
    
    try:
        preview = build_mint_transaction(
            args.address,
            args.wallet,
            chain=args.chain,
            quantity=args.quantity,
            mint_function=args.function,
            mint_price_wei=args.price_wei,
        )
        display_tx_preview(preview)
        
        if DRY_RUN:
            print("  [DRY_RUN=true] Transaction NOT sent.")
            print("  To execute, set DRY_RUN=false and use approval_queue.")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

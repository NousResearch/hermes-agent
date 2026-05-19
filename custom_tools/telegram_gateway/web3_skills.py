"""
web3_skills.py - Evelyn Web3 Skills Pack for Telegram
======================================================
Commands:
  /contract <address> [chain] - Analyze NFT contract
  /wallet <address> [chain]   - Wallet summary
  /floor <collection>         - Floor price via OpenSea API
  /risk <contract> [chain]    - AI risk analysis

Auto-detection:
  - 0x addresses auto-detected (contract vs wallet)
  - opensea.io links auto-detected for floor check

SAFETY: Read-only blockchain queries. NEVER executes transactions.
"""

import os
import re
import json

import httpx
from web3 import Web3
from web3.exceptions import ContractLogicError

from custom_tools.check_wallet import get_web3, validate_address
from custom_tools.telegram_gateway.ai_chat import get_ai_response

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")
OPENSEA_API_KEY = os.getenv("OPENSEA_API_KEY", "")

ETHERSCAN_APIS = {
    "ethereum": "https://api.etherscan.io/api",
    "base": "https://api.basescan.org/api",
    "arbitrum": "https://api.arbiscan.io/api",
    "polygon": "https://api.polygonscan.com/api",
}

EXPLORERS = {
    "ethereum": "https://etherscan.io/address/",
    "base": "https://basescan.org/address/",
    "arbitrum": "https://arbiscan.io/address/",
    "polygon": "https://polygonscan.com/address/",
}

ERC721_ABI = [
    {"inputs": [], "name": "name", "outputs": [{"type": "string"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "symbol", "outputs": [{"type": "string"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "totalSupply", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [{"type": "bytes4"}], "name": "supportsInterface", "outputs": [{"type": "bool"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "owner", "outputs": [{"type": "address"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "maxSupply", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "mintPrice", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "publicPrice", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "cost", "outputs": [{"type": "uint256"}], "stateMutability": "view", "type": "function"},
]

ERC721_INTERFACE_ID = "0x80ac58cd"
ERC1155_INTERFACE_ID = "0xd9b67a26"


def _safe_call(contract, method_name, *args):
    try:
        return getattr(contract.functions, method_name)(*args).call()
    except Exception:
        return None


async def analyze_contract(address: str, chain: str = "ethereum") -> str:
    """Analyze NFT contract — returns Telegram HTML formatted string."""
    try:
        checksummed = validate_address(address)
    except Exception:
        return "❌ Address invalid sayang. Cek lagi ya."

    try:
        w3 = get_web3(chain)
    except Exception as e:
        return f"❌ Ga bisa connect ke {chain} RPC: {e}"

    try:
        code = w3.eth.get_code(checksummed)
        if code == b"" or code == b"0x":
            return f"❌ <code>{checksummed[:10]}...</code> bukan contract sayang (EOA wallet)."
    except Exception as e:
        return f"❌ Error: {e}"

    contract = w3.eth.contract(address=checksummed, abi=ERC721_ABI)

    name = _safe_call(contract, "name") or "Unknown"
    symbol = _safe_call(contract, "symbol") or "?"
    total_supply = _safe_call(contract, "totalSupply")
    max_supply = _safe_call(contract, "maxSupply")
    owner = _safe_call(contract, "owner")

    is_erc721 = _safe_call(contract, "supportsInterface", bytes.fromhex(ERC721_INTERFACE_ID[2:]))
    is_erc1155 = _safe_call(contract, "supportsInterface", bytes.fromhex(ERC1155_INTERFACE_ID[2:]))

    token_type = "ERC-721" if is_erc721 else ("ERC-1155" if is_erc1155 else "Unknown")

    # Price detection
    price = None
    for fn in ["mintPrice", "publicPrice", "cost"]:
        val = _safe_call(contract, fn)
        if val is not None and val > 0:
            price = val
            break

    # Minted percentage
    minted_pct = ""
    if total_supply is not None and max_supply and max_supply > 0:
        pct = round((total_supply / max_supply) * 100, 1)
        minted_pct = f" ({pct}% minted)"

    # Mint function detection via Etherscan
    mint_fns = []
    if ETHERSCAN_API_KEY:
        try:
            api_url = ETHERSCAN_APIS.get(chain, ETHERSCAN_APIS["ethereum"])
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(api_url, params={
                    "module": "contract", "action": "getabi",
                    "address": checksummed, "apikey": ETHERSCAN_API_KEY,
                })
                data = resp.json()
                if data.get("status") == "1":
                    abi = json.loads(data["result"])
                    for item in abi:
                        if item.get("type") == "function":
                            fn_name = item.get("name", "").lower()
                            if "mint" in fn_name or "claim" in fn_name:
                                payable = "💰" if item.get("stateMutability") == "payable" else ""
                                mint_fns.append(f"{item['name']}{payable}")
        except Exception:
            pass

    # Mint status
    mint_status = "Not detected"
    if mint_fns:
        mint_status = "Mint detected"

    # Price display
    price_str = "Free" if price is None or price == 0 else f"{Web3.from_wei(price, 'ether')} ETH"

    # Risk notes
    risks = []
    if price is None or price == 0:
        risks.append("Free mint detected")
    if owner:
        risks.append("Owner functions present")
    if not mint_fns and ETHERSCAN_API_KEY:
        risks.append("No verified mint function found")

    explorer_url = EXPLORERS.get(chain, EXPLORERS["ethereum"]) + checksummed

    lines = [
        f"🧠 <b>Evelyn Contract Scan</b>",
        f"",
        f"💎 <b>Name:</b> {name} ({symbol})",
        f"⛓ <b>Chain:</b> {chain.capitalize()}",
        f"🧩 <b>Type:</b> {token_type}",
        f"",
        f"📦 <b>Supply:</b>",
        f"  • Current: {total_supply if total_supply is not None else 'N/A'}{minted_pct}",
        f"  • Max: {max_supply if max_supply else 'N/A'}",
        f"",
        f"💰 <b>Mint:</b>",
        f"  • Price: {price_str}",
        f"  • Status: {mint_status}",
    ]

    if mint_fns:
        lines.append(f"  • Functions: {', '.join(mint_fns[:5])}")

    if risks:
        lines.append(f"")
        lines.append(f"⚠️ <b>Risk Notes:</b>")
        for r in risks:
            lines.append(f"  • {r}")

    lines.append(f"")
    lines.append(f"🔗 <a href='{explorer_url}'>Explorer</a>")

    return "\n".join(lines)


async def analyze_wallet(address: str, chain: str = "ethereum") -> str:
    """Analyze wallet — returns Telegram HTML."""
    try:
        checksummed = validate_address(address)
    except Exception:
        return "❌ Address invalid sayang."

    try:
        w3 = get_web3(chain)
    except Exception as e:
        return f"❌ Ga bisa connect ke {chain}: {e}"

    try:
        balance_wei = w3.eth.get_balance(checksummed)
        balance_eth = Web3.from_wei(balance_wei, "ether")
        tx_count = w3.eth.get_transaction_count(checksummed)
        code = w3.eth.get_code(checksummed)
        is_contract = code != b"" and code != b"0x"

        lines = [
            f"👛 <b>Wallet Summary</b>",
            f"",
            f"<b>Address:</b> <code>{checksummed}</code>",
            f"<b>Chain:</b> {chain.capitalize()}",
            f"<b>Type:</b> {'Contract' if is_contract else 'EOA (Wallet)'}",
            f"<b>Balance:</b> {balance_eth:.6f} ETH",
            f"<b>TX Count:</b> {tx_count}",
        ]

        if balance_wei == 0:
            lines.append(f"\n⚠️ Wallet kosong nih sayang")

        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error: {e}"


async def get_floor_price(collection_slug: str) -> str:
    """Get floor price from OpenSea API."""
    headers = {}
    if OPENSEA_API_KEY:
        headers["X-API-KEY"] = OPENSEA_API_KEY

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://api.opensea.io/api/v2/collections/{collection_slug}/stats",
                headers=headers,
            )
            if resp.status_code == 200:
                data = resp.json()
                total = data.get("total", {})
                floor = total.get("floor_price", 0)
                volume = total.get("volume", 0)
                sales = total.get("sales", 0)
                num_owners = total.get("num_owners", 0)
                supply = total.get("supply", 0)

                lines = [
                    f"📊 <b>Floor Price: {collection_slug}</b>",
                    f"",
                    f"<b>Floor:</b> {floor:.4f} ETH" if floor else "<b>Floor:</b> N/A",
                    f"<b>Volume:</b> {volume:.2f} ETH" if volume else "<b>Volume:</b> N/A",
                    f"<b>Sales:</b> {sales}",
                    f"<b>Owners:</b> {num_owners}",
                    f"<b>Supply:</b> {supply}",
                    f"",
                    f"🔗 <a href='https://opensea.io/collection/{collection_slug}'>OpenSea</a>",
                ]
                return "\n".join(lines)
            elif resp.status_code == 404:
                return f"❌ Collection '{collection_slug}' ga ketemu di OpenSea beb."
            else:
                return f"❌ OpenSea API error: {resp.status_code}"
    except httpx.TimeoutException:
        return "❌ OpenSea timeout, coba lagi ntar ya sayang."
    except Exception as e:
        return f"❌ Error: {str(e)[:100]}"


async def analyze_risk(address: str, user_id: int, chain: str = "ethereum") -> str:
    """AI-powered risk analysis of a contract."""
    contract_info = await analyze_contract(address, chain)

    risk_prompt = (
        f"Analyze risiko dari NFT contract ini dan kasih summary singkat:\n\n"
        f"{contract_info}\n\n"
        f"Fokus pada: red flags, mint safety, owner privileges, scam indicators.\n"
        f"Kasih risk level: LOW/MEDIUM/HIGH/CRITICAL.\n"
        f"Jawab 3-5 kalimat, bahasa Indonesia casual."
    )

    ai_analysis = await get_ai_response(user_id, risk_prompt)

    return f"{contract_info}\n\n━━━━━━━━━━━━━━━\n🤖 <b>Evelyn's Take:</b>\n{ai_analysis}"


# === Auto-Detection Helpers ===

def detect_address(text: str) -> str | None:
    match = re.search(r'0x[a-fA-F0-9]{40}', text)
    return match.group(0) if match else None


def detect_opensea_slug(text: str) -> str | None:
    match = re.search(r'opensea\.io/collection/([a-zA-Z0-9\-]+)', text)
    if match:
        return match.group(1)
    return None


def detect_chain_from_text(text: str) -> str:
    t = text.lower()
    if "base" in t:
        return "base"
    elif "arb" in t or "arbitrum" in t:
        return "arbitrum"
    elif "polygon" in t or "matic" in t:
        return "polygon"
    return "ethereum"

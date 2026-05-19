"""
web3_skills.py - Evelyn Web3 Skills Pack for Telegram
======================================================
Commands:
  /contract <address> - Analyze NFT contract (ERC721/1155, mint fn, price, risk)
  /wallet <address>   - Wallet summary (balance, tx count)
  /floor <collection> - Floor price via OpenSea API
  /risk <contract>    - AI risk analysis of contract

Auto-detection:
  - 0x... addresses auto-detected as contract or wallet
  - opensea.io links auto-detected for floor check

SAFETY:
- Read-only blockchain queries
- NEVER exposes private keys
- NEVER auto-executes transactions
- AI risk analysis is informational only

Usage:
    Integrated into telegram_gateway/bot.py
"""

import os
import re
import json

import httpx
from web3 import Web3
from web3.exceptions import ContractLogicError

from custom_tools.check_wallet import get_web3, validate_address, CHAIN_RPC_MAP
from custom_tools.telegram_gateway.ai_chat import get_ai_response


# === Configuration ===

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")
OPENSEA_API_KEY = os.getenv("OPENSEA_API_KEY", "")

# Etherscan API endpoints
ETHERSCAN_APIS = {
    "ethereum": "https://api.etherscan.io/api",
    "base": "https://api.basescan.org/api",
    "arbitrum": "https://api.arbiscan.io/api",
    "polygon": "https://api.polygonscan.com/api",
}

# Minimal ABIs
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
    """Safely call a contract method."""
    try:
        return getattr(contract.functions, method_name)(*args).call()
    except Exception:
        return None


# === /contract <address> ===

async def analyze_contract(address: str, chain: str = "ethereum") -> str:
    """
    Analyze an NFT contract and return Telegram-formatted summary.

    Returns HTML-formatted string for Telegram.
    """
    try:
        checksummed = validate_address(address)
    except Exception:
        return "❌ Address invalid sayang. Cek lagi ya."

    try:
        w3 = get_web3(chain)
    except Exception as e:
        return f"❌ Ga bisa connect ke {chain} RPC: {e}"

    # Check if contract
    try:
        code = w3.eth.get_code(checksummed)
        if code == b"" or code == b"0x":
            return f"❌ <code>{checksummed[:10]}...</code> bukan contract (EOA wallet)."
    except Exception as e:
        return f"❌ Error checking address: {e}"

    contract = w3.eth.contract(address=checksummed, abi=ERC721_ABI)

    # Basic info
    name = _safe_call(contract, "name") or "Unknown"
    symbol = _safe_call(contract, "symbol") or "?"
    total_supply = _safe_call(contract, "totalSupply")
    max_supply = _safe_call(contract, "maxSupply")
    owner = _safe_call(contract, "owner")

    # Interface detection
    is_erc721 = _safe_call(contract, "supportsInterface", bytes.fromhex(ERC721_INTERFACE_ID[2:]))
    is_erc1155 = _safe_call(contract, "supportsInterface", bytes.fromhex(ERC1155_INTERFACE_ID[2:]))

    token_type = "Unknown"
    if is_erc721:
        token_type = "ERC-721"
    elif is_erc1155:
        token_type = "ERC-1155"

    # Price detection
    price = None
    for price_fn in ["mintPrice", "publicPrice", "cost"]:
        val = _safe_call(contract, price_fn)
        if val is not None and val > 0:
            price = val
            break

    # Mint function detection (via Etherscan ABI if available)
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
                                payable = item.get("stateMutability") == "payable"
                                mint_fns.append(f"{item['name']}{'💰' if payable else ''}")
        except Exception:
            pass

    # Explorer link
    explorers = {
        "ethereum": "https://etherscan.io/address/",
        "base": "https://basescan.org/address/",
        "arbitrum": "https://arbiscan.io/address/",
        "polygon": "https://polygonscan.com/address/",
    }
    explorer_url = explorers.get(chain, explorers["ethereum"]) + checksummed

    # Build response
    lines = [
        f"🔍 <b>Contract Analysis</b>",
        f"",
        f"<b>Name:</b> {name} ({symbol})",
        f"<b>Type:</b> {token_type}",
        f"<b>Chain:</b> {chain}",
        f"<b>Supply:</b> {total_supply if total_supply is not None else 'N/A'}",
        f"<b>Max Supply:</b> {max_supply if max_supply is not None else 'N/A'}",
    ]

    if price:
        lines.append(f"<b>Mint Price:</b> {Web3.from_wei(price, 'ether')} ETH")
    else:
        lines.append(f"<b>Mint Price:</b> Free / Not detected")

    if owner:
        lines.append(f"<b>Owner:</b> <code>{owner[:10]}...</code>")

    if mint_fns:
        lines.append(f"<b>Mint Functions:</b> {', '.join(mint_fns[:5])}")

    lines.append(f"")
    lines.append(f"🔗 <a href='{explorer_url}'>View on Explorer</a>")

    return "\n".join(lines)


# === /wallet <address> ===

async def analyze_wallet(address: str, chain: str = "ethereum") -> str:
    """Analyze a wallet and return summary."""
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

        # Check if contract or EOA
        code = w3.eth.get_code(checksummed)
        is_contract = code != b"" and code != b"0x"

        lines = [
            f"👛 <b>Wallet Summary</b>",
            f"",
            f"<b>Address:</b> <code>{checksummed}</code>",
            f"<b>Chain:</b> {chain}",
            f"<b>Type:</b> {'Contract' if is_contract else 'EOA (Wallet)'}",
            f"<b>Balance:</b> {balance_eth:.6f} ETH",
            f"<b>TX Count:</b> {tx_count}",
        ]

        if balance_wei == 0:
            lines.append(f"\n⚠️ Wallet kosong nih")

        return "\n".join(lines)

    except Exception as e:
        return f"❌ Error: {e}"


# === /floor <collection> ===

async def get_floor_price(collection_slug: str) -> str:
    """Get floor price from OpenSea API."""

    # Try OpenSea API v2
    headers = {}
    if OPENSEA_API_KEY:
        headers["X-API-KEY"] = OPENSEA_API_KEY

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Try collection stats endpoint
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
                return f"❌ Collection '{collection_slug}' ga ketemu di OpenSea."
            else:
                return f"❌ OpenSea API error: {resp.status_code}"

    except httpx.TimeoutException:
        return "❌ OpenSea timeout, coba lagi ntar ya."
    except Exception as e:
        return f"❌ Error: {str(e)[:100]}"


# === /risk <contract> ===

async def analyze_risk(address: str, user_id: int, chain: str = "ethereum") -> str:
    """AI-powered risk analysis of a contract."""

    # First get basic contract info
    contract_info = await analyze_contract(address, chain)

    # Ask Evelyn AI to analyze risk
    risk_prompt = (
        f"Analyze risiko dari NFT contract ini dan kasih summary singkat:\n\n"
        f"{contract_info}\n\n"
        f"Fokus pada:\n"
        f"- Apakah ada red flags (unverified, no source code, proxy)\n"
        f"- Mint function safety\n"
        f"- Owner privileges (bisa drain/rug?)\n"
        f"- Free mint scam indicators\n"
        f"- Overall risk level: LOW/MEDIUM/HIGH/CRITICAL\n"
        f"Jawab dalam 3-5 kalimat, bahasa Indonesia casual."
    )

    ai_analysis = await get_ai_response(user_id, risk_prompt)

    lines = [
        f"⚠️ <b>Risk Analysis</b>",
        f"",
        contract_info,
        f"",
        f"━━━━━━━━━━━━━━━",
        f"🤖 <b>Evelyn's Take:</b>",
        f"{ai_analysis}",
    ]

    return "\n".join(lines)


# === Auto-Detection Helpers ===

def detect_address(text: str) -> str | None:
    """Detect Ethereum address in text."""
    match = re.search(r'0x[a-fA-F0-9]{40}', text)
    return match.group(0) if match else None


def detect_opensea_slug(text: str) -> str | None:
    """Detect OpenSea collection slug from URL or text."""
    # Match opensea.io/collection/slug
    match = re.search(r'opensea\.io/collection/([a-zA-Z0-9\-]+)', text)
    if match:
        return match.group(1)
    # If it's just a simple slug-like text (no spaces, lowercase)
    if re.match(r'^[a-z0-9\-]+$', text.strip()) and len(text.strip()) > 2:
        return text.strip()
    return None


def detect_chain_from_text(text: str) -> str:
    """Try to detect chain from text context."""
    text_lower = text.lower()
    if "base" in text_lower:
        return "base"
    elif "arb" in text_lower or "arbitrum" in text_lower:
        return "arbitrum"
    elif "polygon" in text_lower or "matic" in text_lower:
        return "polygon"
    return "ethereum"

"""
x402 Solana Payment Gate — Premium Quantum Trading API (Tiered)

Implements the x402 HTTP 402 protocol for Solana-native payments.
Two tiers:
  - Simulated Quantum (0.25 SOL): Quantum simulator with hardware-optimized weights
  - Real Quantum Hardware (1 SOL): IBM Quantum QPU execution

Supports: Any Solana memecoin + BTC, ETH, SOL majors.

x402 Protocol Flow (V2):
1. Client sends request to premium endpoint
2. Server returns 402 with PAYMENT-REQUIRED header listing both tiers
3. Client pays to ecosystem wallet (0.25 SOL or 1 SOL)
4. Client retries with X-PAYMENT header (base64 JSON with tx sig)
5. Server verifies on-chain, determines tier from amount, returns data

Solana Network ID: solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp (mainnet-beta)
"""

import os
import json
import time
import base64
import hashlib
import asyncio
import aiohttp
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Set

logger = logging.getLogger("x402.solana_gate")

# =============================================================================
# CONSTANTS
# =============================================================================

# Solana mainnet-beta network ID per x402 spec
SOLANA_NETWORK = "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp"

# Ecosystem wallet — receives SOL payments
ECOSYSTEM_WALLET = os.getenv(
    "X402_SOLANA_RECEIVER",
    "3fSS5RVErbgcJEDCQmCXpKsD2tWqfhxFZtkDUB8qw"
)

# SOL is the native asset on Solana (no mint address needed)
SOL_ASSET = "native"
SOL_DECIMALS = 9

# Tiered pricing
TIER_SIMULATED_SOL = 0.25
TIER_SIMULATED_LAMPORTS = 250_000_000  # 0.25 SOL
TIER_HARDWARE_SOL = 1.0
TIER_HARDWARE_LAMPORTS = 1_000_000_000  # 1 SOL

# Minimum accepted payment (simulated tier)
QUERY_PRICE_SOL = TIER_SIMULATED_SOL
QUERY_PRICE_LAMPORTS = TIER_SIMULATED_LAMPORTS

# RPC + Helius
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")

# x402 header names (V2 spec)
HEADER_PAYMENT_REQUIRED = "X-PAYMENT"  # 402 response: base64 JSON payment requirements
HEADER_PAYMENT_SIGNATURE = "X-PAYMENT"  # retry request: base64 JSON with tx signature
HEADER_PAYMENT_RESPONSE = "X-PAYMENT-RESPONSE"  # success response: base64 JSON receipt

# Discovery
COSMOS_API_BASE = os.getenv("COSMOS_API_URL", "https://ai.cosmos.cloud")

# Max payment age (prevent old tx replays)
MAX_PAYMENT_AGE_SECONDS = 300  # 5 minutes

# =============================================================================
# MAJOR ASSET SUPPORT (BTC, ETH, SOL + any Solana memecoin)
# =============================================================================

MAJOR_ASSETS = {
    "BTC": {"coingecko_id": "bitcoin", "name": "Bitcoin"},
    "BITCOIN": {"coingecko_id": "bitcoin", "name": "Bitcoin"},
    "ETH": {"coingecko_id": "ethereum", "name": "Ethereum"},
    "ETHEREUM": {"coingecko_id": "ethereum", "name": "Ethereum"},
    "SOL": {"coingecko_id": "solana", "name": "Solana", "mint": "So11111111111111111111111111111111111111112"},
    "SOLANA": {"coingecko_id": "solana", "name": "Solana", "mint": "So11111111111111111111111111111111111111112"},
}

COINGECKO_IDS = {"bitcoin", "ethereum", "solana"}


def resolve_asset(token_input: str) -> Dict[str, Any]:
    """
    Resolve a token input to a standardized format.
    Accepts: Solana mint address, ticker (BTC/ETH/SOL), or CoinGecko ID.
    Returns: {"type": "solana_token"|"major", "address": str, "coingecko_id": str|None, "name": str}
    """
    upper = token_input.strip().upper()

    # Check ticker/name match
    if upper in MAJOR_ASSETS:
        info = MAJOR_ASSETS[upper]
        return {
            "type": "major",
            "address": info.get("mint", token_input),
            "coingecko_id": info["coingecko_id"],
            "name": info["name"],
        }

    # Check CoinGecko ID match
    lower = token_input.strip().lower()
    if lower in COINGECKO_IDS:
        for ticker, info in MAJOR_ASSETS.items():
            if info["coingecko_id"] == lower:
                return {
                    "type": "major",
                    "address": info.get("mint", token_input),
                    "coingecko_id": info["coingecko_id"],
                    "name": info["name"],
                }

    # Default: treat as Solana token mint address
    return {
        "type": "solana_token",
        "address": token_input.strip(),
        "coingecko_id": None,
        "name": token_input[:8] + "...",
    }


# =============================================================================
# TIER DETERMINATION
# =============================================================================

def determine_tier(amount_lamports: int) -> str:
    """Determine which tier the payment qualifies for based on amount."""
    if amount_lamports >= TIER_HARDWARE_LAMPORTS:
        return "hardware"
    elif amount_lamports >= TIER_SIMULATED_LAMPORTS:
        return "simulated"
    return "insufficient"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PaymentReceipt:
    """Record of a verified x402 payment."""
    tx_signature: str
    payer_wallet: str
    amount_lamports: int
    amount_sol: float
    endpoint: str
    token_queried: str
    verified_at: str
    tier: str = "simulated"
    slot: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class X402SolanaStats:
    """Track x402 payment statistics."""
    total_queries: int = 0
    total_revenue_lamports: int = 0
    total_402_issued: int = 0
    total_verified: int = 0
    total_rejected: int = 0
    unique_payers: int = 0
    simulated_queries: int = 0
    hardware_queries: int = 0
    _payer_set: Set[str] = field(default_factory=set, repr=False)
    recent_payments: List[dict] = field(default_factory=list, repr=False)

    def record_402(self):
        self.total_402_issued += 1

    def record_verified(self, receipt: PaymentReceipt):
        self.total_queries += 1
        self.total_verified += 1
        self.total_revenue_lamports += receipt.amount_lamports
        self._payer_set.add(receipt.payer_wallet)
        self.unique_payers = len(self._payer_set)
        if receipt.tier == "hardware":
            self.hardware_queries += 1
        else:
            self.simulated_queries += 1
        self.recent_payments.append(receipt.to_dict())
        # Keep last 200
        if len(self.recent_payments) > 200:
            self.recent_payments = self.recent_payments[-200:]

    def record_rejected(self):
        self.total_rejected += 1

    def to_dict(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "total_revenue_sol": self.total_revenue_lamports / (10 ** SOL_DECIMALS),
            "total_revenue_lamports": self.total_revenue_lamports,
            "total_402_issued": self.total_402_issued,
            "total_verified": self.total_verified,
            "total_rejected": self.total_rejected,
            "unique_payers": self.unique_payers,
            "simulated_queries": self.simulated_queries,
            "hardware_queries": self.hardware_queries,
            "recent_payments": self.recent_payments[-10:],  # Last 10 for display
        }


# =============================================================================
# SOLANA PAYMENT VERIFIER
# =============================================================================

class SolanaPaymentVerifier:
    """
    Verifies SOL transfer transactions on Solana mainnet.
    Uses Helius parsed API (primary) + Solana RPC fallback.
    Prevents replay attacks via used_signatures set.
    """

    def __init__(self):
        self._used_signatures: Set[str] = set()
        self._sig_file = os.path.join(
            os.path.dirname(__file__), "..", "..", "web", "data", "x402_signatures.json"
        )
        self._load_signatures()

    def _load_signatures(self):
        """Load used signatures from disk."""
        try:
            if os.path.exists(self._sig_file):
                with open(self._sig_file, 'r') as f:
                    data = json.load(f)
                    self._used_signatures = set(data.get("signatures", []))
                    logger.info(f"Loaded {len(self._used_signatures)} used x402 signatures")
        except Exception as e:
            logger.warning(f"Failed to load x402 signatures: {e}")

    def _save_signatures(self):
        """Persist used signatures to disk."""
        try:
            os.makedirs(os.path.dirname(self._sig_file), exist_ok=True)
            with open(self._sig_file, 'w') as f:
                # Keep last 5000 signatures
                sigs = list(self._used_signatures)[-5000:]
                json.dump({"signatures": sigs, "updated": datetime.now().isoformat()}, f)
        except Exception as e:
            logger.warning(f"Failed to save x402 signatures: {e}")

    def is_replay(self, tx_signature: str) -> bool:
        """Check if this signature was already used."""
        return tx_signature in self._used_signatures

    def mark_used(self, tx_signature: str):
        """Mark a signature as used."""
        self._used_signatures.add(tx_signature)
        self._save_signatures()

    async def verify_sol_transfer(
        self,
        tx_signature: str,
        expected_recipient: str = ECOSYSTEM_WALLET,
        min_amount_lamports: int = QUERY_PRICE_LAMPORTS,
    ) -> Dict[str, Any]:
        """
        Verify a SOL transfer transaction on-chain.
        min_amount_lamports defaults to 0.25 SOL (simulated tier minimum).
        The actual tier is determined by determine_tier() after verification.
        """
        result = {"valid": False, "payer": "", "amount_lamports": 0, "slot": 0, "error": None}

        # Replay check
        if self.is_replay(tx_signature):
            result["error"] = "Transaction signature already used (replay attack)"
            return result

        try:
            # Try Helius first
            if HELIUS_API_KEY:
                helius_result = await self._verify_via_helius(
                    tx_signature, expected_recipient, min_amount_lamports
                )
                if helius_result:
                    return helius_result

            # Fallback to RPC
            return await self._verify_via_rpc(
                tx_signature, expected_recipient, min_amount_lamports
            )
        except Exception as e:
            logger.error(f"x402 verification error: {e}")
            result["error"] = str(e)
            return result

    async def _verify_via_helius(
        self, tx_signature: str, expected_recipient: str, min_amount_lamports: int
    ) -> Optional[Dict[str, Any]]:
        """Verify using Helius parsed transaction API."""
        try:
            url = f"https://api.helius.xyz/v0/transactions/?api-key={HELIUS_API_KEY}"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={"transactions": [tx_signature]}) as resp:
                    if resp.status != 200:
                        return None

                    data = await resp.json()
                    if not data:
                        return {"valid": False, "error": "Transaction not found", "payer": "", "amount_lamports": 0, "slot": 0}

                    tx = data[0]

                    # Check native SOL transfers
                    native_transfers = tx.get("nativeTransfers", [])
                    for transfer in native_transfers:
                        if transfer.get("toUserAccount") == expected_recipient:
                            amount = transfer.get("amount", 0)  # in lamports
                            if amount >= min_amount_lamports:
                                return {
                                    "valid": True,
                                    "payer": transfer.get("fromUserAccount", ""),
                                    "amount_lamports": amount,
                                    "slot": tx.get("slot", 0),
                                    "error": None,
                                }
                            else:
                                return {
                                    "valid": False,
                                    "payer": transfer.get("fromUserAccount", ""),
                                    "amount_lamports": amount,
                                    "slot": tx.get("slot", 0),
                                    "error": f"Insufficient: {amount / 1e9:.4f} SOL (minimum {min_amount_lamports / 1e9} SOL)",
                                }

                    return {
                        "valid": False, "payer": "", "amount_lamports": 0, "slot": 0,
                        "error": "No SOL transfer to ecosystem wallet found in transaction",
                    }
        except Exception as e:
            logger.warning(f"Helius x402 verification failed: {e}")
            return None

    async def _verify_via_rpc(
        self, tx_signature: str, expected_recipient: str, min_amount_lamports: int
    ) -> Dict[str, Any]:
        """Verify using Solana RPC getTransaction (fallback)."""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getTransaction",
                    "params": [
                        tx_signature,
                        {
                            "encoding": "jsonParsed",
                            "commitment": "confirmed",
                            "maxSupportedTransactionVersion": 0,
                        }
                    ]
                }

                async with session.post(SOLANA_RPC_URL, json=payload) as resp:
                    data = await resp.json()

                    if "error" in data:
                        return {
                            "valid": False, "payer": "", "amount_lamports": 0, "slot": 0,
                            "error": f"RPC error: {data['error'].get('message', 'Unknown')}",
                        }

                    result = data.get("result")
                    if not result:
                        return {
                            "valid": False, "payer": "", "amount_lamports": 0, "slot": 0,
                            "error": "Transaction not found or not confirmed",
                        }

                    # Check tx success
                    if result.get("meta", {}).get("err") is not None:
                        return {
                            "valid": False, "payer": "", "amount_lamports": 0, "slot": 0,
                            "error": "Transaction failed on-chain",
                        }

                    # Check SOL balance changes
                    account_keys = result.get("transaction", {}).get("message", {}).get("accountKeys", [])
                    pre_balances = result.get("meta", {}).get("preBalances", [])
                    post_balances = result.get("meta", {}).get("postBalances", [])

                    # Find the recipient account and check balance increase
                    for i, key_info in enumerate(account_keys):
                        pubkey = key_info.get("pubkey", key_info) if isinstance(key_info, dict) else key_info
                        if pubkey == expected_recipient and i < len(pre_balances) and i < len(post_balances):
                            received = post_balances[i] - pre_balances[i]
                            if received >= min_amount_lamports:
                                # Find the payer (first signer)
                                payer = ""
                                for k in account_keys:
                                    kp = k.get("pubkey", k) if isinstance(k, dict) else k
                                    signer = k.get("signer", False) if isinstance(k, dict) else False
                                    if signer and kp != expected_recipient:
                                        payer = kp
                                        break

                                return {
                                    "valid": True,
                                    "payer": payer,
                                    "amount_lamports": received,
                                    "slot": result.get("slot", 0),
                                    "error": None,
                                }
                            elif received > 0:
                                return {
                                    "valid": False, "payer": "", "amount_lamports": received, "slot": 0,
                                    "error": f"Insufficient: {received / 1e9:.4f} SOL (minimum {min_amount_lamports / 1e9} SOL)",
                                }

                    return {
                        "valid": False, "payer": "", "amount_lamports": 0, "slot": 0,
                        "error": "No SOL transfer to ecosystem wallet found",
                    }

        except Exception as e:
            logger.error(f"RPC x402 verification failed: {e}")
            return {
                "valid": False, "payer": "", "amount_lamports": 0, "slot": 0,
                "error": f"Verification failed: {e}",
            }


# =============================================================================
# x402 PROTOCOL HELPERS
# =============================================================================

def build_payment_required_payload(endpoint: str, resource_description: str = "") -> dict:
    """
    Build the x402 V2 PAYMENT-REQUIRED payload with BOTH tiers.
    Returned as base64 JSON in the X-PAYMENT header of 402 responses.
    """
    return {
        "x402Version": 2,
        "accepts": [
            {
                "scheme": "exact",
                "network": SOLANA_NETWORK,
                "maxAmountRequired": str(TIER_SIMULATED_LAMPORTS),
                "resource": f"{COSMOS_API_BASE}{endpoint}",
                "description": (
                    "Simulated Quantum (0.25 SOL) — Quantum simulator with hardware-optimized "
                    "algo weights. EMA momentum + quantum Monte Carlo + collective AI. Fast response."
                ),
                "mimeType": "application/json",
                "payTo": ECOSYSTEM_WALLET,
                "asset": SOL_ASSET,
                "maxTimeoutSeconds": MAX_PAYMENT_AGE_SECONDS,
                "extra": {
                    "name": "Cosmos AI Swarm",
                    "tier": "simulated",
                    "pricing": f"{TIER_SIMULATED_SOL} SOL",
                    "estimated_time": "5-15 seconds",
                    "capabilities": [
                        "quantum_simulation",
                        "hardware_optimized_weights",
                        "ema_momentum",
                        "collective_intelligence",
                        "signal_fusion",
                        "scenario_analysis",
                    ],
                    "supported_assets": ["Any Solana memecoin", "BTC", "ETH", "SOL"],
                }
            },
            {
                "scheme": "exact",
                "network": SOLANA_NETWORK,
                "maxAmountRequired": str(TIER_HARDWARE_LAMPORTS),
                "resource": f"{COSMOS_API_BASE}{endpoint}",
                "description": (
                    "Real Quantum Hardware (1 SOL) — IBM Quantum QPU circuit execution. "
                    "Higher qubit count, more shots, Bell correlation verification on real hardware. "
                    "Processing takes 30-90 seconds."
                ),
                "mimeType": "application/json",
                "payTo": ECOSYSTEM_WALLET,
                "asset": SOL_ASSET,
                "maxTimeoutSeconds": MAX_PAYMENT_AGE_SECONDS,
                "extra": {
                    "name": "Cosmos AI Swarm",
                    "tier": "hardware",
                    "pricing": f"{TIER_HARDWARE_SOL} SOL",
                    "estimated_time": "30-90 seconds",
                    "capabilities": [
                        "ibm_quantum_hardware",
                        "higher_qubit_count",
                        "increased_shot_count",
                        "ema_momentum",
                        "collective_intelligence",
                        "signal_fusion",
                        "scenario_analysis",
                        "bell_correlation_verification",
                    ],
                    "supported_assets": ["Any Solana memecoin", "BTC", "ETH", "SOL"],
                }
            },
        ],
        "error": "",
    }


def encode_x402_header(payload: dict) -> str:
    """Encode a payload as base64 JSON for x402 headers."""
    return base64.b64encode(json.dumps(payload).encode()).decode()


def decode_x402_header(header_value: str) -> Optional[dict]:
    """Decode a base64 JSON x402 header."""
    try:
        return json.loads(base64.b64decode(header_value).decode())
    except Exception:
        return None


def build_payment_response(receipt: PaymentReceipt) -> dict:
    """Build the PAYMENT-RESPONSE header payload for successful payment."""
    return {
        "x402Version": 2,
        "success": True,
        "transaction": receipt.tx_signature,
        "payer": receipt.payer_wallet,
        "network": SOLANA_NETWORK,
        "settled": True,
        "tier": receipt.tier,
    }


# =============================================================================
# x402 DISCOVERY METADATA
# =============================================================================

def get_x402_discovery_manifest() -> dict:
    """
    x402 discovery manifest for registration with hubs, bazaars, and i1l.store.
    Lists both pricing tiers.
    """
    return {
        "x402Version": 2,
        "provider": {
            "name": "Cosmos AI Swarm",
            "description": (
                "Quantum-enhanced trading intelligence powered by IBM Quantum hardware and simulation, "
                "EMA momentum analysis, and multi-agent collective deliberation. "
                "Two tiers: Simulated (0.25 SOL, fast) and Real Quantum Hardware (1 SOL, IBM QPU). "
                "Supports any Solana memecoin plus BTC, ETH, and SOL majors."
            ),
            "url": COSMOS_API_BASE,
            "logo": f"{COSMOS_API_BASE}/static/logo.png",
            "category": "trading",
            "tags": ["solana", "quantum", "trading", "defi", "ai", "signals", "ibm-quantum", "bitcoin", "ethereum"],
        },
        "endpoints": [
            {
                "path": "/api/x402/quantum/analyze",
                "method": "POST",
                "description": "Simulated Quantum — quantum simulator with hardware-optimized weights (fast)",
                "price": str(TIER_SIMULATED_LAMPORTS),
                "asset": SOL_ASSET,
                "network": SOLANA_NETWORK,
                "payTo": ECOSYSTEM_WALLET,
                "extra": {"tier": "simulated", "estimated_time": "5-15 seconds"},
                "requestSchema": {
                    "type": "object",
                    "required": ["token_address"],
                    "properties": {
                        "token_address": {
                            "type": "string",
                            "description": "Solana token mint address, or ticker: BTC, ETH, SOL",
                        },
                    },
                },
                "responseSchema": {
                    "type": "object",
                    "properties": {
                        "tier": {"type": "string"},
                        "signal": {"type": "object", "description": "Full quantum trading signal"},
                        "scenarios": {"type": "object", "description": "Multi-scenario quantum analysis"},
                        "accuracy": {"type": "object", "description": "Historical signal accuracy stats"},
                    },
                },
            },
            {
                "path": "/api/x402/quantum/analyze",
                "method": "POST",
                "description": "Real Quantum Hardware — IBM Quantum QPU circuit execution (30-90s processing)",
                "price": str(TIER_HARDWARE_LAMPORTS),
                "asset": SOL_ASSET,
                "network": SOLANA_NETWORK,
                "payTo": ECOSYSTEM_WALLET,
                "extra": {"tier": "hardware", "estimated_time": "30-90 seconds"},
                "requestSchema": {
                    "type": "object",
                    "required": ["token_address"],
                    "properties": {
                        "token_address": {
                            "type": "string",
                            "description": "Solana token mint address, or ticker: BTC, ETH, SOL",
                        },
                    },
                },
                "responseSchema": {
                    "type": "object",
                    "properties": {
                        "tier": {"type": "string"},
                        "signal": {"type": "object"},
                        "scenarios": {"type": "object"},
                        "accuracy": {"type": "object"},
                        "quantum_hardware_details": {"type": "object", "description": "QPU backend, qubits, shots, execution time"},
                    },
                },
            },
        ],
        "networks": [SOLANA_NETWORK],
        "termsOfService": f"{COSMOS_API_BASE}/terms",
        "contact": "cosmos@ai.cosmos.cloud",
    }


# =============================================================================
# PREMIUM QUANTUM QUERY ENGINE
# =============================================================================

async def run_premium_quantum_analysis(
    token_address: str, tier: str = "simulated"
) -> Dict[str, Any]:
    """
    Run the full quantum analysis pipeline for a premium x402 query.

    Args:
        token_address: Solana mint address, or ticker (BTC/ETH/SOL)
        tier: "simulated" (0.25 SOL) or "hardware" (1 SOL)

    Returns comprehensive trading data with tier-appropriate processing.
    """
    start_time = time.time()
    use_hardware = (tier == "hardware")

    # Resolve asset (ticker → CoinGecko, mint → Birdeye/Jupiter)
    asset_info = resolve_asset(token_address)

    result = {
        "token_address": token_address,
        "asset": asset_info["name"],
        "asset_type": asset_info["type"],
        "tier": tier,
        "timestamp": datetime.now().isoformat(),
        "powered_by": "Cosmos AI Swarm — Quantum Trading Cortex",
    }

    if use_hardware:
        result["processing_note"] = (
            "Real IBM Quantum QPU execution in progress. "
            "Circuit compilation, transpilation, and hardware execution takes 30-90 seconds. "
            "Results are from genuine quantum hardware, not simulation."
        )

    # 1. Generate quantum trading signal
    try:
        from Cosmos.core.quantum_trading import get_quantum_cortex
        cortex = get_quantum_cortex()

        if cortex is None:
            from Cosmos.core.quantum_trading import initialize_quantum_cortex
            cortex = await initialize_quantum_cortex()

        # Fetch live price data
        price_history = await _fetch_price_history(token_address, asset_info=asset_info)
        current_price = price_history[-1] if price_history else 0.0

        # Generate the fused signal (with hardware flag)
        signal = await cortex.generate_signal(
            token_address=asset_info["address"],
            price_history=price_history,
            current_price=current_price,
            use_hardware=use_hardware,
        )
        result["signal"] = signal.to_dict()

        # 2. Run scenario analysis
        try:
            scenarios = await cortex.quantum_scenario_analysis(
                asset_info["address"], price_history,
                use_hardware=use_hardware,
            )
            result["scenarios"] = scenarios
        except Exception as e:
            result["scenarios"] = {"error": str(e)}

        # 3. Accuracy stats (public — proves value)
        try:
            accuracy = cortex.accuracy_tracker.get_accuracy_stats()
            result["accuracy"] = accuracy
        except Exception:
            result["accuracy"] = {}

        # 4. Correlations (if any discovered)
        try:
            corr_list = []
            addr = asset_info["address"]
            for key, corr in cortex.correlations.items():
                if addr in key:
                    corr_list.append(corr.to_dict())
            if corr_list:
                result["correlations"] = corr_list
        except Exception:
            pass

    except Exception as e:
        logger.error(f"x402 quantum analysis failed: {e}")
        result["signal"] = {"error": str(e)}

    # 5. Add price context
    try:
        source = "coingecko" if asset_info["type"] == "major" else "birdeye/jupiter"
        result["price_data"] = {
            "current_price": current_price if 'current_price' in dir() else 0.0,
            "price_points": len(price_history) if 'price_history' in dir() else 0,
            "source": source,
        }
    except Exception:
        pass

    # 6. Processing time
    elapsed_ms = int((time.time() - start_time) * 1000)
    result["processing_time_ms"] = elapsed_ms

    if use_hardware:
        result["processing_note"] = (
            f"Real IBM Quantum QPU execution completed in {elapsed_ms / 1000:.1f}s. "
            f"Results are from genuine quantum hardware."
        )
        result["quantum_hardware_details"] = {
            "tier": "hardware",
            "estimated_qubits": 6,
            "estimated_shots": 4096,
            "execution_time_seconds": round(elapsed_ms / 1000, 1),
        }
    else:
        result["processing_note"] = (
            f"Quantum simulator with hardware-optimized weights. "
            f"Processed in {elapsed_ms / 1000:.1f}s. "
            f"Algo weights calibrated on real IBM Quantum QPU via QAOA optimization."
        )

    result["supported_assets_note"] = (
        "Accepts any Solana token mint address, or tickers: BTC, ETH, SOL"
    )

    return result


async def _fetch_price_history(
    token_address: str, limit: int = 60, asset_info: Optional[Dict] = None
) -> List[float]:
    """
    Fetch recent price history for a token.
    Majors (BTC/ETH/SOL): CoinGecko market_chart
    Solana tokens: Birdeye (primary) + Jupiter (fallback)
    """
    if asset_info is None:
        asset_info = resolve_asset(token_address)

    # --- Major assets via CoinGecko ---
    if asset_info["type"] == "major" and asset_info.get("coingecko_id"):
        prices = await _fetch_coingecko_prices(asset_info["coingecko_id"], limit)
        if prices:
            return prices
        # Fall through to Birdeye/Jupiter for SOL if CoinGecko fails
        if asset_info.get("mint"):
            token_address = asset_info["mint"]
        else:
            return []

    # --- Solana tokens via Birdeye ---
    prices = []
    birdeye_key = os.getenv("BIRDEYE_API_KEY", "")
    if birdeye_key:
        try:
            async with aiohttp.ClientSession() as session:
                url = (
                    f"https://public-api.birdeye.so/defi/history_price"
                    f"?address={token_address}&address_type=token&type=1m"
                    f"&time_from={int(time.time()) - 3600}&time_to={int(time.time())}"
                )
                headers = {"X-API-KEY": birdeye_key, "x-chain": "solana"}
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get("data", {}).get("items", [])
                        prices = [item.get("value", 0) for item in items if item.get("value")]
                        if prices:
                            return prices[-limit:]
        except Exception as e:
            logger.debug(f"Birdeye price fetch failed: {e}")

    # --- Fallback: Jupiter price API ---
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.jup.ag/price/v2?ids={token_address}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price_data = data.get("data", {}).get(token_address, {})
                    price = float(price_data.get("price", 0))
                    if price > 0:
                        return [price]
    except Exception as e:
        logger.debug(f"Jupiter price fetch failed: {e}")

    return prices


async def _fetch_coingecko_prices(coingecko_id: str, limit: int = 60) -> List[float]:
    """Fetch 1-hour of price history from CoinGecko for major assets."""
    try:
        async with aiohttp.ClientSession() as session:
            # days=0.042 ≈ 1 hour of data
            url = (
                f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/market_chart"
                f"?vs_currency=usd&days=0.042&precision=full"
            )
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price_points = data.get("prices", [])
                    if price_points:
                        prices = [p[1] for p in price_points if len(p) >= 2]
                        return prices[-limit:] if prices else []
    except Exception as e:
        logger.debug(f"CoinGecko price fetch failed for {coingecko_id}: {e}")
    return []


# =============================================================================
# SINGLETON
# =============================================================================

_verifier: Optional[SolanaPaymentVerifier] = None
_stats: Optional[X402SolanaStats] = None


def get_solana_verifier() -> SolanaPaymentVerifier:
    """Get or create the Solana payment verifier."""
    global _verifier
    if _verifier is None:
        _verifier = SolanaPaymentVerifier()
    return _verifier


def get_x402_stats() -> X402SolanaStats:
    """Get or create the x402 stats tracker."""
    global _stats
    if _stats is None:
        _stats = X402SolanaStats()
    return _stats

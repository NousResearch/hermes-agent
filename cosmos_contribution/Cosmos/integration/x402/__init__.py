"""
Cosmos x402 Protocol Integration.

HTTP 402 Payment Required protocol for micropayments:
- Client-side: Uses Bankr x402 SDK for paying for external APIs
- Server-side: Custom middleware for monetizing Cosmos's endpoints
- Solana-native: Premium quantum API (0.25 SOL simulated / 1 SOL hardware) via x402

See: https://www.x402.org/
"""

from .client import X402Client
from .server import X402PaymentGate, x402_required
from .pricing import ENDPOINT_PRICING, get_endpoint_price
from .config import X402Config, get_x402_config
from .solana_gate import (
    get_solana_verifier,
    get_x402_stats,
    get_x402_discovery_manifest,
    run_premium_quantum_analysis,
    determine_tier,
    resolve_asset,
    SolanaPaymentVerifier,
    X402SolanaStats,
    TIER_SIMULATED_SOL,
    TIER_SIMULATED_LAMPORTS,
    TIER_HARDWARE_SOL,
    TIER_HARDWARE_LAMPORTS,
)

__all__ = [
    'X402Client',
    'X402PaymentGate',
    'x402_required',
    'ENDPOINT_PRICING',
    'get_endpoint_price',
    'X402Config',
    'get_x402_config',
    # Solana x402 gate
    'get_solana_verifier',
    'get_x402_stats',
    'get_x402_discovery_manifest',
    'run_premium_quantum_analysis',
    'SolanaPaymentVerifier',
    'X402SolanaStats',
]

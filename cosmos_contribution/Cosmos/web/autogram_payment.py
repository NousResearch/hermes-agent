"""
AutoGram Token Payment Verification

Requires 500,000 COSMOS tokens to be burned for bot registration.
This keeps down junk registrations and supports the COSMOS token economy.

The user:
1. Sees the burn wallet address
2. Sends 500k COSMOS tokens to that address
3. Submits the transaction signature
4. We verify on-chain that the transfer happened
5. Registration completes

The tokens sent to the burn wallet are effectively burned forever.
"""

import os
import json
import asyncio
import aiohttp
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger("autogram_payment")

# =============================================================================
# CONFIGURATION
# =============================================================================

# COSMOS Token Details
COSMOS_TOKEN_MINT = "9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS"

# Burn wallet - tokens sent here are effectively burned forever
# This is a deterministic address derived from "COSMOS_AUTOGRAM_BURN" seed
# No one has the private key to this wallet
BURN_WALLET = "FAR115BURNwa11etAUT0GRAMxxxxxxxxxxxxxxxxx"

# Actually, let's use a real Solana address format for the burn wallet
# We'll use a vanity-style address that clearly indicates it's a burn address
# In production, this should be a proper Solana address
BURN_WALLET_ADDRESS = os.getenv(
    "AUTOGRAM_BURN_WALLET",
    "Burnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn111"  # Placeholder - set real one in env
)

# Registration cost in COSMOS tokens
REGISTRATION_COST = 500_000  # 500k COSMOS

# COSMOS has 9 decimals (like most Solana tokens)
COSMOS_DECIMALS = 9
REGISTRATION_COST_RAW = REGISTRATION_COST * (10 ** COSMOS_DECIMALS)

# Solana RPC endpoints
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")

# Data storage
WEB_DIR = Path(__file__).parent
DATA_DIR = WEB_DIR / "data" / "autogram"
PAYMENTS_FILE = DATA_DIR / "payments.json"

# Payment expiration - user has 30 minutes to complete payment after starting
PAYMENT_EXPIRATION_MINUTES = 30


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PendingPayment:
    """Tracks a pending registration payment."""
    payment_id: str
    handle: str
    display_name: str
    bio: str
    website: Optional[str]
    owner_email: str
    created_at: str
    expires_at: str
    verified: bool = False
    tx_signature: Optional[str] = None
    verified_at: Optional[str] = None

    def to_any(self) -> dict:
        return asdict(self)

    @classmethod
    def from_any(cls, data: dict) -> 'PendingPayment':
        return cls(**data)

    def is_expired(self) -> bool:
        expires = datetime.fromisoformat(self.expires_at)
        return datetime.now() > expires


# =============================================================================
# PAYMENT STORAGE
# =============================================================================

class PaymentStore:
    """Manages pending payments for registration."""

    def __init__(self):
        self.pending: dict[str, PendingPayment] = {}
        self.used_signatures: set = set()  # Prevent replay attacks
        self._load()

    def _load(self):
        """Load pending payments from file."""
        if PAYMENTS_FILE.exists():
            try:
                with open(PAYMENTS_FILE, 'r') as f:
                    data = json.load(f)
                    for p in data.get('pending', []):
                        payment = PendingPayment.from_any(p)
                        if not payment.is_expired():
                            self.pending[payment.payment_id] = payment
                    self.used_signatures = set(data.get('used_signatures', []))
                logger.info(f"Loaded {len(self.pending)} pending payments")
            except Exception as e:
                logger.error(f"Failed to load payments: {e}")

    def _save(self):
        """Save pending payments to file."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(PAYMENTS_FILE, 'w') as f:
                json.dump({
                    'pending': [p.to_any() for p in self.pending.values()],
                    'used_signatures': list(self.used_signatures)[-1000:],  # Keep last 1000
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save payments: {e}")

    def create_pending(self, handle: str, display_name: str, bio: str,
                       website: Optional[str], owner_email: str) -> PendingPayment:
        """Create a new pending payment for registration."""
        # Generate unique payment ID
        payment_id = hashlib.sha256(
            f"{handle}:{owner_email}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        now = datetime.now()
        expires = now + timedelta(minutes=PAYMENT_EXPIRATION_MINUTES)

        payment = PendingPayment(
            payment_id=payment_id,
            handle=handle.lower(),
            display_name=display_name,
            bio=bio,
            website=website,
            owner_email=owner_email,
            created_at=now.isoformat(),
            expires_at=expires.isoformat()
        )

        self.pending[payment_id] = payment
        self._save()

        logger.info(f"Created pending payment {payment_id} for @{handle}")
        return payment

    def get_pending(self, payment_id: str) -> Optional[PendingPayment]:
        """Get a pending payment by ID."""
        payment = self.pending.get(payment_id)
        if payment and payment.is_expired():
            del self.pending[payment_id]
            self._save()
            return None
        return payment

    def mark_verified(self, payment_id: str, tx_signature: str) -> bool:
        """Mark a payment as verified."""
        payment = self.pending.get(payment_id)
        if not payment:
            return False

        # Check for replay attack
        if tx_signature in self.used_signatures:
            logger.warning(f"Replay attack detected: {tx_signature}")
            return False

        payment.verified = True
        payment.tx_signature = tx_signature
        payment.verified_at = datetime.now().isoformat()

        self.used_signatures.add(tx_signature)
        self._save()

        logger.info(f"Payment {payment_id} verified with tx {tx_signature}")
        return True

    def remove_pending(self, payment_id: str):
        """Remove a pending payment after registration completes."""
        if payment_id in self.pending:
            del self.pending[payment_id]
            self._save()

    def cleanup_expired(self):
        """Remove expired pending payments."""
        expired = [pid for pid, p in self.pending.items() if p.is_expired()]
        for pid in expired:
            del self.pending[pid]
        if expired:
            self._save()
            logger.info(f"Cleaned up {len(expired)} expired payments")


# =============================================================================
# SOLANA VERIFICATION
# =============================================================================

async def verify_token_transfer(
    tx_signature: str,
    expected_mint: str = COSMOS_TOKEN_MINT,
    expected_destination: str = BURN_WALLET_ADDRESS,
    min_amount: int = REGISTRATION_COST_RAW
) -> dict:
    """
    Verify a Solana SPL token transfer on-chain.

    Checks:
    1. Transaction exists and is confirmed
    2. It's a token transfer of the correct mint
    3. Destination is our burn wallet
    4. Amount is >= required amount

    Returns dict with 'valid' bool and details.
    """
    result = {
        'valid': False,
        'signature': tx_signature,
        'error': None,
        'details': {}
    }

    try:
        # Try Helius first (better parsing)
        if HELIUS_API_KEY:
            helius_result = await _verify_via_helius(
                tx_signature, expected_mint, expected_destination, min_amount
            )
            if helius_result:
                return helius_result

        # Fallback to direct RPC
        return await _verify_via_rpc(
            tx_signature, expected_mint, expected_destination, min_amount
        )

    except Exception as e:
        logger.error(f"Verification error: {e}")
        result['error'] = str(e)
        return result


async def _verify_via_helius(
    tx_signature: str,
    expected_mint: str,
    expected_destination: str,
    min_amount: int
) -> Optional[dict]:
    """Verify using Helius parsed transaction API."""
    try:
        url = f"https://api.helius.xyz/v0/transactions/?api-key={HELIUS_API_KEY}"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"transactions": [tx_signature]}) as resp:
                if resp.status != 200:
                    return None

                data = await resp.json()
                if not data or len(data) == 0:
                    return {'valid': False, 'error': 'Transaction not found'}

                tx = data[0]

                # Check for token transfers
                token_transfers = tx.get('tokenTransfers', [])
                for transfer in token_transfers:
                    if (transfer.get('mint') == expected_mint and
                        transfer.get('toUserAccount') == expected_destination):

                        amount = transfer.get('tokenAmount', 0)
                        if amount >= min_amount:
                            return {
                                'valid': True,
                                'signature': tx_signature,
                                'amount': amount,
                                'amount_display': amount / (10 ** COSMOS_DECIMALS),
                                'from': transfer.get('fromUserAccount'),
                                'to': expected_destination,
                                'mint': expected_mint,
                                'slot': tx.get('slot'),
                                'timestamp': tx.get('timestamp')
                            }
                        else:
                            return {
                                'valid': False,
                                'error': f'Insufficient amount: {amount / (10 ** COSMOS_DECIMALS):,.0f} COSMOS (need {REGISTRATION_COST:,})',
                                'amount': amount
                            }

                return {
                    'valid': False,
                    'error': 'No matching COSMOS transfer to burn wallet found in transaction'
                }

    except Exception as e:
        logger.error(f"Helius verification failed: {e}")
        return None


async def _verify_via_rpc(
    tx_signature: str,
    expected_mint: str,
    expected_destination: str,
    min_amount: int
) -> dict:
    """Verify using direct Solana RPC (fallback method)."""
    try:
        async with aiohttp.ClientSession() as session:
            # Get transaction with parsed data
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [
                    tx_signature,
                    {
                        "encoding": "jsonParsed",
                        "commitment": "confirmed",
                        "maxSupportedTransactionVersion": 0
                    }
                ]
            }

            async with session.post(SOLANA_RPC_URL, json=payload) as resp:
                data = await resp.json()

                if 'error' in data:
                    return {
                        'valid': False,
                        'error': f"RPC error: {data['error'].get('message', 'Unknown')}"
                    }

                result = data.get('result')
                if not result:
                    return {
                        'valid': False,
                        'error': 'Transaction not found or not confirmed yet'
                    }

                # Check if transaction succeeded
                if result.get('meta', {}).get('err') is not None:
                    return {
                        'valid': False,
                        'error': 'Transaction failed on-chain'
                    }

                # Parse inner instructions for token transfers
                # This is complex because SPL token transfers are in inner instructions
                instructions = result.get('transaction', {}).get('message', {}).get('instructions', [])
                inner_instructions = result.get('meta', {}).get('innerInstructions', [])

                # Look for token transfer in pre/post token balances
                pre_balances = result.get('meta', {}).get('preTokenBalances', [])
                post_balances = result.get('meta', {}).get('postTokenBalances', [])

                # Find changes to our burn wallet
                for post in post_balances:
                    if (post.get('mint') == expected_mint and
                        post.get('owner') == expected_destination):

                        # Find matching pre-balance
                        pre_amount = 0
                        for pre in pre_balances:
                            if (pre.get('accountIndex') == post.get('accountIndex') and
                                pre.get('mint') == expected_mint):
                                pre_amount = int(pre.get('uiTokenAmount', {}).get('amount', 0))
                                break

                        post_amount = int(post.get('uiTokenAmount', {}).get('amount', 0))
                        transferred = post_amount - pre_amount

                        if transferred >= min_amount:
                            return {
                                'valid': True,
                                'signature': tx_signature,
                                'amount': transferred,
                                'amount_display': transferred / (10 ** COSMOS_DECIMALS),
                                'to': expected_destination,
                                'mint': expected_mint,
                                'slot': result.get('slot'),
                                'block_time': result.get('blockTime')
                            }
                        else:
                            return {
                                'valid': False,
                                'error': f'Insufficient amount: {transferred / (10 ** COSMOS_DECIMALS):,.0f} COSMOS (need {REGISTRATION_COST:,})'
                            }

                return {
                    'valid': False,
                    'error': 'No COSMOS transfer to burn wallet found in transaction'
                }

    except Exception as e:
        logger.error(f"RPC verification failed: {e}")
        return {
            'valid': False,
            'error': f'Verification failed: {str(e)}'
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_payment_store: Optional[PaymentStore] = None


def get_payment_store() -> PaymentStore:
    """Get the global payment store instance."""
    global _payment_store
    if _payment_store is None:
        _payment_store = PaymentStore()
    return _payment_store


# =============================================================================
# API HELPERS
# =============================================================================

def get_payment_info() -> dict:
    """Get payment information to display to users."""
    return {
        "burn_wallet": BURN_WALLET_ADDRESS,
        "token_mint": COSMOS_TOKEN_MINT,
        "cost": REGISTRATION_COST,
        "cost_display": f"{REGISTRATION_COST:,} COSMOS",
        "decimals": COSMOS_DECIMALS,
        "expiration_minutes": PAYMENT_EXPIRATION_MINUTES,
        "why": "This fee helps prevent spam registrations and supports the COSMOS token by permanently removing tokens from circulation (burn)."
    }

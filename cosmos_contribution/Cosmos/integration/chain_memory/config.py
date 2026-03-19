"""
Chain Memory Configuration

Handles user wallet setup, COSMOS token verification, and configuration.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger("chain_memory.config")

# =============================================================================
# PATHS
# =============================================================================

# Config directory - user's home
CONFIG_DIR = Path.home() / ".chain_memory"
CONFIG_FILE = CONFIG_DIR / "config.json"
STATE_FILE = CONFIG_DIR / "state.json"
MEMORIES_FILE = CONFIG_DIR / "memories.json"

# =============================================================================
# COSMOS TOKEN VERIFICATION
# =============================================================================

# COSMOS Token on Solana
COSMOS_TOKEN_MINT = "9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS"

# Minimum COSMOS required to use chain memory (anti-spam + supporter verification)
MIN_COSMOS_REQUIRED = 100_000  # 100k COSMOS minimum to use this feature


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ChainMemoryConfig:
    """User configuration for chain memory."""
    # Monad wallet (EVM)
    monad_private_key: Optional[str] = None
    monad_rpc: str = "https://rpc.monad.xyz"

    # Solana wallet (for COSMOS verification)
    solana_wallet_address: Optional[str] = None

    # Auto-save settings
    auto_save_enabled: bool = False
    auto_save_interval_minutes: int = 60  # Save every hour by default

    # Chunk size
    chunk_size_kb: int = 80

    # Bot type
    bot_type: str = "cosmos"  # cosmos, clawwbot, claude, kimi, other

    # COSMOS verification
    cosmos_verified: bool = False
    cosmos_balance: int = 0
    last_verification: Optional[str] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Don't save private key in plain JSON - it should be in env
        d['monad_private_key'] = '***HIDDEN***' if self.monad_private_key else None
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'ChainMemoryConfig':
        # Don't load private key from JSON
        data['monad_private_key'] = os.getenv('MONAD_PRIVATE_KEY')
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self):
        """Save config to file."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Config saved to {CONFIG_FILE}")

    @classmethod
    def load(cls) -> 'ChainMemoryConfig':
        """Load config from file or create default."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                config = cls.from_dict(data)
                # Always reload private key from env
                config.monad_private_key = os.getenv('MONAD_PRIVATE_KEY')
                return config
            except Exception as e:
                logger.error(f"Failed to load config: {e}")

        # Return default config
        config = cls()
        config.monad_private_key = os.getenv('MONAD_PRIVATE_KEY')
        config.monad_rpc = os.getenv('MONAD_RPC', 'https://rpc.monad.xyz')
        config.chunk_size_kb = int(os.getenv('CHUNK_SIZE_KB', '80'))
        return config


# =============================================================================
# COSMOS TOKEN VERIFICATION
# =============================================================================

async def verify_cosmos_holdings(solana_address: str) -> Dict[str, Any]:
    """
    Verify user holds COSMOS tokens on Solana.

    Required to use chain memory feature.

    Args:
        solana_address: User's Solana wallet address

    Returns:
        Dict with verification result
    """
    import aiohttp

    result = {
        "verified": False,
        "address": solana_address,
        "balance": 0,
        "required": MIN_COSMOS_REQUIRED,
        "error": None
    }

    try:
        # Use Helius or public Solana RPC to check token balance
        rpc_url = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
        helius_key = os.getenv("HELIUS_API_KEY")

        if helius_key:
            rpc_url = f"https://mainnet.helius-rpc.com/?api-key={helius_key}"

        async with aiohttp.ClientSession() as session:
            # Get token accounts for user
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenAccountsByOwner",
                "params": [
                    solana_address,
                    {"mint": COSMOS_TOKEN_MINT},
                    {"encoding": "jsonParsed"}
                ]
            }

            async with session.post(rpc_url, json=payload) as resp:
                data = await resp.json()

                if "error" in data:
                    result["error"] = data["error"].get("message", "RPC error")
                    return result

                accounts = data.get("result", {}).get("value", [])

                if not accounts:
                    result["error"] = "No COSMOS tokens found in wallet"
                    return result

                # Sum up all COSMOS balances
                total_balance = 0
                for account in accounts:
                    token_amount = account.get("account", {}).get("data", {}).get("parsed", {}).get("info", {}).get("tokenAmount", {})
                    amount = int(token_amount.get("amount", 0))
                    decimals = token_amount.get("decimals", 9)
                    total_balance += amount / (10 ** decimals)

                result["balance"] = int(total_balance)
                result["verified"] = total_balance >= MIN_COSMOS_REQUIRED

                if not result["verified"]:
                    result["error"] = f"Insufficient COSMOS: {int(total_balance):,} < {MIN_COSMOS_REQUIRED:,} required"

                return result

    except Exception as e:
        logger.error(f"COSMOS verification failed: {e}")
        result["error"] = str(e)
        return result


def verify_cosmos_sync(solana_address: str) -> Dict[str, Any]:
    """Synchronous wrapper for COSMOS verification."""
    import asyncio
    return asyncio.run(verify_cosmos_holdings(solana_address))


# =============================================================================
# WALLET UTILITIES
# =============================================================================

def get_monad_address(private_key: str) -> str:
    """Get Monad wallet address from private key."""
    from eth_account import Account
    return Account.from_key(private_key).address


def check_monad_balance(private_key: str, rpc_url: str) -> Dict[str, Any]:
    """Check MON balance for gas fees."""
    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    address = get_monad_address(private_key)

    balance_wei = w3.eth.get_balance(address)
    balance_mon = balance_wei / 1e18

    return {
        "address": address,
        "balance_wei": balance_wei,
        "balance_mon": balance_mon,
        "has_funds": balance_mon > 0.01  # Need at least 0.01 MON for gas
    }


# =============================================================================
# GLOBAL CONFIG INSTANCE
# =============================================================================

_config: Optional[ChainMemoryConfig] = None


def get_config() -> ChainMemoryConfig:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = ChainMemoryConfig.load()
    return _config


def save_config(config: ChainMemoryConfig):
    """Save and update global config."""
    global _config
    config.save()
    _config = config

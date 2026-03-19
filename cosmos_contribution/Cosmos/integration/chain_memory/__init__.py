"""
Chain Memory - On-Chain AI Memory Storage

Store your AI bot's memory permanently on Monad blockchain.
Uses memvid for efficient encoding and BetterClips-style calldata storage.

REQUIREMENTS:
- Must hold 100,000+ COSMOS tokens (Solana) to use this feature
- Monad wallet with MON for gas fees

SUPPORTED BOTS:
- Cosmos (full support, recommended)
- ClawwBot/HermesAgent (partial support)
- Claude Code (partial support)
- Kimi (partial support)

SETUP:
    python -m cosmos.integration.chain_memory.setup

USAGE:
    from Cosmos.integration.chain_memory import ChainMemory

    cm = ChainMemory()
    result = await cm.push_memory(title="My Bot Backup")

CLI:
    python -m cosmos.integration.chain_memory push
    python -m cosmos.integration.chain_memory pull --wallet 0x...
    python -m cosmos.integration.chain_memory list
"""

from .config import (
    ChainMemoryConfig,
    get_config,
    save_config,
    verify_cosmos_holdings,
    check_monad_balance,
    MIN_COSMOS_REQUIRED,
    COSMOS_TOKEN_MINT,
)

from .memory_manager import ChainMemory
from .memvid_bridge import MemvidBridge, BotMemoryPackage
from .state_capture import StateCapture, StateRestore, CosmosState
from .auto_save import AutoSaveManager, enable_auto_save, disable_auto_save
from .startup import prompt_memory_load, auto_load_memories

__version__ = "1.0.0"
__all__ = [
    # Config
    "ChainMemoryConfig",
    "get_config",
    "save_config",
    "verify_cosmos_holdings",
    "check_monad_balance",
    "MIN_COSMOS_REQUIRED",
    "COSMOS_TOKEN_MINT",

    # Core
    "ChainMemory",
    "MemvidBridge",
    "BotMemoryPackage",

    # State
    "StateCapture",
    "StateRestore",
    "CosmosState",

    # Auto-save
    "AutoSaveManager",
    "enable_auto_save",
    "disable_auto_save",

    # Startup
    "prompt_memory_load",
    "auto_load_memories",
]

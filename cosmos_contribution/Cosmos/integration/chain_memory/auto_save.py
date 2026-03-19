"""
Chain Memory Auto-Save

Automatically backs up Cosmos state to Monad blockchain.

Features:
- Periodic backups (configurable interval)
- Pre-crash detection backup
- Balance checking before upload
- COSMOS verification before each save
"""

import os
import sys
import asyncio
import atexit
import signal
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, Callable
from pathlib import Path

from .config import get_config, ChainMemoryConfig, verify_cosmos_sync, check_monad_balance
from .state_capture import StateCapture, CosmosState
from .memory_manager import ChainMemory

logger = logging.getLogger("chain_memory.auto_save")


class AutoSaveManager:
    """
    Manages automatic state backups to chain.

    Runs in background, periodically capturing and uploading state.
    """

    def __init__(
        self,
        config: Optional[ChainMemoryConfig] = None,
        cosmos_root: Optional[str] = None
    ):
        """
        Initialize auto-save manager.

        Args:
            config: Configuration (uses global if not provided)
            cosmos_root: Root directory of Cosmos
        """
        self.config = config or get_config()
        self.capture = StateCapture(cosmos_root)
        self.chain_memory = None  # Lazy init

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_save: Optional[datetime] = None
        self._save_count = 0

        # Callbacks
        self.on_save_start: Optional[Callable] = None
        self.on_save_complete: Optional[Callable] = None
        self.on_save_error: Optional[Callable] = None

    def _init_chain_memory(self):
        """Initialize chain memory client."""
        if self.chain_memory is None:
            if not self.config.monad_private_key:
                raise ValueError("Monad private key not configured. Run setup first.")

            self.chain_memory = ChainMemory(
                wallet_key=self.config.monad_private_key,
                rpc_url=self.config.monad_rpc,
                bot_type=self.config.bot_type
            )

    def verify_requirements(self) -> dict:
        """
        Verify all requirements for auto-save.

        Returns:
            Dict with verification results
        """
        result = {
            "ready": False,
            "cosmos_verified": False,
            "has_funds": False,
            "config_valid": False,
            "errors": []
        }

        # Check config
        if not self.config.monad_private_key:
            result["errors"].append("Monad private key not configured")
        elif not self.config.solana_wallet_address:
            result["errors"].append("Solana wallet not configured")
        else:
            result["config_valid"] = True

        # Check COSMOS
        if self.config.solana_wallet_address:
            try:
                cosmos_result = verify_cosmos_sync(self.config.solana_wallet_address)
                result["cosmos_verified"] = cosmos_result.get("verified", False)
                if not result["cosmos_verified"]:
                    result["errors"].append(f"COSMOS verification failed: {cosmos_result.get('error')}")
            except Exception as e:
                result["errors"].append(f"COSMOS check failed: {e}")

        # Check MON balance
        if self.config.monad_private_key:
            try:
                balance = check_monad_balance(
                    self.config.monad_private_key,
                    self.config.monad_rpc
                )
                result["has_funds"] = balance.get("has_funds", False)
                result["mon_balance"] = balance.get("balance_mon", 0)
                if not result["has_funds"]:
                    result["errors"].append("Insufficient MON balance for gas")
            except Exception as e:
                result["errors"].append(f"Balance check failed: {e}")

        result["ready"] = (
            result["config_valid"] and
            result["cosmos_verified"] and
            result["has_funds"]
        )

        return result

    async def save_state_now(self, title: Optional[str] = None) -> dict:
        """
        Immediately capture and save state to chain.

        Args:
            title: Optional title for this backup

        Returns:
            Dict with save result
        """
        self._init_chain_memory()

        # Verify requirements
        verify = self.verify_requirements()
        if not verify["ready"]:
            return {
                "success": False,
                "error": "Requirements not met",
                "details": verify["errors"]
            }

        try:
            if self.on_save_start:
                self.on_save_start()

            # Capture state
            logger.info("Capturing Cosmos state...")
            state = self.capture.capture_full_state()

            # Generate title
            if title is None:
                title = f"Auto-backup {datetime.now().strftime('%Y-%m-%d %H:%M')}"

            # Convert state to memory package format
            from .memvid_bridge import BotMemoryPackage, MemoryChunk
            import json

            # Create chunks from state
            chunks = []

            # Add full state as single JSON chunk
            state_json = json.dumps(state.to_dict())
            chunks.append(MemoryChunk(
                content=state_json,
                chunk_type="full_state",
                timestamp=state.captured_at,
                metadata={"version": state.version, "machine_id": state.machine_id}
            ))

            package = BotMemoryPackage(
                bot_name="Cosmos",
                bot_type=self.config.bot_type,
                version="1.0",
                created_at=datetime.now().isoformat(),
                chunks=chunks,
                personality=state.personality_state,
                settings={"auto_save": True}
            )

            # Push to chain
            logger.info("Pushing state to chain...")

            def progress(current, total, status):
                logger.info(f"[{current}/{total}] {status}")

            record = await self.chain_memory.push_memory(
                title=title,
                on_progress=progress
            )

            self._last_save = datetime.now()
            self._save_count += 1

            if self.on_save_complete:
                self.on_save_complete(record)

            return {
                "success": True,
                "memory_id": record.memory_id,
                "tx_count": len(record.tx_hashes),
                "size": record.total_size,
                "saved_at": self._last_save.isoformat()
            }

        except Exception as e:
            logger.error(f"Auto-save failed: {e}")

            if self.on_save_error:
                self.on_save_error(e)

            return {
                "success": False,
                "error": str(e)
            }

    def _background_loop(self):
        """Background thread loop for periodic saves."""
        logger.info("Auto-save background loop started")

        interval = timedelta(minutes=self.config.auto_save_interval_minutes)

        while self._running:
            try:
                # Check if it's time to save
                should_save = (
                    self._last_save is None or
                    datetime.now() - self._last_save >= interval
                )

                if should_save:
                    logger.info("Auto-save triggered")
                    asyncio.run(self.save_state_now())

                # Sleep in small increments to allow clean shutdown
                for _ in range(60):  # Check every second
                    if not self._running:
                        break
                    threading.Event().wait(1)

            except Exception as e:
                logger.error(f"Auto-save loop error: {e}")
                threading.Event().wait(60)  # Wait a minute on error

        logger.info("Auto-save background loop stopped")

    def start(self):
        """Start auto-save background thread."""
        if self._running:
            logger.warning("Auto-save already running")
            return

        if not self.config.auto_save_enabled:
            logger.info("Auto-save is disabled in config")
            return

        # Verify requirements first
        verify = self.verify_requirements()
        if not verify["ready"]:
            logger.error(f"Cannot start auto-save: {verify['errors']}")
            return

        self._running = True
        self._thread = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()

        logger.info(f"Auto-save started (interval: {self.config.auto_save_interval_minutes} min)")

    def stop(self):
        """Stop auto-save background thread."""
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

        logger.info("Auto-save stopped")

    def get_status(self) -> dict:
        """Get auto-save status."""
        return {
            "running": self._running,
            "enabled": self.config.auto_save_enabled,
            "interval_minutes": self.config.auto_save_interval_minutes,
            "last_save": self._last_save.isoformat() if self._last_save else None,
            "save_count": self._save_count
        }


# =============================================================================
# CRASH HANDLER
# =============================================================================

_auto_save_manager: Optional[AutoSaveManager] = None


def _emergency_save():
    """Emergency save on crash/exit."""
    global _auto_save_manager

    if _auto_save_manager is None:
        return

    try:
        logger.warning("Emergency save triggered!")
        asyncio.run(_auto_save_manager.save_state_now("Emergency backup (crash/exit)"))
    except Exception as e:
        logger.error(f"Emergency save failed: {e}")


def register_crash_handler(manager: AutoSaveManager):
    """Register crash handlers for emergency saves."""
    global _auto_save_manager
    _auto_save_manager = manager

    # Register atexit handler
    atexit.register(_emergency_save)

    # Register signal handlers
    def signal_handler(signum, frame):
        logger.warning(f"Received signal {signum}, triggering emergency save...")
        _emergency_save()
        sys.exit(1)

    try:
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, signal_handler)
    except Exception as e:
        logger.warning(f"Could not register signal handlers: {e}")

    logger.info("Crash handlers registered")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def enable_auto_save(
    interval_minutes: int = 60,
    crash_save: bool = True
) -> AutoSaveManager:
    """
    Enable auto-save with default settings.

    Args:
        interval_minutes: Backup interval
        crash_save: Enable emergency save on crash

    Returns:
        AutoSaveManager instance
    """
    config = get_config()
    config.auto_save_enabled = True
    config.auto_save_interval_minutes = interval_minutes

    manager = AutoSaveManager(config)

    if crash_save:
        register_crash_handler(manager)

    manager.start()
    return manager


def disable_auto_save():
    """Disable auto-save."""
    global _auto_save_manager

    if _auto_save_manager:
        _auto_save_manager.stop()
        _auto_save_manager = None

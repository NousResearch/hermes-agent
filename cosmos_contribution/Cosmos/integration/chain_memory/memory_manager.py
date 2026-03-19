"""
Chain Memory Manager - High-level API for on-chain memory storage.

Provides a simple interface for:
- Pushing bot memory to Monad blockchain
- Pulling and reconstructing memory from chain
- Loading memory into Cosmos/HermesAgent bots

REQUIREMENTS:
- Must hold 100,000+ COSMOS tokens to use push features
- Monad wallet with MON for gas fees
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict

from .memvid_bridge import MemvidBridge, BotMemoryPackage
from .protected import ChainUploader, ChainDownloader, get_fingerprint
from .config import (
    get_config,
    verify_cosmos_holdings,
    MIN_COSMOS_REQUIRED,
    COSMOS_TOKEN_MINT
)

logger = logging.getLogger("chain_memory")

# =============================================================================
# LOCAL STORAGE
# =============================================================================

# Store memory records locally for easy access
DATA_DIR = Path(__file__).parent / "data"
MEMORIES_FILE = DATA_DIR / "chain_memories.json"


def _load_local_memories() -> Dict:
    """Load local memory index."""
    if MEMORIES_FILE.exists():
        with open(MEMORIES_FILE, 'r') as f:
            return json.load(f)
    return {"memories": [], "updated_at": None}


def _save_local_memories(data: Dict):
    """Save local memory index."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now().isoformat()
    with open(MEMORIES_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# =============================================================================
# CHAIN MEMORY CLASS
# =============================================================================

@dataclass
class MemoryRecord:
    """Record of a memory stored on-chain."""
    memory_id: str
    title: str
    tx_hashes: List[str]
    total_chunks: int
    total_size: int
    bot_type: str  # "cosmos" or "HermesAgent"
    wallet_address: str
    chain: str
    uploaded_at: str
    local_path: Optional[str] = None  # Path to reconstructed files

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryRecord':
        return cls(**data)


class ChainMemory:
    """
    Main interface for on-chain memory storage.

    Usage:
        cm = ChainMemory(wallet_key="0x...")

        # Push memory to chain
        result = cm.push_memory(title="My Bot Memory")

        # Pull memory from chain
        memory = cm.pull_memory(tx_ids=["0x...", "0x..."])
        # OR load all memories for a wallet
        memories = cm.pull_all_memories(wallet_address="0x...")
    """

    def __init__(
        self,
        wallet_key: Optional[str] = None,
        rpc_url: Optional[str] = None,
        bot_type: str = "cosmos",
        skip_cosmos_check: bool = False
    ):
        """
        Initialize ChainMemory.

        Args:
            wallet_key: Private key for uploads (required for push)
            rpc_url: Monad RPC URL (uses default if not specified)
            bot_type: Type of bot ("cosmos", "clawwbot", "claude", "kimi")
            skip_cosmos_check: Skip COSMOS verification (for testing only)
        """
        # Load config
        self.config = get_config()

        # Use provided values or fall back to config/env
        self.wallet_key = wallet_key or self.config.monad_private_key or os.getenv("MONAD_PRIVATE_KEY")
        self.rpc_url = rpc_url or self.config.monad_rpc or os.getenv("MONAD_RPC", "https://rpc.monad.xyz")
        self.bot_type = bot_type or self.config.bot_type

        self.memvid = MemvidBridge()
        self._uploader = None
        self._downloader = None
        self._cosmos_verified = skip_cosmos_check

        logger.info(f"ChainMemory initialized for {self.bot_type}")
        logger.debug(f"Device fingerprint: {get_fingerprint()}")

    async def verify_cosmos(self) -> Dict:
        """
        Verify COSMOS token holdings.

        Required before pushing to chain.

        Returns:
            Dict with verification result
        """
        if not self.config.solana_wallet_address:
            return {
                "verified": False,
                "error": "Solana wallet not configured. Run setup first."
            }

        result = await verify_cosmos_holdings(self.config.solana_wallet_address)
        self._cosmos_verified = result.get("verified", False)

        if self._cosmos_verified:
            logger.info(f"COSMOS verified: {result.get('balance', 0):,} COSMOS")
        else:
            logger.warning(f"COSMOS verification failed: {result.get('error')}")

        return result

    def _require_cosmos(self):
        """Raise error if COSMOS not verified."""
        if not self._cosmos_verified:
            raise PermissionError(
                f"COSMOS token verification required!\n\n"
                f"You must hold at least {MIN_COSMOS_REQUIRED:,} COSMOS tokens to use Chain Memory.\n"
                f"COSMOS Token: {COSMOS_TOKEN_MINT}\n"
                f"Buy COSMOS: https://pump.fun/coin/{COSMOS_TOKEN_MINT}\n\n"
                f"Run 'python -m cosmos.integration.chain_memory.setup' to configure."
            )

    @property
    def uploader(self) -> ChainUploader:
        """Get uploader (lazy init)."""
        if self._uploader is None:
            if not self.wallet_key:
                raise ValueError(
                    "Wallet key required for uploads. "
                    "Set MONAD_PRIVATE_KEY env var or pass wallet_key parameter."
                )
            self._uploader = ChainUploader(self.wallet_key, self.rpc_url)
        return self._uploader

    @property
    def downloader(self) -> ChainDownloader:
        """Get downloader (lazy init)."""
        if self._downloader is None:
            self._downloader = ChainDownloader(self.rpc_url)
        return self._downloader

    @property
    def wallet_address(self) -> Optional[str]:
        """Get wallet address if key is set."""
        if self.wallet_key:
            from eth_account import Account
            return Account.from_key(self.wallet_key).address
        return None

    # -------------------------------------------------------------------------
    # PUSH MEMORY
    # -------------------------------------------------------------------------

    def estimate_push_cost(self, memory_path: Optional[str] = None) -> Dict:
        """
        Estimate the cost to push memory to chain.

        Args:
            memory_path: Path to memory directory (uses default for bot_type)

        Returns:
            Cost estimate dict
        """
        # Extract memory
        if self.bot_type == "cosmos":
            package = self.memvid.extract_cosmos_memory(memory_path)
        else:
            package = self.memvid.extract_hermes_memory(memory_path)

        # Get stats
        stats = self.memvid.get_memory_stats(package)

        # Estimate MP4 size (rough estimate: ~10% of text size after compression)
        estimated_mp4_size = int(stats["total_characters"] * 0.1)

        return self.uploader.estimate_cost(estimated_mp4_size)

    async def push_memory(
        self,
        title: str = "Bot Memory",
        memory_path: Optional[str] = None,
        on_progress: Optional[Callable[[int, int, str], None]] = None
    ) -> MemoryRecord:
        """
        Push bot memory to Monad blockchain.

        REQUIRES: Must hold 100,000+ COSMOS tokens!

        Steps:
        1. Verify COSMOS holdings
        2. Extract memory from bot
        3. Encode to MP4 using memvid
        4. Upload to Monad as chunked calldata
        5. Save record locally

        Args:
            title: Title for this memory backup
            memory_path: Path to memory directory
            on_progress: Progress callback(current, total, status)

        Returns:
            MemoryRecord with transaction details

        Raises:
            PermissionError: If COSMOS verification fails
        """
        logger.info(f"Pushing memory: {title}")

        # Step 0: Verify COSMOS holdings
        if not self._cosmos_verified:
            if on_progress:
                on_progress(0, 5, "Verifying COSMOS holdings...")

            result = await self.verify_cosmos()
            if not result.get("verified"):
                self._require_cosmos()

        # Step 1: Extract memory
        if on_progress:
            on_progress(1, 5, "Extracting memory...")

        if self.bot_type == "cosmos":
            package = self.memvid.extract_cosmos_memory(memory_path)
        else:
            package = self.memvid.extract_hermes_memory(memory_path)

        stats = self.memvid.get_memory_stats(package)
        logger.info(f"Extracted {stats['total_chunks']} chunks ({stats['total_characters']:,} chars)")

        # Step 2: Encode to MP4
        if on_progress:
            on_progress(1, 4, "Encoding to MP4...")

        mp4_path, index_path = self.memvid.encode_to_mp4(package)
        logger.info(f"Encoded to: {mp4_path}")

        # Step 3: Upload to chain
        if on_progress:
            on_progress(2, 4, "Uploading to Monad...")

        def upload_progress(current, total, status):
            if on_progress:
                # Map upload progress to step 2-3
                progress = 2 + (current / total)
                on_progress(int(progress * 100), 400, status)

        result = await self.uploader.upload(
            mp4_path=mp4_path,
            index_path=index_path,
            title=title,
            on_progress=upload_progress
        )

        if not result.success:
            raise RuntimeError(f"Upload failed: {result.error}")

        # Step 4: Save record
        if on_progress:
            on_progress(4, 4, "Saving record...")

        record = MemoryRecord(
            memory_id=result.memory_id,
            title=title,
            tx_hashes=result.tx_hashes,
            total_chunks=result.total_chunks,
            total_size=result.total_size,
            bot_type=self.bot_type,
            wallet_address=self.wallet_address,
            chain="monad",
            uploaded_at=datetime.now().isoformat()
        )

        # Save locally
        local_data = _load_local_memories()
        local_data["memories"].append(record.to_dict())
        _save_local_memories(local_data)

        # Cleanup temp files
        try:
            os.remove(mp4_path)
            os.remove(index_path)
        except Exception:
            pass

        logger.info(f"Memory pushed! ID: {result.memory_id}")
        logger.info(f"Cost: {result.total_cost:.4f} {result.cost_currency}")
        logger.info(f"TX hashes: {len(result.tx_hashes)} transactions")

        return record

    # -------------------------------------------------------------------------
    # PULL MEMORY
    # -------------------------------------------------------------------------

    async def pull_memory(
        self,
        tx_ids: Optional[List[str]] = None,
        memory_id: Optional[str] = None,
        output_dir: Optional[str] = None,
        on_progress: Optional[Callable[[int, int, str], None]] = None
    ) -> BotMemoryPackage:
        """
        Pull and reconstruct memory from blockchain.

        Args:
            tx_ids: List of transaction hashes (in order)
            memory_id: Memory ID to look up from local records
            output_dir: Directory to save reconstructed files
            on_progress: Progress callback

        Returns:
            BotMemoryPackage ready to load
        """
        # Get TX hashes
        if tx_ids is None:
            if memory_id is None:
                raise ValueError("Either tx_ids or memory_id required")

            # Look up from local records
            local_data = _load_local_memories()
            record = next(
                (m for m in local_data["memories"] if m["memory_id"] == memory_id),
                None
            )
            if not record:
                raise ValueError(f"Memory {memory_id} not found in local records")

            tx_ids = record["tx_hashes"]

        # Set output dir
        if output_dir is None:
            output_dir = str(DATA_DIR / "reconstructed")

        logger.info(f"Pulling memory from {len(tx_ids)} transactions...")

        # Download from chain
        result = await self.downloader.download(
            tx_hashes=tx_ids,
            output_dir=output_dir,
            on_progress=on_progress
        )

        if not result.success:
            raise RuntimeError(f"Download failed: {result.error}")

        # Decode MP4 back to memory
        logger.info("Decoding memory from MP4...")
        package = self.memvid.decode_from_mp4(result.mp4_path, result.index_path)

        # Update local record with path
        local_data = _load_local_memories()
        for mem in local_data["memories"]:
            if mem.get("memory_id") == result.memory_id:
                mem["local_path"] = output_dir
        _save_local_memories(local_data)

        stats = self.memvid.get_memory_stats(package)
        logger.info(f"Reconstructed: {stats['total_chunks']} chunks")

        return package

    async def pull_all_memories(
        self,
        wallet_address: Optional[str] = None,
        on_progress: Optional[Callable[[int, int, str], None]] = None
    ) -> List[BotMemoryPackage]:
        """
        Pull all memories for a wallet address.

        Args:
            wallet_address: Wallet to scan (uses own wallet if not specified)
            on_progress: Progress callback

        Returns:
            List of BotMemoryPackage objects
        """
        address = wallet_address or self.wallet_address
        if not address:
            raise ValueError("Wallet address required")

        logger.info(f"Finding all memories for {address}...")

        # Find memories on chain
        memories = await self.downloader.find_memories_by_wallet(address)
        logger.info(f"Found {len(memories)} memory uploads")

        # Pull each one
        packages = []
        for i, mem in enumerate(memories):
            if on_progress:
                on_progress(i + 1, len(memories), f"Pulling {mem['memory_id']}...")

            try:
                package = await self.pull_memory(tx_ids=mem["tx_hashes"])
                packages.append(package)
            except Exception as e:
                logger.error(f"Failed to pull {mem['memory_id']}: {e}")

        return packages

    # -------------------------------------------------------------------------
    # LOAD INTO BOT
    # -------------------------------------------------------------------------

    def load_into_cosmos(
        self,
        package: BotMemoryPackage,
        memory_path: Optional[str] = None,
        merge: bool = True
    ):
        """
        Load memory package into Cosmos's memory system.

        Args:
            package: The memory package to load
            memory_path: Path to Cosmos memory directory
            merge: If True, merge with existing; if False, replace
        """
        if memory_path is None:
            memory_path = Path(__file__).parent.parent.parent / "memory"

        memory_path = Path(memory_path)
        memory_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading {len(package.chunks)} chunks into Cosmos...")

        # Group chunks by type
        by_type = {}
        for chunk in package.chunks:
            if chunk.chunk_type not in by_type:
                by_type[chunk.chunk_type] = []
            by_type[chunk.chunk_type].append(chunk)

        # Load archival memory
        if "archival" in by_type:
            archival_file = memory_path / "archival_memory.json"
            existing = {"entries": []}
            if merge and archival_file.exists():
                with open(archival_file, 'r') as f:
                    existing = json.load(f)

            for chunk in by_type["archival"]:
                existing["entries"].append({
                    "content": chunk.content,
                    "timestamp": chunk.timestamp,
                    "importance": chunk.metadata.get("importance", 0.5),
                    "tags": chunk.metadata.get("tags", []),
                    "source": "chain_memory"
                })

            with open(archival_file, 'w') as f:
                json.dump(existing, f, indent=2)

            logger.info(f"Loaded {len(by_type['archival'])} archival memories")

        # Load dialogue history
        if "dialogue" in by_type:
            dialogue_file = memory_path / "dialogue_history.json"
            existing = {"conversations": []}
            if merge and dialogue_file.exists():
                with open(dialogue_file, 'r') as f:
                    existing = json.load(f)

            for chunk in by_type["dialogue"]:
                # Parse messages from content
                messages = []
                for line in chunk.content.split('\n'):
                    if ': ' in line:
                        role, content = line.split(': ', 1)
                        messages.append({"role": role, "content": content})

                existing["conversations"].append({
                    "messages": messages,
                    "timestamp": chunk.timestamp,
                    "participants": chunk.metadata.get("participants", []),
                    "topic": chunk.metadata.get("topic", ""),
                    "source": "chain_memory"
                })

            with open(dialogue_file, 'w') as f:
                json.dump(existing, f, indent=2)

            logger.info(f"Loaded {len(by_type['dialogue'])} conversations")

        # Load episodic memory
        if "episodic" in by_type:
            episodic_file = memory_path / "episodic_memory.json"
            existing = {"episodes": []}
            if merge and episodic_file.exists():
                with open(episodic_file, 'r') as f:
                    existing = json.load(f)

            for chunk in by_type["episodic"]:
                existing["episodes"].append({
                    "description": chunk.content,
                    "timestamp": chunk.timestamp,
                    "emotion": chunk.metadata.get("emotion", ""),
                    "significance": chunk.metadata.get("significance", 0.5),
                    "source": "chain_memory"
                })

            with open(episodic_file, 'w') as f:
                json.dump(existing, f, indent=2)

            logger.info(f"Loaded {len(by_type['episodic'])} episodes")

        # Load personality
        if package.personality:
            personality_file = memory_path / "personality_state.json"
            with open(personality_file, 'w') as f:
                json.dump(package.personality, f, indent=2)
            logger.info("Loaded personality state")

        logger.info("Memory loaded into Cosmos!")

    def load_into_HermesAgent(
        self,
        package: BotMemoryPackage,
        memory_path: Optional[str] = None,
        merge: bool = True
    ):
        """
        Load memory package into HermesAgent's memory system.

        Args:
            package: The memory package to load
            memory_path: Path to HermesAgent memory directory
            merge: If True, merge with existing; if False, replace
        """
        if memory_path is None:
            possible_paths = [
                Path.home() / ".hermes" / "memory",
                Path.home() / ".config" / "HermesAgent" / "memory",
            ]
            memory_path = possible_paths[0]

        memory_path = Path(memory_path)
        memory_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading {len(package.chunks)} chunks into HermesAgent...")

        memory_file = memory_path / "memory.json"
        existing = {"long_term": [], "sessions": []}
        if merge and memory_file.exists():
            with open(memory_file, 'r') as f:
                existing = json.load(f)

        for chunk in package.chunks:
            if chunk.chunk_type in ["archival", "episodic"]:
                existing["long_term"].append({
                    "content": chunk.content,
                    "created_at": chunk.timestamp,
                    "metadata": chunk.metadata,
                    "source": "chain_memory"
                })
            elif chunk.chunk_type == "dialogue":
                existing["sessions"].append({
                    "messages": json.loads(chunk.content) if chunk.content.startswith('[') else [],
                    "timestamp": chunk.timestamp,
                    "id": chunk.metadata.get("session_id", ""),
                    "source": "chain_memory"
                })

        with open(memory_file, 'w') as f:
            json.dump(existing, f, indent=2)

        logger.info("Memory loaded into HermesAgent!")

    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------

    def list_local_memories(self) -> List[MemoryRecord]:
        """List all locally tracked memories."""
        local_data = _load_local_memories()
        return [MemoryRecord.from_dict(m) for m in local_data["memories"]]

    def get_memory_info(self, memory_id: str) -> Optional[MemoryRecord]:
        """Get info about a specific memory."""
        local_data = _load_local_memories()
        for m in local_data["memories"]:
            if m["memory_id"] == memory_id:
                return MemoryRecord.from_dict(m)
        return None

    def export_tx_list(self, memory_id: str) -> str:
        """
        Export transaction list for sharing.

        Returns a string that can be shared to allow others to
        reconstruct the memory.
        """
        record = self.get_memory_info(memory_id)
        if not record:
            raise ValueError(f"Memory {memory_id} not found")

        export_data = {
            "format": "cosmos_chain_memory_v1",
            "memory_id": record.memory_id,
            "title": record.title,
            "bot_type": record.bot_type,
            "chain": record.chain,
            "tx_hashes": record.tx_hashes,
            "total_chunks": record.total_chunks,
            "uploaded_at": record.uploaded_at
        }

        return json.dumps(export_data, indent=2)

    def import_tx_list(self, export_string: str) -> MemoryRecord:
        """
        Import a shared transaction list.

        This doesn't download the memory, just saves the record locally
        so it can be pulled later.
        """
        data = json.loads(export_string)

        if data.get("format") != "cosmos_chain_memory_v1":
            raise ValueError("Invalid export format")

        record = MemoryRecord(
            memory_id=data["memory_id"],
            title=data.get("title", "Imported Memory"),
            tx_hashes=data["tx_hashes"],
            total_chunks=data.get("total_chunks", len(data["tx_hashes"])),
            total_size=0,
            bot_type=data.get("bot_type", "cosmos"),
            wallet_address="",
            chain=data.get("chain", "monad"),
            uploaded_at=data.get("uploaded_at", datetime.now().isoformat())
        )

        # Save locally
        local_data = _load_local_memories()
        local_data["memories"].append(record.to_dict())
        _save_local_memories(local_data)

        logger.info(f"Imported memory: {record.memory_id}")
        return record

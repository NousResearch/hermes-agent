"""
Memvid Bridge - Convert bot memory to/from MP4 video format.

Uses the memvid library to encode text-based memory into compact MP4 files
that can be stored on-chain efficiently.
"""

import os
import json
import tempfile
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger("chain_memory.memvid")

# =============================================================================
# MEMORY DATA STRUCTURES
# =============================================================================

@dataclass
class MemoryChunk:
    """A single memory chunk with metadata."""
    content: str
    chunk_type: str  # "dialogue", "archival", "episodic", "personality"
    timestamp: str
    metadata: Dict[str, Any]

    def to_text(self) -> str:
        """Convert to indexed text for memvid encoding."""
        meta_str = json.dumps(self.metadata, separators=(',', ':'))
        return f"[{self.chunk_type}|{self.timestamp}|{meta_str}]\n{self.content}"

    @classmethod
    def from_text(cls, text: str) -> 'MemoryChunk':
        """Parse from memvid-encoded text."""
        try:
            # Parse header: [type|timestamp|metadata]
            header_end = text.index(']\n')
            header = text[1:header_end]
            content = text[header_end + 2:]

            parts = header.split('|', 2)
            chunk_type = parts[0] if len(parts) > 0 else "unknown"
            timestamp = parts[1] if len(parts) > 1 else datetime.now().isoformat()
            metadata = json.loads(parts[2]) if len(parts) > 2 else {}

            return cls(
                content=content,
                chunk_type=chunk_type,
                timestamp=timestamp,
                metadata=metadata
            )
        except Exception as e:
            # Fallback for plain text
            return cls(
                content=text,
                chunk_type="unknown",
                timestamp=datetime.now().isoformat(),
                metadata={}
            )


@dataclass
class BotMemoryPackage:
    """Complete memory package for a bot."""
    bot_name: str
    bot_type: str  # "cosmos" or "HermesAgent"
    version: str
    created_at: str
    chunks: List[MemoryChunk]
    personality: Optional[Dict] = None
    settings: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "bot_name": self.bot_name,
            "bot_type": self.bot_type,
            "version": self.version,
            "created_at": self.created_at,
            "chunk_count": len(self.chunks),
            "personality": self.personality,
            "settings": self.settings
        }


# =============================================================================
# MEMVID BRIDGE
# =============================================================================

class MemvidBridge:
    """
    Bridge between bot memory and memvid MP4 encoding.

    Handles:
    - Extracting memory from Cosmos/HermesAgent format
    - Encoding to MP4 using memvid
    - Decoding MP4 back to memory format
    """

    # Memvid encoding parameters optimized for on-chain storage
    DEFAULT_CHUNK_SIZE = 512  # Characters per chunk
    DEFAULT_OVERLAP = 50
    DEFAULT_FPS = 30
    DEFAULT_FRAME_SIZE = (640, 360)  # Small for efficiency

    def __init__(self, temp_dir: Optional[str] = None):
        """Initialize the memvid bridge."""
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self._check_memvid()

    def _check_memvid(self):
        """Check if memvid is installed."""
        try:
            import memvid
            self.memvid = memvid
            logger.info("Memvid library loaded successfully")
        except ImportError:
            logger.warning("Memvid not installed. Run: pip install memvid")
            self.memvid = None

    # -------------------------------------------------------------------------
    # COSMOS MEMORY EXTRACTION
    # -------------------------------------------------------------------------

    def extract_cosmos_memory(self, memory_path: Optional[str] = None) -> BotMemoryPackage:
        """
        Extract memory from Cosmos's memory system.

        Reads from:
        - archival_memory.json (long-term storage)
        - dialogue_history.json (conversations)
        - episodic_memory.json (experiences)
        - personality evolved state
        """
        if memory_path is None:
            # Default Cosmos memory location
            memory_path = Path(__file__).parent.parent.parent / "memory"

        memory_path = Path(memory_path)
        chunks = []

        # 1. Archival Memory
        archival_file = memory_path / "archival_memory.json"
        if archival_file.exists():
            try:
                with open(archival_file, 'r', encoding='utf-8') as f:
                    archival = json.load(f)
                    for entry in archival.get('entries', []):
                        chunks.append(MemoryChunk(
                            content=entry.get('content', ''),
                            chunk_type="archival",
                            timestamp=entry.get('timestamp', datetime.now().isoformat()),
                            metadata={
                                "importance": entry.get('importance', 0.5),
                                "tags": entry.get('tags', [])
                            }
                        ))
                logger.info(f"Extracted {len(archival.get('entries', []))} archival memories")
            except Exception as e:
                logger.error(f"Failed to read archival memory: {e}")

        # 2. Dialogue History
        dialogue_file = memory_path / "dialogue_history.json"
        if dialogue_file.exists():
            try:
                with open(dialogue_file, 'r', encoding='utf-8') as f:
                    dialogue = json.load(f)
                    for conv in dialogue.get('conversations', []):
                        content = "\n".join([
                            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
                            for m in conv.get('messages', [])
                        ])
                        chunks.append(MemoryChunk(
                            content=content,
                            chunk_type="dialogue",
                            timestamp=conv.get('timestamp', datetime.now().isoformat()),
                            metadata={
                                "participants": conv.get('participants', []),
                                "topic": conv.get('topic', '')
                            }
                        ))
                logger.info(f"Extracted {len(dialogue.get('conversations', []))} conversations")
            except Exception as e:
                logger.error(f"Failed to read dialogue history: {e}")

        # 3. Episodic Memory
        episodic_file = memory_path / "episodic_memory.json"
        if episodic_file.exists():
            try:
                with open(episodic_file, 'r', encoding='utf-8') as f:
                    episodic = json.load(f)
                    for episode in episodic.get('episodes', []):
                        chunks.append(MemoryChunk(
                            content=episode.get('description', ''),
                            chunk_type="episodic",
                            timestamp=episode.get('timestamp', datetime.now().isoformat()),
                            metadata={
                                "emotion": episode.get('emotion', ''),
                                "significance": episode.get('significance', 0.5)
                            }
                        ))
                logger.info(f"Extracted {len(episodic.get('episodes', []))} episodes")
            except Exception as e:
                logger.error(f"Failed to read episodic memory: {e}")

        # 4. Personality State
        personality = None
        personality_file = memory_path / "personality_state.json"
        if personality_file.exists():
            try:
                with open(personality_file, 'r', encoding='utf-8') as f:
                    personality = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read personality: {e}")

        return BotMemoryPackage(
            bot_name="Cosmos",
            bot_type="cosmos",
            version="1.0",
            created_at=datetime.now().isoformat(),
            chunks=chunks,
            personality=personality
        )

    # -------------------------------------------------------------------------
    # HermesAgent memory EXTRACTION
    # -------------------------------------------------------------------------

    def extract_hermes_memory(self, memory_path: Optional[str] = None) -> BotMemoryPackage:
        """
        Extract memory from HermesAgent's memory system.

        HermesAgent uses a similar but slightly different memory structure.
        """
        if memory_path is None:
            # Try common HermesAgent locations
            possible_paths = [
                Path.home() / ".hermes" / "memory",
                Path.home() / ".config" / "HermesAgent" / "memory",
                Path("/workspace/Hermes/memory")
            ]
            memory_path = next((p for p in possible_paths if p.exists()), possible_paths[0])

        memory_path = Path(memory_path)
        chunks = []

        # HermesAgent memory format
        memory_file = memory_path / "memory.json"
        if memory_file.exists():
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    memory = json.load(f)

                    # Long-term memories
                    for entry in memory.get('long_term', []):
                        chunks.append(MemoryChunk(
                            content=entry.get('content', ''),
                            chunk_type="archival",
                            timestamp=entry.get('created_at', datetime.now().isoformat()),
                            metadata=entry.get('metadata', {})
                        ))

                    # Session memories
                    for entry in memory.get('sessions', []):
                        chunks.append(MemoryChunk(
                            content=json.dumps(entry.get('messages', [])),
                            chunk_type="dialogue",
                            timestamp=entry.get('timestamp', datetime.now().isoformat()),
                            metadata={"session_id": entry.get('id', '')}
                        ))

                logger.info(f"Extracted {len(chunks)} HermesAgent memories")
            except Exception as e:
                logger.error(f"Failed to read HermesAgent memory: {e}")

        return BotMemoryPackage(
            bot_name="HermesAgent",
            bot_type="HermesAgent",
            version="1.0",
            created_at=datetime.now().isoformat(),
            chunks=chunks
        )

    # -------------------------------------------------------------------------
    # MEMVID ENCODING
    # -------------------------------------------------------------------------

    def encode_to_mp4(
        self,
        memory_package: BotMemoryPackage,
        output_path: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Encode a memory package to MP4 video using memvid.

        Returns:
            Tuple of (mp4_path, index_path)
        """
        if self.memvid is None:
            raise RuntimeError("Memvid not installed. Run: pip install memvid")

        # Generate output paths
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{memory_package.bot_name.lower()}_memory_{timestamp}"
            output_path = os.path.join(self.temp_dir, base_name)

        mp4_path = f"{output_path}.mp4"
        index_path = f"{output_path}_index.json"

        # Convert chunks to text for encoding
        text_chunks = []

        # Add metadata header as first chunk
        header = json.dumps({
            "type": "memory_package",
            "version": "1.0",
            "bot": memory_package.to_dict()
        })
        text_chunks.append(f"[HEADER]\n{header}")

        # Add all memory chunks
        for chunk in memory_package.chunks:
            text_chunks.append(chunk.to_text())

        # Add personality as final chunk if present
        if memory_package.personality:
            personality_json = json.dumps(memory_package.personality)
            text_chunks.append(f"[personality|{datetime.now().isoformat()}|{{}}]\n{personality_json}")

        logger.info(f"Encoding {len(text_chunks)} chunks to MP4...")

        # Use memvid encoder
        from memvid import MemvidEncoder

        encoder = MemvidEncoder(
            chunk_size=self.DEFAULT_CHUNK_SIZE,
            overlap=self.DEFAULT_OVERLAP
        )

        encoder.add_chunks(text_chunks)
        encoder.build_video(
            mp4_path,
            index_path,
            fps=self.DEFAULT_FPS
        )

        # Get file sizes
        mp4_size = os.path.getsize(mp4_path)
        index_size = os.path.getsize(index_path)

        logger.info(f"Encoded to MP4: {mp4_path} ({mp4_size:,} bytes)")
        logger.info(f"Index file: {index_path} ({index_size:,} bytes)")

        return mp4_path, index_path

    # -------------------------------------------------------------------------
    # MEMVID DECODING
    # -------------------------------------------------------------------------

    def decode_from_mp4(
        self,
        mp4_path: str,
        index_path: str
    ) -> BotMemoryPackage:
        """
        Decode memory from MP4 video using memvid.

        Returns:
            BotMemoryPackage with reconstructed memory
        """
        if self.memvid is None:
            raise RuntimeError("Memvid not installed. Run: pip install memvid")

        from memvid import MemvidRetriever

        logger.info(f"Decoding MP4: {mp4_path}")

        retriever = MemvidRetriever(mp4_path, index_path)

        # Get all chunks - search with empty query to get everything
        # This is a workaround since memvid doesn't have a "get all" method
        all_results = retriever.search("", top_k=10000)

        chunks = []
        header_data = None
        personality = None

        for result in all_results:
            text = result.get('text', '') if isinstance(result, dict) else str(result)

            # Parse header
            if text.startswith("[HEADER]"):
                try:
                    header_json = text.split('\n', 1)[1]
                    header_data = json.loads(header_json)
                except Exception:
                    pass
                continue

            # Parse regular chunks
            chunk = MemoryChunk.from_text(text)

            if chunk.chunk_type == "personality":
                try:
                    personality = json.loads(chunk.content)
                except (json.JSONDecodeError, ValueError):
                    pass
            else:
                chunks.append(chunk)

        # Build package from header or defaults
        bot_info = header_data.get('bot', {}) if header_data else {}

        return BotMemoryPackage(
            bot_name=bot_info.get('bot_name', 'Unknown'),
            bot_type=bot_info.get('bot_type', 'unknown'),
            version=bot_info.get('version', '1.0'),
            created_at=bot_info.get('created_at', datetime.now().isoformat()),
            chunks=chunks,
            personality=personality,
            settings=bot_info.get('settings')
        )

    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------

    def get_memory_stats(self, memory_package: BotMemoryPackage) -> Dict:
        """Get statistics about a memory package."""
        type_counts = {}
        total_chars = 0

        for chunk in memory_package.chunks:
            type_counts[chunk.chunk_type] = type_counts.get(chunk.chunk_type, 0) + 1
            total_chars += len(chunk.content)

        return {
            "total_chunks": len(memory_package.chunks),
            "total_characters": total_chars,
            "chunk_types": type_counts,
            "has_personality": memory_package.personality is not None,
            "bot_name": memory_package.bot_name,
            "bot_type": memory_package.bot_type
        }

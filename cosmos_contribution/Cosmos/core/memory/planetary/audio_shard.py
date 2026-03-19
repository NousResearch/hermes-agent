"""
cosmos Planetary Audio Shard - Distributed TTS Cache
--------------------------------------------------------

"Good news, everyone! My voice can now echo across the planetary network!"

This module implements distributed caching for TTS audio files, allowing
cosmos instances to share pre-generated audio across the P2P network.

Features:
- **Audio Sharding**: Distributed storage using consistent hashing
- **P2P Sharing**: Gossip audio metadata to peers for cache hits
- **Privacy Safe**: Only shares text hashes, not actual text content
- **Lazy Loading**: Audio files transferred on-demand via P2P requests
"""

import asyncio
import json
import hashlib
import base64
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import  Optional
from pathlib import Path
from enum import Enum

from loguru import logger

# Try to import Nexus for signal handling
try:
    from Cosmos.core.nexus import nexus, Signal, SignalType
    NEXUS_AVAILABLE = True
except ImportError:
    NEXUS_AVAILABLE = False

# Try to import sharding
try:
    from Cosmos.memory.sharding import ShardManager
    SHARDING_AVAILABLE = True
except ImportError:
    SHARDING_AVAILABLE = False


class AudioScope(Enum):
    LOCAL_ONLY = "local_only"       # Never shared (personal voice samples)
    PLANETARY = "planetary"          # Shared with global swarm


@dataclass
class AudioMetadata:
    """Metadata for a cached audio file (shareable across P2P)."""
    text_hash: str              # MD5 hash of the text (privacy-preserving)
    audio_hash: str             # SHA256 hash of audio file (for verification)
    file_size: int              # Size in bytes
    duration_ms: int            # Estimated duration
    voice_id: str               # Voice identifier (e.g., "cosmos")
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    author_node: str = ""       # Node that generated this audio
    scope: str = "planetary"    # Sharing scope

    def to_any(self) -> dict:
        return {
            "text_hash": self.text_hash,
            "audio_hash": self.audio_hash,
            "file_size": self.file_size,
            "duration_ms": self.duration_ms,
            "voice_id": self.voice_id,
            "created_at": self.created_at,
            "author_node": self.author_node,
            "scope": self.scope
        }

    @classmethod
    def from_any(cls, data: dict) -> "AudioMetadata":
        return cls(
            text_hash=data["text_hash"],
            audio_hash=data["audio_hash"],
            file_size=data.get("file_size", 0),
            duration_ms=data.get("duration_ms", 0),
            voice_id=data.get("voice_id", "cosmos"),
            created_at=data.get("created_at", ""),
            author_node=data.get("author_node", ""),
            scope=data.get("scope", "planetary")
        )


class PlanetaryAudioShard:
    """
    Distributed audio cache for TTS voice cloning.

    Integrates with:
    - ShardManager for local distributed storage
    - P2P SwarmFabric for gossip-based metadata sharing
    - Nexus signal system for event handling
    """

    def __init__(
        self,
        cache_dir: Path,
        node_id: Optional[str] = None,
        num_shards: int = 4,
        use_p2p: bool = True
    ):
        self.cache_dir = Path(cache_dir)
        self.node_id = node_id or hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
        self.use_p2p = use_p2p

        # Shard manager for distributed local storage
        self.shard_manager = ShardManager(num_shards) if SHARDING_AVAILABLE else None
        self.num_shards = num_shards

        # Local metadata index (text_hash -> AudioMetadata)
        self.local_index: dict[str, AudioMetadata] = {}

        # Global cache from P2P network (text_hash -> AudioMetadata + peer info)
        self.global_index: dict[str, dict] = {}

        # Pending audio requests (for P2P fetching)
        self.pending_requests: dict[str, asyncio.Future] = {}

        # Initialize directories
        self._init_shard_dirs()

        # Load existing cache index
        self._load_index()

        # Setup signal handlers for P2P
        self._setup_signal_handlers()

        logger.info(f"PlanetaryAudioShard initialized: node={self.node_id}, shards={num_shards}")

    def _init_shard_dirs(self):
        """Create shard directories."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for i in range(self.num_shards):
            shard_dir = self.cache_dir / f"shard_{i}"
            shard_dir.mkdir(exist_ok=True)

        # Index file
        self.index_path = self.cache_dir / "audio_index.json"

    def _load_index(self):
        """Load existing audio index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    data = json.load(f)
                    for text_hash, meta_dict in data.get("local", {}).items():
                        self.local_index[text_hash] = AudioMetadata.from_any(meta_dict)
                    logger.info(f"Loaded {len(self.local_index)} cached audio entries")
            except Exception as e:
                logger.warning(f"Failed to load audio index: {e}")

    def _save_index(self):
        """Persist audio index to disk."""
        try:
            data = {
                "local": {k: v.to_any() for k, v in self.local_index.items()},
                "node_id": self.node_id,
                "updated_at": datetime.now().isoformat()
            }
            with open(self.index_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save audio index: {e}")

    def _setup_signal_handlers(self):
        """Register handlers for P2P audio events."""
        if not NEXUS_AVAILABLE:
            return

        def on_audio_metadata_received(signal):
            """Handle incoming audio metadata from P2P."""
            payload = signal.payload
            if payload.get("event") == "planetary_audio_metadata":
                meta_data = payload.get("metadata", {})
                peer_id = payload.get("peer_id", "unknown")

                if meta_data and meta_data.get("text_hash"):
                    text_hash = meta_data["text_hash"]

                    # Don't overwrite local entries
                    if text_hash not in self.local_index:
                        self.global_index[text_hash] = {
                            "metadata": AudioMetadata.from_any(meta_data),
                            "peer_id": peer_id,
                            "received_at": datetime.now().isoformat()
                        }
                        logger.debug(f"AudioShard: Cached metadata {text_hash[:8]}... from {peer_id}")

        def on_audio_request(signal):
            """Handle audio file request from peer."""
            payload = signal.payload
            if payload.get("event") == "audio_request":
                text_hash = payload.get("text_hash")
                requester = payload.get("requester_id")

                if text_hash and text_hash in self.local_index:
                    # We have this audio, send it back
                    asyncio.create_task(self._respond_to_audio_request(text_hash, requester))

        def on_audio_response(signal):
            """Handle audio file response from peer."""
            payload = signal.payload
            if payload.get("event") == "audio_response":
                text_hash = payload.get("text_hash")
                audio_data = payload.get("audio_data")  # Base64 encoded

                if text_hash and audio_data:
                    # Fulfill pending request
                    if text_hash in self.pending_requests:
                        future = self.pending_requests.pop(text_hash)
                        if not future.done():
                            future.set_result(audio_data)

        nexus.subscribe(SignalType.EXTERNAL_EVENT, on_audio_metadata_received)
        nexus.subscribe(SignalType.EXTERNAL_EVENT, on_audio_request)
        nexus.subscribe(SignalType.EXTERNAL_EVENT, on_audio_response)

    def get_shard_path(self, text_hash: str) -> Path:
        """Get the shard directory for a given text hash."""
        if self.shard_manager:
            shard_id = self.shard_manager.get_shard_id(text_hash)
        else:
            shard_id = int(hashlib.md5(text_hash.encode()).hexdigest(), 16) % self.num_shards
        return self.cache_dir / f"shard_{shard_id}"

    def get_audio_path(self, text_hash: str) -> Path:
        """Get the full path for an audio file."""
        shard_path = self.get_shard_path(text_hash)
        return shard_path / f"{text_hash}.wav"

    def has_audio(self, text_hash: str) -> bool:
        """Check if audio exists locally."""
        return text_hash in self.local_index and self.get_audio_path(text_hash).exists()

    def has_remote_audio(self, text_hash: str) -> bool:
        """Check if audio metadata exists in global cache (from peers)."""
        return text_hash in self.global_index

    def get_audio(self, text_hash: str) -> Optional[Path]:
        """Get local audio file path if it exists."""
        if self.has_audio(text_hash):
            return self.get_audio_path(text_hash)
        return None

    async def store_audio(
        self,
        text_hash: str,
        audio_data: bytes,
        voice_id: str = "cosmos",
        duration_ms: int = 0,
        scope: AudioScope = AudioScope.PLANETARY
    ) -> Path:
        """
        Store generated audio and optionally broadcast to P2P network.

        Args:
            text_hash: MD5 hash of the source text
            audio_data: Raw WAV audio bytes
            voice_id: Identifier for the voice used
            duration_ms: Audio duration in milliseconds
            scope: Sharing scope (LOCAL_ONLY or PLANETARY)

        Returns:
            Path to the stored audio file
        """
        # Determine shard and save file
        audio_path = self.get_audio_path(text_hash)
        audio_path.parent.mkdir(parents=True, exist_ok=True)

        with open(audio_path, "wb") as f:
            f.write(audio_data)

        # Calculate audio hash for verification
        audio_hash = hashlib.sha256(audio_data).hexdigest()

        # Create metadata
        metadata = AudioMetadata(
            text_hash=text_hash,
            audio_hash=audio_hash,
            file_size=len(audio_data),
            duration_ms=duration_ms,
            voice_id=voice_id,
            author_node=self.node_id,
            scope=scope.value
        )

        # Update local index
        self.local_index[text_hash] = metadata
        self._save_index()

        logger.info(f"AudioShard: Stored {text_hash[:8]}... in shard ({len(audio_data)} bytes)")

        # Broadcast metadata to P2P network if planetary scope
        if scope == AudioScope.PLANETARY and self.use_p2p:
            await self._broadcast_metadata(metadata)

        return audio_path

    async def _broadcast_metadata(self, metadata: AudioMetadata):
        """Broadcast audio metadata to P2P network."""
        try:
            from Cosmos.core.swarm.p2p import swarm_fabric

            # Send via P2P fabric
            message = {
                "type": "GOSSIP_AUDIO",
                "metadata": metadata.to_any(),
                "node_id": self.node_id
            }

            await swarm_fabric.broadcast_message(message)
            logger.debug(f"AudioShard: Broadcasted metadata {metadata.text_hash[:8]}...")

        except ImportError:
            logger.debug("P2P fabric not available for audio broadcast")
        except Exception as e:
            logger.warning(f"AudioShard: Broadcast failed: {e}")

    async def request_audio_from_peer(self, text_hash: str, timeout: float = 10.0) -> Optional[bytes]:
        """
        Request audio file from a peer that has it cached.

        Args:
            text_hash: The text hash to request
            timeout: Timeout in seconds

        Returns:
            Audio data bytes if received, None otherwise
        """
        if text_hash not in self.global_index:
            return None

        peer_info = self.global_index[text_hash]
        peer_id = peer_info.get("peer_id")

        if not peer_id:
            return None

        try:
            from Cosmos.core.swarm.p2p import swarm_fabric

            # Create future for response
            future = asyncio.Future()
            self.pending_requests[text_hash] = future

            # Send request
            request = {
                "type": "AUDIO_REQUEST",
                "text_hash": text_hash,
                "requester_id": self.node_id
            }

            await swarm_fabric.send_to_peer(peer_id, request)

            # Wait for response with timeout
            try:
                audio_b64 = await asyncio.wait_for(future, timeout=timeout)
                audio_data = base64.b64decode(audio_b64)

                # Store locally for future use
                metadata = peer_info["metadata"]
                await self.store_audio(
                    text_hash,
                    audio_data,
                    voice_id=metadata.voice_id,
                    duration_ms=metadata.duration_ms,
                    scope=AudioScope.LOCAL_ONLY  # Don't re-broadcast
                )

                return audio_data

            except asyncio.TimeoutError:
                logger.warning(f"AudioShard: Timeout requesting {text_hash[:8]}... from {peer_id}")
                self.pending_requests.pop(text_hash, None)
                return None

        except ImportError:
            logger.debug("P2P fabric not available for audio request")
            return None
        except Exception as e:
            logger.warning(f"AudioShard: Request failed: {e}")
            return None

    async def _respond_to_audio_request(self, text_hash: str, requester_id: str):
        """Send audio file to requesting peer."""
        try:
            from Cosmos.core.swarm.p2p import swarm_fabric

            audio_path = self.get_audio_path(text_hash)
            if not audio_path.exists():
                return

            # Read and encode audio
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            audio_b64 = base64.b64encode(audio_data).decode()

            # Send response
            response = {
                "type": "AUDIO_RESPONSE",
                "text_hash": text_hash,
                "audio_data": audio_b64,
                "sender_id": self.node_id
            }

            await swarm_fabric.send_to_peer(requester_id, response)
            logger.info(f"AudioShard: Sent {text_hash[:8]}... to {requester_id}")

        except Exception as e:
            logger.warning(f"AudioShard: Failed to respond to request: {e}")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_size = 0
        for text_hash in self.local_index:
            path = self.get_audio_path(text_hash)
            if path.exists():
                total_size += path.stat().st_size

        return {
            "local_entries": len(self.local_index),
            "global_entries": len(self.global_index),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "node_id": self.node_id,
            "num_shards": self.num_shards
        }

    def clear_cache(self, older_than_days: int = 30):
        """Clear old cache entries."""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=older_than_days)

        removed = 0
        for text_hash, metadata in list(self.local_index.items()):
            try:
                created = datetime.fromisoformat(metadata.created_at)
                if created < cutoff:
                    # Remove file
                    audio_path = self.get_audio_path(text_hash)
                    if audio_path.exists():
                        audio_path.unlink()
                    # Remove from index
                    del self.local_index[text_hash]
                    removed += 1
            except Exception:
                pass

        if removed > 0:
            self._save_index()
            logger.info(f"AudioShard: Cleared {removed} old cache entries")

        return removed


# Global instance (lazy loaded)
_audio_shard: Optional[PlanetaryAudioShard] = None

def get_audio_shard(cache_dir: Optional[Path] = None) -> PlanetaryAudioShard:
    """Get or create the global audio shard instance."""
    global _audio_shard
    if _audio_shard is None:
        if cache_dir is None:
            # Default to web static audio cache
            cache_dir = Path(__file__).parent.parent.parent.parent / "web" / "static" / "audio" / "cache"
        _audio_shard = PlanetaryAudioShard(cache_dir)
    return _audio_shard

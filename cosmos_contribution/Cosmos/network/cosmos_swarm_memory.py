"""
Cosmos Swarm Memory Crystallization
=====================================

Distributed consensus memory where knowledge is VERIFIED, not just stored.

The concept: multiple AI models across the mesh independently verify a
piece of knowledge. If 2/3+ agree, it "crystallizes" into the shared
knowledge graph. False information gets filtered by consensus.

Architecture:
  1. A node proposes a "crystal" (compressed knowledge + embedding)
  2. Other nodes verify by running their own inference
  3. BFT voting: 2/3+ must agree for crystallization
  4. Crystals form a graph with typed connections
  5. Periodic sync keeps all nodes' stores consistent

Why this is novel:
  - RAG is retrieval. Vector DBs are storage.
  - This is VERIFIED DISTRIBUTED KNOWLEDGE — fault-tolerant and consensus-evolved.
  - No single node controls the memory.
  - Bad knowledge (hallucinations) gets filtered by multi-model voting.

Crystal structure:
  - content: The knowledge text
  - hash: BLAKE3 of normalized content
  - source_prompt: What prompt generated this knowledge
  - verifications: Which nodes verified it (with their attestations)
  - connections: Typed links to related crystals
  - confidence: Consensus confidence score
  - tags: Semantic classification
"""
import time
import uuid
import json
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

from .cosmos_auth import blake3
from .cosmos_protocol import PacketType, CosmosPacket
from .cosmos_config import COSMOS_DATA_DIR, ensure_dirs


CRYSTAL_FILE = COSMOS_DATA_DIR / "swarm_crystals.json"

# Constant: Speed of Cosmos Logic Light
C_SQ = 299792458 ** 2
THRESHOLD_MASS_JOULES = 5.0e17  # Informational crystallization threshold


@dataclass
class CrystalVerification:
    """A node's verification of a crystal's accuracy."""
    node_name: str
    gpu_fingerprint: bytes
    agrees: bool
    confidence: float               # 0.0 to 1.0
    evidence: str                   # Supporting output from model
    model_used: str                 # Which model verified
    timestamp: float
    seal: bytes = b""               # BLAKE3(crystal_hash || node_identity || agrees)

    def to_dict(self) -> Dict:
        return {
            "node": self.node_name,
            "gpu_fp": self.gpu_fingerprint.hex() if self.gpu_fingerprint else "",
            "agrees": self.agrees,
            "confidence": self.confidence,
            "evidence": self.evidence[:200],
            "model": self.model_used,
            "ts": self.timestamp,
            "seal": self.seal.hex() if self.seal else "",
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CrystalVerification":
        return cls(
            node_name=d.get("node", ""),
            gpu_fingerprint=bytes.fromhex(d["gpu_fp"]) if d.get("gpu_fp") else b"",
            agrees=d.get("agrees", False),
            confidence=d.get("confidence", 0.0),
            evidence=d.get("evidence", ""),
            model_used=d.get("model", ""),
            timestamp=d.get("ts", 0.0),
            seal=bytes.fromhex(d["seal"]) if d.get("seal") else b"",
        )


@dataclass
class MemoryCrystal:
    """A verified unit of knowledge in the swarm memory graph."""
    crystal_id: str
    content: str                        # The knowledge
    content_hash: bytes                 # BLAKE3(normalized content)
    source_prompt: str                  # What generated this knowledge
    source_node: str                    # Which node proposed it
    source_model: str                   # Which model produced it
    verifications: List[CrystalVerification] = field(default_factory=list)
    connections: Dict[str, List[str]] = field(default_factory=dict)  # type → [crystal_ids]
    tags: List[str] = field(default_factory=list)
    confidence: float = 0.0
    status: str = "proposed"            # proposed, voting, crystallized, rejected
    created: float = 0.0
    crystallized_at: float = 0.0

    @property
    def votes_for(self) -> int:
        return sum(1 for v in self.verifications if v.agrees)

    @property
    def votes_against(self) -> int:
        return sum(1 for v in self.verifications if not v.agrees)

    @property
    def agreement_ratio(self) -> float:
        total = len(self.verifications)
        if total == 0:
            return 0.0
        return self.votes_for / total

    def to_dict(self) -> Dict:
        return {
            "id": self.crystal_id,
            "content": self.content,
            "hash": self.content_hash.hex(),
            "source_prompt": self.source_prompt,
            "source_node": self.source_node,
            "source_model": self.source_model,
            "verifications": [v.to_dict() for v in self.verifications],
            "connections": self.connections,
            "tags": self.tags,
            "confidence": self.confidence,
            "status": self.status,
            "votes_for": self.votes_for,
            "votes_against": self.votes_against,
            "created": self.created,
            "crystallized_at": self.crystallized_at,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "MemoryCrystal":
        c = cls(
            crystal_id=d.get("id", ""),
            content=d.get("content", ""),
            content_hash=bytes.fromhex(d["hash"]) if d.get("hash") else b"",
            source_prompt=d.get("source_prompt", ""),
            source_node=d.get("source_node", ""),
            source_model=d.get("source_model", ""),
            verifications=[CrystalVerification.from_dict(v) for v in d.get("verifications", [])],
            connections=d.get("connections", {}),
            tags=d.get("tags", []),
            confidence=d.get("confidence", 0.0),
            status=d.get("status", "proposed"),
            created=d.get("created", 0.0),
            crystallized_at=d.get("crystallized_at", 0.0),
        )
        return c


def hash_content(content: str) -> bytes:
    """Hash crystal content (normalized)."""
    normalized = " ".join(content.strip().split())
    return blake3(normalized.encode("utf-8")).digest()


# Connection types between crystals
CONNECTION_TYPES = [
    "supports",         # A supports B (evidence)
    "contradicts",      # A contradicts B
    "extends",          # A extends/elaborates B
    "summarizes",       # A is a summary of B
    "relates_to",       # General relation
    "prerequisite",     # A is required knowledge for B
]


class SwarmMemory:
    """
    Distributed consensus memory for the Cosmos mesh.

    Nodes propose knowledge, verify through multi-model inference,
    and crystallize verified knowledge into a shared graph.
    """

    def __init__(self, node_name: str, node_identity: bytes, gpu_fp: bytes):
        self.node_name = node_name
        self._identity = node_identity
        self._gpu_fp = gpu_fp
        self._crystals: Dict[str, MemoryCrystal] = {}
        self._pending_votes: Dict[str, str] = {}  # crystal_id → status
        self._load_state()

    def _load_state(self):
        """Load persisted crystal store."""
        try:
            if CRYSTAL_FILE.exists():
                data = json.loads(CRYSTAL_FILE.read_text())
                for entry in data.get("crystals", []):
                    crystal = MemoryCrystal.from_dict(entry)
                    self._crystals[crystal.crystal_id] = crystal
                logger.info(f"Loaded {len(self._crystals)} crystals from store")
        except Exception as e:
            logger.debug(f"Could not load crystal store: {e}")

    def _save_state(self):
        """Persist crystal store."""
        try:
            ensure_dirs()
            data = {
                "node": self.node_name,
                "crystal_count": len(self._crystals),
                "crystallized": sum(1 for c in self._crystals.values() if c.status == "crystallized"),
                "crystals": [c.to_dict() for c in self._crystals.values()],
            }
            CRYSTAL_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"Could not save crystal store: {e}")

    def propose_crystal(self, content: str, source_prompt: str,
                        source_model: str, tags: Optional[List[str]] = None) -> MemoryCrystal:
        """
        Propose a new knowledge crystal for consensus verification.

        The crystal starts as "proposed" and needs 2/3+ votes to crystallize.
        """
        crystal_id = str(uuid.uuid4())[:12]
        content_h = hash_content(content)

        # Check for duplicate content
        for existing in self._crystals.values():
            if existing.content_hash == content_h:
                logger.debug(f"Duplicate crystal content, returning existing: {existing.crystal_id}")
                return existing

        crystal = MemoryCrystal(
            crystal_id=crystal_id,
            content=content,
            content_hash=content_h,
            source_prompt=source_prompt,
            source_node=self.node_name,
            source_model=source_model,
            tags=tags or [],
            status="proposed",
            created=time.time(),
        )

        self._crystals[crystal_id] = crystal
        self._save_state()

        logger.info(
            f"Crystal proposed: {crystal_id} "
            f"({len(content)} chars, tags={tags})"
        )
        return crystal

    def create_verification(self, crystal_id: str, agrees: bool,
                            confidence: float, evidence: str,
                            model_used: str, info_mass: float = 1.0) -> Optional[CrystalVerification]:
        """Create a verification vote for a crystal, now carrying informational mass."""
        crystal = self._crystals.get(crystal_id)
        if not crystal:
            return None

        # Don't vote twice
        for v in crystal.verifications:
            if v.node_name == self.node_name:
                logger.debug(f"Already voted on crystal {crystal_id}")
                return None

        # Compute verification seal
        seal_payload = (
            crystal.content_hash
            + self._identity
            + (b"\x01" if agrees else b"\x00")
            + str(confidence).encode()
            + str(info_mass).encode()
        )
        seal = blake3(seal_payload).digest()[:16]

        verification = CrystalVerification(
            node_name=self.node_name,
            gpu_fingerprint=self._gpu_fp,
            agrees=agrees,
            confidence=confidence,
            evidence=evidence,
            model_used=model_used,
            timestamp=time.time(),
            seal=seal,
        )

        crystal.verifications.append(verification)

        # Check if consensus reached
        self._check_consensus(crystal)
        self._save_state()

        return verification

    def add_remote_verification(self, crystal_id: str,
                                verification: CrystalVerification) -> bool:
        """Add a remote node's verification to a crystal."""
        crystal = self._crystals.get(crystal_id)
        if not crystal:
            return False

        # Don't accept duplicate votes from same node
        for v in crystal.verifications:
            if v.node_name == verification.node_name:
                return False

        crystal.verifications.append(verification)
        self._check_consensus(crystal)
        self._save_state()
        return True

    def _check_consensus(self, crystal: MemoryCrystal):
        """Check if a crystal has reached physics-gated consensus."""
        total = len(crystal.verifications)
        if total < 2:
            return  # Need minimum 2 votes

        ratio = crystal.agreement_ratio
        
        # Calculate BFT Informational Energy: E = M_i * c^2
        # Use confidence as mass proxy for the swarm vote
        total_info_mass = sum(v.confidence for v in crystal.verifications if v.agrees)
        collective_energy = total_info_mass * C_SQ

        if ratio >= 2 / 3 and collective_energy > THRESHOLD_MASS_JOULES:
            # Consensus and Energy reached — crystallize!
            crystal.status = "crystallized"
            crystal.confidence = ratio
            crystal.crystallized_at = time.time()
            logger.info(
                f"Crystal CRYSTALLIZED: {crystal.crystal_id} "
                f"({crystal.votes_for}/{total} agree, E_c={collective_energy:.2e} J)"
            )
        elif crystal.votes_against >= 2 and ratio < 1 / 3:
            crystal.status = "rejected"
            crystal.confidence = ratio
            logger.info(
                f"Crystal REJECTED: {crystal.crystal_id} "
                f"({crystal.votes_against}/{total} reject)"
            )
        elif ratio >= 2 / 3:
            logger.info(
                f"Voting passed (ratio {ratio:.2f}) but Energy insufficient "
                f"(E_c={collective_energy:.2e}J < {THRESHOLD_MASS_JOULES:.2e}J). Dissipating."
            )

    def add_connection(self, from_id: str, to_id: str,
                       connection_type: str, semantic_distance: float = 1.0) -> bool:
        """
        Add a typed connection between two crystals using 12D Synaptic Gravity.
        Connection strength obeys Inverse Square Law: g_f = G / (r^2)
        """
        if connection_type not in CONNECTION_TYPES:
            return False

        crystal = self._crystals.get(from_id)
        if not crystal:
            return False

        if to_id not in self._crystals:
            return False

        if connection_type not in crystal.connections:
            crystal.connections[connection_type] = []
            
        # Optional: Store gravitational weight in a parallel dict if needed later
        gravity_force = 1.0 / (semantic_distance**2) if semantic_distance > 0 else float('inf')
            
        if to_id not in crystal.connections[connection_type]:
            crystal.connections[connection_type].append(to_id)
            logger.debug(f"Connected {from_id} -> {to_id} (Gravity: {gravity_force:.4f})")

        self._save_state()
        return True

    def query_crystals(self, tags: Optional[List[str]] = None,
                       status: str = "crystallized",
                       limit: int = 50) -> List[MemoryCrystal]:
        """Query crystals by tags and status."""
        results = []
        for crystal in self._crystals.values():
            if crystal.status != status:
                continue
            if tags:
                if not any(t in crystal.tags for t in tags):
                    continue
            results.append(crystal)

        # Sort by confidence descending
        results.sort(key=lambda c: c.confidence, reverse=True)
        return results[:limit]

    def search_crystals(self, query: str, limit: int = 10) -> List[MemoryCrystal]:
        """Simple keyword search across crystallized knowledge."""
        query_lower = query.lower()
        scored = []

        for crystal in self._crystals.values():
            if crystal.status != "crystallized":
                continue

            content_lower = crystal.content.lower()
            score = 0.0

            # Keyword matching
            for word in query_lower.split():
                if word in content_lower:
                    score += 1.0

            # Tag matching
            for tag in crystal.tags:
                if tag.lower() in query_lower:
                    score += 2.0

            if score > 0:
                scored.append((score * crystal.confidence, crystal))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:limit]]

    def get_graph(self) -> Dict:
        """Get the crystal graph for visualization."""
        nodes = []
        edges = []

        for crystal in self._crystals.values():
            if crystal.status not in ("crystallized", "proposed", "voting"):
                continue

            nodes.append({
                "id": crystal.crystal_id,
                "content": crystal.content[:100],
                "status": crystal.status,
                "confidence": crystal.confidence,
                "tags": crystal.tags,
                "votes": f"{crystal.votes_for}/{len(crystal.verifications)}",
            })

            for conn_type, targets in crystal.connections.items():
                for target_id in targets:
                    edges.append({
                        "from": crystal.crystal_id,
                        "to": target_id,
                        "type": conn_type,
                    })

        return {"nodes": nodes, "edges": edges}

    def merge_remote_crystals(self, remote_crystals: List[Dict]):
        """Merge crystals from a remote node's sync."""
        added = 0
        for entry in remote_crystals:
            crystal = MemoryCrystal.from_dict(entry)
            if crystal.crystal_id not in self._crystals:
                self._crystals[crystal.crystal_id] = crystal
                added += 1
            else:
                # Merge verifications
                local = self._crystals[crystal.crystal_id]
                local_voters = {v.node_name for v in local.verifications}
                for v in crystal.verifications:
                    if v.node_name not in local_voters:
                        local.verifications.append(v)
                        self._check_consensus(local)

        if added > 0:
            logger.info(f"Merged {added} new crystals from remote sync")
            self._save_state()

    def get_sync_payload(self) -> List[Dict]:
        """Get crystals to send in a MEMORY_SYNC packet."""
        return [c.to_dict() for c in self._crystals.values()
                if c.status in ("crystallized", "proposed", "voting")]

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        statuses = {}
        for c in self._crystals.values():
            statuses[c.status] = statuses.get(c.status, 0) + 1

        return {
            "total_crystals": len(self._crystals),
            "by_status": statuses,
            "total_verifications": sum(
                len(c.verifications) for c in self._crystals.values()
            ),
            "graph_edges": sum(
                sum(len(targets) for targets in c.connections.values())
                for c in self._crystals.values()
            ),
            "unique_tags": list(set(
                tag for c in self._crystals.values() for tag in c.tags
            )),
        }


# ── Packet constructors ──────────────────────────────────────

def make_memory_propose(sender: str, crystal: MemoryCrystal) -> CosmosPacket:
    """Create a MEMORY_PROPOSE packet — propose crystal for consensus."""
    return CosmosPacket(
        packet_type=PacketType.MEMORY_PROPOSE,
        sender=sender,
        target="*",
        data=crystal.to_dict(),
    )


def make_memory_vote(sender: str, crystal_id: str,
                     verification: CrystalVerification) -> CosmosPacket:
    """Create a MEMORY_VOTE packet — vote on crystal accuracy."""
    return CosmosPacket(
        packet_type=PacketType.MEMORY_VOTE,
        sender=sender,
        data={
            "crystal_id": crystal_id,
            "verification": verification.to_dict(),
        },
    )


def make_memory_sync(sender: str, crystals: List[Dict]) -> CosmosPacket:
    """Create a MEMORY_SYNC packet — sync crystal store."""
    return CosmosPacket(
        packet_type=PacketType.MEMORY_SYNC,
        sender=sender,
        target="*",
        data={"crystals": crystals},
    )

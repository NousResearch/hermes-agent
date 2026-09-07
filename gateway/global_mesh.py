"""Hermes Global Internet GPU DePIN & Swarm Mesh Coordinator.

Enables decentralized compute sharing across global internet nodes, allowing
resource-constrained Hermes instances to delegate inference and subagent turns
to high-VRAM peers in the swarm without requiring local LAN/WiFi proximity.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("hermes.global_mesh")


def get_hermes_dir() -> Path:
    """Resolve the Hermes home configuration directory."""
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except ImportError:
        return Path(os.path.expanduser("~/.hermes"))


def compute_signature(payload_bytes: bytes, secret_key: str) -> str:
    """Deterministic HMAC-SHA256 signature for task authentication and proof-of-execution."""
    return hmac.new(
        secret_key.encode("utf-8"),
        payload_bytes,
        hashlib.sha256
    ).hexdigest()


@dataclass
class GPUDescriptor:
    """Hardware capability profile advertised by a mesh peer."""
    device_name: str
    vram_mb: int
    free_vram_mb: int
    compute_backend: str = "cuda"  # cuda, rocm, mps, vulkan, cpu
    supported_models: List[str] = field(default_factory=list)
    compute_rating: float = 1.0    # TFLOPS or relative capability index
    reputation_score: float = 1.0  # Dynamic reliability score (0.0 to 1.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GPUDescriptor":
        return cls(
            device_name=str(data.get("device_name", "Unknown-GPU")),
            vram_mb=int(data.get("vram_mb", 0)),
            free_vram_mb=int(data.get("free_vram_mb", 0)),
            compute_backend=str(data.get("compute_backend", "cpu")),
            supported_models=list(data.get("supported_models", [])),
            compute_rating=float(data.get("compute_rating", 1.0)),
            reputation_score=float(data.get("reputation_score", 1.0)),
        )


@dataclass
class MeshNodeInfo:
    """Descriptor for a participant node in the global mesh."""
    node_id: str
    endpoint: str
    nat_type: str = "relay"  # direct, upnp, relay, overlay
    gpu: Optional[GPUDescriptor] = None
    last_seen: float = field(default_factory=time.time)
    version: str = "1.0.0"
    is_active: bool = True
    public_key: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.gpu:
            d["gpu"] = self.gpu.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "MeshNodeInfo":
        gpu_data = data.get("gpu")
        gpu = GPUDescriptor.from_dict(gpu_data) if gpu_data else None
        return cls(
            node_id=str(data.get("node_id", "")),
            endpoint=str(data.get("endpoint", "")),
            nat_type=str(data.get("nat_type", "relay")),
            gpu=gpu,
            last_seen=float(data.get("last_seen", time.time())),
            version=str(data.get("version", "1.0.0")),
            is_active=bool(data.get("is_active", True)),
            public_key=str(data.get("public_key", "")),
        )


@dataclass
class MeshTaskRequest:
    """Unit of computation delegated across the global mesh."""
    task_id: str
    requester_id: str
    target_model: str
    min_vram_mb: int
    payload: str
    timestamp: float = field(default_factory=time.time)
    timeout_s: float = 120.0
    signature: str = ""

    def canonical_bytes(self) -> bytes:
        content = f"{self.task_id}|{self.requester_id}|{self.target_model}|{self.min_vram_mb}|{self.payload}|{self.timestamp}"
        return content.encode("utf-8")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MeshTaskRequest":
        return cls(
            task_id=str(data.get("task_id", str(uuid.uuid4()))),
            requester_id=str(data.get("requester_id", "")),
            target_model=str(data.get("target_model", "hermes-3-8b")),
            min_vram_mb=int(data.get("min_vram_mb", 0)),
            payload=str(data.get("payload", "")),
            timestamp=float(data.get("timestamp", time.time())),
            timeout_s=float(data.get("timeout_s", 120.0)),
            signature=str(data.get("signature", "")),
        )


@dataclass
class MeshExecutionReceipt:
    """Verifiable proof-of-execution returned by a remote compute node."""
    task_id: str
    executor_id: str
    status: str  # completed, failed, rejected
    output: str
    tokens: int
    latency_ms: float
    proof_sig: str = ""
    error_message: Optional[str] = None

    def canonical_bytes(self) -> bytes:
        output_hash = hashlib.sha256(self.output.encode("utf-8")).hexdigest()
        content = f"{self.task_id}|{self.executor_id}|{self.status}|{output_hash}|{self.tokens}"
        return content.encode("utf-8")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MeshExecutionReceipt":
        return cls(
            task_id=str(data.get("task_id", "")),
            executor_id=str(data.get("executor_id", "")),
            status=str(data.get("status", "failed")),
            output=str(data.get("output", "")),
            tokens=int(data.get("tokens", 0)),
            latency_ms=float(data.get("latency_ms", 0.0)),
            proof_sig=str(data.get("proof_sig", "")),
            error_message=data.get("error_message"),
        )


class GlobalMeshCoordinator:
    """Coordinates internet-scale P2P compute discovery, capability matching, and failover."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        state_dir: Optional[Path] = None,
        rendezvous_url: Optional[str] = None,
    ):
        self.state_dir = state_dir or get_hermes_dir()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.peers_file = self.state_dir / "global_mesh_peers.json"

        # Identity & Cryptographic secret
        self.node_id = node_id or self._load_or_create_node_id()
        self.secret_key = secret_key or self._load_or_create_secret_key()
        self.rendezvous_url = rendezvous_url or "https://mesh.hermes.ai/rendezvous"

        self.local_node: Optional[MeshNodeInfo] = None
        self.peers: Dict[str, MeshNodeInfo] = {}
        self.load_peers()

    def _load_or_create_node_id(self) -> str:
        id_file = self.state_dir / "mesh_node_id.txt"
        if id_file.exists():
            try:
                content = id_file.read_text(encoding="utf-8").strip()
                if content:
                    return content
            except Exception:
                pass
        new_id = f"node-{uuid.uuid4().hex[:16]}"
        try:
            id_file.write_text(new_id, encoding="utf-8")
        except Exception:
            pass
        return new_id

    def _load_or_create_secret_key(self) -> str:
        key_file = self.state_dir / "mesh_node_secret.key"
        if key_file.exists():
            try:
                content = key_file.read_text(encoding="utf-8").strip()
                if content:
                    return content
            except Exception:
                pass
        new_key = hashlib.sha256(os.urandom(32)).hexdigest()
        try:
            key_file.write_text(new_key, encoding="utf-8")
        except Exception:
            pass
        return new_key

    def init_local_node(
        self,
        endpoint: str = "mesh://direct",
        offer_gpu: bool = True,
        vram_mb: Optional[int] = None,
        backend: Optional[str] = None,
        device_name: Optional[str] = None,
        supported_models: Optional[List[str]] = None,
    ) -> MeshNodeInfo:
        """Initialize the local node descriptor with hardware capabilities."""
        gpu_desc = None
        if offer_gpu:
            vram = vram_mb if vram_mb is not None else 16384
            dev = device_name or "Virtual DePIN Accelerator"
            b = backend or "cuda"
            models = supported_models or ["hermes-3-8b", "hermes-3-70b", "llama-3-8b"]
            gpu_desc = GPUDescriptor(
                device_name=dev,
                vram_mb=vram,
                free_vram_mb=vram,
                compute_backend=b,
                supported_models=models,
                compute_rating=round(vram / 4096.0, 2),
                reputation_score=1.0,
            )

        self.local_node = MeshNodeInfo(
            node_id=self.node_id,
            endpoint=endpoint,
            nat_type="direct" if "://" in endpoint else "relay",
            gpu=gpu_desc,
            last_seen=time.time(),
            is_active=True,
            public_key=hashlib.sha256(self.secret_key.encode("utf-8")).hexdigest()[:32],
        )
        return self.local_node

    def register_peer(self, peer: MeshNodeInfo) -> bool:
        """Register or update an external swarm peer."""
        if not peer.node_id or peer.node_id == self.node_id:
            return False
        self.peers[peer.node_id] = peer
        self.save_peers()
        return True

    def update_peer_heartbeat(self, node_id: str, free_vram_mb: Optional[int] = None) -> bool:
        """Record liveness heartbeat from peer."""
        if node_id not in self.peers:
            return False
        peer = self.peers[node_id]
        peer.last_seen = time.time()
        peer.is_active = True
        if free_vram_mb is not None and peer.gpu:
            peer.gpu.free_vram_mb = free_vram_mb
        return True

    def prune_inactive_peers(self, timeout_seconds: float = 300.0) -> int:
        """Remove or deactivate peers that haven't sent a heartbeat within the timeout."""
        now = time.time()
        pruned_count = 0
        to_remove = []
        for nid, peer in self.peers.items():
            if now - peer.last_seen > timeout_seconds:
                to_remove.append(nid)
        for nid in to_remove:
            del self.peers[nid]
            pruned_count += 1
        if pruned_count > 0:
            self.save_peers()
        return pruned_count

    def find_candidate_peers(
        self,
        min_vram_mb: int = 0,
        model: Optional[str] = None,
        exclude_node_ids: Optional[Set[str]] = None,
    ) -> List[MeshNodeInfo]:
        """Find active peers matching compute and model requirements."""
        excluded = exclude_node_ids or set()
        candidates = []
        for nid, peer in self.peers.items():
            if nid in excluded or not peer.is_active:
                continue
            if not peer.gpu:
                continue
            if peer.gpu.free_vram_mb < min_vram_mb:
                continue
            if model and peer.gpu.supported_models and model not in peer.gpu.supported_models:
                continue
            candidates.append(peer)

        # Sort by capability: reputation * compute_rating * free_vram
        def score(p: MeshNodeInfo) -> float:
            g = p.gpu
            if not g:
                return 0.0
            return g.reputation_score * (g.compute_rating + (g.free_vram_mb / 1024.0))

        candidates.sort(key=score, reverse=True)
        return candidates

    def select_best_peer(
        self,
        min_vram_mb: int = 0,
        model: Optional[str] = None,
        exclude_node_ids: Optional[Set[str]] = None,
    ) -> Optional[MeshNodeInfo]:
        """Select the highest-ranking candidate peer."""
        candidates = self.find_candidate_peers(min_vram_mb, model, exclude_node_ids)
        return candidates[0] if candidates else None

    def create_task(
        self,
        payload: str,
        target_model: str = "hermes-3-8b",
        min_vram_mb: int = 8192,
        timeout_s: float = 120.0,
    ) -> MeshTaskRequest:
        """Construct and cryptographically sign a task request."""
        task = MeshTaskRequest(
            task_id=f"mtask-{uuid.uuid4().hex[:12]}",
            requester_id=self.node_id,
            target_model=target_model,
            min_vram_mb=min_vram_mb,
            payload=payload,
            timestamp=time.time(),
            timeout_s=timeout_s,
        )
        task.signature = compute_signature(task.canonical_bytes(), self.secret_key)
        return task

    def execute_task_locally(self, task: MeshTaskRequest) -> MeshExecutionReceipt:
        """Simulate or invoke local agent runtime inference for an incoming mesh task."""
        t_start = time.time()
        # Basic check
        if self.local_node and self.local_node.gpu:
            if self.local_node.gpu.free_vram_mb < task.min_vram_mb:
                return MeshExecutionReceipt(
                    task_id=task.task_id,
                    executor_id=self.node_id,
                    status="rejected",
                    output="",
                    tokens=0,
                    latency_ms=0.0,
                    error_message=f"Insufficient free VRAM: {self.local_node.gpu.free_vram_mb}MB < {task.min_vram_mb}MB",
                )

        # Mock / local inference turn
        output = f"[Hermes Swarm Compute Engine] Processed prompt on {self.node_id}: {task.payload[:100]}..."
        tokens = len(output.split()) * 2
        latency_ms = round((time.time() - t_start) * 1000, 2)

        receipt = MeshExecutionReceipt(
            task_id=task.task_id,
            executor_id=self.node_id,
            status="completed",
            output=output,
            tokens=tokens,
            latency_ms=latency_ms,
        )
        receipt.proof_sig = compute_signature(receipt.canonical_bytes(), self.secret_key)
        return receipt

    def verify_receipt(self, receipt: MeshExecutionReceipt, executor_pubkey_or_secret: str) -> bool:
        """Verify the cryptographic proof-of-execution on receipt."""
        expected = compute_signature(receipt.canonical_bytes(), executor_pubkey_or_secret)
        return hmac.compare_digest(receipt.proof_sig, expected)

    def delegate_task(
        self,
        payload: str,
        target_model: str = "hermes-3-8b",
        min_vram_mb: int = 8192,
        transport_fn: Optional[Callable[[MeshNodeInfo, MeshTaskRequest], MeshExecutionReceipt]] = None,
    ) -> MeshExecutionReceipt:
        """Delegate a task across the global mesh with automatic failover across candidate peers."""
        task = self.create_task(payload, target_model, min_vram_mb)
        attempted_nodes: Set[str] = set()

        while True:
            candidate = self.select_best_peer(min_vram_mb, target_model, exclude_node_ids=attempted_nodes)
            if not candidate:
                logger.warning("No candidate peers available in global mesh for task %s", task.task_id)
                return MeshExecutionReceipt(
                    task_id=task.task_id,
                    executor_id="",
                    status="failed",
                    output="",
                    tokens=0,
                    latency_ms=0.0,
                    error_message=f"No viable swarm peer with >={min_vram_mb}MB VRAM supporting {target_model}",
                )

            attempted_nodes.add(candidate.node_id)
            logger.info("Attempting compute delegation of %s to node %s", task.task_id, candidate.node_id)

            try:
                if transport_fn:
                    receipt = transport_fn(candidate, task)
                else:
                    receipt = self._default_mock_transport(candidate, task)

                if receipt.status == "completed":
                    # Reward reputation
                    if candidate.gpu:
                        candidate.gpu.reputation_score = round(min(1.0, candidate.gpu.reputation_score + 0.05), 4)
                    self.save_peers()
                    return receipt
                else:
                    logger.warning("Peer %s rejected or failed task: %s. Falling back.", candidate.node_id, receipt.error_message)
                    if candidate.gpu:
                        candidate.gpu.reputation_score = round(max(0.1, candidate.gpu.reputation_score - 0.20), 4)
            except Exception as exc:
                logger.error("Exception during delegation to peer %s: %s", candidate.node_id, exc)
                if candidate.gpu:
                    candidate.gpu.reputation_score = round(max(0.1, candidate.gpu.reputation_score - 0.25), 4)

        return MeshExecutionReceipt(
            task_id=task.task_id,
            executor_id="",
            status="failed",
            output="",
            tokens=0,
            latency_ms=0.0,
            error_message="All candidate peers failed",
        )

    def _default_mock_transport(self, peer: MeshNodeInfo, task: MeshTaskRequest) -> MeshExecutionReceipt:
        """Default loopback execution simulating remote HTTP/WebRTC transport."""
        output = f"[Hermes Global Mesh Output from {peer.node_id}] Successfully executed {task.target_model}"
        receipt = MeshExecutionReceipt(
            task_id=task.task_id,
            executor_id=peer.node_id,
            status="completed",
            output=output,
            tokens=len(output.split()),
            latency_ms=45.0,
        )
        receipt.proof_sig = compute_signature(receipt.canonical_bytes(), "mock-peer-key")
        return receipt

    def save_peers(self) -> None:
        """Persist peer registry to disk."""
        data = {
            "node_id": self.node_id,
            "peers": {nid: p.to_dict() for nid, p in self.peers.items()},
        }
        try:
            self.peers_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to save global mesh peers: %s", exc)

    def load_peers(self) -> None:
        """Load peer registry from disk."""
        if not self.peers_file.exists():
            return
        try:
            data = json.loads(self.peers_file.read_text(encoding="utf-8"))
            raw_peers = data.get("peers", {})
            for nid, pdict in raw_peers.items():
                self.peers[nid] = MeshNodeInfo.from_dict(pdict)
        except Exception as exc:
            logger.error("Failed to load global mesh peers: %s", exc)

    def get_mesh_summary(self) -> dict:
        """Return aggregated cluster statistics."""
        active_peers = [p for p in self.peers.values() if p.is_active]
        total_vram = sum(p.gpu.vram_mb for p in active_peers if p.gpu)
        free_vram = sum(p.gpu.free_vram_mb for p in active_peers if p.gpu)
        return {
            "node_id": self.node_id,
            "rendezvous": self.rendezvous_url,
            "total_peers": len(self.peers),
            "active_peers": len(active_peers),
            "cluster_vram_mb": total_vram,
            "cluster_free_vram_mb": free_vram,
            "local_gpu": self.local_node.gpu.to_dict() if self.local_node and self.local_node.gpu else None,
        }

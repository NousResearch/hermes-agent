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
import stat
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

def _get_ed25519():
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.exceptions import InvalidSignature
        return ed25519, InvalidSignature
    except ImportError:
        return None, None

logger = logging.getLogger("hermes.global_mesh")


def get_hermes_dir() -> Path:
    """Resolve the Hermes home configuration directory."""
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except ImportError:
        return Path(os.path.expanduser("~/.hermes"))


def _write_secure_file(path: Path, content: str, mode: int = 0o600) -> None:
    """Write text content to file with strict owner-only permissions (0600)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(str(path), flags, mode)
    try:
        with open(fd, "w", encoding="utf-8", closefd=False) as f:
            f.write(content)
    finally:
        os.close(fd)
    try:
        os.chmod(str(path), mode)
    except (OSError, NotImplementedError):
        pass


def compute_signature(payload_bytes: bytes, secret_or_private_key: str) -> str:
    """Compute signature using asymmetric Ed25519 private key or HMAC-SHA256 fallback."""
    ed25519_mod, _ = _get_ed25519()
    if ed25519_mod and len(secret_or_private_key) == 64:
        try:
            priv = ed25519_mod.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(secret_or_private_key))
            return priv.sign(payload_bytes).hex()
        except Exception:
            pass
    return hmac.new(
        secret_or_private_key.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()


def verify_signature(payload_bytes: bytes, signature_hex: str, public_key_or_secret: str) -> bool:
    """Verify cryptographic signature using Ed25519 public key or HMAC fallback."""
    ed25519_mod, invalid_sig = _get_ed25519()
    if ed25519_mod and len(public_key_or_secret) == 64:
        # Try interpreting as Ed25519 public key
        try:
            pub = ed25519_mod.Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_or_secret))
            pub.verify(bytes.fromhex(signature_hex), payload_bytes)
            return True
        except (invalid_sig, ValueError):
            pass
        except Exception:
            pass
        # Try interpreting as Ed25519 private key (derives public key)
        try:
            priv = ed25519_mod.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(public_key_or_secret))
            priv.public_key().verify(bytes.fromhex(signature_hex), payload_bytes)
            return True
        except (invalid_sig, ValueError):
            pass
        except Exception:
            pass

    # HMAC fallback comparison
    try:
        expected = hmac.new(
            public_key_or_secret.encode("utf-8"),
            payload_bytes,
            hashlib.sha256,
        ).hexdigest()
        if hmac.compare_digest(signature_hex, expected):
            return True
    except Exception:
        pass
    return False


def probe_local_gpu() -> Optional[GPUDescriptor]:
    """Probe the host system for physical GPU/accelerator hardware.

    Checks in order:
    1. PyTorch CUDA / ROCm / MPS
    2. nvidia-smi CLI
    Returns None if no physical hardware accelerator is detected.
    """
    # 1. PyTorch check
    try:
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_mb = int(props.total_memory / (1024 * 1024))
            try:
                free_b, _ = torch.cuda.mem_get_info()
                free_mb = int(free_b / (1024 * 1024))
            except Exception:
                free_mb = total_mb
            backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
            supported = ["hermes-3-8b"]
            if total_mb >= 32768:
                supported.append("hermes-3-70b")
            return GPUDescriptor(
                device_name=dev_name,
                vram_mb=total_mb,
                free_vram_mb=free_mb,
                compute_backend=backend,
                supported_models=supported,
                compute_rating=round(total_mb / 4096.0, 2),
                reputation_score=1.0,
            )
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            import psutil
            total_ram_mb = int(psutil.virtual_memory().total / (1024 * 1024))
            free_ram_mb = int(psutil.virtual_memory().available / (1024 * 1024))
            mps_vram = int(total_ram_mb * 0.75)
            mps_free = int(free_ram_mb * 0.75)
            supported = ["hermes-3-8b"]
            if mps_vram >= 32768:
                supported.append("hermes-3-70b")
            return GPUDescriptor(
                device_name="Apple Silicon (MPS)",
                vram_mb=mps_vram,
                free_vram_mb=mps_free,
                compute_backend="mps",
                supported_models=supported,
                compute_rating=round(mps_vram / 4096.0, 2),
                reputation_score=1.0,
            )
    except (ImportError, Exception):
        pass

    # 2. nvidia-smi CLI
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            line = res.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                name = parts[0]
                total_mb = int(float(parts[1]))
                free_mb = int(float(parts[2]))
                supported = ["hermes-3-8b"]
                if total_mb >= 32768:
                    supported.append("hermes-3-70b")
                return GPUDescriptor(
                    device_name=name,
                    vram_mb=total_mb,
                    free_vram_mb=free_mb,
                    compute_backend="cuda",
                    supported_models=supported,
                    compute_rating=round(total_mb / 4096.0, 2),
                    reputation_score=1.0,
                )
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    return None


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


def http_transport(
    peer: MeshNodeInfo,
    task: MeshTaskRequest,
    timeout_s: float = 60.0,
) -> MeshExecutionReceipt:
    """Execute a task remotely on a peer node via HTTP/HTTPS transport."""
    url = peer.endpoint.rstrip("/")
    if not (url.startswith("http://") or url.startswith("https://")):
        return MeshExecutionReceipt(
            task_id=task.task_id,
            executor_id=peer.node_id,
            status="failed",
            output="",
            tokens=0,
            latency_ms=0.0,
            error_message=f"Unsupported endpoint scheme: {peer.endpoint}. Must be http:// or https://",
        )

    task_url = f"{url}/mesh/task"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Hermes-Global-Mesh/1.0",
        "X-Hermes-Requester-Id": task.requester_id,
        "X-Hermes-Task-Signature": task.signature,
    }
    payload_bytes = json.dumps(task.to_dict()).encode("utf-8")
    req = urllib.request.Request(task_url, data=payload_bytes, headers=headers, method="POST")

    t_start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            receipt = MeshExecutionReceipt.from_dict(data)
            receipt.latency_ms = round((time.time() - t_start) * 1000, 2)
            return receipt
    except urllib.error.HTTPError as exc:
        err_msg = f"HTTP {exc.code}: {exc.reason}"
        try:
            body = exc.read().decode("utf-8")
            err_data = json.loads(body)
            if "error" in err_data:
                err_msg += f" - {err_data['error']}"
        except Exception:
            pass
        return MeshExecutionReceipt(
            task_id=task.task_id,
            executor_id=peer.node_id,
            status="failed",
            output="",
            tokens=0,
            latency_ms=round((time.time() - t_start) * 1000, 2),
            error_message=err_msg,
        )
    except Exception as exc:
        return MeshExecutionReceipt(
            task_id=task.task_id,
            executor_id=peer.node_id,
            status="failed",
            output="",
            tokens=0,
            latency_ms=round((time.time() - t_start) * 1000, 2),
            error_message=f"Network transport error: {exc}",
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

        # Identity & Cryptographic asymmetric keypair
        self.node_id = node_id or self._load_or_create_node_id()
        if secret_key:
            self.secret_key = secret_key
            ed25519_mod, _ = _get_ed25519()
            if ed25519_mod and len(secret_key) == 64:
                try:
                    priv = ed25519_mod.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(secret_key))
                    self.public_key = priv.public_key().public_bytes_raw().hex()
                except Exception:
                    self.public_key = hashlib.sha256(secret_key.encode("utf-8")).hexdigest()[:32]
            else:
                self.public_key = hashlib.sha256(secret_key.encode("utf-8")).hexdigest()[:32]
        else:
            self.secret_key, self.public_key = self._load_or_create_keypair()

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
            _write_secure_file(id_file, new_id, 0o644)
        except Exception:
            pass
        return new_id

    def _load_or_create_keypair(self) -> Tuple[str, str]:
        """Load or securely generate an asymmetric Ed25519 keypair."""
        key_file = self.state_dir / "mesh_node_secret.key"
        ed25519_mod, _ = _get_ed25519()
        if key_file.exists():
            try:
                content = key_file.read_text(encoding="utf-8").strip()
                if len(content) == 64:
                    if ed25519_mod:
                        try:
                            priv = ed25519_mod.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(content))
                            pub_hex = priv.public_key().public_bytes_raw().hex()
                            return content, pub_hex
                        except Exception:
                            pass
                    pub_hex = hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]
                    return content, pub_hex
            except Exception:
                pass

        if ed25519_mod:
            priv = ed25519_mod.Ed25519PrivateKey.generate()
            priv_hex = priv.private_bytes_raw().hex()
            pub_hex = priv.public_key().public_bytes_raw().hex()
        else:
            priv_hex = hashlib.sha256(os.urandom(32)).hexdigest()
            pub_hex = hashlib.sha256(priv_hex.encode("utf-8")).hexdigest()[:32]

        try:
            _write_secure_file(key_file, priv_hex, 0o600)
        except Exception as exc:
            logger.error("Failed to write mesh secret key securely: %s", exc)

        return priv_hex, pub_hex

    def init_local_node(
        self,
        endpoint: str = "mesh://direct",
        offer_gpu: Optional[bool] = None,
        vram_mb: Optional[int] = None,
        backend: Optional[str] = None,
        device_name: Optional[str] = None,
        supported_models: Optional[List[str]] = None,
    ) -> MeshNodeInfo:
        """Initialize local node descriptor with verified hardware capabilities."""
        gpu_desc: Optional[GPUDescriptor] = None

        if offer_gpu is False:
            gpu_desc = None
        elif vram_mb is not None or device_name is not None or backend is not None:
            # Explicit hardware configuration provided by caller
            vram = vram_mb if vram_mb is not None else 8192
            dev = device_name or "Custom GPU Accelerator"
            b = backend or "cuda"
            models = supported_models or ["hermes-3-8b"]
            gpu_desc = GPUDescriptor(
                device_name=dev,
                vram_mb=vram,
                free_vram_mb=vram,
                compute_backend=b,
                supported_models=models,
                compute_rating=round(vram / 4096.0, 2),
                reputation_score=1.0,
            )
        elif offer_gpu is True or offer_gpu is None:
            # Auto-probe host hardware for physical accelerator
            probed = probe_local_gpu()
            if probed:
                gpu_desc = probed
            elif offer_gpu is True:
                logger.warning("No physical GPU detected on host. Defaulting to client-only mode.")
                gpu_desc = None
            else:
                gpu_desc = None

        self.local_node = MeshNodeInfo(
            node_id=self.node_id,
            endpoint=endpoint,
            nat_type="direct" if "://" in endpoint else "relay",
            gpu=gpu_desc,
            last_seen=time.time(),
            is_active=True,
            public_key=self.public_key,
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
        """Remove peers that haven't sent a heartbeat within the timeout."""
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
        """Verify the cryptographic proof-of-execution on receipt using asymmetric Ed25519 public key."""
        return verify_signature(receipt.canonical_bytes(), receipt.proof_sig, executor_pubkey_or_secret)

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
                elif candidate.endpoint.startswith("http://") or candidate.endpoint.startswith("https://"):
                    receipt = http_transport(candidate, task, timeout_s=task.timeout_s)
                else:
                    receipt = self._default_mock_transport(candidate, task)

                if receipt.status == "completed":
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
        """Loopback execution simulating remote transport for test and mesh:// mock endpoints."""
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

    def sync_rendezvous(
        self,
        rendezvous_url: Optional[str] = None,
        timeout_s: float = 10.0,
    ) -> Tuple[bool, List[MeshNodeInfo]]:
        """Announce local node to rendezvous server and fetch active swarm peers."""
        url = (rendezvous_url or self.rendezvous_url).rstrip("/")
        if not (url.startswith("http://") or url.startswith("https://")):
            logger.warning("Rendezvous URL must start with http:// or https://: %s", url)
            return False, []

        announced_peers: List[MeshNodeInfo] = []
        if self.local_node:
            announce_url = f"{url}/announce"
            body = json.dumps(self.local_node.to_dict()).encode("utf-8")
            req = urllib.request.Request(
                announce_url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Hermes-Global-Mesh/1.0",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    pass
            except Exception as exc:
                logger.warning("Failed to announce to rendezvous %s: %s", announce_url, exc)

        peers_url = f"{url}/peers"
        req = urllib.request.Request(
            peers_url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Hermes-Global-Mesh/1.0",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                peer_list = data if isinstance(data, list) else data.get("peers", [])
                for p_data in peer_list:
                    peer = MeshNodeInfo.from_dict(p_data)
                    if peer.node_id != self.node_id:
                        self.register_peer(peer)
                        announced_peers.append(peer)
                return True, announced_peers
        except Exception as exc:
            logger.warning("Failed to fetch peers from rendezvous %s: %s", peers_url, exc)
            return False, []

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

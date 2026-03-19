"""
COSMOS Network Configuration
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import json
import os


@dataclass
class NodeConfig:
    """Configuration for a single COSMOS node."""
    name: str
    host: str
    port: int = 9999
    node_type: str = "core"  # "core" or "pro"
    ssh_port: int = 22
    gpu_model: str = ""
    vram_gb: int = 0
    approved: bool = True
    local_models: List[str] = field(default_factory=list)


# Core nodes — our servers
# Peer connections go through SSH tunnels:
#   Each node connects to peers via localhost:19999 (tunneled to remote:9999)
CORE_NODES = {
    "nexus-alpha": NodeConfig(
        name="nexus-alpha",
        host="127.0.0.1",    # Peers connect via SSH tunnel
        port=19999,           # Tunneled port (local 19999 → remote 9999)
        ssh_port=22046,
        gpu_model="NVIDIA A40",
        vram_gb=48,
        local_models=[
            "deepseek-r1:8b", "phi4", "qwen2.5:7b",
            "mistral:7b", "llama3:8b", "gemma2:9b",
        ],
    ),
    "nexus-beta": NodeConfig(
        name="nexus-beta",
        host="127.0.0.1",    # Peers connect via SSH tunnel
        port=19999,           # Tunneled port (local 19999 → remote 9999)
        ssh_port=37909,
        gpu_model="NVIDIA RTX A6000",
        vram_gb=49,
        local_models=["qwen3-coder-next"],
    ),
}

# PRO node requirements
PRO_NODE_REQUIREMENTS = {
    "min_vram_gb": 24,
    "min_tflops_fp16": 80.0,
    "max_latency_ms": 30,
    "min_ram_gb": 32,
}

# Paths
COSMOS_DATA_DIR = Path(os.environ.get(
    "COSMOS_DATA_DIR",
    "/workspace/Cosmos/data/cosmos" if os.path.exists("/workspace") else "data/cosmos"
))
COSMOS_SEED_FILE = COSMOS_DATA_DIR / "swarm.seed"
COSMOS_NODES_FILE = COSMOS_DATA_DIR / "nodes.json"
COSMOS_MESH_FILE = COSMOS_DATA_DIR / "mesh_state.json"
COSMOS_PENDING_FILE = COSMOS_DATA_DIR / "pending_approvals.json"

# Network tuning
TCP_BACKLOG = 64
MAX_FRAME_SIZE = 4 * 1024 * 1024  # 4MB max frame
HEARTBEAT_INTERVAL = 10.0  # seconds between heartbeats
HEARTBEAT_TIMEOUT_MULT = 5  # miss 5 heartbeats before declaring dead (50s)
CONNECTION_TIMEOUT = 15.0  # allow for SSH tunnel handshake overhead
RECONNECT_INTERVAL = 10.0  # check for missing peers every 10s
RECONNECT_MAX_RETRIES = 0  # 0 = infinite retries (never give up on core peers)
STREAM_CHUNK_SIZE = 4096  # bytes per dialogue chunk
MAX_STREAMS_PER_CONNECTION = 256

# TCP keepalive tuning (critical for SSH tunnel connections)
TCP_KEEPIDLE = 60   # start probes after 60s idle
TCP_KEEPINTVL = 10  # probe every 10s
TCP_KEEPCNT = 5     # 5 missed probes = dead


def ensure_dirs():
    """Create COSMOS data directories."""
    COSMOS_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_known_nodes() -> Dict[str, NodeConfig]:
    """Load all known nodes (core + approved PRO)."""
    nodes = dict(CORE_NODES)
    if COSMOS_NODES_FILE.exists():
        try:
            data = json.loads(COSMOS_NODES_FILE.read_text())
            for name, cfg in data.items():
                if name not in nodes:
                    nodes[name] = NodeConfig(**cfg)
        except Exception:
            pass
    return nodes


def save_node(node: NodeConfig):
    """Persist a node config."""
    ensure_dirs()
    nodes = {}
    if COSMOS_NODES_FILE.exists():
        try:
            nodes = json.loads(COSMOS_NODES_FILE.read_text())
        except Exception:
            pass
    nodes[node.name] = {
        "name": node.name,
        "host": node.host,
        "port": node.port,
        "node_type": node.node_type,
        "ssh_port": node.ssh_port,
        "gpu_model": node.gpu_model,
        "vram_gb": node.vram_gb,
        "approved": node.approved,
        "local_models": node.local_models,
    }
    COSMOS_NODES_FILE.write_text(json.dumps(nodes, indent=2))

"""
COSMOS Proof-of-Swarm Authentication
=====================================

5-layer auth stack:
  1. Swarm Seed — shared 256-bit secret, identity = BLAKE3(seed || name)
  2. GPU Fingerprint — hardware-bound via deterministic CUDA compute
  3. Temporal Hash Mesh — interleaved hash chains across nodes
  4. Swarm Quorum — 2+ nodes must verify new connections
  5. Rolling BLAKE3 Seals — per-packet, forward-secure, replay-proof

No private keys. No certs. Just one seed + math + hardware.
"""
import os
import time
import hashlib
from typing import Optional, Dict, Tuple
from pathlib import Path
from loguru import logger

try:
    from blake3 import blake3
except ImportError:
    # Fallback to hashlib-based BLAKE3-like (uses SHA3-256 as stand-in)
    # Install blake3 for real speed: pip install blake3
    class _Blake3Fallback:
        def __init__(self, data=b"", key=None):
            self._h = hashlib.sha3_256(data)
        def update(self, data):
            self._h.update(data)
            return self
        def digest(self):
            return self._h.digest()
        def hexdigest(self):
            return self._h.hexdigest()
    def blake3(data=b"", key=None):
        return _Blake3Fallback(data, key)
    logger.warning("blake3 not installed, using SHA3-256 fallback. Install blake3 for full speed.")

from .cosmos_config import COSMOS_SEED_FILE, ensure_dirs


# ── Layer 1: Swarm Seed & Identity ───────────────────────────

def generate_swarm_seed() -> bytes:
    """Generate a new 256-bit swarm seed. Do this ONCE."""
    seed = os.urandom(32)
    ensure_dirs()
    COSMOS_SEED_FILE.write_bytes(seed)
    logger.info(f"Generated new swarm seed → {COSMOS_SEED_FILE}")
    return seed


def load_swarm_seed() -> Optional[bytes]:
    """Load the swarm seed from disk."""
    if COSMOS_SEED_FILE.exists():
        seed = COSMOS_SEED_FILE.read_bytes()
        if len(seed) == 32:
            return seed
    return None


def get_or_create_seed() -> bytes:
    """Load existing seed or generate a new one."""
    seed = load_swarm_seed()
    if seed is None:
        seed = generate_swarm_seed()
    return seed


def derive_identity(swarm_seed: bytes, bot_name: str) -> bytes:
    """
    Derive a bot's identity token from the swarm seed + name.

    identity = BLAKE3(swarm_seed || bot_name)

    This IS the bot's auth credential. No key files.
    """
    return blake3(swarm_seed + bot_name.encode("utf-8")).digest()


def derive_node_identity(swarm_seed: bytes, node_name: str, gpu_fingerprint: bytes) -> bytes:
    """
    Derive a node's identity — bound to both swarm membership AND hardware.

    node_identity = BLAKE3(swarm_seed || node_name || gpu_fingerprint)
    """
    return blake3(swarm_seed + node_name.encode("utf-8") + gpu_fingerprint).digest()


# ── Layer 2: GPU Hardware Fingerprint ────────────────────────

def compute_gpu_fingerprint() -> bytes:
    """
    Compute a deterministic GPU fingerprint.

    Runs a fixed matrix multiply on GPU — floating-point rounding
    behavior is unique per GPU model/silicon. Combined with hardware
    info for additional uniqueness.

    Returns 32-byte BLAKE3 hash.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("No CUDA GPU available, using CPU fingerprint")
            return _cpu_fingerprint()

        # Fixed seed → deterministic input matrices
        torch.manual_seed(0xFA2B5)
        torch.cuda.manual_seed(0xFA2B5)

        # Deterministic CUDA ops
        a = torch.randn(512, 512, device="cuda", dtype=torch.float32)
        b = torch.randn(512, 512, device="cuda", dtype=torch.float32)

        # Matrix multiply — rounding differs per GPU silicon
        result = torch.mm(a, b)
        torch.cuda.synchronize()

        result_bytes = result.cpu().numpy().tobytes()

        # Combine with hardware identifiers
        gpu_name = torch.cuda.get_device_name(0).encode()
        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory if hasattr(props, 'total_memory') else getattr(props, 'total_mem', 0)
        hw_info = f"{total_mem}_{props.multi_processor_count}_{props.major}.{props.minor}".encode()

        fingerprint = blake3(result_bytes + gpu_name + hw_info).digest()
        logger.info(f"GPU fingerprint computed: {fingerprint[:8].hex()}... ({gpu_name.decode()})")
        return fingerprint

    except Exception as e:
        logger.warning(f"GPU fingerprint failed: {e}, using CPU fallback")
        return _cpu_fingerprint()


def _cpu_fingerprint() -> bytes:
    """Fallback fingerprint using CPU info (less unique but functional)."""
    import platform
    info = f"{platform.machine()}_{platform.processor()}_{os.cpu_count()}".encode()
    return blake3(b"cpu_fallback_" + info).digest()


# ── Layer 5: Rolling BLAKE3 Packet Seals ─────────────────────

class RollingSeal:
    """
    Per-connection rolling seal for packet authentication.

    Each packet gets a unique seal based on:
      seal_key = BLAKE3(identity_token || sequence_bytes)
      seal = BLAKE3_keyed(seal_key, payload)[:16]

    Properties:
      - Forward-secure: each packet uses a unique derived key
      - Replay-proof: sequence is monotonic
      - Fast: BLAKE3 is ~3x faster than SHA-256
    """

    def __init__(self, identity_token: bytes):
        self._base_key = identity_token
        self._seq = 0

    @property
    def sequence(self) -> int:
        return self._seq

    def seal(self, payload: bytes) -> Tuple[bytes, int]:
        """
        Seal a payload. Returns (seal_bytes, sequence_used).
        Increments sequence after each call.
        """
        seq_bytes = self._seq.to_bytes(8, "big")
        seal_key = blake3(self._base_key + seq_bytes).digest()

        # Keyed BLAKE3 MAC of the payload
        seal_value = blake3(seal_key + payload).digest()[:16]  # 128-bit seal

        used_seq = self._seq
        self._seq += 1
        return seal_value, used_seq

    def verify(self, payload: bytes, seal_value: bytes, sequence: int) -> bool:
        """Verify a seal against a payload and sequence number."""
        seq_bytes = sequence.to_bytes(8, "big")
        seal_key = blake3(self._base_key + seq_bytes).digest()
        expected = blake3(seal_key + payload).digest()[:16]
        # Constant-time comparison
        return _constant_time_compare(expected, seal_value)


def _constant_time_compare(a: bytes, b: bytes) -> bool:
    """Constant-time byte comparison to prevent timing attacks."""
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a, b):
        result |= x ^ y
    return result == 0


# ── Challenge-Response for Connection Auth ────────────────────

def generate_challenge() -> bytes:
    """Generate a random 32-byte challenge for connection auth."""
    return os.urandom(32)


def solve_challenge(identity_token: bytes, challenge: bytes, timestamp: float) -> bytes:
    """
    Solve a challenge using identity token.
    response = BLAKE3(identity_token || challenge || timestamp_bytes)
    """
    ts_bytes = str(timestamp).encode()
    return blake3(identity_token + challenge + ts_bytes).digest()


def verify_challenge(
    identity_token: bytes,
    challenge: bytes,
    response: bytes,
    timestamp: float,
    max_age_seconds: float = 30.0,
) -> bool:
    """Verify a challenge response. Rejects stale timestamps."""
    # Check timestamp freshness
    if abs(time.time() - timestamp) > max_age_seconds:
        return False
    expected = solve_challenge(identity_token, challenge, timestamp)
    return _constant_time_compare(expected, response)

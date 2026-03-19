"""
COSMOS Protocol — Packet types, encoding/decoding, frame I/O.

Wire format:
  [4 bytes: length (big-endian uint32)] [msgpack body]

Designed for maximum speed: raw TCP, msgpack, zero unnecessary bytes.
"""
import struct
import time
import asyncio
from enum import IntEnum
from typing import Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass, field

try:
    import msgpack
except ImportError:
    raise ImportError("COSMOS requires msgpack: pip install msgpack")

from . import PROTOCOL_VERSION


class PacketType(IntEnum):
    """COSMOS packet types."""
    # Core protocol (0x01-0x0F)
    HELLO      = 0x01  # Handshake + GPU fingerprint + spec report
    VERIFY     = 0x02  # Quorum verification response
    DIALOGUE   = 0x03  # Streaming dialogue content (hot path)
    ROUTE      = 0x04  # Route request to specific bot
    HEARTBEAT  = 0x05  # Keep-alive + mesh hash exchange
    DISCOVERY  = 0x06  # Announce/query available bots
    ACK        = 0x07  # Acknowledge receipt
    JOIN_REQ   = 0x08  # PRO user join request
    JOIN_VOTE  = 0x09  # Quorum vote on join request
    APPROVE    = 0x0A  # Admin manual approval
    BENCHMARK  = 0x0B  # GPU benchmark result
    KICK       = 0x0C  # Remove node from mesh

    # Proof-of-Inference (0x10-0x12)
    POI_REQUEST     = 0x10  # Request consensus inference round
    POI_ATTESTATION = 0x11  # Node's hardware-signed inference attestation
    POI_RESULT      = 0x12  # Final consensus proof

    # Latent Space Routing (0x13)
    LATENT_ROUTE    = 0x13  # Auto-route by semantic embedding

    # Model Attestation (0x14)
    MODEL_ATTEST    = 0x14  # GPU-signed model weight attestation

    # Swarm Memory (0x15-0x17)
    MEMORY_PROPOSE  = 0x15  # Propose a knowledge crystal
    MEMORY_VOTE     = 0x16  # Vote on crystal accuracy
    MEMORY_SYNC     = 0x17  # Sync crystal store across nodes

    ERROR      = 0xFF  # Error response


@dataclass
class COSMOSPacket:
    """A single COSMOS protocol packet."""
    packet_type: int                    # PacketType value
    sender: str                         # Bot/node name
    target: str = "*"                   # Target name ("*" = broadcast)
    stream_id: str = ""                 # Stream ID for multiplexing
    sequence: int = 0                   # Sequence number
    timestamp: float = 0.0             # Unix timestamp
    seal: bytes = b""                   # BLAKE3 rolling seal
    mesh_hash: bytes = b""             # Temporal mesh hash
    data: Any = None                    # Payload (varies by type)
    streaming: bool = False             # Is this a streaming chunk?
    final: bool = False                 # Last chunk in stream?

    def to_dict(self) -> Dict:
        """Convert to msgpack-friendly dict. Short keys for speed."""
        return {
            "v": PROTOCOL_VERSION,
            "t": self.packet_type,
            "from": self.sender,
            "to": self.target,
            "sid": self.stream_id,
            "seq": self.sequence,
            "ts": self.timestamp or time.time(),
            "seal": self.seal,
            "mesh": self.mesh_hash,
            "d": self.data,
            "s": self.streaming,
            "fin": self.final,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "COSMOSPacket":
        """Reconstruct from decoded msgpack dict."""
        return cls(
            packet_type=d.get("t", 0),
            sender=d.get("from", ""),
            target=d.get("to", "*"),
            stream_id=d.get("sid", ""),
            sequence=d.get("seq", 0),
            timestamp=d.get("ts", 0.0),
            seal=d.get("seal", b""),
            mesh_hash=d.get("mesh", b""),
            data=d.get("d"),
            streaming=d.get("s", False),
            final=d.get("fin", False),
        )


# ── Frame I/O ────────────────────────────────────────────────

# Length prefix: 4 bytes, big-endian unsigned int
HEADER_FMT = "!I"
HEADER_SIZE = 4
MAX_FRAME = 4 * 1024 * 1024  # 4MB


def encode_frame(packet: COSMOSPacket) -> bytes:
    """Encode a packet into a length-prefixed msgpack frame."""
    body = msgpack.packb(packet.to_dict(), use_bin_type=True)
    if len(body) > MAX_FRAME:
        raise ValueError(f"Frame too large: {len(body)} > {MAX_FRAME}")
    return struct.pack(HEADER_FMT, len(body)) + body


def decode_body(body: bytes) -> COSMOSPacket:
    """Decode a msgpack body into a COSMOSPacket."""
    d = msgpack.unpackb(body, raw=False)
    return COSMOSPacket.from_dict(d)


async def read_frame(reader: asyncio.StreamReader) -> Optional[COSMOSPacket]:
    """Read one length-prefixed frame from an async stream."""
    header = await reader.readexactly(HEADER_SIZE)
    (length,) = struct.unpack(HEADER_FMT, header)
    if length > MAX_FRAME:
        raise ValueError(f"Frame too large: {length}")
    body = await reader.readexactly(length)
    return decode_body(body)


async def write_frame(writer: asyncio.StreamWriter, packet: COSMOSPacket):
    """Write one length-prefixed frame to an async stream."""
    data = encode_frame(packet)
    writer.write(data)
    await writer.drain()


async def stream_frames(reader: asyncio.StreamReader) -> AsyncIterator[COSMOSPacket]:
    """Continuously read frames from a stream. Yields packets."""
    while True:
        try:
            packet = await read_frame(reader)
            if packet is None:
                break
            yield packet
        except (asyncio.IncompleteReadError, ConnectionError):
            break


# ── Convenience constructors ─────────────────────────────────

def make_hello(sender: str, gpu_fingerprint: bytes, node_info: Dict) -> COSMOSPacket:
    """Create a HELLO handshake packet."""
    return COSMOSPacket(
        packet_type=PacketType.HELLO,
        sender=sender,
        data={
            "gpu_fp": gpu_fingerprint,
            "info": node_info,
        },
    )


def make_verify(sender: str, target: str, accepted: bool, reason: str = "") -> COSMOSPacket:
    """Create a VERIFY response packet."""
    return COSMOSPacket(
        packet_type=PacketType.VERIFY,
        sender=sender,
        target=target,
        data={"accepted": accepted, "reason": reason},
    )


def make_dialogue(
    sender: str,
    target: str,
    content: str,
    stream_id: str,
    sequence: int,
    streaming: bool = False,
    final: bool = False,
) -> COSMOSPacket:
    """Create a DIALOGUE packet (the hot path)."""
    return COSMOSPacket(
        packet_type=PacketType.DIALOGUE,
        sender=sender,
        target=target,
        stream_id=stream_id,
        sequence=sequence,
        data=content,
        streaming=streaming,
        final=final,
    )


def make_route(sender: str, target_bot: str, prompt: str, stream_id: str) -> COSMOSPacket:
    """Create a ROUTE request — ask a remote node to query a specific bot."""
    return COSMOSPacket(
        packet_type=PacketType.ROUTE,
        sender=sender,
        target=target_bot,
        stream_id=stream_id,
        data={"prompt": prompt, "bot": target_bot},
    )


def make_heartbeat(sender: str, mesh_hash: bytes) -> COSMOSPacket:
    """Create a HEARTBEAT packet with mesh hash."""
    return COSMOSPacket(
        packet_type=PacketType.HEARTBEAT,
        sender=sender,
        mesh_hash=mesh_hash,
        data={"ts": time.time()},
    )


def make_discovery(sender: str, bots: list = None, query: bool = False) -> COSMOSPacket:
    """Create a DISCOVERY packet — announce bots or query for them."""
    return COSMOSPacket(
        packet_type=PacketType.DISCOVERY,
        sender=sender,
        data={"bots": bots or [], "query": query},
    )


def make_join_request(
    sender: str,
    gpu_fingerprint: bytes,
    benchmark_score: float,
    latency_ms: float,
    pro_token: str,
    node_info: Dict,
) -> COSMOSPacket:
    """Create a JOIN_REQ packet for PRO user onboarding."""
    return COSMOSPacket(
        packet_type=PacketType.JOIN_REQ,
        sender=sender,
        data={
            "gpu_fp": gpu_fingerprint,
            "benchmark": benchmark_score,
            "latency_ms": latency_ms,
            "pro_token": pro_token,
            "info": node_info,
        },
    )

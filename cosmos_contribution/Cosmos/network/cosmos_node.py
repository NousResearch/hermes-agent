"""
COSMOS Node Daemon
==================

The heart of the COSMOS mesh. Each server runs one node that:
  - Listens on TCP port 9999 (TCP_NODELAY for speed)
  - Manages connections to peer nodes (persistent, multiplexed)
  - Routes DIALOGUE requests to local bots (Ollama, API agents)
  - Streams responses back as DIALOGUE chunks (bidirectional)
  - Exchanges mesh hashes via heartbeat
  - Handles quorum voting for new connections
  - Processes PRO user join requests

Usage:
    python -m cosmos.network.cosmos_node --name nexus-alpha
"""
import os
import sys
import json
import time
import uuid
import socket
import asyncio
import argparse
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
from dataclasses import dataclass, field
from loguru import logger

from . import COSMOS_PORT, PROTOCOL_VERSION
from .cosmos_protocol import (
    PacketType, COSMOSPacket,
    read_frame, write_frame, encode_frame,
    make_hello, make_verify, make_dialogue, make_heartbeat,
    make_discovery,
)
from .cosmos_auth import (
    get_or_create_seed, derive_identity, derive_node_identity,
    compute_gpu_fingerprint, RollingSeal,
    generate_challenge, solve_challenge, verify_challenge,
)
from .cosmos_mesh import TemporalHashMesh, SwarmQuorum
from .cosmos_poi import (
    ProofOfInference, InferenceAttestation,
    make_poi_request, make_poi_attestation, make_poi_result,
    hash_prompt, hash_model, hash_output,
)
from .cosmos_latent_router import LatentRouter
from .cosmos_attestation import (
    ModelAttestor, ModelAttestation, make_model_attest,
    compute_model_hash_with_weights,
)
from .cosmos_swarm_memory import (
    SwarmMemory, MemoryCrystal, CrystalVerification,
    make_memory_propose, make_memory_vote, make_memory_sync,
)
from .cosmos_config import (
    CORE_NODES, NodeConfig, load_known_nodes, save_node,
    HEARTBEAT_INTERVAL, HEARTBEAT_TIMEOUT_MULT, CONNECTION_TIMEOUT,
    RECONNECT_INTERVAL, RECONNECT_MAX_RETRIES,
    STREAM_CHUNK_SIZE, MAX_STREAMS_PER_CONNECTION,
    COSMOS_PENDING_FILE, ensure_dirs,
    TCP_KEEPIDLE, TCP_KEEPINTVL, TCP_KEEPCNT,
)

import random  # for jitter in reconnect backoff


def _tune_keepalive(sock):
    """Apply aggressive TCP keepalive settings for SSH tunnel reliability."""
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Platform-specific keepalive tuning
        if hasattr(socket, "TCP_KEEPIDLE"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, TCP_KEEPIDLE)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, TCP_KEEPINTVL)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, TCP_KEEPCNT)
        elif sys.platform == "darwin":
            # macOS uses TCP_KEEPALIVE instead of TCP_KEEPIDLE
            TCP_KEEPALIVE_MAC = 0x10
            sock.setsockopt(socket.IPPROTO_TCP, TCP_KEEPALIVE_MAC, TCP_KEEPIDLE)
    except Exception as e:
        logger.debug(f"Could not tune TCP keepalive: {e}")


@dataclass
class PeerConnection:
    """Active connection to a peer node."""
    node_name: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    seal: RollingSeal
    verified: bool = False
    remote_bots: List[str] = field(default_factory=list)
    last_heartbeat: float = 0.0
    active_streams: Dict[str, asyncio.Queue] = field(default_factory=dict)


class COSMOSNode:
    """
    COSMOS mesh node daemon.

    Handles:
    - Incoming peer connections (server mode)
    - Outgoing peer connections (client mode)
    - Local bot registry and routing
    - Mesh hash maintenance
    - Quorum voting
    - PRO user approval queue
    """

    def __init__(self, node_name: str, host: str = "0.0.0.0", port: int = COSMOS_PORT):
        self.node_name = node_name
        self.host = host
        self.port = port

        # Auth
        self._seed = get_or_create_seed()
        self._gpu_fp = b""  # Computed async on start
        self._identity = derive_identity(self._seed, node_name)

        # Mesh
        self._mesh = TemporalHashMesh(node_name, self._identity)
        self._quorum = SwarmQuorum(min_votes=2)

        # Peers
        self._peers: Dict[str, PeerConnection] = {}
        self._peer_lock = asyncio.Lock()
        self._connecting: Set[str] = set()  # peers with active connect tasks

        # Local bot registry: bot_name → query_function
        self._local_bots: Dict[str, Callable] = {}

        # Pending PRO approvals
        self._pending_approvals: List[Dict] = []

        # Stream response queues: stream_id → Queue
        self._response_queues: Dict[str, asyncio.Queue] = {}

        # v2.0 Subsystems
        self._poi: Optional[ProofOfInference] = None      # Initialized after GPU fp
        self._latent_router = LatentRouter()
        self._attestor: Optional[ModelAttestor] = None     # Initialized after GPU fp
        self._swarm_memory: Optional[SwarmMemory] = None   # Initialized after GPU fp

        # Server
        self._server = None
        self._running = False

        logger.info(f"COSMOS Node '{node_name}' initialized (identity: {self._identity[:8].hex()}...)")

    # ── Local Bot Registry ────────────────────────────────────

    def register_bot(self, bot_name: str, query_fn: Callable):
        """
        Register a local bot that can be queried via COSMOS.

        query_fn should be: async def(prompt: str, max_tokens: int) -> str
        """
        self._local_bots[bot_name] = query_fn
        logger.info(f"Registered local bot: {bot_name}")

        # Register with latent router
        self._latent_router.add_model(bot_name)

        # Create GPU-signed model attestation
        if self._attestor:
            att = self._attestor.attest_model(bot_name, use_weights=True)
            # Broadcast attestation to peers
            asyncio.ensure_future(self._broadcast_attestation(att))

    async def _broadcast_attestation(self, attestation: ModelAttestation):
        """Broadcast a model attestation to all connected peers."""
        pkt = make_model_attest(self.node_name, attestation)
        async with self._peer_lock:
            for p in self._peers.values():
                try:
                    await write_frame(p.writer, pkt)
                except Exception:
                    pass

    def get_local_bots(self) -> List[str]:
        """Get list of locally available bots."""
        return list(self._local_bots.keys())

    def get_all_bots(self) -> Dict[str, str]:
        """Get all bots (local + remote) with their location."""
        bots = {name: self.node_name for name in self._local_bots}
        for peer in self._peers.values():
            for bot in peer.remote_bots:
                bots[bot] = peer.node_name
        return bots

    # ── Server ────────────────────────────────────────────────

    async def start(self):
        """Start the COSMOS node daemon."""
        self._running = True

        # Compute GPU fingerprint
        logger.info("Computing GPU fingerprint...")
        self._gpu_fp = await asyncio.get_event_loop().run_in_executor(
            None, compute_gpu_fingerprint
        )
        self._identity = derive_node_identity(self._seed, self.node_name, self._gpu_fp)
        logger.info(f"Node identity (with GPU): {self._identity[:8].hex()}...")

        # Initialize v2.0 subsystems now that we have GPU fingerprint
        gpu_name = ""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass
        self._poi = ProofOfInference(self.node_name, self._identity, self._gpu_fp)
        self._attestor = ModelAttestor(self.node_name, self._identity, self._gpu_fp, gpu_name)
        self._swarm_memory = SwarmMemory(self.node_name, self._identity, self._gpu_fp)
        logger.info("v2.0 subsystems initialized: PoI, LatentRouter, Attestation, SwarmMemory")

        # Start TCP server with SO_REUSEADDR to avoid "address already in use"
        self._server = await asyncio.start_server(
            self._handle_incoming,
            self.host, self.port,
            reuse_address=True,
        )
        # Tune TCP keepalive on server socket for SSH tunnel reliability
        for sock in self._server.sockets:
            _tune_keepalive(sock)

        logger.info(f"COSMOS Node listening on {self.host}:{self.port}")

        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._mesh_advance_loop())
        asyncio.create_task(self._memory_sync_loop())
        asyncio.create_task(self._reconnect_loop())

        # Connect to known peers
        asyncio.create_task(self._connect_to_peers())

    async def stop(self):
        """Gracefully stop the node."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Close all peer connections
        async with self._peer_lock:
            for peer in self._peers.values():
                peer.writer.close()
            self._peers.clear()

        logger.info(f"COSMOS Node '{self.node_name}' stopped")

    # ── Incoming Connections ──────────────────────────────────

    async def _handle_incoming(self, reader: asyncio.StreamReader,
                               writer: asyncio.StreamWriter):
        """Handle an incoming peer connection."""
        addr = writer.get_extra_info("peername")
        logger.info(f"Incoming connection from {addr}")

        # Tune TCP for SSH tunnel reliability
        sock = writer.get_extra_info("socket")
        if sock:
            _tune_keepalive(sock)

        try:
            # Expect HELLO packet first
            packet = await asyncio.wait_for(read_frame(reader), timeout=CONNECTION_TIMEOUT)
            if not packet or packet.packet_type != PacketType.HELLO:
                logger.warning(f"Expected HELLO from {addr}, got {packet.packet_type if packet else 'nothing'}")
                writer.close()
                return

            peer_name = packet.sender
            peer_data = packet.data or {}
            peer_gpu_fp = peer_data.get("gpu_fp", b"")

            # Verify identity
            peer_identity = derive_node_identity(self._seed, peer_name, peer_gpu_fp)
            challenge = generate_challenge()

            # Send challenge
            await write_frame(writer, COSMOSPacket(
                packet_type=PacketType.VERIFY,
                sender=self.node_name,
                target=peer_name,
                data={"challenge": challenge, "step": "challenge"},
            ))

            # Receive challenge response
            resp = await asyncio.wait_for(read_frame(reader), timeout=CONNECTION_TIMEOUT)
            if not resp or resp.packet_type != PacketType.VERIFY:
                writer.close()
                return

            resp_data = resp.data or {}
            if not verify_challenge(
                peer_identity,
                challenge,
                resp_data.get("response", b""),
                resp_data.get("timestamp", 0),
            ):
                logger.warning(f"Challenge verification FAILED for {peer_name}")
                await write_frame(writer, make_verify(self.node_name, peer_name, False, "auth failed"))
                writer.close()
                return

            # Verified! Send acceptance
            await write_frame(writer, make_verify(self.node_name, peer_name, True))

            # Register peer — extract bots from HELLO data (nested in info)
            peer_info = peer_data.get("info", {})
            if isinstance(peer_info, dict):
                hello_bots = peer_info.get("bots", [])
            else:
                hello_bots = []
            seal = RollingSeal(peer_identity)
            peer = PeerConnection(
                node_name=peer_name,
                reader=reader,
                writer=writer,
                seal=seal,
                verified=True,
                remote_bots=hello_bots,
            )

            async with self._peer_lock:
                self._peers[peer_name] = peer

            logger.info(f"Peer {peer_name} verified and connected (bots: {peer.remote_bots})")

            # Exchange bot discovery — send ALL known bots (local + mesh)
            await write_frame(writer, make_discovery(
                self.node_name, bots=list(self.get_all_bots().keys())
            ))

            # Handle packets from this peer
            await self._handle_peer_packets(peer)

        except (asyncio.TimeoutError, ConnectionError, asyncio.IncompleteReadError) as e:
            logger.debug(f"Connection from {addr} ended: {e}")
        finally:
            async with self._peer_lock:
                # Clean up if peer was registered
                for name, p in list(self._peers.items()):
                    if p.writer is writer:
                        del self._peers[name]
                        self._mesh.remove_peer(name)
                        break
            writer.close()

    # ── Outgoing Connections ──────────────────────────────────

    async def _connect_to_peers(self):
        """Connect to all known peer nodes."""
        await asyncio.sleep(2)  # Let server start first
        nodes = load_known_nodes()

        for name, cfg in nodes.items():
            if name == self.node_name:
                continue
            if not cfg.approved:
                continue
            asyncio.create_task(self._connect_to_peer(name, cfg.host, cfg.port))

    async def _connect_to_peer(self, peer_name: str, host: str, port: int,
                               max_retries: int = RECONNECT_MAX_RETRIES):
        """Connect to a specific peer with retry. max_retries=0 means infinite."""
        if peer_name in self._connecting:
            return  # Another task is already connecting to this peer
        self._connecting.add(peer_name)
        try:
            await self._connect_to_peer_inner(peer_name, host, port, max_retries)
        finally:
            self._connecting.discard(peer_name)

    async def _connect_to_peer_inner(self, peer_name: str, host: str, port: int,
                                     max_retries: int):
        """Inner connect logic with retry."""
        attempt = 0
        while max_retries == 0 or attempt < max_retries:
            if not self._running:
                return
            if peer_name in self._peers:
                return  # Already connected (maybe they connected to us)

            try:
                logger.info(f"Connecting to peer {peer_name} at {host}:{port} (attempt {attempt + 1})")
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=CONNECTION_TIMEOUT,
                )

                # Tune TCP for SSH tunnel reliability
                sock = writer.get_extra_info("socket")
                if sock:
                    _tune_keepalive(sock)

                # Send HELLO
                await write_frame(writer, make_hello(
                    self.node_name,
                    self._gpu_fp,
                    {"version": PROTOCOL_VERSION, "bots": self.get_local_bots()},
                ))

                # Receive challenge
                resp = await asyncio.wait_for(read_frame(reader), timeout=CONNECTION_TIMEOUT)
                if not resp or resp.packet_type != PacketType.VERIFY:
                    writer.close()
                    continue

                challenge = resp.data.get("challenge", b"")
                ts = time.time()

                # Solve challenge
                my_identity = derive_node_identity(self._seed, self.node_name, self._gpu_fp)
                response = solve_challenge(my_identity, challenge, ts)

                await write_frame(writer, COSMOSPacket(
                    packet_type=PacketType.VERIFY,
                    sender=self.node_name,
                    target=peer_name,
                    data={"response": response, "timestamp": ts, "step": "response"},
                ))

                # Receive verification result
                result = await asyncio.wait_for(read_frame(reader), timeout=CONNECTION_TIMEOUT)
                if not result or result.packet_type != PacketType.VERIFY:
                    writer.close()
                    continue

                if not result.data.get("accepted", False):
                    logger.warning(f"Peer {peer_name} rejected us: {result.data.get('reason')}")
                    writer.close()
                    continue

                # Connected!
                peer_identity = derive_node_identity(self._seed, peer_name, b"")
                seal = RollingSeal(peer_identity)
                peer = PeerConnection(
                    node_name=peer_name,
                    reader=reader,
                    writer=writer,
                    seal=seal,
                    verified=True,
                )

                async with self._peer_lock:
                    self._peers[peer_name] = peer

                logger.info(f"Connected to peer {peer_name}")

                # Send our bot list to the peer (all known bots)
                await write_frame(writer, make_discovery(
                    self.node_name, bots=list(self.get_all_bots().keys())
                ))

                # Handle packets (will receive peer's DISCOVERY here)
                await self._handle_peer_packets(peer)
                return

            except (ConnectionError, asyncio.TimeoutError, OSError) as e:
                logger.debug(f"Failed to connect to {peer_name}: {e}")
                # Exponential backoff with jitter (cap at 30s)
                delay = min(2 ** attempt, 30) + random.uniform(0, 2)
                await asyncio.sleep(delay)
                attempt += 1

        logger.warning(f"Could not connect to peer {peer_name} after {attempt} attempts")

    # ── Packet Handler ────────────────────────────────────────

    async def _handle_peer_packets(self, peer: PeerConnection):
        """Main loop: read and dispatch packets from a connected peer."""
        try:
            while self._running:
                packet = await asyncio.wait_for(
                    read_frame(peer.reader),
                    timeout=HEARTBEAT_INTERVAL * HEARTBEAT_TIMEOUT_MULT,
                )
                if packet is None:
                    break

                peer.last_heartbeat = time.time()
                await self._dispatch_packet(peer, packet)

        except asyncio.TimeoutError:
            logger.warning(f"Peer {peer.node_name} timed out after {HEARTBEAT_INTERVAL * HEARTBEAT_TIMEOUT_MULT}s — will reconnect")
        except (ConnectionError, asyncio.IncompleteReadError) as e:
            logger.info(f"Peer {peer.node_name} disconnected: {e}")
        finally:
            async with self._peer_lock:
                self._peers.pop(peer.node_name, None)
            self._mesh.remove_peer(peer.node_name)

    async def _dispatch_packet(self, peer: PeerConnection, packet: COSMOSPacket):
        """Dispatch a received packet to the appropriate handler."""
        t = packet.packet_type

        if t == PacketType.HEARTBEAT:
            await self._handle_heartbeat(peer, packet)
        elif t == PacketType.DISCOVERY:
            await self._handle_discovery(peer, packet)
        elif t == PacketType.ROUTE:
            await self._handle_route(peer, packet)
        elif t == PacketType.DIALOGUE:
            await self._handle_dialogue(peer, packet)
        elif t == PacketType.JOIN_REQ:
            await self._handle_join_request(peer, packet)
        elif t == PacketType.JOIN_VOTE:
            await self._handle_join_vote(peer, packet)
        elif t == PacketType.LATENT_ROUTE:
            await self._handle_latent_route(peer, packet)
        elif t == PacketType.POI_REQUEST:
            await self._handle_poi_request(peer, packet)
        elif t == PacketType.POI_ATTESTATION:
            await self._handle_poi_attestation(peer, packet)
        elif t == PacketType.POI_RESULT:
            await self._handle_poi_result(peer, packet)
        elif t == PacketType.MODEL_ATTEST:
            await self._handle_model_attest(peer, packet)
        elif t == PacketType.MEMORY_PROPOSE:
            await self._handle_memory_propose(peer, packet)
        elif t == PacketType.MEMORY_VOTE:
            await self._handle_memory_vote(peer, packet)
        elif t == PacketType.MEMORY_SYNC:
            await self._handle_memory_sync(peer, packet)
        elif t == PacketType.ACK:
            pass  # Acknowledged
        else:
            logger.debug(f"Unhandled packet type {t} from {peer.node_name}")

    # ── Handlers ──────────────────────────────────────────────

    async def _handle_heartbeat(self, peer: PeerConnection, packet: COSMOSPacket):
        """Handle heartbeat — update mesh state."""
        peer.last_heartbeat = time.time()
        hb_data = packet.data or {}

        self._mesh.update_peer(
            peer.node_name,
            packet.mesh_hash or hb_data.get("hash", b""),
            hb_data.get("seq", 0),
        )

        # Check mesh consistency
        claimed_root = hb_data.get("mesh_root", b"")
        if claimed_root:
            self._mesh.verify_peer_mesh(peer.node_name, claimed_root)

    async def _handle_discovery(self, peer: PeerConnection, packet: COSMOSPacket):
        """Handle bot discovery — exchange available bots."""
        data = packet.data or {}
        if data.get("query"):
            # They're asking what we have — send ALL bots (local + mesh)
            await write_frame(peer.writer, make_discovery(
                self.node_name, bots=list(self.get_all_bots().keys())
            ))
        else:
            # They're telling us what they have
            peer.remote_bots = data.get("bots", [])
            logger.info(f"Peer {peer.node_name} has bots: {peer.remote_bots}")

    async def _handle_route(self, peer: PeerConnection, packet: COSMOSPacket):
        """
        Handle ROUTE request — query a local bot OR forward to the correct peer.
        This is the HOT PATH for dialogue.
        """
        data = packet.data or {}
        bot_name = data.get("bot", packet.target)
        prompt = data.get("prompt", "")
        stream_id = packet.stream_id or str(uuid.uuid4())[:8]
        max_tokens = data.get("max_tokens", 4000)

        if bot_name not in self._local_bots:
            # Bot not local — try forwarding to the peer that has it
            await self._forward_route(peer, bot_name, prompt, stream_id, max_tokens)
            return

        query_fn = self._local_bots[bot_name]

        try:
            # Query the local bot
            response = await query_fn(prompt, max_tokens)

            if not response:
                response = "[No response from bot]"

            # Stream the response in chunks for maximum speed
            chunks = [response[i:i + STREAM_CHUNK_SIZE]
                      for i in range(0, len(response), STREAM_CHUNK_SIZE)]

            for idx, chunk in enumerate(chunks):
                is_final = (idx == len(chunks) - 1)
                pkt = make_dialogue(
                    sender=bot_name,
                    target=peer.node_name,
                    content=chunk,
                    stream_id=stream_id,
                    sequence=idx,
                    streaming=not is_final,
                    final=is_final,
                )

                # Seal the packet
                payload_bytes = chunk.encode("utf-8") if isinstance(chunk, str) else chunk
                my_seal = RollingSeal(self._identity)
                pkt.seal, _ = my_seal.seal(payload_bytes)
                pkt.mesh_hash = self._mesh.my_hash

                await write_frame(peer.writer, pkt)

            logger.debug(f"Streamed {len(chunks)} chunks for {bot_name} → {peer.node_name}")

        except Exception as e:
            logger.error(f"Route handler error for {bot_name}: {e}")
            await write_frame(peer.writer, COSMOSPacket(
                packet_type=PacketType.ERROR,
                sender=self.node_name,
                stream_id=stream_id,
                data={"error": str(e)},
            ))

    async def _forward_route(self, requester: PeerConnection, bot_name: str,
                             prompt: str, stream_id: str, max_tokens: int):
        """
        Forward a ROUTE request to the peer that owns the bot.
        Relays DIALOGUE chunks back to the original requester.
        """
        # Find which peer has this bot
        target_peer = None
        async with self._peer_lock:
            for p in self._peers.values():
                if bot_name in p.remote_bots:
                    target_peer = p
                    break

        if not target_peer:
            await write_frame(requester.writer, COSMOSPacket(
                packet_type=PacketType.ERROR,
                sender=self.node_name,
                target=requester.node_name,
                stream_id=stream_id,
                data={"error": f"Bot '{bot_name}' not found on any peer"},
            ))
            return

        logger.info(f"Forwarding ROUTE for '{bot_name}' to {target_peer.node_name}")

        # Set up a relay queue to collect responses from the target peer
        relay_q: asyncio.Queue = asyncio.Queue()
        self._response_queues[stream_id] = relay_q

        try:
            # Forward the ROUTE to the target peer
            route_pkt = COSMOSPacket(
                packet_type=PacketType.ROUTE,
                sender=self.node_name,
                target=bot_name,
                stream_id=stream_id,
                data={"prompt": prompt, "bot": bot_name, "max_tokens": max_tokens},
            )
            await write_frame(target_peer.writer, route_pkt)

            # Relay responses back to the original requester
            # Large models (80B+) can take 2+ minutes, so use generous timeout
            deadline = time.time() + 300.0
            while time.time() < deadline:
                try:
                    remaining = max(1, deadline - time.time())
                    pkt = await asyncio.wait_for(relay_q.get(), timeout=remaining)
                    if pkt is None:
                        break

                    # Forward the packet to the requester as-is
                    await write_frame(requester.writer, pkt)

                    if pkt.final or pkt.packet_type == PacketType.ERROR:
                        break
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout relaying for stream {stream_id}")
                    break

        except Exception as e:
            logger.error(f"Forward error for {bot_name}: {e}")
            await write_frame(requester.writer, COSMOSPacket(
                packet_type=PacketType.ERROR,
                sender=self.node_name,
                stream_id=stream_id,
                data={"error": f"Forward failed: {e}"},
            ))
        finally:
            self._response_queues.pop(stream_id, None)

    async def _handle_dialogue(self, peer: PeerConnection, packet: COSMOSPacket):
        """Handle incoming DIALOGUE packets — reassemble streamed responses."""
        stream_id = packet.stream_id
        if not stream_id:
            return

        # Put into the response queue for the waiting client
        q = self._response_queues.get(stream_id)
        if q:
            await q.put(packet)
            if packet.final:
                await q.put(None)  # Signal end of stream
        else:
            logger.debug(f"No listener for stream {stream_id}")

    async def _handle_join_request(self, peer: PeerConnection, packet: COSMOSPacket):
        """Handle PRO user join request — queue for manual approval."""
        data = packet.data or {}
        request = {
            "id": str(uuid.uuid4())[:8],
            "name": packet.sender,
            "gpu_fp": data.get("gpu_fp", b"").hex() if isinstance(data.get("gpu_fp"), bytes) else str(data.get("gpu_fp", "")),
            "benchmark": data.get("benchmark", 0),
            "latency_ms": data.get("latency_ms", 999),
            "pro_token": data.get("pro_token", ""),
            "info": data.get("info", {}),
            "timestamp": time.time(),
            "status": "pending",
        }

        self._pending_approvals.append(request)
        self._save_pending_approvals()

        logger.info(f"PRO join request from {packet.sender}: benchmark={data.get('benchmark')}, latency={data.get('latency_ms')}ms")

        # Auto-reject if specs don't meet minimum
        from .cosmos_config import PRO_NODE_REQUIREMENTS
        if data.get("latency_ms", 999) > PRO_NODE_REQUIREMENTS["max_latency_ms"]:
            request["status"] = "rejected"
            request["reason"] = f"Latency {data.get('latency_ms')}ms exceeds {PRO_NODE_REQUIREMENTS['max_latency_ms']}ms max"
            logger.info(f"Auto-rejected {packet.sender}: {request['reason']}")

    async def _handle_join_vote(self, peer: PeerConnection, packet: COSMOSPacket):
        """Handle quorum vote from another node."""
        data = packet.data or {}
        request_id = data.get("request_id", "")
        approve = data.get("approve", False)

        decision = self._quorum.cast_vote(request_id, peer.node_name, approve)
        if decision == "accepted":
            logger.info(f"Quorum accepted join request {request_id}")
        elif decision == "rejected":
            logger.info(f"Quorum rejected join request {request_id}")

    def _save_pending_approvals(self):
        """Persist pending approvals for the admin dashboard."""
        ensure_dirs()
        try:
            COSMOS_PENDING_FILE.write_text(json.dumps(self._pending_approvals, indent=2))
        except Exception as e:
            logger.debug(f"Could not save pending approvals: {e}")

    # ── v2.0 Handlers: PoI, Latent Routing, Attestation, Memory ──

    async def _handle_latent_route(self, peer: PeerConnection, packet: COSMOSPacket):
        """Handle LATENT_ROUTE — auto-route by semantic embedding."""
        data = packet.data or {}
        prompt = data.get("prompt", "")
        stream_id = packet.stream_id or str(uuid.uuid4())[:8]
        max_tokens = data.get("max_tokens", 4000)

        all_bots = list(self.get_all_bots().keys())
        decision = self._latent_router.route(prompt, all_bots)

        if not decision.selected_bot:
            await write_frame(peer.writer, COSMOSPacket(
                packet_type=PacketType.ERROR,
                sender=self.node_name,
                stream_id=stream_id,
                data={"error": "No model available for latent routing"},
            ))
            return

        logger.info(
            f"Latent route: '{prompt[:40]}...' → {decision.selected_bot} "
            f"(confidence={decision.confidence:.3f})"
        )

        # Rewrite as a standard ROUTE to the selected bot
        route_pkt = COSMOSPacket(
            packet_type=PacketType.ROUTE,
            sender=packet.sender,
            target=decision.selected_bot,
            stream_id=stream_id,
            data={
                "prompt": prompt,
                "bot": decision.selected_bot,
                "max_tokens": max_tokens,
                "latent_route": True,
                "route_confidence": decision.confidence,
                "route_scores": decision.all_scores,
            },
        )
        await self._handle_route(peer, route_pkt)

    async def _handle_poi_request(self, peer: PeerConnection, packet: COSMOSPacket):
        """Handle POI_REQUEST — participate in consensus inference round."""
        data = packet.data or {}
        round_id = data.get("round_id", "")
        prompt = data.get("prompt", "")
        model_name = data.get("model", "")

        if not round_id or not prompt or not model_name:
            return

        # Check if we have this model locally
        if model_name not in self._local_bots:
            logger.debug(f"PoI: don't have model {model_name}, skipping round {round_id}")
            return

        logger.info(f"PoI: participating in round {round_id} with {model_name}")

        # Run inference
        query_fn = self._local_bots[model_name]
        start = time.time()
        try:
            output = await query_fn(prompt, 4000)
            inference_ms = (time.time() - start) * 1000

            if not output:
                return

            # Create hardware-bound attestation
            attestation = self._poi.create_attestation(
                round_id, prompt, model_name, output, inference_ms
            )

            # Send attestation back
            await write_frame(peer.writer, make_poi_attestation(
                self.node_name, attestation
            ))
            logger.info(f"PoI: sent attestation for round {round_id}, output_hash={attestation.output_hash[:8].hex()}")

        except Exception as e:
            logger.error(f"PoI inference error: {e}")

    async def _handle_poi_attestation(self, peer: PeerConnection, packet: COSMOSPacket):
        """Handle POI_ATTESTATION — collect attestation from a validator."""
        data = packet.data or {}
        attestation = InferenceAttestation.from_dict(data)

        if not attestation.round_id:
            return

        if self._poi.add_attestation(attestation.round_id, attestation):
            # Try to reach consensus
            proof = self._poi.try_consensus(attestation.round_id)
            if proof:
                # Broadcast consensus proof to all peers
                result_pkt = make_poi_result(self.node_name, proof)
                async with self._peer_lock:
                    for p in self._peers.values():
                        try:
                            await write_frame(p.writer, result_pkt)
                        except Exception:
                            pass
                logger.info(f"PoI: consensus reached and broadcast for round {attestation.round_id}")

    async def _handle_poi_result(self, peer: PeerConnection, packet: COSMOSPacket):
        """Handle POI_RESULT — receive consensus proof from another node."""
        data = packet.data or {}
        logger.info(
            f"PoI proof received from {peer.node_name}: "
            f"round={data.get('round_id')}, "
            f"agreement={data.get('agreement_ratio', 0):.0%}, "
            f"verified={data.get('verified', False)}"
        )

    async def _handle_model_attest(self, peer: PeerConnection, packet: COSMOSPacket):
        """Handle MODEL_ATTEST — receive model attestation from peer."""
        data = packet.data or {}
        attestation = ModelAttestation.from_dict(data)

        if self._attestor:
            self._attestor.add_remote_attestation(attestation)
            trust = self._attestor.get_trust_score(peer.node_name)
            logger.info(
                f"Model attestation from {peer.node_name}: "
                f"model={attestation.model_name}, "
                f"trust_score={trust:.3f}"
            )

    async def _handle_memory_propose(self, peer: PeerConnection, packet: COSMOSPacket):
        """Handle MEMORY_PROPOSE — verify and vote on proposed crystal."""
        data = packet.data or {}
        crystal = MemoryCrystal.from_dict(data)

        if not crystal.crystal_id or not crystal.content:
            return

        # Store the proposed crystal
        if self._swarm_memory:
            if crystal.crystal_id not in self._swarm_memory._crystals:
                self._swarm_memory._crystals[crystal.crystal_id] = crystal

            # Auto-verify using a local model if available
            if self._local_bots:
                asyncio.create_task(
                    self._verify_crystal(peer, crystal)
                )

    async def _verify_crystal(self, peer: PeerConnection, crystal: MemoryCrystal):
        """Verify a crystal by running inference and comparing."""
        # Pick the best local model for verification
        available = list(self._local_bots.keys())
        if not available:
            return

        # Use latent router to pick the best verifier
        verify_prompt = (
            f"Verify this statement is accurate. Respond with AGREE or DISAGREE "
            f"and a brief explanation:\n\n{crystal.content}"
        )
        decision = self._latent_router.route(verify_prompt, available)
        model = decision.selected_bot or available[0]
        query_fn = self._local_bots[model]

        try:
            output = await query_fn(verify_prompt, 500)
            if not output:
                return

            output_lower = output.lower()
            agrees = "agree" in output_lower and "disagree" not in output_lower
            confidence = 0.8 if agrees else 0.3

            verification = self._swarm_memory.create_verification(
                crystal.crystal_id, agrees, confidence, output, model
            )

            if verification:
                # Send vote back to the proposer
                vote_pkt = make_memory_vote(
                    self.node_name, crystal.crystal_id, verification
                )
                await write_frame(peer.writer, vote_pkt)

        except Exception as e:
            logger.error(f"Crystal verification error: {e}")

    async def _handle_memory_vote(self, peer: PeerConnection, packet: COSMOSPacket):
        """Handle MEMORY_VOTE — receive vote on a crystal."""
        data = packet.data or {}
        crystal_id = data.get("crystal_id", "")
        v_data = data.get("verification", {})

        if not crystal_id or not v_data:
            return

        verification = CrystalVerification.from_dict(v_data)
        if self._swarm_memory:
            self._swarm_memory.add_remote_verification(crystal_id, verification)
            logger.info(
                f"Memory vote from {peer.node_name}: "
                f"crystal={crystal_id}, agrees={verification.agrees}"
            )

    async def _handle_memory_sync(self, peer: PeerConnection, packet: COSMOSPacket):
        """Handle MEMORY_SYNC — merge crystal stores."""
        data = packet.data or {}
        remote_crystals = data.get("crystals", [])

        if self._swarm_memory and remote_crystals:
            self._swarm_memory.merge_remote_crystals(remote_crystals)
            logger.info(f"Memory sync from {peer.node_name}: {len(remote_crystals)} crystals")

    # ── v2.0 API Methods ──────────────────────────────────────

    async def consensus_query(self, prompt: str, model_name: str,
                              min_validators: int = 2,
                              timeout: float = 300.0) -> Optional[Dict]:
        """
        Run a Proof-of-Inference consensus query.

        Sends prompt to all peers with the model, collects attestations,
        returns consensus proof.
        """
        if not self._poi:
            return None

        round_id = self._poi.create_round(prompt, model_name, min_validators)

        # If we have the model locally, create our own attestation
        if model_name in self._local_bots:
            query_fn = self._local_bots[model_name]
            start = time.time()
            try:
                output = await query_fn(prompt, 4000)
                inference_ms = (time.time() - start) * 1000
                if output:
                    att = self._poi.create_attestation(
                        round_id, prompt, model_name, output, inference_ms
                    )
                    self._poi.add_attestation(round_id, att)
            except Exception as e:
                logger.error(f"Local PoI inference error: {e}")

        # Send POI_REQUEST to all peers
        req_pkt = make_poi_request(self.node_name, round_id, prompt, model_name, min_validators)
        async with self._peer_lock:
            for p in self._peers.values():
                try:
                    await write_frame(p.writer, req_pkt)
                except Exception:
                    pass

        # Wait for consensus
        deadline = time.time() + timeout
        while time.time() < deadline:
            proof = self._poi.try_consensus(round_id)
            if proof:
                return proof.to_dict()
            status = self._poi.get_round_status(round_id)
            if status and status["status"] in ("no_consensus",):
                return {"status": "no_consensus", "round_id": round_id}
            await asyncio.sleep(1.0)

        return {"status": "timeout", "round_id": round_id}

    async def latent_query(self, prompt: str,
                           max_tokens: int = 4000) -> Optional[Tuple[str, Dict]]:
        """
        Query with automatic latent space routing.

        Returns (response_text, route_decision_dict).
        """
        all_bots = list(self.get_all_bots().keys())
        decision = self._latent_router.route(prompt, all_bots)

        if not decision.selected_bot:
            return None

        # Route to selected bot
        if decision.selected_bot in self._local_bots:
            query_fn = self._local_bots[decision.selected_bot]
            response = await query_fn(prompt, max_tokens)
        else:
            response = await self.query_remote_bot(
                decision.selected_bot, prompt, max_tokens
            )

        return (response or "", decision.__dict__) if response is not None else None

    async def propose_memory(self, content: str, source_prompt: str,
                             source_model: str,
                             tags: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Propose a knowledge crystal for consensus verification.

        Returns crystal dict.
        """
        if not self._swarm_memory:
            return None

        crystal = self._swarm_memory.propose_crystal(
            content, source_prompt, source_model, tags
        )

        # Broadcast to peers for voting
        propose_pkt = make_memory_propose(self.node_name, crystal)
        async with self._peer_lock:
            for p in self._peers.values():
                try:
                    await write_frame(p.writer, propose_pkt)
                except Exception:
                    pass

        return crystal.to_dict()

    # ── Remote Bot Query (Client-side) ────────────────────────

    async def query_remote_bot(self, bot_name: str, prompt: str,
                               max_tokens: int = 4000,
                               timeout: float = 120.0) -> Optional[str]:
        """
        Query a bot on a remote node. Returns complete response.
        Automatically finds which peer has the bot.
        """
        # Find which peer has this bot
        peer = None
        async with self._peer_lock:
            for p in self._peers.values():
                if bot_name in p.remote_bots:
                    peer = p
                    break

        if not peer:
            logger.warning(f"Bot '{bot_name}' not found on any connected peer")
            return None

        stream_id = str(uuid.uuid4())[:8]
        q: asyncio.Queue = asyncio.Queue()
        self._response_queues[stream_id] = q

        try:
            # Send ROUTE request
            route_pkt = COSMOSPacket(
                packet_type=PacketType.ROUTE,
                sender=self.node_name,
                target=bot_name,
                stream_id=stream_id,
                data={"prompt": prompt, "bot": bot_name, "max_tokens": max_tokens},
            )
            await write_frame(peer.writer, route_pkt)

            # Collect streamed response
            chunks = []
            deadline = time.time() + timeout

            while time.time() < deadline:
                try:
                    pkt = await asyncio.wait_for(q.get(), timeout=min(90, deadline - time.time()))
                    if pkt is None:
                        break  # End of stream
                    if pkt.packet_type == PacketType.ERROR:
                        error = pkt.data.get("error", "Unknown error") if isinstance(pkt.data, dict) else str(pkt.data)
                        logger.error(f"Remote error for {bot_name}: {error}")
                        return None
                    if pkt.packet_type == PacketType.DIALOGUE:
                        chunks.append(pkt.data if isinstance(pkt.data, str) else str(pkt.data))
                        if pkt.final:
                            break
                except asyncio.TimeoutError:
                    break

            return "".join(chunks) if chunks else None

        finally:
            self._response_queues.pop(stream_id, None)

    async def stream_remote_bot(self, bot_name: str, prompt: str,
                                max_tokens: int = 4000):
        """
        Stream dialogue from a remote bot. Yields chunks as they arrive.
        This is the FAST PATH — minimum latency per chunk.
        """
        peer = None
        async with self._peer_lock:
            for p in self._peers.values():
                if bot_name in p.remote_bots:
                    peer = p
                    break

        if not peer:
            return

        stream_id = str(uuid.uuid4())[:8]
        q: asyncio.Queue = asyncio.Queue()
        self._response_queues[stream_id] = q

        try:
            route_pkt = COSMOSPacket(
                packet_type=PacketType.ROUTE,
                sender=self.node_name,
                target=bot_name,
                stream_id=stream_id,
                data={"prompt": prompt, "bot": bot_name, "max_tokens": max_tokens},
            )
            await write_frame(peer.writer, route_pkt)

            while True:
                pkt = await asyncio.wait_for(q.get(), timeout=120.0)
                if pkt is None:
                    break
                if pkt.packet_type == PacketType.DIALOGUE:
                    yield pkt.data if isinstance(pkt.data, str) else str(pkt.data)
                    if pkt.final:
                        break
                elif pkt.packet_type == PacketType.ERROR:
                    break

        except asyncio.TimeoutError:
            pass
        finally:
            self._response_queues.pop(stream_id, None)

    # ── Background Loops ──────────────────────────────────────

    async def _heartbeat_loop(self):
        """Send heartbeats to all peers periodically."""
        while self._running:
            await asyncio.sleep(HEARTBEAT_INTERVAL)

            mesh_data = self._mesh.get_heartbeat_data()
            hb = make_heartbeat(self.node_name, self._mesh.my_hash)
            hb.data = mesh_data

            async with self._peer_lock:
                dead = []
                for name, peer in self._peers.items():
                    try:
                        await write_frame(peer.writer, hb)
                    except (ConnectionError, OSError):
                        dead.append(name)

                for name in dead:
                    logger.info(f"Peer {name} dead (heartbeat failed)")
                    self._peers.pop(name, None)
                    self._mesh.remove_peer(name)

    async def _mesh_advance_loop(self):
        """Advance the mesh hash chain periodically."""
        while self._running:
            await asyncio.sleep(HEARTBEAT_INTERVAL * 2)
            self._mesh.advance()
            self._mesh.prune_stale_peers()
            if self._poi:
                self._poi.cleanup_stale()

    async def _reconnect_loop(self):
        """Periodically check for missing peers and reconnect."""
        while self._running:
            await asyncio.sleep(RECONNECT_INTERVAL)

            nodes = load_known_nodes()
            for name, cfg in nodes.items():
                if name == self.node_name or not cfg.approved:
                    continue
                if name not in self._peers:
                    logger.info(f"Peer {name} missing, attempting reconnect...")
                    # Core peers get infinite retries, PRO peers get limited
                    retries = 0 if cfg.node_type == "core" else 5
                    asyncio.create_task(
                        self._connect_to_peer(name, cfg.host, cfg.port, max_retries=retries)
                    )

    async def _memory_sync_loop(self):
        """Periodically sync swarm memory crystals across peers."""
        while self._running:
            await asyncio.sleep(60)  # Sync every 60 seconds

            if not self._swarm_memory:
                continue

            crystals = self._swarm_memory.get_sync_payload()
            if not crystals:
                continue

            sync_pkt = make_memory_sync(self.node_name, crystals)
            async with self._peer_lock:
                for p in self._peers.values():
                    try:
                        await write_frame(p.writer, sync_pkt)
                    except Exception:
                        pass

    # ── Admin API ─────────────────────────────────────────────

    def approve_join(self, request_id: str) -> bool:
        """Admin approves a PRO user join request."""
        for req in self._pending_approvals:
            if req["id"] == request_id and req["status"] == "pending":
                req["status"] = "approved"

                # Add to known nodes
                node = NodeConfig(
                    name=req["name"],
                    host=req.get("info", {}).get("host", ""),
                    port=req.get("info", {}).get("port", COSMOS_PORT),
                    node_type="pro",
                    gpu_model=req.get("info", {}).get("gpu_model", ""),
                    vram_gb=req.get("info", {}).get("vram_gb", 0),
                    approved=True,
                )
                save_node(node)
                self._save_pending_approvals()
                logger.info(f"Admin approved PRO node: {req['name']}")
                return True
        return False

    def reject_join(self, request_id: str, reason: str = "") -> bool:
        """Admin rejects a PRO user join request."""
        for req in self._pending_approvals:
            if req["id"] == request_id and req["status"] == "pending":
                req["status"] = "rejected"
                req["reason"] = reason
                self._save_pending_approvals()
                logger.info(f"Admin rejected PRO node: {req['name']} ({reason})")
                return True
        return False

    def get_status(self) -> Dict:
        """Get node status for API/dashboard."""
        status = {
            "node_name": self.node_name,
            "version": "2.0.0",
            "identity": self._identity[:8].hex(),
            "gpu_fingerprint": self._gpu_fp[:8].hex() if self._gpu_fp else "none",
            "port": self.port,
            "connected_peers": list(self._peers.keys()),
            "peer_count": len(self._peers),
            "local_bots": self.get_local_bots(),
            "all_bots": self.get_all_bots(),
            "mesh_root": self._mesh.mesh_root[:8].hex() if self._mesh.mesh_root else "none",
            "mesh_peers": self._mesh.peer_count,
            "mesh_sequence": self._mesh.my_sequence,
            "pending_approvals": len([a for a in self._pending_approvals if a["status"] == "pending"]),
        }

        # v2.0 subsystem status
        if self._poi:
            status["poi"] = {
                "active_rounds": len(self._poi._rounds),
                "completed_proofs": len(self._poi._proofs),
            }
        if self._latent_router:
            status["latent_router"] = self._latent_router.get_routing_stats()
        if self._attestor:
            status["attestation"] = self._attestor.get_chain_summary()
        if self._swarm_memory:
            status["swarm_memory"] = self._swarm_memory.get_stats()

        return status


# ── Singleton ─────────────────────────────────────────────────

_cosmos_node: Optional[COSMOSNode] = None


def get_cosmos_node() -> Optional[COSMOSNode]:
    """Get the running COSMOS node (if any)."""
    return _cosmos_node


async def start_cosmos_node(node_name: str, host: str = "0.0.0.0",
                           port: int = COSMOS_PORT) -> COSMOSNode:
    """Start the COSMOS node daemon."""
    global _cosmos_node
    ensure_dirs()
    _cosmos_node = COSMOSNode(node_name, host, port)
    await _cosmos_node.start()
    return _cosmos_node


# ── CLI Entry Point ───────────────────────────────────────────

async def _main():
    parser = argparse.ArgumentParser(description="COSMOS Node Daemon")
    parser.add_argument("--name", required=True, help="Node name (e.g., nexus-alpha)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=COSMOS_PORT, help="Listen port")
    args = parser.parse_args()

    node = await start_cosmos_node(args.name, args.host, args.port)

    # Register local bots from Ollama
    try:
        await _register_ollama_bots(node)
    except Exception as e:
        logger.warning(f"Could not register Ollama bots: {e}")

    logger.info(f"COSMOS Node '{args.name}' running. Press Ctrl+C to stop.")

    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        await node.stop()


async def _register_ollama_bots(node: COSMOSNode):
    """Auto-register local Ollama models as COSMOS bots."""
    import httpx

    def _make_query_fn(model: str):
        """Create a query function for an Ollama model."""
        async def query_fn(prompt: str, max_tokens: int = 4000) -> str:
            async with httpx.AsyncClient() as c:
                r = await c.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {"num_predict": max_tokens},
                    },
                    timeout=120.0,
                )
                if r.status_code == 200:
                    return r.json().get("message", {}).get("content", "")
                return ""
        return query_fn

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:11434/api/tags", timeout=5.0)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                for model_info in models:
                    model_name = model_info["name"]
                    bot_name = model_name.replace(":", "-")
                    node.register_bot(bot_name, _make_query_fn(model_name))

                logger.info(f"Registered {len(models)} Ollama models as COSMOS bots")
    except Exception as e:
        logger.warning(f"Ollama not available: {e}")


if __name__ == "__main__":
    asyncio.run(_main())

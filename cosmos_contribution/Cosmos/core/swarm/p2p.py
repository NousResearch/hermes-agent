"""
cosmos P2P Swarm v2.6 - The "Professorial" Gossip Protocol.

"I decided to build my own internet, with blackjack and... wait, wrong catchphrase."

Improvements:
1. TCP-based Multiplexed Streams: High reliability peer communication.
2. Gossipsub Simulation: Efficient broadcast of Knowledge Fragments (DKG).
3. Kademlia-inspired DHT: Distributed routing of agent capabilities.
4. Auto-Discovery: Async UDP beaconing for seamless swarm entry.
5. WAN Bootstrap: WebSocket connection to relay server with password auth.
"""

import asyncio
import json
import uuid
import socket
import os
from typing import   Optional, Set
from dataclasses import dataclass, field
from loguru import logger

# Optional websockets for bootstrap connection
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from Cosmos.core.nexus import nexus, Signal, SignalType
from Cosmos.core.swarm.dkg import DecentralizedKnowledgeGraph

@dataclass
class PeerInfo:
    id: str
    addr: str
    port: int
    capabilities: list[str]
    writer: Optional[asyncio.StreamWriter] = None
    last_seen: float = field(default_factory=lambda: 0.0)
    missed_pings: int = 0  # V4.0: Heartbeat tracking


@dataclass
class StateCheckpoint:
    """V4.0: Periodic state snapshot for fault-tolerant recovery."""
    node_id: str
    timestamp: float
    epoch: int
    dark_matter_state: Optional[dict[str, float]] = None
    plasticity_hash: Optional[str] = None
    awareness_alignment: Optional[float] = None

class SwarmFabric:
    """
    Advanced Decentralized Networking Layer.
    """
    def __init__(self, node_id: Optional[str] = None, port: int = 9999,
                 bootstrap_url: Optional[str] = None, bootstrap_password: Optional[str] = None):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.port = port
        self.peers: dict[str, PeerInfo] = {}
        self.dkg = DecentralizedKnowledgeGraph(self.node_id)
        self.seen_messages: Set[str] = set()  # For gossip deduplication

        # Bootstrap configuration (for WAN connectivity)
        self.bootstrap_url = bootstrap_url or os.getenv("cosmos_BOOTSTRAP_PEER")
        self.bootstrap_password = bootstrap_password or os.getenv("cosmos_BOOTSTRAP_PASSWORD")
        self.bootstrap_ws = None
        self.bootstrap_authenticated = False

    async def start(self):
        """Boot the P2P Fabric."""
        logger.info(f"Swarm Fabric: Starting node {self.node_id} on port {self.port}")

        # 1. Start TCP Server (The listener)
        self.server = await asyncio.start_server(self._handle_peer_conn, '0.0.0.0', self.port)

        # 2. Start UDP Beacon (Discovery for LAN)
        asyncio.create_task(self._udp_beacon())
        asyncio.create_task(self._udp_listener())

        # 3. Start Peer Maintenance
        asyncio.create_task(self._maintain_peer_health())

        # 4. Connect to Bootstrap Node (WAN)
        if self.bootstrap_url and WEBSOCKETS_AVAILABLE:
            asyncio.create_task(self._connect_to_bootstrap())

        async with self.server:
            await self.server.serve_forever()

    async def _connect_to_bootstrap(self):
        """Connect to remote bootstrap node for WAN P2P."""
        while True:
            try:
                logger.info(f"P2P: Connecting to bootstrap {self.bootstrap_url}...")
                async with websockets.connect(self.bootstrap_url) as ws:
                    self.bootstrap_ws = ws
                    self.bootstrap_authenticated = False

                    # Send HELLO with password
                    hello_msg = {
                        "type": "HELLO",
                        "node_id": self.node_id,
                        "version": "2.9.0",
                        "capabilities": ["CV", "NLP", "P2P", "PLANETARY"]
                    }
                    if self.bootstrap_password:
                        hello_msg["password"] = self.bootstrap_password

                    await ws.send(json.dumps(hello_msg))

                    # Listen for messages
                    async for message in ws:
                        await self._process_bootstrap_message(message)

            except Exception as e:
                logger.warning(f"P2P: Bootstrap connection failed: {e}")
                self.bootstrap_ws = None
                self.bootstrap_authenticated = False

            # Reconnect after delay
            await asyncio.sleep(30)

    async def _process_bootstrap_message(self, message: str):
        """Process messages from bootstrap server."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "AUTH_FAILED":
                logger.error(f"P2P: Bootstrap auth failed - {data.get('message', 'Invalid password')}")
                self.bootstrap_authenticated = False
                return

            if msg_type == "WELCOME":
                self.bootstrap_authenticated = True
                peer_count = data.get("peer_count", 0)
                logger.info(f"P2P: Connected to bootstrap! {peer_count} peers in network")
                return

            if msg_type == "AUTH_REQUIRED":
                logger.warning("P2P: Bootstrap requires authentication")
                return

            if msg_type == "GOSSIP":
                # Received planetary skill from network
                skill_data = data.get("skill")
                if skill_data:
                    nexus.emit(Signal(
                        type=SignalType.EXTERNAL_EVENT,
                        payload={
                            "event": "planetary_skill_received",
                            "skill": skill_data
                        },
                        source="bootstrap"
                    ))
                    logger.info(f"P2P: Received skill from bootstrap: {skill_data.get('id', 'unknown')[:8]}...")

            if msg_type == "PEER_RES":
                # list of peers for direct connection
                peers = data.get("peers", [])
                logger.debug(f"P2P: Received {len(peers)} peers from bootstrap")
                for peer in peers:
                    asyncio.create_task(
                        self._connect_to_peer(peer["id"], peer["ip"], self.port, [])
                    )

        except json.JSONDecodeError:
            logger.error("P2P: Malformed message from bootstrap")
        except Exception as e:
            logger.error(f"P2P: Error processing bootstrap message: {e}")

    async def broadcast_to_bootstrap(self, msg: dict):
        """Send message to bootstrap for WAN distribution."""
        if self.bootstrap_ws and self.bootstrap_authenticated:
            try:
                await self.bootstrap_ws.send(json.dumps(msg))
            except Exception as e:
                logger.debug(f"P2P: Bootstrap send failed: {e}")

    async def _udp_beacon(self):
        """Broadcast presence via UDP."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while True:
            beacon = json.dumps({
                "type": "BEACON",
                "id": self.node_id,
                "port": self.port,
                "caps": ["CV", "NLP", "P2P"]
            })
            sock.sendto(beacon.encode(), ('<broadcast>', 8888))
            await asyncio.sleep(15)

    async def _udp_listener(self):
        """Listen for peer beacons."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', 8888))
        sock.setblocking(False)
        loop = asyncio.get_event_loop()
        while True:
            data, addr = await loop.sock_recvfrom(sock, 1024)
            msg = json.loads(data.decode())
            if msg["id"] != self.node_id:
                asyncio.create_task(self._connect_to_peer(msg["id"], addr[0], msg["port"], msg.get("caps", [])))

    async def _connect_to_peer(self, peer_id: str, host: str, port: int, caps: list[str]):
        if peer_id in self.peers: return
        try:
            reader, writer = await asyncio.open_connection(host, port)
            self._register_peer(peer_id, host, port, caps, writer)
            # Handshake
            await self._send_to_peer(peer_id, {"type": "HELLO", "id": self.node_id})
        except Exception as e:
            logger.trace(f"P2P: Failed to connect to {peer_id}: {e}")

    def _register_peer(self, pid, host, port, caps, writer):
        self.peers[pid] = PeerInfo(id=pid, addr=host, port=port, capabilities=caps, writer=writer)
        logger.info(f"P2P: Securely linked to peer '{pid}'")

    async def _handle_peer_conn(self, reader, writer):
        """Inbound TCP Connection handler."""
        peer_addr = writer.get_extra_info('peername')
        try:
            while True:
                data = await reader.read(4096)
                if not data: break
                msg = json.loads(data.decode())
                await self._process_peer_message(msg, writer)
        except Exception as e:
            pass
        finally:
            writer.close()

    async def _process_peer_message(self, msg: dict, writer: asyncio.StreamWriter):
        m_type = msg.get("type")
        m_id = msg.get("msg_id", str(uuid.uuid4()))

        if m_id in self.seen_messages:
            return
        self.seen_messages.add(m_id)

        if m_type == "HELLO":
            # Handshake acknowledgment
            peer_id = msg.get("id")
            if peer_id:
                logger.debug(f"P2P: Handshake from {peer_id}")
                # Send our capabilities back
                await self._send_to_writer(writer, {
                    "type": "HELLO_ACK",
                    "id": self.node_id,
                    "caps": ["CV", "NLP", "P2P", "PLANETARY"]
                })

        elif m_type == "HELLO_ACK":
            peer_id = msg.get("id")
            logger.debug(f"P2P: Handshake ACK from {peer_id}")

        elif m_type == "GOSSIP_DKG":
            # 1. Merge locally
            fragment = msg.get("fragment")
            if fragment:
                self.dkg.merge_fragment(fragment)
                logger.debug(f"P2P: Merged DKG fragment from gossip")
            # 2. Re-broadcast (Gossip)
            await self.gossip(msg)

        elif m_type == "GOSSIP_SKILL":
            # Planetary Memory skill sharing
            skill_data = msg.get("skill")
            if skill_data:
                # Signal to planetary memory
                nexus.emit(Signal(
                    type=SignalType.EXTERNAL_EVENT,
                    payload={
                        "event": "planetary_skill_received",
                        "skill": skill_data
                    },
                    source="p2p_fabric"
                ))
                logger.info(f"P2P: Received planetary skill: {skill_data.get('id', 'unknown')[:8]}...")
            await self.gossip(msg)

        elif m_type == "DHT_QUERY":
            # Find node with capability
            requested_cap = msg.get("capability")
            if requested_cap in ["CV", "NLP", "P2P", "PLANETARY"]:
                await self._send_to_writer(writer, {
                    "type": "DHT_RESPONSE",
                    "node_id": self.node_id,
                    "capability": requested_cap,
                    "available": True
                })

        elif m_type == "TASK_AUCTION":
            # Distributed Task Auction
            task_desc = msg.get("task")
            task_id = msg.get("task_id")
            # Emit for local agents to bid
            nexus.emit(Signal(
                type=SignalType.TASK_RECEIVED,
                payload={
                    "task_id": task_id,
                    "description": task_desc,
                    "from_peer": msg.get("from_node")
                },
                source="p2p_fabric"
            ))
            logger.info(f"P2P: Task auction received: {task_id}")

        elif m_type == "GOSSIP_AUDIO":
            # Planetary Audio Shard - metadata sharing
            metadata = msg.get("metadata")
            peer_id = msg.get("node_id", "unknown")
            if metadata:
                nexus.emit(Signal(
                    type=SignalType.EXTERNAL_EVENT,
                    payload={
                        "event": "planetary_audio_metadata",
                        "metadata": metadata,
                        "peer_id": peer_id
                    },
                    source="p2p_fabric"
                ))
                logger.debug(f"P2P: Received audio metadata: {metadata.get('text_hash', 'unknown')[:8]}...")
            await self.gossip(msg)

        elif m_type == "AUDIO_REQUEST":
            # Peer requesting audio file
            text_hash = msg.get("text_hash")
            requester_id = msg.get("requester_id")
            if text_hash and requester_id:
                nexus.emit(Signal(
                    type=SignalType.EXTERNAL_EVENT,
                    payload={
                        "event": "audio_request",
                        "text_hash": text_hash,
                        "requester_id": requester_id
                    },
                    source="p2p_fabric"
                ))
                logger.debug(f"P2P: Audio request for {text_hash[:8]}... from {requester_id}")

        elif m_type == "AUDIO_RESPONSE":
            # Peer sending audio file data
            text_hash = msg.get("text_hash")
            audio_data = msg.get("audio_data")
            sender_id = msg.get("sender_id")
            if text_hash and audio_data:
                nexus.emit(Signal(
                    type=SignalType.EXTERNAL_EVENT,
                    payload={
                        "event": "audio_response",
                        "text_hash": text_hash,
                        "audio_data": audio_data,
                        "sender_id": sender_id
                    },
                    source="p2p_fabric"
                ))
                logger.debug(f"P2P: Audio response for {text_hash[:8]}... from {sender_id}")

        elif m_type == "GOSSIP_LEARNING":
            # Shared learning data from other nodes
            cycle = msg.get("cycle", 0)
            concepts = msg.get("concepts", [])
            peer_id = msg.get("node_id", "unknown")
            nexus.emit(Signal(
                type=SignalType.EXTERNAL_EVENT,
                payload={
                    "event": "planetary_learning_received",
                    "cycle": cycle,
                    "concepts": concepts,
                    "tool_stats": msg.get("tool_stats", {}),
                    "user_count": msg.get("user_count", 0),
                    "peer_id": peer_id,
                    "timestamp": msg.get("timestamp")
                },
                source="p2p_fabric"
            ))
            logger.info(f"P2P: Received learning from {peer_id} - cycle {cycle}, {len(concepts)} concepts")
            # Re-gossip to propagate through network
            await self.gossip(msg)

        elif m_type == "GOSSIP_CONVERSATION":
            # Bot-to-bot conversation shared from other nodes
            conversation = msg.get("conversation", [])
            peer_id = msg.get("node_id", "unknown")
            if conversation:
                nexus.emit(Signal(
                    type=SignalType.EXTERNAL_EVENT,
                    payload={
                        "event": "planetary_conversation_received",
                        "conversation": conversation,
                        "peer_id": peer_id,
                        "timestamp": msg.get("timestamp")
                    },
                    source="p2p_fabric"
                ))
                logger.debug(f"P2P: Received conversation from {peer_id} ({len(conversation)} messages)")
            await self.gossip(msg)

        elif m_type == "GOSSIP_PLASTICITY":
            # V4.0: Shared Hebbian weights from peer
            weights_data = msg.get("weights")
            peer_id = msg.get("node_id", "unknown")
            if weights_data:
                nexus.emit(Signal(
                    type=SignalType.EXTERNAL_EVENT,
                    payload={
                        "event": "planetary_plasticity_received",
                        "weights": weights_data,
                        "peer_id": peer_id,
                    },
                    source="p2p_fabric"
                ))
                logger.info(f"P2P: Received plasticity weights from {peer_id} (epoch={weights_data.get('epoch', '?')})")
            await self.gossip(msg)

        elif m_type == "GOSSIP_STATE_CHECKPOINT":
            # V4.0: Fault-tolerant state checkpoint from peer
            checkpoint = msg.get("checkpoint")
            peer_id = msg.get("node_id", "unknown")
            if checkpoint:
                nexus.emit(Signal(
                    type=SignalType.EXTERNAL_EVENT,
                    payload={
                        "event": "state_checkpoint_received",
                        "checkpoint": checkpoint,
                        "peer_id": peer_id,
                    },
                    source="p2p_fabric"
                ))
                logger.debug(f"P2P: State checkpoint from {peer_id} (epoch={checkpoint.get('epoch', '?')})")

        elif m_type == "PING":
            # V4.0: Heartbeat response
            await self._send_to_writer(writer, {
                "type": "PONG",
                "id": self.node_id,
                "timestamp": asyncio.get_event_loop().time()
            })

        elif m_type == "PONG":
            # V4.0: Heartbeat acknowledgment — reset missed pings
            peer_id = msg.get("id")
            if peer_id and peer_id in self.peers:
                self.peers[peer_id].last_seen = asyncio.get_event_loop().time()
                self.peers[peer_id].missed_pings = 0

    async def _send_to_writer(self, writer: asyncio.StreamWriter, msg: dict):
        """Send message directly to a writer."""
        try:
            msg["msg_id"] = msg.get("msg_id", str(uuid.uuid4()))
            payload = json.dumps(msg).encode()
            writer.write(payload)
            await writer.drain()
        except Exception as e:
            logger.debug(f"P2P: Send failed: {e}")

    async def broadcast_skill(self, skill_data: dict):
        """Broadcast a planetary skill to the swarm (LAN + WAN)."""
        msg = {
            "type": "GOSSIP_SKILL",
            "skill": skill_data,
            "from_node": self.node_id
        }

        # Local peers (LAN)
        await self.gossip(msg)

        # Bootstrap (WAN) - send as GOSSIP for relay
        bootstrap_msg = {
            "type": "GOSSIP",
            "skill_id": skill_data.get("id", "unknown"),
            "skill": skill_data,
            "from_node": self.node_id
        }
        await self.broadcast_to_bootstrap(bootstrap_msg)

        total_peers = len(self.peers) + (1 if self.bootstrap_authenticated else 0)
        logger.info(f"P2P: Broadcasted skill to {total_peers} peers (LAN + WAN)")

    async def submit_task_auction(self, task_id: str, task_desc: str):
        """Submit a task for distributed auction."""
        msg = {
            "type": "TASK_AUCTION",
            "task_id": task_id,
            "task": task_desc,
            "from_node": self.node_id
        }
        await self.gossip(msg)
        logger.info(f"P2P: Submitted task auction: {task_id}")

    async def broadcast_message(self, msg: dict):
        """Broadcast dict message to all peers (LAN + WAN)."""
        # Add node_id if not present
        if "node_id" not in msg:
            msg["node_id"] = self.node_id

        # Local peers (LAN)
        await self.gossip(msg)

        # Bootstrap (WAN)
        await self.broadcast_to_bootstrap(msg)

        total_peers = len(self.peers) + (1 if self.bootstrap_authenticated else 0)
        logger.debug(f"P2P: Broadcasted message type '{msg.get('type')}' to {total_peers} peers")

    async def send_to_peer(self, peer_id: str, msg: dict):
        """Send a message to a specific peer by ID."""
        await self._send_to_peer(peer_id, msg)

    async def broadcast_conversation(self, conversation: list[dict]):
        """Share bot conversation excerpts across the planetary network."""
        if not conversation:
            return

        msg = {
            "type": "GOSSIP_CONVERSATION",
            "conversation": conversation[-10:],  # Last 10 messages
            "node_id": self.node_id,
            "timestamp": asyncio.get_event_loop().time()
        }

        # Local peers (LAN)
        await self.gossip(msg)

        # Bootstrap (WAN)
        await self.broadcast_to_bootstrap(msg)

        total_peers = len(self.peers) + (1 if self.bootstrap_authenticated else 0)
        logger.info(f"P2P: Shared conversation ({len(conversation)} msgs) with {total_peers} peers")

    async def gossip(self, msg: dict):
        """Propagate message through the fabric."""
        pids = list(self.peers.keys())
        # Gossip to random subset of peers (simplified scale)
        for pid in pids:
            await self._send_to_peer(pid, msg)

    async def _send_to_peer(self, peer_id: str, msg: dict):
        peer = self.peers.get(peer_id)
        if peer and peer.writer:
            try:
                msg["msg_id"] = msg.get("msg_id", str(uuid.uuid4()))
                payload = json.dumps(msg).encode()
                peer.writer.write(payload)
                await peer.writer.drain()
            except Exception:
                # Remove dead peer
                del self.peers[peer_id]

    async def _maintain_peer_health(self):
        """V4.0: Enhanced with heartbeat ping/pong and eviction."""
        while True:
            # Prune old seen messages
            if len(self.seen_messages) > 1000:
                self.seen_messages.clear()

            # Heartbeat: Ping all peers
            dead_peers = []
            for pid, peer in list(self.peers.items()):
                if peer.missed_pings >= 3:
                    # Evict after 3 missed pings (~90s unresponsive)
                    dead_peers.append(pid)
                    logger.warning(f"P2P: Evicting unresponsive peer {pid} (3 missed pings)")
                else:
                    peer.missed_pings += 1
                    try:
                        await self._send_to_peer(pid, {
                            "type": "PING",
                            "id": self.node_id,
                            "timestamp": asyncio.get_event_loop().time()
                        })
                    except Exception:
                        peer.missed_pings += 1

            for pid in dead_peers:
                if pid in self.peers:
                    try:
                        self.peers[pid].writer.close()
                    except Exception:
                        pass
                    del self.peers[pid]

            await asyncio.sleep(30)  # Ping every 30s

    async def broadcast_state_checkpoint(self, checkpoint: dict):
        """V4.0: Broadcast a state checkpoint for fault-tolerant recovery."""
        msg = {
            "type": "GOSSIP_STATE_CHECKPOINT",
            "checkpoint": checkpoint,
            "node_id": self.node_id,
        }
        await self.broadcast_message(msg)
        logger.info(f"P2P: Broadcasted state checkpoint (epoch={checkpoint.get('epoch', '?')})")

# Global Instance (will use env vars for bootstrap config)
swarm_fabric = SwarmFabric(
    bootstrap_url=os.getenv("cosmos_BOOTSTRAP_PEER"),
    bootstrap_password=os.getenv("cosmos_BOOTSTRAP_PASSWORD")
)

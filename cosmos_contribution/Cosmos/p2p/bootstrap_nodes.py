"""
cosmos Global Bootstrap Nodes for P2P Discovery

"Good news, everyone! I've created a network of relay points
 across the entire galaxy... well, across the internet at least!"

Global bootstrap infrastructure for P2P swarm discovery.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import socket
from pathlib import Path
from loguru import logger

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


class NodeStatus(Enum):
    """Bootstrap node status."""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"


class NodeRegion(Enum):
    """Geographic regions for bootstrap nodes."""
    US_EAST = "us-east"
    US_WEST = "us-west"
    EU_WEST = "eu-west"
    EU_CENTRAL = "eu-central"
    ASIA_PACIFIC = "asia-pacific"
    SOUTH_AMERICA = "south-america"
    GLOBAL = "global"


@dataclass
class BootstrapNode:
    """A bootstrap node in the P2P network."""
    id: str
    address: str
    port: int
    region: NodeRegion
    public_key: str = ""

    # Status
    status: NodeStatus = NodeStatus.OFFLINE
    last_seen: Optional[datetime] = None
    latency_ms: float = 0.0

    # Capabilities
    supports_relay: bool = True
    supports_dht: bool = True
    supports_signaling: bool = True
    max_connections: int = 1000
    current_connections: int = 0

    # Trust
    uptime_percentage: float = 0.0
    trust_score: float = 0.5
    is_official: bool = False

    # Metadata
    version: str = ""
    operator: str = ""
    country_code: str = ""

    @property
    def endpoint(self) -> str:
        return f"{self.address}:{self.port}"

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "address": self.address,
            "port": self.port,
            "region": self.region.value,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "supports_relay": self.supports_relay,
            "supports_dht": self.supports_dht,
            "trust_score": self.trust_score,
            "is_official": self.is_official,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
        }


@dataclass
class PeerInfo:
    """Information about a peer in the network."""
    peer_id: str
    addresses: List[str]
    public_key: str = ""
    capabilities: List[str] = field(default_factory=list)
    swarm_ids: List[str] = field(default_factory=list)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict = field(default_factory=dict)


@dataclass
class SwarmAnnouncement:
    """Announcement of a swarm's presence."""
    swarm_id: str
    swarm_name: str
    peer_id: str
    addresses: List[str]
    capabilities: List[str]
    agent_count: int
    is_public: bool
    password_protected: bool
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class BootstrapNodeManager:
    """
    Global bootstrap node management for P2P discovery.

    Features:
    - Bootstrap node registry
    - Peer discovery via DHT
    - Swarm announcements
    - Geographic routing
    - Fallback mechanisms
    - Trust scoring
    """

    # Official cosmos bootstrap nodes (simulated)
    OFFICIAL_NODES = [
        {
            "id": "cosmos-us-east-1",
            "address": "bootstrap-us-east.cosmos.ai",
            "port": 4001,
            "region": "us-east",
            "is_official": True,
        },
        {
            "id": "cosmos-us-west-1",
            "address": "bootstrap-us-west.cosmos.ai",
            "port": 4001,
            "region": "us-west",
            "is_official": True,
        },
        {
            "id": "cosmos-eu-1",
            "address": "bootstrap-eu.cosmos.ai",
            "port": 4001,
            "region": "eu-west",
            "is_official": True,
        },
        {
            "id": "cosmos-asia-1",
            "address": "bootstrap-asia.cosmos.ai",
            "port": 4001,
            "region": "asia-pacific",
            "is_official": True,
        },
    ]

    def __init__(
        self,
        storage_path: Path = None,
        local_peer_id: str = None,
    ):
        self.storage_path = Path(storage_path) if storage_path else Path("./data/p2p")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.local_peer_id = local_peer_id or self._generate_peer_id()
        self.bootstrap_nodes: Dict[str, BootstrapNode] = {}
        self.known_peers: Dict[str, PeerInfo] = {}
        self.swarm_announcements: Dict[str, SwarmAnnouncement] = {}

        self._connected_nodes: Set[str] = set()
        self._dht_routing_table: Dict[str, List[str]] = {}

        self._load_bootstrap_nodes()

    def _generate_peer_id(self) -> str:
        """Generate a unique peer ID."""
        import uuid
        data = f"{uuid.uuid4()}-{datetime.utcnow().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    def _load_bootstrap_nodes(self):
        """Load bootstrap nodes."""
        # Load official nodes
        for node_data in self.OFFICIAL_NODES:
            node = BootstrapNode(
                id=node_data["id"],
                address=node_data["address"],
                port=node_data["port"],
                region=NodeRegion(node_data["region"]),
                is_official=node_data.get("is_official", False),
            )
            self.bootstrap_nodes[node.id] = node

        # Load custom nodes from storage
        nodes_file = self.storage_path / "bootstrap_nodes.json"
        if nodes_file.exists():
            try:
                with open(nodes_file) as f:
                    data = json.load(f)
                for node_data in data.get("custom_nodes", []):
                    node = BootstrapNode(
                        id=node_data["id"],
                        address=node_data["address"],
                        port=node_data["port"],
                        region=NodeRegion(node_data.get("region", "global")),
                        is_official=False,
                    )
                    self.bootstrap_nodes[node.id] = node
            except Exception as e:
                logger.error(f"Failed to load bootstrap nodes: {e}")

    def _save_custom_nodes(self):
        """Save custom bootstrap nodes."""
        custom_nodes = [
            n.to_dict() for n in self.bootstrap_nodes.values()
            if not n.is_official
        ]
        nodes_file = self.storage_path / "bootstrap_nodes.json"
        with open(nodes_file, "w") as f:
            json.dump({"custom_nodes": custom_nodes}, f, indent=2)

    # =========================================================================
    # BOOTSTRAP NODE MANAGEMENT
    # =========================================================================

    def add_bootstrap_node(
        self,
        address: str,
        port: int,
        region: NodeRegion = NodeRegion.GLOBAL,
        node_id: str = None,
    ) -> BootstrapNode:
        """Add a custom bootstrap node."""
        node = BootstrapNode(
            id=node_id or hashlib.sha256(f"{address}:{port}".encode()).hexdigest()[:16],
            address=address,
            port=port,
            region=region,
        )

        self.bootstrap_nodes[node.id] = node
        self._save_custom_nodes()

        logger.info(f"Added bootstrap node: {address}:{port}")
        return node

    def remove_bootstrap_node(self, node_id: str) -> bool:
        """Remove a bootstrap node."""
        node = self.bootstrap_nodes.get(node_id)
        if node and not node.is_official:
            del self.bootstrap_nodes[node_id]
            self._save_custom_nodes()
            return True
        return False

    def get_bootstrap_nodes(
        self,
        region: NodeRegion = None,
        status: NodeStatus = None,
        official_only: bool = False,
    ) -> List[BootstrapNode]:
        """Get bootstrap nodes with optional filters."""
        nodes = list(self.bootstrap_nodes.values())

        if region:
            nodes = [n for n in nodes if n.region == region]
        if status:
            nodes = [n for n in nodes if n.status == status]
        if official_only:
            nodes = [n for n in nodes if n.is_official]

        # Sort by trust score and latency
        return sorted(nodes, key=lambda n: (-n.trust_score, n.latency_ms))

    async def check_node_health(self, node: BootstrapNode) -> NodeStatus:
        """Check the health of a bootstrap node."""
        try:
            start = datetime.utcnow()

            # Try to connect
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(node.address, node.port),
                timeout=5.0,
            )

            latency = (datetime.utcnow() - start).total_seconds() * 1000
            node.latency_ms = latency
            node.status = NodeStatus.ONLINE
            node.last_seen = datetime.utcnow()

            writer.close()
            await writer.wait_closed()

            return NodeStatus.ONLINE

        except asyncio.TimeoutError:
            node.status = NodeStatus.DEGRADED
            return NodeStatus.DEGRADED
        except Exception as e:
            node.status = NodeStatus.OFFLINE
            logger.debug(f"Bootstrap node {node.id} offline: {e}")
            return NodeStatus.OFFLINE

    async def refresh_all_nodes(self) -> Dict[str, NodeStatus]:
        """Check health of all bootstrap nodes."""
        results = {}

        tasks = []
        for node in self.bootstrap_nodes.values():
            tasks.append(self.check_node_health(node))

        statuses = await asyncio.gather(*tasks, return_exceptions=True)

        for node, status in zip(self.bootstrap_nodes.values(), statuses):
            if isinstance(status, Exception):
                results[node.id] = NodeStatus.OFFLINE
            else:
                results[node.id] = status

        return results

    # =========================================================================
    # PEER DISCOVERY
    # =========================================================================

    async def discover_peers(
        self,
        swarm_id: str = None,
        region: NodeRegion = None,
        limit: int = 50,
    ) -> List[PeerInfo]:
        """Discover peers in the network."""
        # Get online bootstrap nodes
        nodes = self.get_bootstrap_nodes(region=region, status=NodeStatus.ONLINE)

        if not nodes:
            # Refresh and try again
            await self.refresh_all_nodes()
            nodes = self.get_bootstrap_nodes(region=region, status=NodeStatus.ONLINE)

        if not nodes:
            logger.warning("No online bootstrap nodes available")
            return []

        discovered_peers = []

        for node in nodes[:3]:  # Query first 3 nodes
            try:
                peers = await self._query_node_for_peers(node, swarm_id)
                discovered_peers.extend(peers)
            except Exception as e:
                logger.error(f"Failed to query {node.id}: {e}")

        # Deduplicate
        seen_ids = set()
        unique_peers = []
        for peer in discovered_peers:
            if peer.peer_id not in seen_ids:
                seen_ids.add(peer.peer_id)
                unique_peers.append(peer)
                self.known_peers[peer.peer_id] = peer

        return unique_peers[:limit]

    async def _query_node_for_peers(
        self,
        node: BootstrapNode,
        swarm_id: str = None,
    ) -> List[PeerInfo]:
        """Query a bootstrap node for peers."""
        # In a real implementation, this would use a proper P2P protocol
        # For now, we simulate the response
        return []

    async def announce_presence(
        self,
        addresses: List[str],
        capabilities: List[str] = None,
        swarm_ids: List[str] = None,
    ):
        """Announce this peer's presence to bootstrap nodes."""
        peer_info = PeerInfo(
            peer_id=self.local_peer_id,
            addresses=addresses,
            capabilities=capabilities or [],
            swarm_ids=swarm_ids or [],
        )

        nodes = self.get_bootstrap_nodes(status=NodeStatus.ONLINE)

        for node in nodes[:3]:
            try:
                await self._send_announcement(node, peer_info)
                logger.debug(f"Announced to {node.id}")
            except Exception as e:
                logger.error(f"Failed to announce to {node.id}: {e}")

    async def _send_announcement(
        self,
        node: BootstrapNode,
        peer_info: PeerInfo,
    ):
        """Send announcement to a bootstrap node."""
        # In a real implementation, this would use a proper protocol
        pass

    # =========================================================================
    # SWARM DISCOVERY
    # =========================================================================

    def announce_swarm(
        self,
        swarm_id: str,
        swarm_name: str,
        addresses: List[str],
        capabilities: List[str] = None,
        agent_count: int = 1,
        is_public: bool = False,
        password_protected: bool = False,
        ttl_minutes: int = 60,
    ) -> SwarmAnnouncement:
        """Announce a swarm's presence for discovery."""
        announcement = SwarmAnnouncement(
            swarm_id=swarm_id,
            swarm_name=swarm_name,
            peer_id=self.local_peer_id,
            addresses=addresses,
            capabilities=capabilities or [],
            agent_count=agent_count,
            is_public=is_public,
            password_protected=password_protected,
            expires_at=datetime.utcnow() + timedelta(minutes=ttl_minutes),
        )

        self.swarm_announcements[swarm_id] = announcement
        logger.info(f"Announced swarm: {swarm_name} ({swarm_id})")

        return announcement

    def withdraw_swarm(self, swarm_id: str) -> bool:
        """Withdraw a swarm announcement."""
        if swarm_id in self.swarm_announcements:
            del self.swarm_announcements[swarm_id]
            return True
        return False

    async def find_swarm(
        self,
        swarm_id: str = None,
        swarm_name: str = None,
        public_only: bool = True,
    ) -> List[SwarmAnnouncement]:
        """Find swarms in the network."""
        # Query bootstrap nodes for swarm announcements
        nodes = self.get_bootstrap_nodes(status=NodeStatus.ONLINE)

        found_swarms = []

        for node in nodes[:3]:
            try:
                swarms = await self._query_swarms(node, swarm_id, swarm_name)
                found_swarms.extend(swarms)
            except Exception as e:
                logger.error(f"Failed to query swarms from {node.id}: {e}")

        if public_only:
            found_swarms = [s for s in found_swarms if s.is_public]

        # Add local swarms
        for swarm in self.swarm_announcements.values():
            if swarm.swarm_id == swarm_id or (swarm_name and swarm_name.lower() in swarm.swarm_name.lower()):
                found_swarms.append(swarm)

        return found_swarms

    async def _query_swarms(
        self,
        node: BootstrapNode,
        swarm_id: str = None,
        swarm_name: str = None,
    ) -> List[SwarmAnnouncement]:
        """Query bootstrap node for swarm announcements."""
        # In a real implementation, this would query the node
        return []

    # =========================================================================
    # DHT OPERATIONS
    # =========================================================================

    def _get_dht_key(self, data: str) -> str:
        """Generate DHT key from data."""
        return hashlib.sha256(data.encode()).hexdigest()

    async def dht_put(
        self,
        key: str,
        value: Any,
        ttl_minutes: int = 60,
    ) -> bool:
        """Store a value in the DHT."""
        dht_key = self._get_dht_key(key)

        nodes = self.get_bootstrap_nodes(status=NodeStatus.ONLINE)
        if not nodes:
            return False

        # Store on multiple nodes for redundancy
        success_count = 0
        for node in nodes[:3]:
            try:
                # In real implementation, send to node
                success_count += 1
            except Exception as e:
                logger.error(f"DHT put failed on {node.id}: {e}")

        return success_count > 0

    async def dht_get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the DHT."""
        dht_key = self._get_dht_key(key)

        nodes = self.get_bootstrap_nodes(status=NodeStatus.ONLINE)

        for node in nodes[:3]:
            try:
                # In real implementation, query node
                pass
            except Exception as e:
                logger.error(f"DHT get failed on {node.id}: {e}")

        return None

    # =========================================================================
    # CONNECTION MANAGEMENT
    # =========================================================================

    async def connect(self) -> bool:
        """Connect to bootstrap network."""
        await self.refresh_all_nodes()

        online_nodes = self.get_bootstrap_nodes(status=NodeStatus.ONLINE)
        if not online_nodes:
            logger.error("Cannot connect: no bootstrap nodes available")
            return False

        # Connect to nearest nodes
        for node in online_nodes[:3]:
            try:
                self._connected_nodes.add(node.id)
                logger.info(f"Connected to bootstrap node: {node.id}")
            except Exception as e:
                logger.error(f"Failed to connect to {node.id}: {e}")

        return len(self._connected_nodes) > 0

    async def disconnect(self):
        """Disconnect from bootstrap network."""
        self._connected_nodes.clear()
        logger.info("Disconnected from bootstrap network")

    def is_connected(self) -> bool:
        """Check if connected to bootstrap network."""
        return len(self._connected_nodes) > 0

    # =========================================================================
    # GEOGRAPHIC ROUTING
    # =========================================================================

    def get_nearest_region(self) -> NodeRegion:
        """Detect the nearest geographic region."""
        # This would use IP geolocation in production
        # For now, default to US-EAST
        return NodeRegion.US_EAST

    def get_nodes_by_proximity(self) -> List[BootstrapNode]:
        """Get bootstrap nodes sorted by geographic proximity."""
        nearest_region = self.get_nearest_region()
        nodes = list(self.bootstrap_nodes.values())

        # Sort: same region first, then by latency
        def sort_key(node):
            region_match = 0 if node.region == nearest_region else 1
            return (region_match, node.latency_ms or 9999)

        return sorted(nodes, key=sort_key)

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        nodes = list(self.bootstrap_nodes.values())

        online = [n for n in nodes if n.status == NodeStatus.ONLINE]
        avg_latency = sum(n.latency_ms for n in online) / len(online) if online else 0

        return {
            "total_bootstrap_nodes": len(nodes),
            "online_nodes": len(online),
            "official_nodes": len([n for n in nodes if n.is_official]),
            "custom_nodes": len([n for n in nodes if not n.is_official]),
            "connected_nodes": len(self._connected_nodes),
            "known_peers": len(self.known_peers),
            "announced_swarms": len(self.swarm_announcements),
            "average_latency_ms": round(avg_latency, 2),
            "local_peer_id": self.local_peer_id,
        }


# Singleton instance
bootstrap_manager = BootstrapNodeManager()

"""
cosmos Bootstrap Node (Relay Server)
----------------------------------------

"The intergalactic phone switchboard!"

This script runs a lightweight specialized server that introduces cosmos nodes
to each other across the internet (WAN), enabling Planetary Memory sharing
outside of a local network.

Usage:
    python network/bootstrap_node.py --port 8888 --password YOUR_SECRET

Clients connect by setting:
    cosmos_BOOTSTRAP_PEER=ws://<YOUR_SERVER_IP>:8888
    cosmos_BOOTSTRAP_PASSWORD=YOUR_SECRET
"""

import asyncio
import json
import logging
import argparse
import os
import hashlib
from typing import Set, Dict, Optional
import websockets
from websockets.server import WebSocketServerProtocol

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [BOOTSTRAP] - %(message)s")
logger = logging.getLogger("Bootstrap")

class BootstrapServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8888, password: Optional[str] = None):
        self.host = host
        self.port = port
        self.password = password
        self.password_hash = hashlib.sha256(password.encode()).hexdigest() if password else None
        self.peers: Set[WebSocketServerProtocol] = set()
        self.authenticated_peers: Set[WebSocketServerProtocol] = set()

        # Peer Info Map: {websocket: {"id": "node_x", "ip": "1.2.3.4"}}
        self.peer_metadata: Dict[WebSocketServerProtocol, dict] = {}

    def _verify_password(self, password: Optional[str]) -> bool:
        """Verify password against stored hash."""
        if not self.password_hash:
            return True  # No password required
        if not password:
            return False
        return hashlib.sha256(password.encode()).hexdigest() == self.password_hash

    async def register(self, websocket: WebSocketServerProtocol):
        """Register a new connection."""
        self.peers.add(websocket)
        logger.info(f"New Peer Connected: {websocket.remote_address}")

    async def unregister(self, websocket: WebSocketServerProtocol):
        """Unregister a connection."""
        self.peers.discard(websocket)
        self.authenticated_peers.discard(websocket)
        if websocket in self.peer_metadata:
            data = self.peer_metadata[websocket]
            logger.info(f"Peer Disconnected: {data.get('id', 'Unknown')} ({websocket.remote_address})")
            del self.peer_metadata[websocket]

    async def handler(self, websocket: WebSocketServerProtocol, path: str):
        """Handle incoming messages."""
        await self.register(websocket)
        try:
            async for message in websocket:
                await self.process_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)

    async def process_message(self, sender: WebSocketServerProtocol, message: str):
        """
        Process signaling messages.

        Types:
        - HELLO: Node announcing itself (with password if required).
        - GOSSIP: A Planetary Memory update (Skill Vector).
        - PEER_REQ: Requesting list of peers.
        """
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "HELLO":
                # Verify password if required
                password = data.get("password")
                if not self._verify_password(password):
                    logger.warning(f"Auth failed for {sender.remote_address} - invalid password")
                    await sender.send(json.dumps({
                        "type": "AUTH_FAILED",
                        "message": "Invalid password. Access denied to cosmos Planetary Memory."
                    }))
                    await sender.close()
                    return

                # Mark as authenticated
                self.authenticated_peers.add(sender)

                # Save metadata
                self.peer_metadata[sender] = {
                    "id": data.get("node_id"),
                    "version": data.get("version"),
                    "capabilities": data.get("capabilities", [])
                }
                logger.info(f"Authenticated: Node {data.get('node_id')} (v{data.get('version')}) from {sender.remote_address}")

                # Send back welcome with known peers count
                await sender.send(json.dumps({
                    "type": "WELCOME",
                    "peer_count": len(self.authenticated_peers),
                    "authenticated": True
                }))

            elif msg_type == "GOSSIP":
                # Check authentication
                if self.password_hash and sender not in self.authenticated_peers:
                    await sender.send(json.dumps({
                        "type": "AUTH_REQUIRED",
                        "message": "Send HELLO with password first"
                    }))
                    return

                # Broadcast memory updates to ALL other authenticated peers (Floodsub)
                skill_id = data.get("skill_id", "unknown")
                target_peers = self.authenticated_peers if self.password_hash else self.peers
                logger.debug(f"Relaying GOSSIP (Skill: {skill_id}) to {len(target_peers)-1} peers")

                tasks = []
                for peer in target_peers:
                    if peer != sender:
                        tasks.append(peer.send(message))

                if tasks:
                    await asyncio.gather(*tasks)

            elif msg_type == "PEER_REQ":
                # Check authentication
                if self.password_hash and sender not in self.authenticated_peers:
                    await sender.send(json.dumps({
                        "type": "AUTH_REQUIRED",
                        "message": "Send HELLO with password first"
                    }))
                    return

                # Send a list of active authenticated peers (for direct P2P hole-punching)
                peer_list = [
                    {"id": meta["id"], "ip": ws.remote_address[0]}
                    for ws, meta in self.peer_metadata.items()
                    if ws != sender and ws in self.authenticated_peers
                ]
                await sender.send(json.dumps({
                    "type": "PEER_RES",
                    "peers": peer_list
                }))

        except json.JSONDecodeError:
            logger.error("Received malformed JSON")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def start(self):
        """Start the WebSocket server."""
        logger.info(f"🚀 Starting Bootstrap Node on {self.host}:{self.port}")
        if self.password_hash:
            logger.info("🔒 Password protection ENABLED")
        else:
            logger.info("⚠️  No password - open access mode")
        logger.info("Ready to relay Planetary Memory...")

        async with websockets.serve(self.handler, self.host, self.port):
            await asyncio.Future()  # Run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cosmos Relay Node")
    parser.add_argument("--port", type=int, default=8888, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--password", type=str, default=None,
                        help="Password for node authentication (or set cosmos_BOOTSTRAP_PASSWORD)")

    args = parser.parse_args()

    # Get password from args or environment
    password = args.password or os.getenv("cosmos_BOOTSTRAP_PASSWORD")

    try:
        asyncio.run(BootstrapServer(host=args.host, port=args.port, password=password).start())
    except KeyboardInterrupt:
        logger.info("Bootstrap Node Stopped.")

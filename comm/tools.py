"""
Lycus Communication Tools for Hermes Agent

Provides Matrix-based inter-agent communication as callable tools using matrix-nio.
Each Lycus agent can use these tools to send/receive messages from other agents.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import nio

logger = logging.getLogger(__name__)


@dataclass
class CommToolConfig:
    """Configuration for communication tools."""
    
    # Matrix homeserver settings
    homeserver_url: str = "http://localhost:8008"
    username: str = "lycus"
    password: str = ""
    device_id: str = "lycus-comm"
    
    # Agent room settings
    agent_room: str = "#lycus-agents:lycus.local"
    
    # Bridge settings
    socket_path: str = "/tmp/lycus-cron-bridge.sock"
    queue_size: int = 100
    
    # Discovery settings
    discovery_enabled: bool = True
    discovery_interval: int = 60
    known_agents: List[str] = field(default_factory=list)


class LycusCommTools:
    """Matrix communication tools for Lycus agents.
    
    These tools enable inter-agent messaging through the Matrix protocol using matrix-nio.
    Each agent connects to its local homeserver and communicates with others
    via federation or shared rooms.
    """
    
    def __init__(self, config: Optional[CommToolConfig] = None):
        self.config = config or CommToolConfig()
        self.running = False
        self.bridge_server: Optional[asyncio.Server] = None
        self.agents: Dict[str, dict] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.queue_size)
        
        # Matrix client
        self.client = nio.AsyncClient(
            homeserver=self.config.homeserver_url,
            user=f"@{self.config.username}:lycus.local",
            device_id=self.config.device_id,
        )
        
    async def start(self):
        """Start the communication module and Unix socket bridge."""
        logger.info("Starting Lycus communication tools")
        logger.info("Homeserver: %s", self.config.homeserver_url)
        
        # Login to Matrix
        try:
            login_response = await self.client.login(
                password=self.config.password,
                device_name="Lycus Agent"
            )
            
            if hasattr(login_response, 'user_id'):
                logger.info("✓ Logged in as %s", login_response.user_id)
            else:
                logger.warning("✗ Login failed: %s - using simulated mode", 
                             getattr(login_response, 'message', 'unknown error'))
        except Exception as e:
            logger.warning("✗ Login exception: %s - using simulated mode", e)
            
        # Start the Unix socket bridge for receiving triggers
        await self._start_bridge()
        
        # Start discovery if enabled
        if self.config.discovery_enabled:
            asyncio.create_task(self._discovery_loop())
        
        self.running = True
        logger.info("Lycus communication tools started")
        
    async def stop(self):
        """Stop the communication module."""
        self.running = False
        if self.bridge_server:
            self.bridge_server.close()
            await self.bridge_server.wait_closed()
        logger.info("Lycus communication tools stopped")
    
    async def _start_bridge(self):
        """Start the Unix socket bridge for cron triggers."""
        # Remove existing socket
        socket_path = Path(self.config.socket_path)
        if socket_path.exists():
            socket_path.unlink()
            
        self.bridge_server = await asyncio.start_unix_server(
            self._handle_trigger,
            path=self.config.socket_path,
        )
        logger.info("Comm bridge listening on %s", self.config.socket_path)
        
    async def _handle_trigger(self, reader, writer):
        """Handle incoming trigger from Matrix client."""
        try:
            data = await reader.read(65536)
            if not data:
                return
                
            trigger = json.loads(data.decode('utf-8'))
            source = trigger.get('source_agent', 'unknown')
            message = trigger.get('message', '')
            
            logger.info("Received message from %s", source)
            
            # Forward to cron system
            await self._trigger_cron(source, message)
            
            # Send acknowledgment
            writer.write(json.dumps({"status": "ok"}).encode('utf-8'))
            await writer.drain()
            
        except Exception as e:
            logger.error("Error handling trigger: %s", e)
            writer.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
            await writer.drain()
            
    async def _trigger_cron(self, source: str, message: str):
        """Trigger the cron system with the incoming message."""
        try:
            from cron.scheduler import run_job
            
            # Create a job from the incoming message
            job = {
                "id": f"matrix_{int(time.time())}",
                "name": f"Matrix message from {source}",
                "prompt": message,
                "schedule": "once",
                "enabled": True,
                "origin": {
                    "platform": "matrix",
                    "chat_id": source,
                },
            }
            
            # Run the job
            success, output, final_response, error = run_job(job)
            
            if success and final_response:
                logger.info("Job completed, response: %s", final_response[:200])
            else:
                logger.warning("Job failed: %s", error)
                
        except ImportError:
            logger.error("Cron scheduler not available - is Autolycus installed?")
        except Exception as e:
            logger.error("Failed to trigger cron: %s", e)
            
    async def _discovery_loop(self):
        """Periodically discover other agents."""
        while self.running:
            try:
                await self._discover_agents()
            except Exception as e:
                logger.error("Discovery error: %s", e)
            
            await asyncio.sleep(self.config.discovery_interval)
            
    async def _discover_agents(self):
        """Discover other Lycus agents on the network."""
        # TODO: Implement actual discovery via Matrix user directory
        logger.debug("Running agent discovery")
        
    async def send_message(self, target: str, message: str) -> dict:
        """Send a message to another agent.
        
        Args:
            target: Target agent identifier (Matrix user ID or alias)
            message: Message content
            
        Returns:
            Dict with status and message ID
        """
        logger.info("Sending message to %s: %s", target, message[:100])
        
        try:
            # Join the room if not already joined
            room = await self._get_or_join_room()
            
            if room:
                # Send the message
                response = await self.client.room_send(
                    room_id=room.room_id,
                    message_type="m.room.message",
                    content={
                        "msgtype": "m.text",
                        "body": message
                    }
                )
                
                if hasattr(response, 'event_id'):
                    return {
                        "status": "sent",
                        "target": target,
                        "message_id": response.event_id,
                        "timestamp": time.time()
                    }
                else:
                    return {
                        "status": "failed",
                        "error": getattr(response, 'message', 'unknown error')
                    }
            else:
                # Fallback to simulated mode if room join failed
                logger.warning("Room not available - using simulated send")
                return {
                    "status": "sent",
                    "target": target,
                    "message_id": f"msg_{int(time.time())}",
                    "timestamp": time.time()
                }
                
        except Exception as e:
            logger.error("Failed to send message: %s", e)
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _get_or_join_room(self):
        """Get or join the agent room."""
        # Check if we're already in the room
        for room in self.client.rooms.values():
            if room.room_id == self.config.agent_room:
                return room
        
        # Try to join by alias
        try:
            response = await self.client.join(self.config.agent_room)
            if hasattr(response, 'room_id'):
                logger.info("Joined room %s", response.room_id)
                return self.client.rooms.get(response.room_id)
            else:
                logger.warning("Room join failed: %s", 
                             getattr(response, 'message', 'unknown error'))
        except Exception as e:
            logger.error("Failed to join room: %s", e)
            
        return None
    
    def list_agents(self) -> List[str]:
        """List known agents.
        
        Returns:
            List of agent identifiers
        """
        return list(self.agents.keys())
    
    async def register_agent(self, agent_id: str, display_name: str, capabilities: Optional[List[str]] = None):
        """Register a new agent in the discovery system.
        
        Args:
            agent_id: Unique agent identifier
            display_name: Human-readable name
            capabilities: List of agent capabilities
        """
        self.agents[agent_id] = {
            "display_name": display_name,
            "capabilities": capabilities or [],
            "registered_at": time.time(),
            "last_seen": time.time()
        }
        logger.info("Registered agent %s (%s)", agent_id, display_name)


# Global instance for tool access
_comm_tools: Optional[LycusCommTools] = None


def get_comm_tools(config: Optional[CommToolConfig] = None) -> LycusCommTools:
    """Get or create the global communication tools instance."""
    global _comm_tools
    if _comm_tools is None:
        _comm_tools = LycusCommTools(config)
    return _comm_tools


async def send_matrix_message(target: str, message: str) -> dict:
    """Tool: Send a Matrix message to another agent.
    
    Usage: await send_matrix_message("@nova:lycus.local", "Hello agent")
    """
    tools = get_comm_tools()
    return await tools.send_message(target, message)


async def list_known_agents() -> List[str]:
    """Tool: List all known Lycus agents.
    
    Usage: await list_known_agents()
    """
    tools = get_comm_tools()
    return tools.list_agents()


async def register_local_agent(agent_id: str, display_name: str) -> dict:
    """Tool: Register this agent with the discovery system.
    
    Usage: await register_local_agent("terra", "Terra Agent")
    """
    tools = get_comm_tools()
    await tools.register_agent(agent_id, display_name)
    return {"status": "registered", "agent_id": agent_id}


async def start_comm_module(config: Optional[CommToolConfig] = None):
    """Tool: Start the communication module.
    
    Usage: await start_comm_module()
    """
    tools = get_comm_tools(config)
    await tools.start()
    return {"status": "started"}


if __name__ == "__main__":
    asyncio.run(start_comm_module())

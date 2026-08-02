"""Federation Platform Adapter — integrates federation as a native Hermes platform.

This adapter runs alongside Telegram, Discord, etc., enabling:
- Real-time peer discovery and connection
- Task submission, claiming, and progress streaming
- Cross-device collaboration without external infrastructure

Usage (config.yaml):
    federation:
      enabled: true
      mode: lan                    # shared_db | lan | auto
      device_id: auto              # or explicit device ID
      ws_port: 18765
      auth_token: "${FEDERATION_TOKEN}"
      peers:                       # for 'lan' mode
        - ws://192.168.1.10:18765
        - ws://192.168.1.11:18765
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from gateway.config import FederationConfig
from gateway.federation.federation_protocol import (
    FedMessage,
    MessageType,
    PeerInfo,
)
from gateway.federation.federation_connection import FederationConnectionManager
from gateway.federation.federation_consensus import FederationConsensus
from gateway.federation.federation_relay import TaskExecutorRelay
from gateway.federation.federation_discovery import FederationMDNS

logger = logging.getLogger(__name__)


def _resolve_device_id(configured: Optional[str]) -> str:
    """Resolve device ID from env, config, or auto-detection."""
    if configured and configured != "auto":
        return configured

    env_id = os.environ.get("HERMES_DEVICE_ID")
    if env_id:
        return env_id

    try:
        import subprocess
        name = subprocess.run(
            ["scutil", "--get", "LocalHostName"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        if name:
            return name.replace(" ", "-")
    except Exception:
        pass

    return socket.gethostname()


class FederationAdapter:
    """Federation adapter — manages peer connections and task routing.

    This is NOT a BasePlatformAdapter subclass because federation is not a
    messaging platform — it's a coordination layer.  It integrates with
    GatewayRunner via direct method calls rather than the adapter lifecycle.
    """

    def __init__(self, config: FederationConfig):
        self.config = config
        self.device_id = _resolve_device_id(config.device_id)
        self._conn_manager: Optional[FederationConnectionManager] = None
        self._running = False
        self._task_handlers: Dict[str, Callable] = {}
        self._task_state: Dict[str, dict] = {}  # task_id -> state

        # Phase 3: Consensus + Relay
        self._consensus: Optional[FederationConsensus] = None
        self._relay: Optional[TaskExecutorRelay] = None
        # Phase 5: mDNS Discovery
        self._mdns: Optional[FederationMDNS] = None

        # Register default task handlers
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default message type handlers."""
        self._task_handlers[MessageType.TASK_SUBMIT.value] = self._handle_task_submit
        self._task_handlers[MessageType.TASK_CLAIM.value] = self._handle_task_claim
        self._task_handlers[MessageType.TASK_PROGRESS.value] = self._handle_task_progress
        self._task_handlers[MessageType.TASK_RESULT.value] = self._handle_task_result
        self._task_handlers[MessageType.TASK_HEARTBEAT.value] = self._handle_task_heartbeat
        self._task_handlers[MessageType.TASK_CLAIM_ACK.value] = self._handle_claim_ack
        self._task_handlers[MessageType.TASK_CLAIM_NACK.value] = self._handle_claim_nack
        self._task_handlers[MessageType.PEER_JOIN.value] = self._handle_peer_join
        self._task_handlers[MessageType.PEER_LEAVE.value] = self._handle_peer_leave
        self._task_handlers[MessageType.PEER_PING.value] = self._handle_peer_ping
        self._task_handlers[MessageType.PEER_PONG.value] = self._handle_peer_pong

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    async def start(self) -> None:
        """Start the federation adapter."""
        if self.config.mode == "shared_db":
            logger.info(
                "Federation: shared_db mode (v1) — no real-time connections. "
                "Use mode 'lan' for WebSocket-based federation."
            )
            self._running = True
            return

        logger.info(
            "Federation: starting %s mode (device=%s, port=%d)",
            self.config.mode, self.device_id, self.config.ws_port,
        )

        self._conn_manager = FederationConnectionManager(
            device_id=self.device_id,
            auth_token=self.config.auth_token,
            ws_port=self.config.ws_port,
            on_message=self._on_message,
            on_peer_join=self._on_peer_join,
            on_peer_leave=self._on_peer_leave,
        )

        # Register configured peers
        for peer_url in self.config.peers:
            # Extract device_id from URL or use a placeholder
            # In v2.1, we'll do proper handshake to get device_id
            info = PeerInfo(
                device_id=f"pending:{peer_url}",
                hostname=peer_url,
                ws_url=peer_url,
            )
            self._conn_manager.register_peer(info)

        await self._conn_manager.start(listen=(self.config.mode != "shared_db"))

        # Phase 3: Initialize consensus + relay
        peer_count = self._conn_manager.get_online_count() + 1  # +1 for self
        self._consensus = FederationConsensus(
            device_id=self.device_id,
            total_peers=peer_count,
            vote_timeout=5.0,
        )
        self._relay = TaskExecutorRelay(
            device_id=self.device_id,
            adapter=self,
            consensus=self._consensus,
            progress_interval=10.0,
            checkpoint_interval=30.0,
        )
        await self._relay.start()

        # Phase 5: Start mDNS discovery for auto mode
        if self.config.mode == "auto":
            self._mdns = FederationMDNS(
                device_id=self.device_id,
                ws_port=self.config.ws_port,
                on_discover=self._on_mdns_discover,
                on_forget=self._on_mdns_forget,
            )
            await self._mdns.start()

        # Announce ourselves
        await self._conn_manager.send(
            FedMessage.peer_join(
                self.device_id,
                PeerInfo(
                    device_id=self.device_id,
                    hostname=socket.gethostname(),
                    ws_url=f"ws://{self._get_local_ip()}:{self.config.ws_port}",
                    cpu_cores=os.cpu_count() or 0,
                ),
            )
        )

        self._running = True
        logger.info("Federation: adapter started (device=%s)", self.device_id)

    async def stop(self) -> None:
        """Stop the federation adapter."""
        self._running = False
        if self._mdns:
            await self._mdns.stop()
        if self._relay:
            await self._relay.stop()
        if self._conn_manager:
            await self._conn_manager.stop()
        logger.info("Federation: adapter stopped")

    # ----------------------------------------------------------------
    # Task management API
    # ----------------------------------------------------------------

    async def submit_task(
        self,
        task_id: str,
        title: str,
        description: str = "",
        priority: int = 3,
        context_snapshot: Optional[dict] = None,
    ) -> bool:
        """Submit a task to the federation for execution."""
        if not self._conn_manager:
            logger.warning("Federation: not connected, cannot submit task")
            return False

        msg = FedMessage.task_submit(
            self.device_id, task_id, title, description, priority, context_snapshot,
        )
        ok = await self._conn_manager.send(msg)
        if ok:
            self._task_state[task_id] = {
                "status": "submitted",
                "title": title,
                "submitted_at": time.time(),
            }
        return ok

    async def claim_task(self, task_id: str) -> bool:
        """Claim a task for local execution."""
        if not self._conn_manager:
            return False

        msg = FedMessage.task_claim(self.device_id, task_id)
        ok = await self._conn_manager.send(msg)
        if ok:
            self._task_state[task_id] = {
                "status": "claimed",
                "claimed_by": self.device_id,
                "claimed_at": time.time(),
            }
        return ok

    async def report_progress(
        self, task_id: str, progress: float, note: str = ""
    ) -> bool:
        """Report task progress to all peers."""
        if not self._conn_manager:
            return False

        msg = FedMessage.task_progress(self.device_id, task_id, progress, note)
        return await self._conn_manager.send(msg)

    async def report_result(
        self,
        task_id: str,
        success: bool,
        result_data: Optional[dict] = None,
        error_info: str = "",
    ) -> bool:
        """Report task completion to all peers."""
        if not self._conn_manager:
            return False

        msg = FedMessage.task_result(
            self.device_id, task_id, success, result_data, error_info,
        )
        ok = await self._conn_manager.send(msg)
        if ok:
            self._task_state[task_id] = {
                "status": "completed" if success else "failed",
                "result_data": result_data or {},
                "error_info": error_info,
            }
        return ok

    async def send_task_heartbeat(self, task_id: str) -> bool:
        """Send a task heartbeat — I'm still executing this task."""
        if not self._conn_manager:
            return False

        msg = FedMessage.task_heartbeat(self.device_id, task_id)
        return await self._conn_manager.send(msg)

    # ----------------------------------------------------------------
    # State queries
    # ----------------------------------------------------------------

    def get_peers(self) -> List[PeerInfo]:
        """Get list of connected peers."""
        if not self._conn_manager:
            return []
        return self._conn_manager.get_peers()

    def get_peer_count(self) -> int:
        """Get number of connected peers."""
        if not self._conn_manager:
            return 0
        return self._conn_manager.get_online_count()

    def get_task_state(self, task_id: str) -> Optional[dict]:
        """Get local state for a task."""
        return self._task_state.get(task_id)

    def get_all_task_states(self) -> Dict[str, dict]:
        """Get all task states."""
        return dict(self._task_state)

    def is_connected(self) -> bool:
        """Check if federation is connected."""
        return self._running and self.get_peer_count() > 0

    # ----------------------------------------------------------------
    # Message handlers
    # ----------------------------------------------------------------

    def _on_message(self, msg: FedMessage) -> None:
        """Route incoming message to appropriate handler."""
        handler = self._task_handlers.get(msg.msg_type)
        if handler:
            try:
                handler(msg)
            except Exception as e:
                logger.error(
                    "Federation: handler error for %s: %s", msg.msg_type, e,
                )
        else:
            logger.debug("Federation: no handler for %s", msg.msg_type)

    def _on_peer_join(self, info: PeerInfo) -> None:
        """Called when a new peer joins."""
        logger.info(
            "Federation: peer joined — %s (%s), score=%.1f",
            info.device_id, info.hostname, info.compute_score,
        )

    def _on_peer_leave(self, device_id: str) -> None:
        """Called when a peer leaves."""
        logger.info("Federation: peer left — %s", device_id)

    def _on_mdns_discover(self, peer) -> None:
        """Handle mDNS discovery of a new peer."""
        logger.info(
            "Federation: mDNS discovered %s (%s) at %s",
            peer.device_id, peer.hostname, peer.ws_url,
        )
        # Register with connection manager
        if self._conn_manager:
            from gateway.federation.federation_protocol import PeerInfo
            info = PeerInfo(
                device_id=peer.device_id,
                hostname=peer.hostname,
                ws_url=peer.ws_url,
            )
            self._conn_manager.register_peer(info)

    def _on_mdns_forget(self, device_id: str) -> None:
        """Handle mDNS peer forget — peer has gone silent."""
        logger.info("Federation: mDNS forgot peer — %s", device_id)
        if self._conn_manager:
            self._conn_manager.unregister_peer(device_id, reason="mdns timeout")

    def _handle_task_submit(self, msg: FedMessage) -> None:
        """Handle incoming task submission."""
        payload = msg.payload
        task_id = payload.get("task_id", "")
        title = payload.get("title", "")

        self._task_state[task_id] = {
            "status": "pending",
            "title": title,
            "description": payload.get("description", ""),
            "priority": payload.get("priority", 3),
            "submitted_by": msg.sender_id,
            "submitted_at": msg.timestamp,
            "context_snapshot": payload.get("context_snapshot", {}),
        }

        logger.info(
            "Federation: task submitted — #%s: %s (by %s)",
            task_id, title, msg.sender_id,
        )

    def _handle_task_claim(self, msg: FedMessage) -> None:
        """Handle task claim from a peer."""
        task_id = msg.payload.get("task_id", "")
        claimer = msg.sender_id

        state = self._task_state.get(task_id, {})
        if state.get("status") == "pending":
            state["status"] = "claimed"
            state["claimed_by"] = claimer
            state["claimed_at"] = msg.timestamp
            logger.info(
                "Federation: task claimed — #%s by %s", task_id, claimer,
            )

    def _handle_task_progress(self, msg: FedMessage) -> None:
        """Handle task progress update."""
        task_id = msg.payload.get("task_id", "")
        progress = msg.payload.get("progress", 0)
        note = msg.payload.get("note", "")

        state = self._task_state.get(task_id, {})
        state["status"] = "in_progress"
        state["progress"] = progress
        state["progress_note"] = note
        state["progress_at"] = msg.timestamp

        logger.debug(
            "Federation: task progress — #%s: %.0f%% (%s)",
            task_id, progress * 100, note,
        )

    def _handle_task_result(self, msg: FedMessage) -> None:
        """Handle task completion."""
        task_id = msg.payload.get("task_id", "")
        success = msg.payload.get("success", False)

        state = self._task_state.get(task_id, {})
        state["status"] = "completed" if success else "failed"
        state["result_data"] = msg.payload.get("result_data", {})
        state["error_info"] = msg.payload.get("error_info", "")
        state["completed_at"] = msg.timestamp

        logger.info(
            "Federation: task %s — #%s",
            "completed" if success else "failed", task_id,
        )

    def _handle_task_heartbeat(self, msg: FedMessage) -> None:
        """Handle task heartbeat — executor still alive."""
        task_id = msg.payload.get("task_id", "")
        state = self._task_state.get(task_id, {})
        state["executor_heartbeat_at"] = msg.timestamp
        logger.debug(
            "Federation: task heartbeat — #%s (by %s)",
            task_id, msg.sender_id,
        )

    def _handle_peer_join(self, msg: FedMessage) -> None:
        """Handle peer join announcement."""
        peer_data = msg.payload.get("peer_info", {})
        if peer_data.get("device_id") == self.device_id:
            return  # Ignore self

        if self._conn_manager:
            info = PeerInfo(**peer_data)
            self._conn_manager.register_peer(info)

    def _handle_peer_leave(self, msg: FedMessage) -> None:
        """Handle peer leave announcement."""
        device_id = msg.sender_id
        if self._conn_manager:
            self._conn_manager.unregister_peer(
                device_id, msg.payload.get("reason", "offline"),
            )

    def _handle_claim_ack(self, msg: FedMessage) -> None:
        """Handle task claim ACK vote."""
        if self._consensus:
            self._consensus.handle_vote_response(msg)

    def _handle_claim_nack(self, msg: FedMessage) -> None:
        """Handle task claim NACK vote."""
        if self._consensus:
            self._consensus.handle_vote_response(msg)

    def _handle_peer_ping(self, msg: FedMessage) -> None:
        """Handle peer liveness probe — respond with pong."""
        pong = FedMessage(
            msg_type=MessageType.PEER_PONG.value,
            sender_id=self.device_id,
            target_id=msg.sender_id,
            payload={"timestamp": time.time()},
        )
        if self._conn_manager:
            asyncio.create_task(self._conn_manager.send(pong))

    def _handle_peer_pong(self, msg: FedMessage) -> None:
        """Handle peer liveness response — update last_seen."""
        device_id = msg.sender_id
        info = self._conn_manager.get_peer(device_id) if self._conn_manager else None
        if info:
            info.last_seen = msg.payload.get("timestamp", time.time())

    # ----------------------------------------------------------------
    # Phase 3: Claim & Execute (consensus + relay)
    # ----------------------------------------------------------------

    async def claim_and_execute(
        self,
        task_id: str,
        title: str = "",
        description: str = "",
        priority: int = 3,
        context_snapshot: Optional[dict] = None,
        handler: Optional[Callable] = None,
    ) -> Optional[dict]:
        """Claim a task via consensus and execute it locally.

        This is the core relay mechanism — combines:
        1. Raft-lite consensus claim
        2. Task execution with progress streaming
        3. Checkpoint/relay support

        Returns the task result dict, or None if claim failed.
        """
        if not self._relay:
            logger.warning("Federation: relay not initialized")
            return None

        state = await self._relay.claim_and_execute(
            task_id=task_id,
            title=title,
            description=description,
            priority=priority,
            context_snapshot=context_snapshot,
            handler=handler,
        )

        if state.status == "completed":
            return state.result_data
        else:
            logger.warning(
                "Federation: task %s ended with status=%s: %s",
                task_id, state.status, state.error_info,
            )
            return None

    async def handoff_task(
        self,
        task_id: str,
        reason: str = "device going offline",
    ) -> bool:
        """Hand off a running task to another device.

        Use this when going offline or under resource pressure.
        The task will be re-broadcasted for another peer to claim.
        """
        if not self._relay:
            return False

        state = await self._relay.handoff_task(task_id, reason)
        return state is not None

    def get_relay(self) -> Optional[TaskExecutorRelay]:
        """Get the task executor relay instance."""
        return self._relay

    def get_consensus(self) -> Optional[FederationConsensus]:
        """Get the consensus instance."""
        return self._consensus

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _get_local_ip(self) -> str:
        """Get local IP address for advertising."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"


def create_federation_adapter(
    enabled: bool = False,
    mode: str = "shared_db",
    device_id: Optional[str] = None,
    ws_port: int = 18765,
    auth_token: Optional[str] = None,
    peers: Optional[List[str]] = None,
    db_path: Optional[str] = None,
    offline_threshold_s: int = 30,
    heartbeat_interval_s: int = 60,
) -> FederationAdapter:
    """Factory function to create a federation adapter from config values."""
    config = FederationConfig(
        enabled=enabled,
        mode=mode,
        device_id=device_id,
        ws_port=ws_port,
        auth_token=auth_token,
        peers=peers or [],
        db_path=db_path,
        offline_threshold_s=offline_threshold_s,
        heartbeat_interval_s=heartbeat_interval_s,
    )
    return FederationAdapter(config)

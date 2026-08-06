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
from gateway.federation.federation_collaboration import FederationMemorySync, FederationDistributedSearch
from gateway.federation.federation_compute_pool import FederationComputePool
from gateway.federation.federation_ops import SEV_CRITICAL
from gateway.federation.federation_cron_relay import FederationCronRelay, FederationSkillSync
from gateway.federation.federation_cluster import FederationLeaderElection, FederationConfigSync
from gateway.federation.federation_api import FederationAPI, FederationAPIConfig

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
        # Phase 7: Collaboration
        self._memory_sync: Optional[FederationMemorySync] = None
        self._distributed_search: Optional[FederationDistributedSearch] = None
        # Phase 8: Compute Pool
        self._compute_pool: Optional[FederationComputePool] = None
        # Phase 9: Cron Relay + Skill Sync
        self._cron_relay: Optional[FederationCronRelay] = None
        self._skill_sync: Optional[FederationSkillSync] = None
        # Phase 10: Cluster Management
        self._leader_election: Optional[FederationLeaderElection] = None
        self._config_sync: Optional[FederationConfigSync] = None
        self._api: Optional[FederationAPI] = None

        # Phase 22: Ops layer (health monitoring + lost-contact SOS)
        from gateway.federation.federation_ops import HealthMonitor, LostContactSOS
        self._health = HealthMonitor(device_id=self.device_id,
                                     offline_threshold_s=config.offline_threshold_s)
        self._sos = LostContactSOS(
            device_id=self.device_id,
            health=self._health,
            on_alert=self._on_ops_alert,
        )

        # Register default message handlers
        self._register_default_handlers()

    def _on_ops_alert(self, alert) -> None:
        """Broadcast an ops alert to all peers + log to audit."""
        try:
            msg = FedMessage(
                msg_type=MessageType.OPS_ALERT.value,
                sender_id=self.device_id,
                payload={
                    "alert_id": alert.alert_id,
                    "severity": alert.severity,
                    "source": alert.source_device,
                    "target": alert.target_device,
                    "type": alert.alert_type,
                    "message": alert.message,
                    "created_at": alert.created_at,
                },
            )
            if self._conn_manager:
                asyncio.create_task(self._conn_manager.send(msg))
        except Exception as e:
            logger.debug("Ops alert broadcast failed: %s", e)

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
        self._task_handlers[MessageType.MEMORY_SYNC.value] = self._handle_memory_sync
        self._task_handlers[MessageType.SEARCH_QUERY.value] = self._handle_search_query
        self._task_handlers[MessageType.SEARCH_RESULT.value] = self._handle_search_result
        self._task_handlers[MessageType.COMPUTE_REQUEST.value] = self._handle_compute_request
        self._task_handlers[MessageType.COMPUTE_RESPONSE.value] = self._handle_compute_response
        self._task_handlers[MessageType.CRON_SYNC.value] = self._handle_cron_sync
        self._task_handlers[MessageType.SKILL_SYNC.value] = self._handle_skill_sync
        self._task_handlers[MessageType.ELECTION.value] = self._handle_election
        self._task_handlers[MessageType.ELECTION_OK.value] = self._handle_election_ok
        # Phase 22: Ops layer handlers
        self._task_handlers[MessageType.OPS_HEALTH.value] = self._handle_ops_health
        self._task_handlers[MessageType.OPS_ALERT.value] = self._handle_ops_alert
        self._task_handlers[MessageType.OPS_SOS.value] = self._handle_ops_sos
        self._task_handlers[MessageType.VICTORY.value] = self._handle_victory
        self._task_handlers[MessageType.COORDINATE.value] = self._handle_coordinate
        self._task_handlers[MessageType.CONFIG_SYNC.value] = self._handle_config_sync

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
            tls_cert=self.config.tls_cert,
            tls_key=self.config.tls_key,
            ip_whitelist=self.config.ip_whitelist,
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

        # Phase 7: Start collaboration layer
        self._memory_sync = FederationMemorySync(
            device_id=self.device_id,
            adapter=self,
        )
        await self._memory_sync.start()

        self._distributed_search = FederationDistributedSearch(
            device_id=self.device_id,
            adapter=self,
        )
        await self._distributed_search.start()

        # Phase 8: Start compute pool
        self._compute_pool = FederationComputePool(
            device_id=self.device_id,
            adapter=self,
        )
        await self._compute_pool.start()

        # Phase 9: Start cron relay + skill sync
        self._cron_relay = FederationCronRelay(
            device_id=self.device_id,
            adapter=self,
        )
        await self._cron_relay.start()

        self._skill_sync = FederationSkillSync(
            device_id=self.device_id,
            adapter=self,
        )
        await self._skill_sync.start()

        # Phase 10: Start cluster management
        self._leader_election = FederationLeaderElection(
            device_id=self.device_id,
            adapter=self,
            compute_score=self._compute_pool.compute_score() if self._compute_pool else 0.0,
        )
        await self._leader_election.start()

        self._config_sync = FederationConfigSync(
            device_id=self.device_id,
            adapter=self,
        )
        await self._config_sync.start()

        # Phase 11: Start HTTP API
        self._api = FederationAPI(
            adapter=self,
            config=FederationAPIConfig(
                enabled=True,
                port=getattr(self.config, 'api_port', 18766),
            ),
            hermes_version=self._get_hermes_version(),
        )
        await self._api.start()

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
        if self._memory_sync:
            await self._memory_sync.stop()
        if self._distributed_search:
            await self._distributed_search.stop()
        if self._compute_pool:
            await self._compute_pool.stop()
        if self._cron_relay:
            await self._cron_relay.stop()
        if self._skill_sync:
            await self._skill_sync.stop()
        if self._leader_election:
            await self._leader_election.stop()
        if self._config_sync:
            await self._config_sync.stop()
        if self._api:
            await self._api.stop()
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

    async def send(self, msg: Any) -> bool:
        """Send a message to all peers (delegated to connection manager).

        Public API used by cluster / collaboration / relay components
        (e.g. FederationLeaderElection broadcasts ELECTION messages).
        """
        if not self._conn_manager:
            logger.warning("Federation: not connected, cannot send message")
            return False
        return await self._conn_manager.send(msg)

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
        # Ops layer: seed health snapshot from join metadata
        self._health.update_from_heartbeat(info.device_id, {
            "hostname": info.hostname,
            "cpu_cores": info.cpu_cores,
            "memory_gb": info.memory_gb,
            "gateway_up": True,
            "federation_connected": True,
        })
        self._sos.reset(info.device_id)

    def _on_peer_leave(self, device_id: str) -> None:
        """Called when a peer leaves."""
        logger.info("Federation: peer left — %s", device_id)
        self._health.mark_offline(device_id, reason="peer_leave")

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
                cpu_cores=peer.cpu_cores,
                memory_gb=peer.memory_gb,
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
    # Phase 22: Ops layer handlers
    # ----------------------------------------------------------------

    def _handle_ops_health(self, msg: FedMessage) -> None:
        """Handle an OPS_HEALTH snapshot from a peer."""
        payload = msg.payload
        self._health.update_from_heartbeat(msg.sender_id, payload)
        self._sos.reset(msg.sender_id)

    def _handle_ops_alert(self, msg: FedMessage) -> None:
        """Handle an OPS_ALERT from a peer — log + keep in local alert buffer."""
        payload = msg.payload
        logger.warning(
            "Federation OPS alert from %s [%s] -> %s: %s",
            msg.sender_id, payload.get("severity"), payload.get("target"),
            payload.get("message"),
        )
        # Mirror remote alert into local history
        self._health._emit_alert(
            severity=payload.get("severity", "info"),
            source_device=msg.sender_id,
            target_device=payload.get("target", ""),
            message=payload.get("message", ""),
            alert_type=payload.get("type", "ops"),
        )

    def _handle_ops_sos(self, msg: FedMessage) -> None:
        """Handle an SOS call from a peer — the operator / other peers act."""
        payload = msg.payload
        logger.critical(
            "Federation SOS from %s: %s",
            msg.sender_id, payload.get("message", "peer requesting assistance"),
        )
        self._health._emit_alert(
            severity=SEV_CRITICAL,
            source_device=msg.sender_id,
            target_device=payload.get("target", ""),
            message=payload.get("message", "SOS received"),
            alert_type="lost_contact",
        )
        # Acknowledge the SOS
        ack = FedMessage(
            msg_type=MessageType.OPS_ASSIST_ACK.value,
            sender_id=self.device_id,
            target_id=msg.sender_id,
            payload={"ack": True, "assist_offer": True},
        )
        if self._conn_manager:
            asyncio.create_task(self._conn_manager.send(ack))

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


    def _handle_memory_sync(self, msg: FedMessage) -> None:
        """Handle MEMORY_SYNC message from peer."""
        if self._memory_sync:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._memory_sync.handle_memory_sync(msg))
            except RuntimeError:
                pass

    def _handle_search_query(self, msg: FedMessage) -> None:
        """Handle SEARCH_QUERY from peer."""
        if self._distributed_search:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._distributed_search.handle_search_query(msg))
            except RuntimeError:
                pass

    def _handle_search_result(self, msg: FedMessage) -> None:
        """Handle SEARCH_RESULT from peer."""
        if self._distributed_search:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._distributed_search.handle_search_result(msg))
            except RuntimeError:
                pass

    async def sync_memory(self, node_id: str, content: str, target: str = "memory") -> bool:
        """Sync a memory entry to all peers."""
        if not self._memory_sync:
            return False
        await self._memory_sync.on_local_memory_change(node_id, content, target)
        return True

    async def federated_search(
        self, query: str, limit: int = 10, sort: str = "newest",
        profile: Optional[str] = None,
    ) -> list:
        """Search across all federation peers."""
        if not self._distributed_search:
            return []
        results = await self._distributed_search.search(query, limit, sort, profile)
        return [
            {
                "device_id": r.device_id,
                "session_id": r.session_id,
                "title": r.session_title,
                "snippet": r.snippet,
                "score": r.score,
            }
            for r in results
        ]

    def get_memory_sync(self):
        """Get memory sync instance."""
        return self._memory_sync

    def get_distributed_search(self):
        """Get distributed search instance."""
        return self._distributed_search



    def _handle_compute_request(self, msg: FedMessage) -> None:
        """Handle COMPUTE_REQUEST from peer."""
        if self._compute_pool:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._compute_pool.handle_compute_request(msg))
            except RuntimeError:
                pass

    def _handle_compute_response(self, msg: FedMessage) -> None:
        """Handle COMPUTE_RESPONSE from peer."""
        if self._compute_pool:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._compute_pool.handle_compute_response(msg))
            except RuntimeError:
                pass

    async def distribute_compute(
        self,
        task_type: str,
        items: list,
        chunk_size: int = 10,
        handler_name: Optional[str] = None,
    ) -> dict:
        """Distribute computation across federation peers."""
        if not self._compute_pool:
            return {"error": "compute pool not initialized"}
        result = await self._compute_pool.distribute(task_type, items, chunk_size, handler_name)
        return {
            "task_id": result.task_id,
            "total_chunks": result.total_chunks,
            "successful_chunks": result.successful_chunks,
            "failed_chunks": result.failed_chunks,
            "success_rate": result.success_rate,
            "duration_ms": result.duration_ms,
            "data": result.aggregated_data,
        }

    def register_compute_handler(self, name: str, handler: Callable) -> None:
        """Register a compute handler."""
        if self._compute_pool:
            self._compute_pool.register_handler(name, handler)

    def get_compute_pool(self):
        """Get compute pool instance."""
        return self._compute_pool



    def _handle_cron_sync(self, msg: FedMessage) -> None:
        """Handle CRON_SYNC from peer."""
        if self._cron_relay:
            self._cron_relay.handle_cron_sync(msg)

    def _handle_skill_sync(self, msg: FedMessage) -> None:
        """Handle SKILL_SYNC from peer."""
        if self._skill_sync:
            self._skill_sync.handle_skill_sync(msg)

    async def sync_cron_job(self, job_id: str, name: str, schedule: str) -> bool:
        """Sync a cron job to all peers."""
        if not self._cron_relay:
            return False
        from gateway.federation.federation_cron_relay import CronJobInfo
        job = CronJobInfo(
            job_id=job_id, name=name, schedule=schedule,
            leader_device=self.device_id,
        )
        await self._cron_relay.sync_job(job)
        return True

    async def release_cron_job(self, job_id: str) -> bool:
        """Release leadership of a cron job."""
        if not self._cron_relay:
            return False
        await self._cron_relay.release_leadership(job_id)
        return True

    def get_cron_jobs(self) -> list:
        """Get all known cron jobs."""
        if not self._cron_relay:
            return []
        return self._cron_relay.get_all_jobs()

    async def sync_skill(self, name: str, content: str, category: str = "") -> bool:
        """Sync a skill to all peers."""
        if not self._skill_sync:
            return False
        await self._skill_sync.sync_skill(name, content, category)
        return True

    def get_skill_sync(self):
        """Get skill sync instance."""
        return self._skill_sync



    def _handle_election(self, msg: FedMessage) -> None:
        """Handle ELECTION from peer."""
        if self._leader_election:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._leader_election.handle_election(msg))
            except RuntimeError:
                pass

    def _handle_election_ok(self, msg: FedMessage) -> None:
        """Handle ELECTION_OK from peer."""
        if self._leader_election:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._leader_election.handle_election_ok(msg))
            except RuntimeError:
                pass

    def _handle_victory(self, msg: FedMessage) -> None:
        """Handle VICTORY from new leader."""
        if self._leader_election:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._leader_election.handle_victory(msg))
            except RuntimeError:
                pass

    def _handle_coordinate(self, msg: FedMessage) -> None:
        """Handle COORDINATE heartbeat from leader."""
        if self._leader_election:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._leader_election.handle_coordinate(msg))
            except RuntimeError:
                pass

    def _handle_config_sync(self, msg: FedMessage) -> None:
        """Handle CONFIG_SYNC from leader."""
        if self._config_sync:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._config_sync.handle_config_sync(msg))
            except RuntimeError:
                pass

    async def initiate_leader_election(self) -> str:
        """Start a leader election."""
        if not self._leader_election:
            return ""
        return await self._leader_election.initiate_election()

    def is_leader(self) -> bool:
        """Check if this device is the cluster leader."""
        return self._leader_election.is_leader() if self._leader_election else False

    def get_leader(self) -> str:
        """Get the current cluster leader."""
        return self._leader_election.get_leader() if self._leader_election else ""

    async def sync_config_to_peers(self) -> bool:
        """Sync local config to all federation peers."""
        if not self._config_sync:
            return False
        return await self._config_sync.sync_config()

    def get_leader_election(self):
        """Get leader election instance."""
        return self._leader_election

    def get_api(self):
        """Get federation API instance."""
        return self._api

    def get_peers_dict(self) -> dict:
        """Get peers dictionary for API responses."""
        return getattr(self, '_peers', {})

    def get_relay(self):
        """Get task relay instance for API."""
        return self._relay

    def get_config_sync(self):
        """Get config sync instance."""
        return self._config_sync


    def _get_hermes_version(self) -> str:
        """Get current Hermes version."""
        try:
            import importlib.metadata
            return importlib.metadata.version("hermes-agent")
        except Exception:
            return "unknown"

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
    require_auth: bool = True,
    tls_cert: Optional[str] = None,
    tls_key: Optional[str] = None,
    ip_whitelist: Optional[List[str]] = None,
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
        require_auth=require_auth,
        tls_cert=tls_cert,
        tls_key=tls_key,
        ip_whitelist=ip_whitelist or [],
        peers=peers or [],
        db_path=db_path,
        offline_threshold_s=offline_threshold_s,
        heartbeat_interval_s=heartbeat_interval_s,
    )
    return FederationAdapter(config)

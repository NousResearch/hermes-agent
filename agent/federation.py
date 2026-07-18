# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------------
# Portions adapted from Ruflo federation coordinator protocol
#   Upstream: https://github.com/ruvnet/ruflo (MIT)
#   Ref: plugins/ruflo-federation/agents/federation-coordinator.md
# ---------------------------------------------------------------------------
"""Federation protocol for cross-machine agent communication.

Enables Hermes instances on different machines to discover each other,
exchange tasks, and delegate work across boundaries.

Architecture:
    Peer A  --(HTTP/JSON-RPC)-->  Peer B
        |                             |
    agent/federation.py        agent/federation.py
        |                             |
    tools/federation_tool.py    tools/federation_tool.py

Peer discovery: config-driven peer list in config.yaml under federation.peers.
Each peer has a name, url, and optional shared secret for auth.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


# ═══ Data Models ═══

@dataclass
class FederationPeer:
    """A remote Hermes instance reachable via federation."""
    name: str
    url: str                          # e.g. "http://10.0.0.5:8642"
    shared_secret: str = ""           # HMAC signing key for mutual auth
    enabled: bool = True
    tags: List[str] = field(default_factory=list)  # e.g. ["gpu-heavy", "code-review"]
    last_seen: float = 0.0
    status: str = "unknown"           # online / offline / timeout


@dataclass
class FederationTask:
    """A task delegated to a remote peer."""
    peer_name: str
    goal: str
    context: str = ""
    skills: List[str] = field(default_factory=list)
    timeout_seconds: int = 300


@dataclass
class FederationResult:
    """Result returned from a remote peer."""
    peer_name: str
    task_id: str
    success: bool
    result: str
    error: str = ""
    elapsed_ms: float = 0.0


# ═══ Federation Manager ═══

class FederationManager:
    """Singleton manager for cross-machine agent federation.

    Usage:
        fm = FederationManager.from_config(config_dict)
        result = await fm.delegate(FederationTask(
            peer_name="hermes-gpu-1",
            goal="Run the full test suite and report failures",
        ))
    """

    def __init__(self, peers: List[FederationPeer] = None):
        self._peers: Dict[str, FederationPeer] = {}
        self._client: Optional[httpx.AsyncClient] = None
        if peers:
            for p in peers:
                self._peers[p.name] = p

    @classmethod
    def from_config(cls, federation_config: dict) -> FederationManager:
        """Create manager from config.yaml federation section."""
        peers = []
        for p in (federation_config.get("peers") or []):
            peers.append(FederationPeer(
                name=p.get("name", ""),
                url=p.get("url", ""),
                shared_secret=p.get("shared_secret", ""),
                enabled=p.get("enabled", True),
                tags=p.get("tags", []),
            ))
        return cls(peers=peers)

    @property
    def peers(self) -> List[FederationPeer]:
        return list(self._peers.values())

    def get_peer(self, name: str) -> Optional[FederationPeer]:
        return self._peers.get(name)

    def list_peers(self, tag_filter: str = None) -> List[FederationPeer]:
        """List peers, optionally filtered by tag."""
        result = [p for p in self._peers.values() if p.enabled]
        if tag_filter:
            result = [p for p in result if tag_filter in p.tags]
        return result

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        return self._client

    async def discover(self) -> Dict[str, str]:
        """Probe all configured peers, return status map."""
        client = await self._get_client()
        statuses = {}
        for peer in self._peers.values():
            if not peer.enabled:
                statuses[peer.name] = "disabled"
                continue
            try:
                r = await client.get(f"{peer.url}/federation/ping", headers=self._auth_headers(peer))
                if r.status_code == 200:
                    peer.status = "online"
                    peer.last_seen = time.time()
                else:
                    peer.status = f"http_{r.status_code}"
            except Exception as e:
                peer.status = "timeout" if "timeout" in str(e).lower() else "offline"
            statuses[peer.name] = peer.status
        return statuses

    async def delegate(self, task: FederationTask) -> FederationResult:
        """Send a task to a remote peer and wait for result."""
        peer = self._peers.get(task.peer_name)
        if not peer:
            return FederationResult(
                peer_name=task.peer_name, task_id="",
                success=False, error=f"Unknown peer: {task.peer_name}",
            )
        if not peer.enabled:
            return FederationResult(
                peer_name=task.peer_name, task_id="",
                success=False, error=f"Peer disabled: {task.peer_name}",
            )

        t0 = time.time()
        client = await self._get_client()
        payload = {
            "goal": task.goal,
            "context": task.context,
            "skills": task.skills,
            "timeout": task.timeout_seconds,
        }
        headers = self._auth_headers(peer)
        headers["Content-Type"] = "application/json"
        headers["X-Federation-Task"] = "1"

        try:
            r = await client.post(
                f"{peer.url}/federation/task",
                json=payload,
                headers=headers,
                timeout=task.timeout_seconds,
            )
            data = r.json()
            return FederationResult(
                peer_name=task.peer_name,
                task_id=data.get("task_id", ""),
                success=data.get("success", r.status_code == 200),
                result=data.get("result", ""),
                error=data.get("error", ""),
                elapsed_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            peer.status = "timeout"
            return FederationResult(
                peer_name=task.peer_name, task_id="",
                success=False, error=str(e),
                elapsed_ms=(time.time() - t0) * 1000,
            )

    def _auth_headers(self, peer: FederationPeer) -> dict:
        """Build HMAC-signed auth headers for mutual authentication."""
        headers = {"X-Federation-Peer": peer.name}
        if peer.shared_secret:
            nonce = str(int(time.time()))
            sig = hmac.new(
                peer.shared_secret.encode(),
                f"{peer.name}:{nonce}".encode(),
                hashlib.sha256,
            ).hexdigest()
            headers["X-Federation-Nonce"] = nonce
            headers["X-Federation-Signature"] = sig
        return headers

    async def shutdown(self):
        if self._client:
            await self._client.aclose()
            self._client = None


# ═══ Global singleton ═══

_federation_manager: Optional[FederationManager] = None


def get_federation_manager() -> Optional[FederationManager]:
    return _federation_manager


def init_federation(config: dict) -> FederationManager:
    """Initialize federation from config. Called at agent startup."""
    global _federation_manager
    fed_config = config.get("federation", {})
    if not fed_config.get("enabled", False):
        logger.debug("Federation disabled in config")
        _federation_manager = None
        return None
    _federation_manager = FederationManager.from_config(fed_config)
    logger.info("Federation initialized with %d peers", len(_federation_manager.peers))
    return _federation_manager
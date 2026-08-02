"""Federation Gateway API — HTTP endpoints for federation management.

Exposes federation state, peers, tasks, and health metrics via HTTP API.
Enables:
- External monitoring (Prometheus, Grafana, custom dashboards)
- CLI `hermes fed status` without direct WebSocket access
- Desktop UI bridge (Phase 13) for live device management
- Third-party integrations (CI/CD, orchestration tools)

Design principles:
- Read-only by default (safe for monitoring)
- Admin endpoints require federation auth token
- JSON responses with consistent schema
- CORS enabled for Desktop app (Electron localhost)

Endpoints:
    GET  /api/federation/status       — Overall federation health
    GET  /api/federation/peers        — Connected peers list
    GET  /api/federation/peers/{id}   — Single peer details
    GET  /api/federation/tasks        — Active/completed tasks
    GET  /api/federation/tasks/{id}   — Single task details
    GET  /api/federation/leader       — Current leader info
    POST /api/federation/handoff      — Submit task to federation
    POST /api/federation/config/sync  — Trigger config sync (admin)
    GET  /api/federation/health       — Health check (ping)
    GET  /api/federation/metrics      — Prometheus-compatible metrics

Usage:
    # In config.yaml
    federation:
      enabled: true
      api_port: 18766          # HTTP API port (separate from WebSocket)
      api_cors_origins:        # For Desktop app
        - "http://localhost:5173"
        - "app://-"
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FederationAPIConfig:
    """Configuration for the federation HTTP API."""

    enabled: bool = True
    port: int = 18766
    host: str = "127.0.0.1"  # localhost only (security)
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:5173"])
    require_auth: bool = True  # Admin endpoints require auth token


@dataclass
class PeerStatus:
    """Status of a single federation peer."""

    device_id: str
    hostname: str
    status: str  # online/offline/connecting
    last_seen: float
    latency_ms: float = 0.0
    compute_score: float = 0.0
    cpu_cores: int = 0
    memory_gb: float = 0.0
    is_leader: bool = False
    mode: str = "auto"  # auto/lan/shared_db
    version: str = ""

    def __post_init__(self):
        pass  # All required fields come first

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "hostname": self.hostname,
            "status": self.status,
            "last_seen": self.last_seen,
            "latency_ms": round(self.latency_ms, 1),
            "compute_score": round(self.compute_score, 1),
            "cpu_cores": self.cpu_cores,
            "memory_gb": round(self.memory_gb, 1),
            "is_leader": self.is_leader,
            "mode": self.mode,
            "version": self.version,
        }


@dataclass
class TaskStatus:
    """Status of a federation task."""

    task_id: str
    source_device: str
    status: str  # pending/running/completed/failed
    target_device: str = ""
    created_at: float = 0.0
    completed_at: float = 0.0
    result: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "source_device": self.source_device,
            "target_device": self.target_device,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "duration_sec": round(self.completed_at - self.created_at, 2) if self.completed_at else None,
            "result": self.result[:500] if self.result else "",
            "error": self.error[:500] if self.error else "",
        }


@dataclass
class FederationStatus:
    """Overall federation health status."""

    device_count: int = 0
    online_count: int = 0
    offline_count: int = 0
    leader: str = ""
    mode: str = "auto"
    uptime_sec: float = 0.0
    tasks_total: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_pending: int = 0
    api_version: str = "v1"
    hermes_version: str = ""

    def __post_init__(self):
        pass  # All fields have defaults

    def to_dict(self) -> dict:
        return {
            "device_count": self.device_count,
            "online_count": self.online_count,
            "offline_count": self.offline_count,
            "leader": self.leader,
            "mode": self.mode,
            "uptime_sec": round(self.uptime_sec, 1),
            "tasks": {
                "total": self.tasks_total,
                "completed": self.tasks_completed,
                "failed": self.tasks_failed,
                "pending": self.tasks_pending,
            },
            "api_version": self.api_version,
            "hermes_version": self.hermes_version,
        }


class FederationAPI:
    """HTTP API server for federation management.

    Provides REST endpoints for monitoring and managing federation.
    Uses aiohttp for async HTTP server (already a Hermes dependency).

    Usage:
        api = FederationAPI(adapter, config)
        await api.start()  # Starts HTTP server on configured port
    """

    def __init__(
        self,
        adapter: Any,  # FederationAdapter
        config: Optional[FederationAPIConfig] = None,
        hermes_version: str = "",
    ):
        self.adapter = adapter
        self.config = config or FederationAPIConfig()
        self.hermes_version = hermes_version
        self._start_time = time.time()
        self._server = None

    async def start(self) -> None:
        """Start the HTTP API server."""
        if not self.config.enabled:
            logger.info("Federation API: disabled (config)")
            return

        try:
            from aiohttp import web

            app = web.Application()
            app.router.add_get("/api/federation/health", self._health)
            app.router.add_get("/api/federation/status", self._get_status)
            runner = web.AppRunner(app)
            await runner.setup()

            site = web.TCPSite(
                runner,
                self.config.host,
                self.config.port,
            )
            await site.start()
            self._server = site
            self._runner = runner

            logger.info(
                "Federation API: started on %s:%d",
                self.config.host, self.config.port,
            )
        except ImportError:
            logger.warning(
                "Federation API: aiohttp not installed, API endpoints unavailable"
            )
        except Exception as e:
            logger.error("Federation API: failed to start: %s", e)

    async def stop(self) -> None:
        """Stop the HTTP API server."""
        if self._server:
            try:
                if hasattr(self, '_runner') and self._runner:
                    await self._runner.cleanup()
                logger.info("Federation API: stopped")
            except Exception as e:
                logger.error("Federation API: failed to stop: %s", e)

    async def _health(self, request) -> Any:
        """GET /api/federation/health — Simple health check."""
        from aiohttp import web

        return web.json_response({
            "status": "healthy",
            "uptime_sec": round(time.time() - self._start_time, 1),
            "timestamp": time.time(),
        })

    async def _get_status(self, request) -> Any:
        """GET /api/federation/status — Overall federation status."""
        from aiohttp import web

        status = self._build_status()
        return web.json_response(status.to_dict())

    def _build_status(self) -> FederationStatus:
        """Build current federation status from adapter state."""
        peers = getattr(self.adapter, "_peers", {})
        online = sum(1 for p in peers.values() if getattr(p, "status", "offline") == "online")

        return FederationStatus(
            device_count=len(peers),
            online_count=online,
            offline_count=len(peers) - online,
            leader=getattr(self.adapter, "get_leader", lambda: "")(),
            mode=getattr(self.adapter, "_mode", "auto"),
            uptime_sec=time.time() - self._start_time,
            hermes_version=self.hermes_version,
        )

    def get_peers(self) -> List[dict]:
        """Get list of all peers (for API response)."""
        peers = getattr(self.adapter, "_peers", {})
        result = []
        for peer_id, peer in peers.items():
            result.append({
                "device_id": peer_id,
                "status": getattr(peer, "status", "unknown"),
                "last_seen": getattr(peer, "last_seen", 0),
            })
        return result

    def get_tasks(self) -> List[dict]:
        """Get list of all tasks (for API response)."""
        relay = getattr(self.adapter, "_relay", None)
        if not relay:
            return []
        tasks = getattr(relay, "_tasks", {})
        return [
            {
                "task_id": tid,
                "status": t.get("status", "unknown"),
                "source": t.get("source_device", ""),
                "target": t.get("target_device", ""),
            }
            for tid, t in tasks.items()
        ]

    def get_metrics(self) -> str:
        """Get Prometheus-compatible metrics."""
        status = self._build_status()
        uptime = time.time() - self._start_time

        lines = [
            "# HELP hermes_federation_devices_total Total number of federation devices",
            "# TYPE hermes_federation_devices_total gauge",
            f"hermes_federation_devices_total {status.device_count}",
            "",
            "# HELP hermes_federation_devices_online Number of online devices",
            "# TYPE hermes_federation_devices_online gauge",
            f"hermes_federation_devices_online {status.online_count}",
            "",
            "# HELP hermes_federation_devices_offline Number of offline devices",
            "# TYPE hermes_federation_devices_offline gauge",
            f"hermes_federation_devices_offline {status.offline_count}",
            "",
            "# HELP hermes_federation_uptime_seconds Federation API uptime",
            "# TYPE hermes_federation_uptime_seconds gauge",
            f"hermes_federation_uptime_seconds {uptime:.1f}",
            "",
            "# HELP hermes_federation_is_leader Whether this device is the leader",
            "# TYPE hermes_federation_is_leader gauge",
            f"hermes_federation_is_leader {1 if status.leader else 0}",
        ]
        return "\n".join(lines)

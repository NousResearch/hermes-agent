"""Federation Ops Layer — operational health, lost-contact SOS, stability.

Phase 22 of the P2P federation (#76660). Adds operational-layer features so
the federation can *assist* each member in keeping Hermes healthy:

- ``PeerHealthStatus``: per-peer health snapshot carried in heartbeats
  (gateway up, federation connected, system load, disk, hermes version).
- ``HealthMonitor``: aggregates health across the cluster, computes a
  per-peer health score, and exposes the matrix to the ops API.
- ``LostContactSOS``: "aircraft lost" style escalation. When a peer stops
  heartbeating, surviving peers escalate through soft → confirmed → critical
  and emit OPS_ALERT so the operator / other peers can act (relay tasks,
  notify, attempt remote recovery). Revival emits a recovery alert.

Design principles (from local field experience):
- Health is carried INSIDE the existing heartbeat payload — no extra wires.
- Death detection reuses DeathDetector (3 consecutive failures, CRITICAL-3).
- Alerts are broadcast as OPS_ALERT messages AND written to the audit log.
- Stable connectivity: outbound TLS uses a permissive client context for
  LAN self-signed certs; reconnect uses exponential backoff (already in
  FederationConnectionManager); HealthMonitor tracks connectivity metrics.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from gateway.federation.death_detector import DeathDetector
from gateway.federation.federation_protocol import FedMessage, MessageType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Health status levels
# ---------------------------------------------------------------------------

HEALTH_UNKNOWN = "unknown"      # Never heard from peer
HEALTH_OK = "ok"                # Healthy: gateway up, fed connected, load ok
HEALTH_DEGRADED = "degraded"    # Running but some subsystem unhealthy
HEALTH_CRITICAL = "critical"    # Gateway down / repeated failures
HEALTH_OFFLINE = "offline"      # No heartbeat for >= offline_threshold

# Alert severities
SEV_INFO = "info"
SEV_WARNING = "warning"
SEV_CRITICAL = "critical"


@dataclass
class PeerHealthStatus:
    """Operational health snapshot of a single federation peer."""

    device_id: str
    hostname: str = ""
    level: str = HEALTH_UNKNOWN
    gateway_up: bool = False
    federation_connected: bool = False
    cpu_load: float = 0.0
    cpu_cores: int = 0
    memory_gb: float = 0.0
    memory_used_gb: float = 0.0
    disk_free_gb: float = 0.0
    hermes_version: str = ""
    ws_latency_ms: float = 0.0
    last_heartbeat_at: float = 0.0
    last_alert_at: float = 0.0
    consecutive_failures: int = 0
    metadata: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        """Serialize to heartbeat payload fragment."""
        return {
            "level": self.level,
            "gateway_up": self.gateway_up,
            "federation_connected": self.federation_connected,
            "cpu_load": self.cpu_load,
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "memory_used_gb": self.memory_used_gb,
            "disk_free_gb": self.disk_free_gb,
            "hermes_version": self.hermes_version,
            "ws_latency_ms": self.ws_latency_ms,
        }

    @classmethod
    def from_payload(cls, device_id: str, payload: dict) -> "PeerHealthStatus":
        """Rebuild from a heartbeat payload fragment."""
        return cls(
            device_id=device_id,
            hostname=payload.get("hostname", ""),
            level=payload.get("level", HEALTH_UNKNOWN),
            gateway_up=bool(payload.get("gateway_up", False)),
            federation_connected=bool(payload.get("federation_connected", False)),
            cpu_load=float(payload.get("cpu_load", 0.0)),
            cpu_cores=int(payload.get("cpu_cores", 0)),
            memory_gb=float(payload.get("memory_gb", 0.0)),
            memory_used_gb=float(payload.get("memory_used_gb", 0.0)),
            disk_free_gb=float(payload.get("disk_free_gb", 0.0)),
            hermes_version=payload.get("hermes_version", ""),
            ws_latency_ms=float(payload.get("ws_latency_ms", 0.0)),
            last_heartbeat_at=time.time(),
        )


@dataclass
class OpsAlert:
    """An operational alert emitted by the ops layer."""

    alert_id: str
    severity: str
    source_device: str
    target_device: str
    message: str
    created_at: float
    alert_type: str = "ops"  # ops | lost_contact | recovery | health_degraded


class HealthMonitor:
    """Aggregates per-peer health across the cluster.

    - ``update_from_heartbeat``: called on every received heartbeat.
    - ``mark_failed``: called when a probe/heartbeat to a peer fails.
    - ``compute_health_score``: weighted score for routing/visibility.
    """

    def __init__(self, device_id: str, offline_threshold_s: float = 30.0):
        self.device_id = device_id
        self.offline_threshold_s = offline_threshold_s
        self._peers: Dict[str, PeerHealthStatus] = {}
        self._death = DeathDetector(dead_threshold=3)
        self._alerts: List[OpsAlert] = []
        self._max_alerts = 200

    # -- state ----------------------------------------------------------

    def update_from_heartbeat(self, device_id: str, payload: dict) -> bool:
        """Record health from a peer heartbeat. Returns True on revival."""
        was_dead = self._death.record_success(device_id)
        status = PeerHealthStatus.from_payload(device_id, payload)
        status.last_heartbeat_at = time.time()
        self._peers[device_id] = status
        if was_dead:
            self._emit_alert(
                severity=SEV_INFO,
                source_device=self.device_id,
                target_device=device_id,
                message=f"Recovery: {device_id} resumed heartbeats after loss of contact",
                alert_type="recovery",
            )
        return was_dead

    def mark_failed(self, device_id: str) -> bool:
        """Record a failed probe/heartbeat. Returns True when peer just died."""
        died = self._death.record_failure(device_id)
        status = self._peers.get(device_id)
        death_status = self._death.get_status(device_id)
        if status and death_status:
            status.consecutive_failures = death_status.failure_count
        if died:
            status = self._peers.get(device_id) or PeerHealthStatus(device_id=device_id)
            status.level = HEALTH_CRITICAL
            self._peers[device_id] = status
            self._emit_alert(
                severity=SEV_CRITICAL,
                source_device=self.device_id,
                target_device=device_id,
                message=(
                    f"LOST CONTACT: {device_id} failed {self._death.threshold} "
                    f"consecutive probes — assumed offline (aircraft-lost)"
                ),
                alert_type="lost_contact",
            )
        return died

    def mark_offline(self, device_id: str, reason: str = "heartbeat timeout") -> None:
        """Mark a peer offline (no heartbeat for >= offline_threshold)."""
        status = self._peers.get(device_id) or PeerHealthStatus(device_id=device_id)
        status.level = HEALTH_OFFLINE
        status.last_alert_at = time.time()
        self._peers[device_id] = status
        self._emit_alert(
            severity=SEV_WARNING,
            source_device=self.device_id,
            target_device=device_id,
            message=f"Offline: {device_id} — {reason}",
            alert_type="ops",
        )

    def touch(self, device_id: str) -> None:
        """Refresh last-seen without changing health level (connectivity probe)."""
        status = self._peers.get(device_id)
        if status:
            status.last_heartbeat_at = time.time()

    # -- queries --------------------------------------------------------

    def get_health(self, device_id: str) -> Optional[PeerHealthStatus]:
        return self._peers.get(device_id)

    def get_all_health(self) -> Dict[str, PeerHealthStatus]:
        return dict(self._peers)

    def compute_health_score(self, device_id: str) -> float:
        """0.0..1.0 weighted health score for a peer."""
        status = self._peers.get(device_id)
        if not status:
            return 0.0
        score = 1.0
        if not status.gateway_up:
            score -= 0.5
        if not status.federation_connected:
            score -= 0.3
        if status.cpu_load > 4.0:
            score -= 0.1 * min(status.cpu_load, 8.0) / 4.0
        if status.level in (HEALTH_CRITICAL, HEALTH_OFFLINE):
            score = 0.0
        elif status.level == HEALTH_DEGRADED:
            score = min(score, 0.6)
        return max(score, 0.0)

    def health_summary(self) -> dict:
        """Compact matrix for the ops API."""
        return {
            device: {
                "level": s.level,
                "gateway_up": s.gateway_up,
                "federation_connected": s.federation_connected,
                "cpu_load": round(s.cpu_load, 2),
                "disk_free_gb": round(s.disk_free_gb, 1),
                "hermes_version": s.hermes_version,
                "health_score": round(self.compute_health_score(device), 2),
                "last_heartbeat_at": s.last_heartbeat_at,
            }
            for device, s in self._peers.items()
        }

    # -- alerts ---------------------------------------------------------

    def get_alerts(self, limit: int = 50) -> List[dict]:
        return [
            {
                "alert_id": a.alert_id,
                "severity": a.severity,
                "source": a.source_device,
                "target": a.target_device,
                "type": a.alert_type,
                "message": a.message,
                "created_at": a.created_at,
            }
            for a in self._alerts[-limit:]
        ]

    def _emit_alert(self, severity: str, source_device: str, target_device: str,
                    message: str, alert_type: str = "ops") -> OpsAlert:
        alert = OpsAlert(
            alert_id=f"{int(time.time()*1000)}-{len(self._alerts)}",
            severity=severity,
            source_device=source_device,
            target_device=target_device,
            message=message,
            created_at=time.time(),
            alert_type=alert_type,
        )
        self._alerts.append(alert)
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts:]
        logger.info("Ops alert [%s] %s -> %s: %s", severity, source_device, target_device, message)
        return alert


class LostContactSOS:
    """Aircraft-lost escalation: detect + escalate + broadcast + assist.

    When a peer loses contact:
    1. HealthMonitor marks the peer failed (3 consecutive failures).
    2. LostContactSOS escalates severity by time offline:
       - soft (0-60s): log + info alert
       - confirmed (60-300s): OPS_ALERT broadcast (surviving peers notify)
       - critical (>300s): OPS_ALERT critical + assist callbacks invoked
    3. Assist callbacks let the operator plug recovery actions
       (e.g. SSH probe + remote gateway restart on the lost host).
    4. On revival, a recovery alert is emitted automatically.
    """

    SOFT_WINDOW_S = 60.0
    CONFIRMED_WINDOW_S = 300.0

    def __init__(self, device_id: str, health: HealthMonitor,
                 on_alert: Optional[Callable[[OpsAlert], None]] = None,
                 on_assist: Optional[Callable[[str], None]] = None):
        self.device_id = device_id
        self.health = health
        self.on_alert = on_alert          # e.g. send to operator channel
        self.on_assist = on_assist        # e.g. SSH probe / remote restart
        self._escalated: Dict[str, str] = {}  # device -> level (soft|confirmed|critical)
        self._lost_at: Dict[str, float] = {}

    def update(self) -> List[OpsAlert]:
        """Check all peers against escalation windows. Returns new alerts."""
        alerts: List[OpsAlert] = []
        now = time.time()
        for device_id, status in self.health.get_all_health().items():
            if status.level not in (HEALTH_CRITICAL, HEALTH_OFFLINE):
                # Reset escalation on healthy peers
                self._escalated.pop(device_id, None)
                self._lost_at.pop(device_id, None)
                continue
            lost_since = self._lost_at.get(device_id, status.last_heartbeat_at or now)
            self._lost_at[device_id] = lost_since
            elapsed = now - lost_since

            if elapsed < self.SOFT_WINDOW_S:
                level = "soft"
            elif elapsed < self.CONFIRMED_WINDOW_S:
                level = "confirmed"
            else:
                level = "critical"

            if self._escalated.get(device_id) != level:
                self._escalated[device_id] = level
                alert = self.health._emit_alert(
                    severity=SEV_WARNING if level in ("soft", "confirmed") else SEV_CRITICAL,
                    source_device=self.device_id,
                    target_device=device_id,
                    message=(
                        f"SOS escalation [{level}]: {device_id} lost contact "
                        f"for {int(elapsed)}s — surviving peers standing by"
                    ),
                    alert_type="lost_contact",
                )
                alerts.append(alert)
                if self.on_alert:
                    try:
                        self.on_alert(alert)
                    except Exception as e:
                        logger.warning("SOS on_alert failed: %s", e)
                if level == "critical" and self.on_assist:
                    try:
                        self.on_assist(device_id)
                    except Exception as e:
                        logger.warning("SOS on_assist failed: %s", e)
        return alerts

    def reset(self, device_id: str) -> None:
        self._escalated.pop(device_id, None)
        self._lost_at.pop(device_id, None)


def collect_local_health(hermes_version: str = "") -> dict:
    """Collect this device's health payload fragment. Cross-platform."""
    import os as _os
    import platform as _platform

    # CPU load (Unix-only)
    try:
        load_avg = _os.getloadavg()[0] if hasattr(_os, "getloadavg") else 0.0
    except Exception:
        load_avg = 0.0

    # Memory — platform-specific detection with encoding= for footgun rule
    try:
        system = _platform.system()
        if system == "Darwin":
            import subprocess as _subprocess
            result = _subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, encoding="utf-8", timeout=3,
            )
            raw = result.stdout.strip()
            memory_gb = float(raw) / (1024 ** 3) if raw else 0.0
        elif system == "Linux":
            import subprocess as _subprocess
            # Try cgroup v2 (modern distros), fall back to /proc/meminfo
            result = _subprocess.run(
                ["cat", "/sys/fs/cgroup/memory.max"],
                capture_output=True, encoding="utf-8", timeout=3,
            )
            if result.returncode == 0 and result.stdout.strip() != "max":
                memory_gb = int(result.stdout.strip()) / (1024 ** 3)
            else:
                result = _subprocess.run(
                    ["awk", "/MemTotal/ {print $2}", "/proc/meminfo"],
                    capture_output=True, encoding="utf-8", timeout=3,
                )
                kb = result.stdout.strip()
                memory_gb = float(kb) / (1024 * 1024) if kb else 0.0
        elif system == "Windows":
            import re as _re
            import subprocess as _subprocess
            result = _subprocess.run(
                ["wmic", "OS", "get", "TotalVisibleMemorySize", "/value"],
                capture_output=True, encoding="utf-8", timeout=3,
            )
            m = _re.search(r"=(\d+)", result.stdout)
            memory_gb = int(m.group(1)) / 1024 if m else 0.0
        else:
            memory_gb = 0.0
    except Exception:
        memory_gb = 0.0

    # Disk free
    try:
        import shutil as _shutil
        disk_free = _shutil.disk_usage(_os.path.expanduser("~")).free / (1024 ** 3)
    except Exception:
        disk_free = 0.0

    return {
        "cpu_load": round(load_avg, 2),
        "cpu_cores": _os.cpu_count() or 0,
        "memory_gb": round(memory_gb, 1),
        "disk_free_gb": round(disk_free, 1),
        "hermes_version": hermes_version,
    }

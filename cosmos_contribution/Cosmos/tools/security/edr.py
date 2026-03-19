"""
cosmos EDR (Endpoint Detection and Response)

"Good news, everyone! I've built an endpoint protection system!"

Endpoint monitoring, threat detection, and response capabilities.
"""

import asyncio
import os
import hashlib
import json
import re
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from collections import deque
from loguru import logger


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertType(Enum):
    """Types of security alerts."""
    PROCESS_ANOMALY = "process_anomaly"
    FILE_MODIFICATION = "file_modification"
    NETWORK_ANOMALY = "network_anomaly"
    PERSISTENCE_MECHANISM = "persistence_mechanism"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    DATA_EXFILTRATION = "data_exfiltration"
    MALWARE_DETECTED = "malware_detected"
    POLICY_VIOLATION = "policy_violation"
    AUTHENTICATION_ANOMALY = "authentication_anomaly"


class ResponseAction(Enum):
    """Automated response actions."""
    ALERT_ONLY = "alert_only"
    KILL_PROCESS = "kill_process"
    QUARANTINE_FILE = "quarantine_file"
    BLOCK_NETWORK = "block_network"
    ISOLATE_ENDPOINT = "isolate_endpoint"
    COLLECT_EVIDENCE = "collect_evidence"


@dataclass
class SecurityAlert:
    """Security alert from EDR."""
    id: str
    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    source: str = ""
    process_name: str = ""
    process_id: int = 0
    file_path: str = ""
    network_info: Dict[str, Any] = field(default_factory=dict)
    indicators: List[str] = field(default_factory=list)
    recommended_actions: List[ResponseAction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "process": {"name": self.process_name, "pid": self.process_id},
            "file_path": self.file_path,
            "network_info": self.network_info,
            "indicators": self.indicators,
            "recommended_actions": [a.value for a in self.recommended_actions],
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
        }


@dataclass
class DetectionRule:
    """EDR detection rule."""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    alert_type: AlertType
    condition: Callable[[Dict[str, Any]], bool]
    response_actions: List[ResponseAction] = field(default_factory=list)
    enabled: bool = True
    tags: List[str] = field(default_factory=list)


@dataclass
class EndpointStatus:
    """Current endpoint security status."""
    hostname: str
    ip_address: str
    os_version: str
    agent_version: str = "1.0.0"
    last_seen: datetime = field(default_factory=datetime.now)
    protection_status: str = "active"
    pending_alerts: int = 0
    quarantined_files: int = 0
    monitored_processes: int = 0


class EDREngine:
    """
    Endpoint Detection and Response Engine.

    Capabilities:
    - Real-time process monitoring
    - File system monitoring
    - Network connection monitoring
    - Behavioral analysis
    - Automated response
    - Alert management
    """

    # Suspicious process behaviors
    SUSPICIOUS_BEHAVIORS = {
        "lsass_access": {
            "process": "lsass.exe",
            "description": "Access to LSASS process (credential dumping)",
            "severity": AlertSeverity.CRITICAL,
        },
        "powershell_encoded": {
            "pattern": r"-enc|-encodedcommand",
            "description": "Encoded PowerShell command",
            "severity": AlertSeverity.HIGH,
        },
        "certutil_download": {
            "pattern": r"certutil.*-urlcache",
            "description": "certutil used for downloading (LOLBin)",
            "severity": AlertSeverity.HIGH,
        },
        "mshta_execution": {
            "process": "mshta.exe",
            "description": "MSHTA execution (potential malware)",
            "severity": AlertSeverity.HIGH,
        },
        "wmic_process_call": {
            "pattern": r"wmic.*process.*call.*create",
            "description": "WMIC process creation (lateral movement)",
            "severity": AlertSeverity.HIGH,
        },
        "scheduled_task_create": {
            "pattern": r"schtasks.*/create",
            "description": "Scheduled task creation (persistence)",
            "severity": AlertSeverity.MEDIUM,
        },
        "reg_run_key": {
            "pattern": r"reg.*add.*\\\\run",
            "description": "Registry Run key modification (persistence)",
            "severity": AlertSeverity.MEDIUM,
        },
        "service_creation": {
            "pattern": r"sc.*create",
            "description": "Service creation (persistence/privilege escalation)",
            "severity": AlertSeverity.MEDIUM,
        },
    }

    # Known malicious file hashes (simplified - real implementation uses threat feeds)
    MALICIOUS_HASHES: Set[str] = set()

    # Monitored file paths
    CRITICAL_PATHS = [
        r"C:\Windows\System32",
        r"C:\Windows\SysWOW64",
        r"C:\Windows\Temp",
        "/etc/passwd",
        "/etc/shadow",
        "/etc/cron.d",
    ]

    def __init__(self, data_dir: str = "./data/edr"):
        """Initialize EDR engine."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.alerts: deque = deque(maxlen=10000)
        self.detection_rules: Dict[str, DetectionRule] = {}
        self.quarantine_dir = self.data_dir / "quarantine"
        self.quarantine_dir.mkdir(exist_ok=True)

        self._alert_counter = 0
        self._monitoring = False
        self._alert_callbacks: List[Callable] = []

        # Load default rules
        self._load_default_rules()

    def _load_default_rules(self):
        """Load default detection rules."""
        # Process-based rules
        self.add_rule(DetectionRule(
            id="proc_001",
            name="LSASS Memory Access",
            description="Process accessing LSASS memory (potential credential dumping)",
            severity=AlertSeverity.CRITICAL,
            alert_type=AlertType.PRIVILEGE_ESCALATION,
            condition=lambda e: "lsass" in e.get("target_process", "").lower(),
            response_actions=[ResponseAction.ALERT_ONLY, ResponseAction.COLLECT_EVIDENCE],
            tags=["credential_access", "t1003"],
        ))

        self.add_rule(DetectionRule(
            id="proc_002",
            name="Encoded PowerShell Command",
            description="PowerShell executed with encoded command",
            severity=AlertSeverity.HIGH,
            alert_type=AlertType.PROCESS_ANOMALY,
            condition=lambda e: (
                "powershell" in e.get("process_name", "").lower() and
                any(x in e.get("cmdline", "").lower() for x in ["-enc", "-encodedcommand", "-e "])
            ),
            response_actions=[ResponseAction.ALERT_ONLY],
            tags=["execution", "t1059.001"],
        ))

        self.add_rule(DetectionRule(
            id="proc_003",
            name="Suspicious Process from Temp",
            description="Executable running from temp directory",
            severity=AlertSeverity.MEDIUM,
            alert_type=AlertType.PROCESS_ANOMALY,
            condition=lambda e: (
                "temp" in e.get("process_path", "").lower() or
                "/tmp/" in e.get("process_path", "").lower()
            ),
            response_actions=[ResponseAction.ALERT_ONLY],
            tags=["execution"],
        ))

        self.add_rule(DetectionRule(
            id="file_001",
            name="Executable in Startup",
            description="New executable added to startup location",
            severity=AlertSeverity.HIGH,
            alert_type=AlertType.PERSISTENCE_MECHANISM,
            condition=lambda e: (
                e.get("event_type") == "file_create" and
                "startup" in e.get("file_path", "").lower() and
                e.get("file_path", "").lower().endswith((".exe", ".dll", ".bat", ".ps1"))
            ),
            response_actions=[ResponseAction.ALERT_ONLY, ResponseAction.QUARANTINE_FILE],
            tags=["persistence", "t1547.001"],
        ))

        self.add_rule(DetectionRule(
            id="net_001",
            name="Connection to Known Malicious Port",
            description="Outbound connection to suspicious port",
            severity=AlertSeverity.HIGH,
            alert_type=AlertType.NETWORK_ANOMALY,
            condition=lambda e: (
                e.get("event_type") == "network" and
                e.get("direction") == "outbound" and
                e.get("remote_port") in [4444, 5555, 6666, 1337, 31337]
            ),
            response_actions=[ResponseAction.ALERT_ONLY, ResponseAction.BLOCK_NETWORK],
            tags=["command_and_control", "t1571"],
        ))

    def add_rule(self, rule: DetectionRule):
        """Add a detection rule."""
        self.detection_rules[rule.id] = rule
        logger.debug(f"Added detection rule: {rule.name}")

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a detection rule."""
        if rule_id in self.detection_rules:
            del self.detection_rules[rule_id]
            return True
        return False

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        self._alert_counter += 1
        return f"ALERT-{datetime.now().strftime('%Y%m%d')}-{self._alert_counter:06d}"

    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        description: str,
        **kwargs,
    ) -> SecurityAlert:
        """Create and store a security alert."""
        alert = SecurityAlert(
            id=self._generate_alert_id(),
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            title=title,
            description=description,
            **kwargs,
        )

        self.alerts.append(alert)
        self._persist_alert(alert)

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        logger.warning(f"[{severity.value.upper()}] {title}")

        return alert

    def _persist_alert(self, alert: SecurityAlert):
        """Persist alert to disk."""
        alerts_dir = self.data_dir / "alerts"
        alerts_dir.mkdir(exist_ok=True)

        date_str = alert.timestamp.strftime("%Y%m%d")
        alert_file = alerts_dir / f"alerts_{date_str}.jsonl"

        with open(alert_file, "a") as f:
            f.write(json.dumps(alert.to_dict()) + "\n")

    def register_alert_callback(self, callback: Callable[[SecurityAlert], None]):
        """Register callback for new alerts."""
        self._alert_callbacks.append(callback)

    def process_event(self, event: Dict[str, Any]) -> Optional[SecurityAlert]:
        """
        Process a security event and check against rules.

        Args:
            event: Event data dictionary

        Returns:
            SecurityAlert if triggered, None otherwise
        """
        for rule in self.detection_rules.values():
            if not rule.enabled:
                continue

            try:
                if rule.condition(event):
                    alert = self.create_alert(
                        alert_type=rule.alert_type,
                        severity=rule.severity,
                        title=rule.name,
                        description=rule.description,
                        process_name=event.get("process_name", ""),
                        process_id=event.get("pid", 0),
                        file_path=event.get("file_path", ""),
                        network_info=event.get("network_info", {}),
                        indicators=[f"Rule: {rule.id}"] + rule.tags,
                        recommended_actions=rule.response_actions,
                    )

                    # Execute automated responses
                    self._execute_responses(alert, rule.response_actions)

                    return alert

            except Exception as e:
                logger.error(f"Rule evaluation error ({rule.id}): {e}")

        return None

    def _execute_responses(self, alert: SecurityAlert, actions: List[ResponseAction]):
        """Execute automated response actions."""
        for action in actions:
            if action == ResponseAction.ALERT_ONLY:
                continue

            elif action == ResponseAction.COLLECT_EVIDENCE:
                logger.info(f"Collecting evidence for alert {alert.id}")
                # Would trigger evidence collection

            elif action == ResponseAction.QUARANTINE_FILE:
                if alert.file_path:
                    self.quarantine_file(alert.file_path, alert.id)

            # Other actions would be implemented based on authorization level
            # KILL_PROCESS, BLOCK_NETWORK, ISOLATE_ENDPOINT require elevated privileges

    def quarantine_file(self, file_path: str, reason: str = "") -> bool:
        """
        Quarantine a suspicious file.

        Args:
            file_path: Path to file
            reason: Reason for quarantine

        Returns:
            Success status
        """
        try:
            import shutil

            source = Path(file_path)
            if not source.exists():
                return False

            # Create quarantine entry
            quarantine_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_dir = self.quarantine_dir / quarantine_id
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Move file
            dest_file = dest_dir / source.name
            shutil.move(str(source), str(dest_file))

            # Calculate hash
            sha256 = hashlib.sha256()
            with open(dest_file, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)

            # Write metadata
            metadata = {
                "original_path": str(source),
                "quarantine_time": datetime.now().isoformat(),
                "reason": reason,
                "sha256": sha256.hexdigest(),
            }
            (dest_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

            logger.info(f"Quarantined file: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Quarantine failed: {e}")
            return False

    async def start_monitoring(self):
        """Start real-time endpoint monitoring."""
        self._monitoring = True
        logger.info("EDR monitoring started")

        # Start monitoring tasks
        tasks = [
            self._monitor_processes(),
            self._monitor_network(),
        ]

        await asyncio.gather(*tasks)

    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self._monitoring = False
        logger.info("EDR monitoring stopped")

    async def _monitor_processes(self):
        """Monitor running processes."""
        try:
            import psutil

            known_pids: Set[int] = set()

            while self._monitoring:
                current_pids = set()

                for proc in psutil.process_iter(["pid", "name", "exe", "cmdline", "username"]):
                    try:
                        info = proc.info
                        pid = info["pid"]
                        current_pids.add(pid)

                        # Check new processes
                        if pid not in known_pids:
                            event = {
                                "event_type": "process_start",
                                "pid": pid,
                                "process_name": info["name"] or "",
                                "process_path": info["exe"] or "",
                                "cmdline": " ".join(info["cmdline"]) if info["cmdline"] else "",
                                "username": info["username"] or "",
                            }

                            self.process_event(event)

                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                known_pids = current_pids
                await asyncio.sleep(1)

        except ImportError:
            logger.warning("psutil not installed - process monitoring disabled")

    async def _monitor_network(self):
        """Monitor network connections."""
        try:
            import psutil

            known_connections: Set[tuple] = set()

            while self._monitoring:
                current_connections = set()

                for conn in psutil.net_connections(kind="inet"):
                    try:
                        conn_id = (conn.laddr, conn.raddr, conn.status)
                        current_connections.add(conn_id)

                        if conn_id not in known_connections and conn.raddr:
                            event = {
                                "event_type": "network",
                                "direction": "outbound" if conn.status == "ESTABLISHED" else "inbound",
                                "local_ip": conn.laddr.ip if conn.laddr else "",
                                "local_port": conn.laddr.port if conn.laddr else 0,
                                "remote_ip": conn.raddr.ip if conn.raddr else "",
                                "remote_port": conn.raddr.port if conn.raddr else 0,
                                "status": conn.status,
                                "pid": conn.pid,
                            }

                            self.process_event(event)

                    except Exception:
                        continue

                known_connections = current_connections
                await asyncio.sleep(2)

        except ImportError:
            logger.warning("psutil not installed - network monitoring disabled")

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        alert_type: Optional[AlertType] = None,
        start_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[SecurityAlert]:
        """Get alerts with optional filtering."""
        filtered = []

        for alert in reversed(self.alerts):
            if severity and alert.severity != severity:
                continue
            if alert_type and alert.alert_type != alert_type:
                continue
            if start_time and alert.timestamp < start_time:
                continue

            filtered.append(alert)
            if len(filtered) >= limit:
                break

        return filtered

    def get_endpoint_status(self) -> EndpointStatus:
        """Get current endpoint status."""
        import platform
        import socket

        hostname = socket.gethostname()

        try:
            ip = socket.gethostbyname(hostname)
        except socket.gaierror:
            ip = "127.0.0.1"

        unresolved = sum(1 for a in self.alerts if not a.resolved)
        quarantined = len(list(self.quarantine_dir.iterdir()))

        return EndpointStatus(
            hostname=hostname,
            ip_address=ip,
            os_version=platform.platform(),
            pending_alerts=unresolved,
            quarantined_files=quarantined,
            monitored_processes=len(self.detection_rules),
        )

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def resolve_alert(self, alert_id: str, notes: str = "") -> bool:
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.metadata["resolution_notes"] = notes
                alert.metadata["resolved_at"] = datetime.now().isoformat()
                return True
        return False

    def export_alerts(self, format: str = "json") -> str:
        """Export alerts in various formats."""
        if format == "json":
            return json.dumps([a.to_dict() for a in self.alerts], indent=2)

        elif format == "csv":
            lines = ["id,timestamp,severity,type,title,process,file_path"]
            for alert in self.alerts:
                lines.append(
                    f'"{alert.id}",{alert.timestamp.isoformat()},'
                    f'{alert.severity.value},{alert.alert_type.value},'
                    f'"{alert.title}","{alert.process_name}","{alert.file_path}"'
                )
            return "\n".join(lines)

        else:
            # Plain text
            lines = []
            for alert in self.alerts:
                lines.append(
                    f"[{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"[{alert.severity.value.upper()}] {alert.title}"
                )
            return "\n".join(lines)


# Global instance
edr_engine = EDREngine()

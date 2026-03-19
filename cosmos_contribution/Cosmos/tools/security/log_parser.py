"""
cosmos Security Log Parser

"Good news, everyone! I can read logs faster than Bender drinks!"

Advanced log parsing with security-focused analysis.
"""

import re
import gzip
import json
from typing import Dict, Any, List, Optional, Generator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from collections import Counter, defaultdict
from loguru import logger


class LogType(Enum):
    """Types of security logs."""
    SYSLOG = "syslog"
    WINDOWS_EVENT = "windows_event"
    APACHE_ACCESS = "apache_access"
    APACHE_ERROR = "apache_error"
    NGINX_ACCESS = "nginx_access"
    NGINX_ERROR = "nginx_error"
    AUTH_LOG = "auth_log"
    FIREWALL = "firewall"
    IDS_IPS = "ids_ips"
    AUDIT = "audit"
    APPLICATION = "application"
    JSON = "json"
    CEF = "cef"
    LEEF = "leef"
    EVTX = "evtx"
    CUSTOM = "custom"


class SecurityEventType(Enum):
    """Security-relevant event types."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PRIVILEGE_CHANGE = "privilege_change"
    FILE_ACCESS = "file_access"
    FILE_MODIFICATION = "file_modification"
    PROCESS_START = "process_start"
    PROCESS_STOP = "process_stop"
    NETWORK_CONNECTION = "network_connection"
    FIREWALL_BLOCK = "firewall_block"
    MALWARE_DETECTED = "malware_detected"
    POLICY_VIOLATION = "policy_violation"
    SYSTEM_ERROR = "system_error"
    CONFIG_CHANGE = "config_change"
    USER_CREATED = "user_created"
    USER_DELETED = "user_deleted"
    PASSWORD_CHANGE = "password_change"
    UNKNOWN = "unknown"


@dataclass
class ParsedLogEntry:
    """A parsed log entry with security context."""
    raw: str
    timestamp: Optional[datetime] = None
    source: str = ""
    log_type: LogType = LogType.CUSTOM
    event_type: SecurityEventType = SecurityEventType.UNKNOWN
    severity: str = "info"
    message: str = ""
    username: str = ""
    source_ip: str = ""
    destination_ip: str = ""
    port: int = 0
    action: str = ""
    status: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "source": self.source,
            "log_type": self.log_type.value,
            "event_type": self.event_type.value,
            "severity": self.severity,
            "message": self.message,
            "username": self.username,
            "source_ip": self.source_ip,
            "destination_ip": self.destination_ip,
            "port": self.port,
            "action": self.action,
            "status": self.status,
            "metadata": self.metadata,
        }


@dataclass
class SecurityInsight:
    """Security insight from log analysis."""
    insight_type: str
    title: str
    description: str
    severity: str
    count: int = 0
    examples: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    related_entries: List[ParsedLogEntry] = field(default_factory=list)


@dataclass
class LogAnalysisReport:
    """Comprehensive log analysis report."""
    total_entries: int = 0
    time_range: tuple = (None, None)
    entries_by_type: Dict[str, int] = field(default_factory=dict)
    entries_by_severity: Dict[str, int] = field(default_factory=dict)
    security_insights: List[SecurityInsight] = field(default_factory=list)
    top_source_ips: List[tuple] = field(default_factory=list)
    top_users: List[tuple] = field(default_factory=list)
    failed_logins: int = 0
    successful_logins: int = 0
    anomalies: List[Dict[str, Any]] = field(default_factory=list)


class SecurityLogParser:
    """
    Advanced security log parser.

    Supports:
    - Multiple log formats (syslog, Windows Event, Apache, nginx, etc.)
    - Security-focused event classification
    - Threat detection patterns
    - Statistical analysis
    - Timeline generation
    """

    # Log format patterns
    PATTERNS = {
        LogType.SYSLOG: re.compile(
            r"^(?P<timestamp>\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+"
            r"(?P<host>\S+)\s+"
            r"(?P<process>[\w\-]+)(?:\[(?P<pid>\d+)\])?\s*:\s*"
            r"(?P<message>.*)$"
        ),
        LogType.AUTH_LOG: re.compile(
            r"^(?P<timestamp>\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})\s+"
            r"(?P<host>\S+)\s+"
            r"(?P<service>\S+?)(?:\[(?P<pid>\d+)\])?\s*:\s*"
            r"(?P<message>.*)$"
        ),
        LogType.APACHE_ACCESS: re.compile(
            r'^(?P<ip>[\d\.]+)\s+-\s+(?P<user>\S+)\s+\[(?P<timestamp>[^\]]+)\]\s+'
            r'"(?P<method>\w+)\s+(?P<path>\S+)\s+(?P<protocol>[^"]+)"\s+'
            r'(?P<status>\d+)\s+(?P<size>\d+|-)'
        ),
        LogType.NGINX_ACCESS: re.compile(
            r'^(?P<ip>[\d\.]+)\s+-\s+(?P<user>\S+)\s+\[(?P<timestamp>[^\]]+)\]\s+'
            r'"(?P<request>[^"]+)"\s+(?P<status>\d+)\s+(?P<size>\d+)\s+'
            r'"(?P<referer>[^"]*)"\s+"(?P<user_agent>[^"]*)"'
        ),
        LogType.WINDOWS_EVENT: re.compile(
            r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+"
            r"(?P<level>\w+)\s+"
            r"(?P<source>\S+)\s+"
            r"(?P<event_id>\d+)\s+"
            r"(?P<message>.*)$"
        ),
        LogType.CEF: re.compile(
            r"^CEF:(?P<version>\d+)\|(?P<vendor>[^|]+)\|(?P<product>[^|]+)\|"
            r"(?P<device_version>[^|]+)\|(?P<signature_id>[^|]+)\|"
            r"(?P<name>[^|]+)\|(?P<severity>[^|]+)\|(?P<extension>.*)$"
        ),
        LogType.JSON: re.compile(r"^\{.*\}$"),
    }

    # Security event patterns
    SECURITY_PATTERNS = {
        # Authentication patterns
        "login_success": [
            (r"Accepted\s+(password|publickey|keyboard-interactive)\s+for\s+(\S+)", ["method", "username"]),
            (r"session opened for user (\S+)", ["username"]),
            (r"successful login.*user[=:\s]+(\S+)", ["username"]),
            (r"Logon\s+Type:\s*(\d+).*Account\s+Name:\s*(\S+)", ["logon_type", "username"]),
        ],
        "login_failure": [
            (r"Failed\s+(password|publickey)\s+for\s+(?:invalid\s+user\s+)?(\S+)\s+from\s+(\S+)", ["method", "username", "ip"]),
            (r"authentication\s+failure.*user[=:\s]+(\S+)", ["username"]),
            (r"Invalid\s+user\s+(\S+)\s+from\s+(\S+)", ["username", "ip"]),
            (r"PAM.*authentication\s+error\s+for\s+(\S+)", ["username"]),
        ],
        "sudo": [
            (r"sudo:\s+(\S+)\s+:", ["username"]),
            (r"COMMAND=(.+)", ["command"]),
        ],
        "privilege_escalation": [
            (r"su.*session opened for user (\S+)", ["target_user"]),
            (r"changed user to (\S+)", ["target_user"]),
        ],
        "brute_force": [
            (r"maximum\s+authentication\s+attempts\s+exceeded", []),
            (r"too\s+many\s+authentication\s+failures", []),
            (r"repeated\s+authentication\s+failures", []),
        ],
        "firewall_block": [
            (r"(?:DROP|REJECT|BLOCK)\s+(?:IN|OUT)?.*SRC=(\S+).*DST=(\S+).*DPT=(\d+)", ["src_ip", "dst_ip", "dst_port"]),
            (r"UFW\s+BLOCK.*SRC=(\S+).*DST=(\S+)", ["src_ip", "dst_ip"]),
        ],
        "sql_injection": [
            (r"(?:union\s+select|;\s*drop|;\s*delete|';\s*--)", []),
            (r"(?:or\s+1\s*=\s*1|'\s+or\s+')", []),
        ],
        "xss_attempt": [
            (r"<script[^>]*>", []),
            (r"javascript\s*:", []),
            (r"on(?:load|error|click)\s*=", []),
        ],
        "path_traversal": [
            (r"\.\.\/|\.\.\\", []),
            (r"%2e%2e%2f|%2e%2e/|\.\.%2f", []),
        ],
        "command_injection": [
            (r";\s*(?:cat|ls|id|whoami|pwd|uname)", []),
            (r"\|\s*(?:cat|ls|id|whoami|pwd)", []),
            (r"`[^`]+`", []),
        ],
    }

    # Windows Event IDs of interest
    WINDOWS_SECURITY_EVENTS = {
        4624: ("Login Success", SecurityEventType.LOGIN_SUCCESS),
        4625: ("Login Failure", SecurityEventType.LOGIN_FAILURE),
        4634: ("Logoff", SecurityEventType.LOGOUT),
        4648: ("Explicit Credentials Login", SecurityEventType.LOGIN_SUCCESS),
        4672: ("Special Privileges Assigned", SecurityEventType.PRIVILEGE_CHANGE),
        4720: ("User Account Created", SecurityEventType.USER_CREATED),
        4726: ("User Account Deleted", SecurityEventType.USER_DELETED),
        4728: ("User Added to Security Group", SecurityEventType.CONFIG_CHANGE),
        4732: ("User Added to Local Group", SecurityEventType.CONFIG_CHANGE),
        4756: ("User Added to Universal Group", SecurityEventType.CONFIG_CHANGE),
        4768: ("Kerberos TGT Request", SecurityEventType.LOGIN_SUCCESS),
        4769: ("Kerberos Service Ticket", SecurityEventType.LOGIN_SUCCESS),
        4771: ("Kerberos Pre-Auth Failed", SecurityEventType.LOGIN_FAILURE),
        4776: ("Credential Validation", SecurityEventType.LOGIN_SUCCESS),
        4688: ("Process Created", SecurityEventType.PROCESS_START),
        4689: ("Process Terminated", SecurityEventType.PROCESS_STOP),
        5156: ("Network Connection", SecurityEventType.NETWORK_CONNECTION),
        5157: ("Network Connection Blocked", SecurityEventType.FIREWALL_BLOCK),
    }

    def __init__(self):
        """Initialize log parser."""
        self._compiled_security_patterns = self._compile_security_patterns()

    def _compile_security_patterns(self) -> Dict[str, List[tuple]]:
        """Compile security patterns for faster matching."""
        compiled = {}
        for category, patterns in self.SECURITY_PATTERNS.items():
            compiled[category] = [
                (re.compile(pattern, re.I), groups)
                for pattern, groups in patterns
            ]
        return compiled

    def detect_log_type(self, sample_lines: List[str]) -> LogType:
        """Detect log type from sample lines."""
        for line in sample_lines:
            line = line.strip()
            if not line:
                continue

            # Check JSON first
            if line.startswith("{"):
                try:
                    json.loads(line)
                    return LogType.JSON
                except json.JSONDecodeError:
                    pass

            # Check CEF
            if line.startswith("CEF:"):
                return LogType.CEF

            # Check patterns
            for log_type, pattern in self.PATTERNS.items():
                if log_type in [LogType.JSON, LogType.CEF]:
                    continue
                if pattern.match(line):
                    return log_type

        return LogType.CUSTOM

    def parse_line(
        self,
        line: str,
        log_type: Optional[LogType] = None,
        source: str = "",
    ) -> ParsedLogEntry:
        """Parse a single log line."""
        entry = ParsedLogEntry(raw=line, source=source)
        line = line.strip()

        if not line:
            return entry

        # Auto-detect type if not specified
        if log_type is None:
            log_type = self.detect_log_type([line])

        entry.log_type = log_type

        # Parse based on type
        if log_type == LogType.JSON:
            self._parse_json(line, entry)
        elif log_type == LogType.CEF:
            self._parse_cef(line, entry)
        elif log_type in self.PATTERNS:
            self._parse_structured(line, entry, self.PATTERNS[log_type])
        else:
            entry.message = line

        # Detect security events
        self._classify_security_event(entry)

        return entry

    def _parse_json(self, line: str, entry: ParsedLogEntry):
        """Parse JSON log line."""
        try:
            data = json.loads(line)

            # Extract common fields
            for ts_field in ["timestamp", "@timestamp", "time", "datetime", "date"]:
                if ts_field in data:
                    entry.timestamp = self._parse_timestamp(str(data[ts_field]))
                    break

            for msg_field in ["message", "msg", "log", "text"]:
                if msg_field in data:
                    entry.message = str(data[msg_field])
                    break

            for user_field in ["user", "username", "user_name", "account"]:
                if user_field in data:
                    entry.username = str(data[user_field])
                    break

            for ip_field in ["source_ip", "src_ip", "client_ip", "ip", "remote_addr"]:
                if ip_field in data:
                    entry.source_ip = str(data[ip_field])
                    break

            entry.metadata = data

        except json.JSONDecodeError:
            entry.message = line

    def _parse_cef(self, line: str, entry: ParsedLogEntry):
        """Parse CEF format log."""
        match = self.PATTERNS[LogType.CEF].match(line)
        if match:
            groups = match.groupdict()
            entry.message = groups.get("name", "")
            entry.severity = groups.get("severity", "")

            # Parse extension
            extension = groups.get("extension", "")
            ext_parts = re.findall(r"(\w+)=([^\s]+(?:\s+[^\s=]+)*?)(?=\s+\w+=|$)", extension)
            for key, value in ext_parts:
                entry.metadata[key] = value.strip()

                # Map common CEF fields
                if key in ["src", "sourceAddress"]:
                    entry.source_ip = value.strip()
                elif key in ["dst", "destinationAddress"]:
                    entry.destination_ip = value.strip()
                elif key in ["suser", "sourceUser"]:
                    entry.username = value.strip()

    def _parse_structured(self, line: str, entry: ParsedLogEntry, pattern: re.Pattern):
        """Parse with structured pattern."""
        match = pattern.match(line)
        if match:
            groups = match.groupdict()

            if "timestamp" in groups:
                entry.timestamp = self._parse_timestamp(groups["timestamp"])
            if "message" in groups:
                entry.message = groups["message"]
            if "host" in groups:
                entry.metadata["host"] = groups["host"]
            if "process" in groups:
                entry.metadata["process"] = groups["process"]
            if "ip" in groups:
                entry.source_ip = groups["ip"]
            if "user" in groups:
                entry.username = groups["user"]
            if "status" in groups:
                entry.status = groups["status"]
            if "level" in groups:
                entry.severity = groups["level"].lower()
            if "event_id" in groups:
                entry.metadata["event_id"] = int(groups["event_id"])

            entry.metadata.update(groups)
        else:
            entry.message = line

    def _classify_security_event(self, entry: ParsedLogEntry):
        """Classify the security event type."""
        text = entry.message + " " + entry.raw

        # Check Windows Event IDs first
        event_id = entry.metadata.get("event_id")
        if event_id and event_id in self.WINDOWS_SECURITY_EVENTS:
            name, event_type = self.WINDOWS_SECURITY_EVENTS[event_id]
            entry.event_type = event_type
            return

        # Check security patterns
        for category, patterns in self._compiled_security_patterns.items():
            for pattern, group_names in patterns:
                match = pattern.search(text)
                if match:
                    # Map category to event type
                    category_to_type = {
                        "login_success": SecurityEventType.LOGIN_SUCCESS,
                        "login_failure": SecurityEventType.LOGIN_FAILURE,
                        "sudo": SecurityEventType.PRIVILEGE_CHANGE,
                        "privilege_escalation": SecurityEventType.PRIVILEGE_CHANGE,
                        "brute_force": SecurityEventType.LOGIN_FAILURE,
                        "firewall_block": SecurityEventType.FIREWALL_BLOCK,
                        "sql_injection": SecurityEventType.POLICY_VIOLATION,
                        "xss_attempt": SecurityEventType.POLICY_VIOLATION,
                        "path_traversal": SecurityEventType.POLICY_VIOLATION,
                        "command_injection": SecurityEventType.POLICY_VIOLATION,
                    }

                    entry.event_type = category_to_type.get(category, SecurityEventType.UNKNOWN)

                    # Extract matched groups
                    groups = match.groups()
                    for i, name in enumerate(group_names):
                        if i < len(groups):
                            if name == "username" and not entry.username:
                                entry.username = groups[i]
                            elif name == "ip" and not entry.source_ip:
                                entry.source_ip = groups[i]
                            elif name == "src_ip" and not entry.source_ip:
                                entry.source_ip = groups[i]
                            elif name == "dst_ip" and not entry.destination_ip:
                                entry.destination_ip = groups[i]
                            else:
                                entry.metadata[name] = groups[i]

                    return

    def _parse_timestamp(self, ts_str: str) -> Optional[datetime]:
        """Parse various timestamp formats."""
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%d/%b/%Y:%H:%M:%S %z",
            "%b %d %H:%M:%S",
            "%b  %d %H:%M:%S",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(ts_str.strip(), fmt)
                if dt.year == 1900:
                    dt = dt.replace(year=datetime.now().year)
                return dt
            except ValueError:
                continue

        return None

    def parse_file(
        self,
        file_path: str,
        log_type: Optional[LogType] = None,
        max_lines: Optional[int] = None,
    ) -> Generator[ParsedLogEntry, None, None]:
        """Parse a log file."""
        path = Path(file_path)

        if not path.exists():
            logger.error(f"Log file not found: {file_path}")
            return

        # Handle gzipped files
        opener = gzip.open if path.suffix == ".gz" else open

        try:
            with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
                # Auto-detect type from first lines
                if log_type is None:
                    sample = []
                    for _ in range(10):
                        line = f.readline()
                        if line:
                            sample.append(line)
                    log_type = self.detect_log_type(sample)
                    f.seek(0)  # Reset to beginning

                for i, line in enumerate(f):
                    if max_lines and i >= max_lines:
                        break

                    entry = self.parse_line(line, log_type, source=str(path))
                    yield entry

        except Exception as e:
            logger.error(f"Error parsing log file: {e}")

    def analyze(
        self,
        entries: List[ParsedLogEntry],
        detect_anomalies: bool = True,
    ) -> LogAnalysisReport:
        """Perform comprehensive security analysis."""
        report = LogAnalysisReport(total_entries=len(entries))

        if not entries:
            return report

        # Time range
        timestamps = [e.timestamp for e in entries if e.timestamp]
        if timestamps:
            report.time_range = (min(timestamps), max(timestamps))

        # Count by type
        type_counter = Counter(e.event_type.value for e in entries)
        report.entries_by_type = dict(type_counter)

        # Count by severity
        severity_counter = Counter(e.severity for e in entries)
        report.entries_by_severity = dict(severity_counter)

        # Authentication statistics
        report.failed_logins = type_counter.get(SecurityEventType.LOGIN_FAILURE.value, 0)
        report.successful_logins = type_counter.get(SecurityEventType.LOGIN_SUCCESS.value, 0)

        # Top source IPs
        ip_counter = Counter(e.source_ip for e in entries if e.source_ip)
        report.top_source_ips = ip_counter.most_common(20)

        # Top users
        user_counter = Counter(e.username for e in entries if e.username)
        report.top_users = user_counter.most_common(20)

        # Generate security insights
        report.security_insights = self._generate_insights(entries)

        # Detect anomalies
        if detect_anomalies:
            report.anomalies = self._detect_anomalies(entries)

        return report

    def _generate_insights(self, entries: List[ParsedLogEntry]) -> List[SecurityInsight]:
        """Generate security insights from log entries."""
        insights = []

        # Brute force detection
        failed_by_ip = defaultdict(list)
        for e in entries:
            if e.event_type == SecurityEventType.LOGIN_FAILURE and e.source_ip:
                failed_by_ip[e.source_ip].append(e)

        for ip, failures in failed_by_ip.items():
            if len(failures) >= 5:
                insights.append(SecurityInsight(
                    insight_type="brute_force",
                    title=f"Potential Brute Force Attack from {ip}",
                    description=f"{len(failures)} failed login attempts detected",
                    severity="high",
                    count=len(failures),
                    examples=[f.username for f in failures[:5]],
                    recommendations=[
                        f"Consider blocking IP {ip}",
                        "Implement account lockout policy",
                        "Enable MFA for affected accounts",
                    ],
                    related_entries=failures[:10],
                ))

        # Privilege escalation detection
        priv_changes = [e for e in entries if e.event_type == SecurityEventType.PRIVILEGE_CHANGE]
        if priv_changes:
            insights.append(SecurityInsight(
                insight_type="privilege_changes",
                title="Privilege Escalation Events Detected",
                description=f"{len(priv_changes)} privilege change events found",
                severity="medium",
                count=len(priv_changes),
                examples=[e.message[:100] for e in priv_changes[:5]],
                recommendations=[
                    "Review privilege escalation events for unauthorized access",
                    "Verify sudo/su usage is legitimate",
                ],
            ))

        # Web attack detection
        attack_types = defaultdict(list)
        for e in entries:
            if e.event_type == SecurityEventType.POLICY_VIOLATION:
                # Classify attack type
                text = e.message.lower()
                if any(x in text for x in ["union select", "drop table", "';--"]):
                    attack_types["sql_injection"].append(e)
                elif any(x in text for x in ["<script", "javascript:"]):
                    attack_types["xss"].append(e)
                elif ".." in text:
                    attack_types["path_traversal"].append(e)
                else:
                    attack_types["other"].append(e)

        for attack_type, events in attack_types.items():
            if events:
                insights.append(SecurityInsight(
                    insight_type=f"web_attack_{attack_type}",
                    title=f"Web Attack Detected: {attack_type.replace('_', ' ').title()}",
                    description=f"{len(events)} {attack_type} attempts detected",
                    severity="high",
                    count=len(events),
                    examples=[e.source_ip for e in events[:5]],
                    recommendations=[
                        "Review WAF rules",
                        "Block malicious source IPs",
                        "Update input validation",
                    ],
                ))

        return insights

    def _detect_anomalies(self, entries: List[ParsedLogEntry]) -> List[Dict[str, Any]]:
        """Detect statistical anomalies."""
        anomalies = []

        # Time-based anomaly detection
        if len(entries) < 100:
            return anomalies

        entries_with_ts = [e for e in entries if e.timestamp]
        if not entries_with_ts:
            return anomalies

        # Group by hour
        hourly_counts = Counter()
        for e in entries_with_ts:
            hour_key = e.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_counts[hour_key] += 1

        if len(hourly_counts) > 2:
            values = list(hourly_counts.values())
            avg = sum(values) / len(values)
            std_dev = (sum((x - avg) ** 2 for x in values) / len(values)) ** 0.5

            # Find hours with unusual activity
            for hour, count in hourly_counts.items():
                if std_dev > 0:
                    z_score = (count - avg) / std_dev
                    if abs(z_score) > 2:
                        anomalies.append({
                            "type": "volume_anomaly",
                            "time": hour,
                            "count": count,
                            "average": avg,
                            "z_score": z_score,
                            "description": f"Unusual activity volume at {hour}: {count} events (avg: {avg:.0f})",
                        })

        return anomalies

    def search(
        self,
        entries: List[ParsedLogEntry],
        query: str = "",
        event_type: Optional[SecurityEventType] = None,
        source_ip: Optional[str] = None,
        username: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[ParsedLogEntry]:
        """Search and filter log entries."""
        results = []

        for entry in entries:
            # Event type filter
            if event_type and entry.event_type != event_type:
                continue

            # Source IP filter
            if source_ip and entry.source_ip != source_ip:
                continue

            # Username filter
            if username and entry.username != username:
                continue

            # Time filters
            if entry.timestamp:
                if start_time and entry.timestamp < start_time:
                    continue
                if end_time and entry.timestamp > end_time:
                    continue

            # Query filter
            if query:
                query_lower = query.lower()
                if query_lower not in entry.raw.lower() and query_lower not in entry.message.lower():
                    continue

            results.append(entry)

        return results


# Global instance
security_log_parser = SecurityLogParser()

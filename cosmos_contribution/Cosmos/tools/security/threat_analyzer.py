"""
cosmos Threat Analyzer

"Good news, everyone! I can identify threats before they become catastrophes!"

Threat intelligence and indicator analysis for security research.
"""

import re
import hashlib
import asyncio
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from loguru import logger


class ThreatType(Enum):
    """Types of threats."""
    IP_ADDRESS = "ip_address"
    DOMAIN = "domain"
    URL = "url"
    FILE_HASH = "file_hash"
    EMAIL = "email"
    CVE = "cve"
    MALWARE = "malware"
    CAMPAIGN = "campaign"


class ThreatSeverity(Enum):
    """Threat severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    UNKNOWN = "unknown"


@dataclass
class ThreatIndicator:
    """A threat indicator (IOC)."""
    indicator: str
    type: ThreatType
    severity: ThreatSeverity = ThreatSeverity.UNKNOWN
    description: str = ""
    tags: List[str] = field(default_factory=list)
    source: str = ""
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    confidence: float = 0.0  # 0-1
    related_indicators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatReport:
    """Threat analysis report."""
    indicators: List[ThreatIndicator] = field(default_factory=list)
    analysis_time: datetime = field(default_factory=datetime.now)
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)


class ThreatAnalyzer:
    """
    Threat intelligence analyzer.

    Capabilities:
    - IOC extraction and classification
    - Known malware detection
    - Domain/IP reputation checking
    - File hash analysis
    - CVE lookups
    """

    # Regex patterns for IOC extraction
    PATTERNS = {
        "ip_v4": re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"),
        "ip_v6": re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"),
        "domain": re.compile(r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b"),
        "url": re.compile(r"https?://[^\s<>\"']+"),
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "md5": re.compile(r"\b[a-fA-F0-9]{32}\b"),
        "sha1": re.compile(r"\b[a-fA-F0-9]{40}\b"),
        "sha256": re.compile(r"\b[a-fA-F0-9]{64}\b"),
        "cve": re.compile(r"CVE-\d{4}-\d{4,7}", re.I),
    }

    # Known malicious patterns (simplified - real implementation would use threat feeds)
    KNOWN_MALICIOUS = {
        "domains": {
            "malware-domain.com", "phishing-site.net", "c2server.org",
        },
        "ip_ranges": [
            # Example malicious IP ranges (simplified)
            ("185.220.100.0", "185.220.100.255"),  # Tor exit nodes (example)
        ],
    }

    # Suspicious TLD list
    SUSPICIOUS_TLDS = {
        "tk", "ml", "ga", "cf", "gq",  # Free TLDs often abused
        "xyz", "top", "club", "work", "click",  # Commonly abused
    }

    def __init__(self, data_dir: str = "./data/threat_intel"):
        """Initialize threat analyzer."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Local threat database
        self._threat_db: Dict[str, ThreatIndicator] = {}

    def extract_iocs(self, text: str) -> List[ThreatIndicator]:
        """
        Extract Indicators of Compromise from text.

        Args:
            text: Text to analyze

        Returns:
            List of extracted IOCs
        """
        indicators = []
        seen: Set[str] = set()

        # Extract IPs
        for match in self.PATTERNS["ip_v4"].finditer(text):
            ip = match.group()
            if ip not in seen:
                seen.add(ip)
                indicators.append(ThreatIndicator(
                    indicator=ip,
                    type=ThreatType.IP_ADDRESS,
                ))

        for match in self.PATTERNS["ip_v6"].finditer(text):
            ip = match.group()
            if ip not in seen:
                seen.add(ip)
                indicators.append(ThreatIndicator(
                    indicator=ip,
                    type=ThreatType.IP_ADDRESS,
                ))

        # Extract domains
        for match in self.PATTERNS["domain"].finditer(text):
            domain = match.group().lower()
            if domain not in seen:
                seen.add(domain)
                indicators.append(ThreatIndicator(
                    indicator=domain,
                    type=ThreatType.DOMAIN,
                ))

        # Extract URLs
        for match in self.PATTERNS["url"].finditer(text):
            url = match.group()
            if url not in seen:
                seen.add(url)
                indicators.append(ThreatIndicator(
                    indicator=url,
                    type=ThreatType.URL,
                ))

        # Extract emails
        for match in self.PATTERNS["email"].finditer(text):
            email = match.group().lower()
            if email not in seen:
                seen.add(email)
                indicators.append(ThreatIndicator(
                    indicator=email,
                    type=ThreatType.EMAIL,
                ))

        # Extract file hashes
        for hash_type, pattern in [("sha256", self.PATTERNS["sha256"]),
                                    ("sha1", self.PATTERNS["sha1"]),
                                    ("md5", self.PATTERNS["md5"])]:
            for match in pattern.finditer(text):
                hash_val = match.group().lower()
                if hash_val not in seen:
                    seen.add(hash_val)
                    indicators.append(ThreatIndicator(
                        indicator=hash_val,
                        type=ThreatType.FILE_HASH,
                        metadata={"hash_type": hash_type},
                    ))

        # Extract CVEs
        for match in self.PATTERNS["cve"].finditer(text):
            cve = match.group().upper()
            if cve not in seen:
                seen.add(cve)
                indicators.append(ThreatIndicator(
                    indicator=cve,
                    type=ThreatType.CVE,
                ))

        return indicators

    def analyze_indicator(self, indicator: ThreatIndicator) -> ThreatIndicator:
        """
        Analyze a single indicator for threats.

        Args:
            indicator: The indicator to analyze

        Returns:
            Updated indicator with analysis results
        """
        # Check against local database
        if indicator.indicator in self._threat_db:
            known = self._threat_db[indicator.indicator]
            indicator.severity = known.severity
            indicator.description = known.description
            indicator.tags = known.tags
            indicator.confidence = known.confidence
            return indicator

        # Type-specific analysis
        if indicator.type == ThreatType.IP_ADDRESS:
            self._analyze_ip(indicator)
        elif indicator.type == ThreatType.DOMAIN:
            self._analyze_domain(indicator)
        elif indicator.type == ThreatType.URL:
            self._analyze_url(indicator)
        elif indicator.type == ThreatType.FILE_HASH:
            self._analyze_hash(indicator)
        elif indicator.type == ThreatType.CVE:
            self._analyze_cve(indicator)

        return indicator

    def _analyze_ip(self, indicator: ThreatIndicator):
        """Analyze IP address indicator."""
        ip = indicator.indicator

        # Check private ranges
        try:
            from ipaddress import ip_address as parse_ip
            parsed = parse_ip(ip)

            if parsed.is_private:
                indicator.severity = ThreatSeverity.INFO
                indicator.description = "Private IP address"
                indicator.tags.append("private")
                return

            if parsed.is_loopback:
                indicator.severity = ThreatSeverity.INFO
                indicator.description = "Loopback address"
                indicator.tags.append("loopback")
                return

            if parsed.is_reserved:
                indicator.severity = ThreatSeverity.INFO
                indicator.description = "Reserved IP address"
                indicator.tags.append("reserved")
                return

        except ValueError:
            pass

        # Check against known malicious ranges
        for start, end in self.KNOWN_MALICIOUS["ip_ranges"]:
            if self._ip_in_range(ip, start, end):
                indicator.severity = ThreatSeverity.HIGH
                indicator.description = "IP in known malicious range"
                indicator.tags.append("malicious")
                indicator.confidence = 0.8
                return

        # Default - unknown
        indicator.severity = ThreatSeverity.UNKNOWN
        indicator.description = "External IP - requires further analysis"

    def _analyze_domain(self, indicator: ThreatIndicator):
        """Analyze domain indicator."""
        domain = indicator.indicator.lower()

        # Check against known malicious
        if domain in self.KNOWN_MALICIOUS["domains"]:
            indicator.severity = ThreatSeverity.CRITICAL
            indicator.description = "Known malicious domain"
            indicator.tags.append("malicious")
            indicator.confidence = 0.95
            return

        # Check TLD
        tld = domain.split(".")[-1] if "." in domain else ""
        if tld in self.SUSPICIOUS_TLDS:
            indicator.severity = ThreatSeverity.MEDIUM
            indicator.description = f"Suspicious TLD ({tld})"
            indicator.tags.append("suspicious_tld")
            indicator.confidence = 0.5
            return

        # Check for suspicious patterns
        suspicious_patterns = [
            (r"^[a-z0-9]{20,}\.", "Long random subdomain"),
            (r"\d{4,}", "Many consecutive digits"),
            (r"paypal|amazon|microsoft|google|apple", "Brand impersonation possible"),
            (r"login|signin|verify|secure|account", "Credential harvesting keywords"),
        ]

        for pattern, desc in suspicious_patterns:
            if re.search(pattern, domain, re.I):
                if indicator.severity == ThreatSeverity.UNKNOWN:
                    indicator.severity = ThreatSeverity.LOW
                indicator.description = desc
                indicator.tags.append("suspicious_pattern")
                indicator.confidence = 0.3
                return

        indicator.severity = ThreatSeverity.UNKNOWN

    def _analyze_url(self, indicator: ThreatIndicator):
        """Analyze URL indicator."""
        url = indicator.indicator

        # Check for data exfiltration patterns
        if len(url) > 500:
            indicator.severity = ThreatSeverity.MEDIUM
            indicator.description = "Unusually long URL - possible data exfiltration"
            indicator.tags.append("long_url")
            indicator.confidence = 0.4
            return

        # Check for encoded payloads
        encoded_patterns = [
            (r"base64[,=]", "Base64 encoded data"),
            (r"javascript:", "JavaScript protocol"),
            (r"data:", "Data URI scheme"),
            (r"%3Cscript", "Encoded script tag"),
        ]

        for pattern, desc in encoded_patterns:
            if re.search(pattern, url, re.I):
                indicator.severity = ThreatSeverity.HIGH
                indicator.description = desc
                indicator.tags.append("encoded_payload")
                indicator.confidence = 0.7
                return

        # Check for suspicious paths
        suspicious_paths = [
            (r"/wp-admin/", "WordPress admin path"),
            (r"/phpmyadmin/", "phpMyAdmin path"),
            (r"\.php\?", "PHP with parameters"),
            (r"/cgi-bin/", "CGI path"),
        ]

        for pattern, desc in suspicious_paths:
            if re.search(pattern, url, re.I):
                indicator.severity = ThreatSeverity.LOW
                indicator.description = f"Sensitive path: {desc}"
                indicator.tags.append("sensitive_path")
                indicator.confidence = 0.3

    def _analyze_hash(self, indicator: ThreatIndicator):
        """Analyze file hash indicator."""
        # In real implementation, would check against VirusTotal, etc.
        indicator.severity = ThreatSeverity.UNKNOWN
        indicator.description = "File hash - check against threat intelligence feeds"
        indicator.tags.append(indicator.metadata.get("hash_type", "hash"))

    def _analyze_cve(self, indicator: ThreatIndicator):
        """Analyze CVE indicator."""
        cve = indicator.indicator

        # Extract year
        match = re.match(r"CVE-(\d{4})-", cve)
        if match:
            year = int(match.group(1))
            current_year = datetime.now().year

            if current_year - year <= 1:
                indicator.severity = ThreatSeverity.HIGH
                indicator.description = "Recent CVE - may have active exploits"
            elif current_year - year <= 3:
                indicator.severity = ThreatSeverity.MEDIUM
                indicator.description = "Relatively recent CVE"
            else:
                indicator.severity = ThreatSeverity.LOW
                indicator.description = "Older CVE - check if patched"

            indicator.tags.append(f"cve_{year}")

    def _ip_in_range(self, ip: str, start: str, end: str) -> bool:
        """Check if IP is in range."""
        try:
            from ipaddress import ip_address as parse_ip
            ip_int = int(parse_ip(ip))
            start_int = int(parse_ip(start))
            end_int = int(parse_ip(end))
            return start_int <= ip_int <= end_int
        except ValueError:
            return False

    async def analyze_text(self, text: str) -> ThreatReport:
        """
        Analyze text for threat indicators.

        Args:
            text: Text to analyze

        Returns:
            ThreatReport with findings
        """
        report = ThreatReport()

        # Extract IOCs
        indicators = self.extract_iocs(text)

        # Analyze each indicator
        for indicator in indicators:
            self.analyze_indicator(indicator)
            report.indicators.append(indicator)

        # Generate summary
        critical_count = sum(1 for i in report.indicators if i.severity == ThreatSeverity.CRITICAL)
        high_count = sum(1 for i in report.indicators if i.severity == ThreatSeverity.HIGH)

        report.summary = (
            f"Found {len(report.indicators)} indicators: "
            f"{critical_count} critical, {high_count} high severity"
        )

        # Generate recommendations
        if critical_count > 0:
            report.recommendations.append("Immediate action required - critical threats detected")
        if high_count > 0:
            report.recommendations.append("Block high-severity indicators at perimeter")

        # Type-specific recommendations
        types_found = set(i.type for i in report.indicators)
        if ThreatType.IP_ADDRESS in types_found:
            report.recommendations.append("Consider blocking malicious IPs at firewall")
        if ThreatType.DOMAIN in types_found:
            report.recommendations.append("Add suspicious domains to DNS blocklist")
        if ThreatType.FILE_HASH in types_found:
            report.recommendations.append("Scan systems for files matching suspicious hashes")

        return report

    def add_to_database(self, indicator: ThreatIndicator):
        """Add indicator to local threat database."""
        self._threat_db[indicator.indicator] = indicator
        logger.info(f"Added {indicator.type.value} to threat database: {indicator.indicator}")

    def export_indicators(self, indicators: List[ThreatIndicator], format: str = "csv") -> str:
        """Export indicators in various formats."""
        if format == "csv":
            lines = ["indicator,type,severity,description,tags"]
            for i in indicators:
                tags = "|".join(i.tags)
                lines.append(f'"{i.indicator}",{i.type.value},{i.severity.value},"{i.description}","{tags}"')
            return "\n".join(lines)

        elif format == "stix":
            # Simplified STIX format
            stix = {
                "type": "bundle",
                "objects": [],
            }
            for i in indicators:
                stix["objects"].append({
                    "type": "indicator",
                    "pattern": f"[{i.type.value}:value = '{i.indicator}']",
                    "valid_from": datetime.now().isoformat(),
                })
            import json
            return json.dumps(stix, indent=2)

        elif format == "json":
            import json
            return json.dumps([
                {
                    "indicator": i.indicator,
                    "type": i.type.value,
                    "severity": i.severity.value,
                    "description": i.description,
                    "tags": i.tags,
                    "confidence": i.confidence,
                }
                for i in indicators
            ], indent=2)

        else:
            # Plain text
            lines = []
            for i in indicators:
                lines.append(f"[{i.severity.value.upper()}] {i.type.value}: {i.indicator}")
                if i.description:
                    lines.append(f"  Description: {i.description}")
            return "\n".join(lines)


# Global instance
threat_analyzer = ThreatAnalyzer()

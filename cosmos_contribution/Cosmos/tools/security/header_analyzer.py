"""
cosmos Email Header Analyzer

"Good news, everyone! I can trace emails back through the tubes of the internet!"

Comprehensive email header analysis for security investigation.
"""

import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from ipaddress import ip_address, IPv4Address, IPv6Address
from loguru import logger


class AuthenticationResult(Enum):
    """Email authentication results."""
    PASS = "pass"
    FAIL = "fail"
    SOFTFAIL = "softfail"
    NEUTRAL = "neutral"
    NONE = "none"
    TEMPERROR = "temperror"
    PERMERROR = "permerror"


class ThreatLevel(Enum):
    """Threat assessment level."""
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    LIKELY_PHISHING = "likely_phishing"
    MALICIOUS = "malicious"


@dataclass
class HopInfo:
    """Information about a single mail hop."""
    hop_number: int
    from_host: str
    by_host: str
    with_protocol: str
    timestamp: Optional[datetime] = None
    delay_seconds: float = 0
    ip_address: Optional[str] = None
    is_internal: bool = False


@dataclass
class AuthenticationStatus:
    """Email authentication status."""
    spf_result: AuthenticationResult = AuthenticationResult.NONE
    spf_domain: str = ""
    dkim_result: AuthenticationResult = AuthenticationResult.NONE
    dkim_domain: str = ""
    dkim_selector: str = ""
    dmarc_result: AuthenticationResult = AuthenticationResult.NONE
    dmarc_policy: str = ""
    arc_result: AuthenticationResult = AuthenticationResult.NONE


@dataclass
class EmailHeaderAnalysis:
    """Complete email header analysis."""
    # Basic info
    message_id: str = ""
    subject: str = ""
    from_address: str = ""
    from_display_name: str = ""
    to_addresses: List[str] = field(default_factory=list)
    date: Optional[datetime] = None
    reply_to: str = ""
    return_path: str = ""

    # Routing
    hops: List[HopInfo] = field(default_factory=list)
    total_delay_seconds: float = 0
    originating_ip: str = ""
    originating_country: str = ""

    # Authentication
    authentication: AuthenticationStatus = field(default_factory=AuthenticationStatus)

    # Analysis
    threat_level: ThreatLevel = ThreatLevel.SAFE
    threat_indicators: List[str] = field(default_factory=list)
    suspicious_headers: List[Dict[str, str]] = field(default_factory=list)

    # Raw data
    all_headers: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "subject": self.subject,
            "from": {
                "address": self.from_address,
                "display_name": self.from_display_name,
            },
            "to": self.to_addresses,
            "date": self.date.isoformat() if self.date else None,
            "reply_to": self.reply_to,
            "return_path": self.return_path,
            "routing": {
                "hops": [
                    {
                        "number": h.hop_number,
                        "from": h.from_host,
                        "by": h.by_host,
                        "protocol": h.with_protocol,
                        "timestamp": h.timestamp.isoformat() if h.timestamp else None,
                        "delay_seconds": h.delay_seconds,
                        "ip": h.ip_address,
                    }
                    for h in self.hops
                ],
                "total_delay_seconds": self.total_delay_seconds,
                "originating_ip": self.originating_ip,
            },
            "authentication": {
                "spf": {"result": self.authentication.spf_result.value, "domain": self.authentication.spf_domain},
                "dkim": {"result": self.authentication.dkim_result.value, "domain": self.authentication.dkim_domain},
                "dmarc": {"result": self.authentication.dmarc_result.value, "policy": self.authentication.dmarc_policy},
            },
            "threat_assessment": {
                "level": self.threat_level.value,
                "indicators": self.threat_indicators,
            },
        }


class HeaderAnalyzer:
    """
    Email header analyzer for security investigation.

    Analyzes:
    - Email routing and hops
    - SPF, DKIM, DMARC authentication
    - Suspicious patterns and indicators
    - Threat assessment
    """

    # Known free email providers
    FREE_EMAIL_PROVIDERS = {
        "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
        "protonmail.com", "mail.com", "aol.com", "icloud.com",
        "zoho.com", "gmx.com", "yandex.com",
    }

    # Suspicious header patterns
    SUSPICIOUS_PATTERNS = [
        (r"X-Spam-Status:\s*Yes", "Marked as spam"),
        (r"X-Spam-Score:\s*([5-9]|[1-9]\d)", "High spam score"),
        (r"X-Forefront-Antispam-Report.*SCL:\s*[6-9]", "Microsoft marked suspicious"),
        (r"X-MS-Exchange-Organization-SCL:\s*[6-9]", "Exchange high SCL"),
        (r"Authentication-Results.*spf=fail", "SPF authentication failed"),
        (r"Authentication-Results.*dkim=fail", "DKIM authentication failed"),
        (r"Authentication-Results.*dmarc=fail", "DMARC authentication failed"),
    ]

    # Phishing indicators
    PHISHING_PATTERNS = [
        (r"reply-to.*differs.*from", "Reply-to differs from sender"),
        (r"X-Originating-IP.*\[(\d{1,3}\.){3}\d{1,3}\]", "Exposed originating IP"),
    ]

    def __init__(self):
        """Initialize header analyzer."""
        pass

    def parse_headers(self, raw_headers: str) -> Dict[str, List[str]]:
        """Parse raw email headers into dictionary."""
        headers: Dict[str, List[str]] = {}
        current_header = None
        current_value = []

        for line in raw_headers.split("\n"):
            # Check for header continuation
            if line and line[0] in " \t":
                if current_header:
                    current_value.append(line.strip())
                continue

            # Save previous header
            if current_header:
                value = " ".join(current_value)
                if current_header in headers:
                    headers[current_header].append(value)
                else:
                    headers[current_header] = [value]

            # Parse new header
            if ":" in line:
                parts = line.split(":", 1)
                current_header = parts[0].strip()
                current_value = [parts[1].strip()] if len(parts) > 1 else []
            else:
                current_header = None
                current_value = []

        # Don't forget the last header
        if current_header:
            value = " ".join(current_value)
            if current_header in headers:
                headers[current_header].append(value)
            else:
                headers[current_header] = [value]

        return headers

    def analyze(self, raw_headers: str) -> EmailHeaderAnalysis:
        """
        Perform comprehensive header analysis.

        Args:
            raw_headers: Raw email headers as string

        Returns:
            EmailHeaderAnalysis with complete analysis
        """
        analysis = EmailHeaderAnalysis()
        headers = self.parse_headers(raw_headers)
        analysis.all_headers = headers

        # Extract basic information
        self._extract_basic_info(headers, analysis)

        # Parse routing information
        self._parse_received_headers(headers, analysis)

        # Check authentication
        self._check_authentication(headers, analysis)

        # Assess threats
        self._assess_threats(headers, analysis)

        return analysis

    def _extract_basic_info(self, headers: Dict[str, List[str]], analysis: EmailHeaderAnalysis):
        """Extract basic email information."""
        # Message ID
        if "Message-ID" in headers:
            analysis.message_id = headers["Message-ID"][0].strip("<>")
        elif "Message-Id" in headers:
            analysis.message_id = headers["Message-Id"][0].strip("<>")

        # Subject
        if "Subject" in headers:
            analysis.subject = headers["Subject"][0]

        # From address
        if "From" in headers:
            from_header = headers["From"][0]
            # Parse "Display Name <email@domain.com>" format
            match = re.search(r"([^<]+)?<([^>]+)>", from_header)
            if match:
                analysis.from_display_name = match.group(1).strip().strip('"') if match.group(1) else ""
                analysis.from_address = match.group(2)
            else:
                analysis.from_address = from_header.strip()

        # To addresses
        if "To" in headers:
            to_header = headers["To"][0]
            # Extract all email addresses
            analysis.to_addresses = re.findall(r"[\w\.-]+@[\w\.-]+", to_header)

        # Date
        if "Date" in headers:
            analysis.date = self._parse_date(headers["Date"][0])

        # Reply-To
        if "Reply-To" in headers:
            match = re.search(r"<([^>]+)>|([^\s,]+@[^\s,]+)", headers["Reply-To"][0])
            if match:
                analysis.reply_to = match.group(1) or match.group(2)

        # Return-Path
        if "Return-Path" in headers:
            match = re.search(r"<([^>]*)>", headers["Return-Path"][0])
            if match:
                analysis.return_path = match.group(1)

    def _parse_received_headers(self, headers: Dict[str, List[str]], analysis: EmailHeaderAnalysis):
        """Parse Received headers to trace email path."""
        if "Received" not in headers:
            return

        received_headers = headers["Received"]
        hops = []
        previous_timestamp = None

        # Received headers are in reverse order (newest first)
        for i, received in enumerate(reversed(received_headers)):
            hop = HopInfo(hop_number=i + 1, from_host="", by_host="", with_protocol="")

            # Extract "from" host
            from_match = re.search(r"from\s+([^\s\(]+)", received, re.I)
            if from_match:
                hop.from_host = from_match.group(1)

            # Extract "by" host
            by_match = re.search(r"by\s+([^\s\(]+)", received, re.I)
            if by_match:
                hop.by_host = by_match.group(1)

            # Extract protocol
            with_match = re.search(r"with\s+(\S+)", received, re.I)
            if with_match:
                hop.with_protocol = with_match.group(1)

            # Extract IP address
            ip_match = re.search(r"\[(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\]", received)
            if ip_match:
                hop.ip_address = ip_match.group(1)

            # Extract timestamp
            # Common formats: "Mon, 1 Jan 2024 12:00:00 +0000" or similar
            date_patterns = [
                r";\s*(.+?(?:\+|-)\d{4})",
                r";\s*(.+?(?:GMT|UTC|EST|PST|PDT|EDT|CST|CDT|MST|MDT))",
                r";\s*(\d{1,2}\s+\w{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2})",
            ]
            for pattern in date_patterns:
                date_match = re.search(pattern, received)
                if date_match:
                    hop.timestamp = self._parse_date(date_match.group(1))
                    if hop.timestamp:
                        break

            # Calculate delay
            if hop.timestamp and previous_timestamp:
                hop.delay_seconds = (hop.timestamp - previous_timestamp).total_seconds()

            previous_timestamp = hop.timestamp

            # Check if internal hop
            if hop.from_host and any(
                internal in hop.from_host.lower()
                for internal in ["localhost", "127.0.0.1", "internal", "local"]
            ):
                hop.is_internal = True

            hops.append(hop)

        analysis.hops = hops

        # Calculate total delay
        if hops:
            analysis.total_delay_seconds = sum(h.delay_seconds for h in hops if h.delay_seconds > 0)

            # Find originating IP (first external hop)
            for hop in hops:
                if hop.ip_address and not hop.is_internal:
                    try:
                        ip = ip_address(hop.ip_address)
                        if not ip.is_private and not ip.is_loopback:
                            analysis.originating_ip = hop.ip_address
                            break
                    except ValueError:
                        pass

    def _check_authentication(self, headers: Dict[str, List[str]], analysis: EmailHeaderAnalysis):
        """Check email authentication (SPF, DKIM, DMARC)."""
        auth_results = []
        for key in ["Authentication-Results", "ARC-Authentication-Results"]:
            if key in headers:
                auth_results.extend(headers[key])

        for auth_result in auth_results:
            auth_lower = auth_result.lower()

            # SPF
            spf_match = re.search(r"spf=(\w+)", auth_lower)
            if spf_match:
                result = spf_match.group(1)
                try:
                    analysis.authentication.spf_result = AuthenticationResult(result)
                except ValueError:
                    analysis.authentication.spf_result = AuthenticationResult.NONE

                # Extract SPF domain
                domain_match = re.search(r"spf=\w+.*smtp\.mailfrom=([^\s;]+)", auth_lower)
                if domain_match:
                    analysis.authentication.spf_domain = domain_match.group(1)

            # DKIM
            dkim_match = re.search(r"dkim=(\w+)", auth_lower)
            if dkim_match:
                result = dkim_match.group(1)
                try:
                    analysis.authentication.dkim_result = AuthenticationResult(result)
                except ValueError:
                    analysis.authentication.dkim_result = AuthenticationResult.NONE

                # Extract DKIM domain
                domain_match = re.search(r"dkim=\w+.*header\.d=([^\s;]+)", auth_lower)
                if domain_match:
                    analysis.authentication.dkim_domain = domain_match.group(1)

            # DMARC
            dmarc_match = re.search(r"dmarc=(\w+)", auth_lower)
            if dmarc_match:
                result = dmarc_match.group(1)
                try:
                    analysis.authentication.dmarc_result = AuthenticationResult(result)
                except ValueError:
                    analysis.authentication.dmarc_result = AuthenticationResult.NONE

                # Extract DMARC policy
                policy_match = re.search(r"dmarc=\w+.*p=(\w+)", auth_lower)
                if policy_match:
                    analysis.authentication.dmarc_policy = policy_match.group(1)

    def _assess_threats(self, headers: Dict[str, List[str]], analysis: EmailHeaderAnalysis):
        """Assess threat level based on analysis."""
        indicators = []
        threat_score = 0

        # Check authentication failures
        if analysis.authentication.spf_result == AuthenticationResult.FAIL:
            indicators.append("SPF authentication failed")
            threat_score += 30

        if analysis.authentication.dkim_result == AuthenticationResult.FAIL:
            indicators.append("DKIM authentication failed")
            threat_score += 25

        if analysis.authentication.dmarc_result == AuthenticationResult.FAIL:
            indicators.append("DMARC authentication failed")
            threat_score += 30

        # Check for spoofing indicators
        if analysis.return_path and analysis.from_address:
            return_domain = analysis.return_path.split("@")[-1] if "@" in analysis.return_path else ""
            from_domain = analysis.from_address.split("@")[-1] if "@" in analysis.from_address else ""

            if return_domain and from_domain and return_domain.lower() != from_domain.lower():
                indicators.append(f"Return-Path domain ({return_domain}) differs from From domain ({from_domain})")
                threat_score += 20

        if analysis.reply_to and analysis.from_address:
            reply_domain = analysis.reply_to.split("@")[-1] if "@" in analysis.reply_to else ""
            from_domain = analysis.from_address.split("@")[-1] if "@" in analysis.from_address else ""

            if reply_domain and from_domain and reply_domain.lower() != from_domain.lower():
                indicators.append(f"Reply-To domain ({reply_domain}) differs from From domain ({from_domain})")
                threat_score += 15

        # Check suspicious header patterns
        raw_headers = "\n".join(
            f"{k}: {v}" for k, values in headers.items() for v in values
        )

        for pattern, description in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, raw_headers, re.I):
                indicators.append(description)
                threat_score += 10

        # Check for phishing patterns
        for pattern, description in self.PHISHING_PATTERNS:
            if re.search(pattern, raw_headers, re.I):
                indicators.append(description)
                threat_score += 20

        # Check for excessive hops (possible relay abuse)
        if len(analysis.hops) > 10:
            indicators.append(f"Unusually high number of mail hops ({len(analysis.hops)})")
            threat_score += 10

        # Check for long delays (possible queuing at malicious servers)
        if analysis.total_delay_seconds > 86400:  # More than 24 hours
            indicators.append(f"Long delivery delay ({analysis.total_delay_seconds / 3600:.1f} hours)")
            threat_score += 10

        # Store findings
        analysis.threat_indicators = indicators

        # Determine threat level
        if threat_score >= 70:
            analysis.threat_level = ThreatLevel.MALICIOUS
        elif threat_score >= 50:
            analysis.threat_level = ThreatLevel.LIKELY_PHISHING
        elif threat_score >= 25:
            analysis.threat_level = ThreatLevel.SUSPICIOUS
        else:
            analysis.threat_level = ThreatLevel.SAFE

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats."""
        from email.utils import parsedate_to_datetime

        try:
            return parsedate_to_datetime(date_str.strip())
        except Exception:
            pass

        # Try additional formats
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%d %b %Y %H:%M:%S %z",
            "%Y-%m-%d %H:%M:%S",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue

        return None

    def generate_report(self, analysis: EmailHeaderAnalysis) -> str:
        """Generate human-readable analysis report."""
        lines = [
            "=" * 60,
            "EMAIL HEADER ANALYSIS REPORT",
            "=" * 60,
            "",
            "BASIC INFORMATION",
            "-" * 40,
            f"From: {analysis.from_display_name} <{analysis.from_address}>",
            f"To: {', '.join(analysis.to_addresses)}",
            f"Subject: {analysis.subject}",
            f"Date: {analysis.date}",
            f"Message-ID: {analysis.message_id}",
            "",
        ]

        if analysis.reply_to:
            lines.append(f"Reply-To: {analysis.reply_to}")
        if analysis.return_path:
            lines.append(f"Return-Path: {analysis.return_path}")

        lines.extend([
            "",
            "ROUTING ANALYSIS",
            "-" * 40,
            f"Number of hops: {len(analysis.hops)}",
            f"Total delay: {analysis.total_delay_seconds:.0f} seconds",
            f"Originating IP: {analysis.originating_ip or 'Unknown'}",
            "",
        ])

        for hop in analysis.hops:
            lines.append(
                f"  Hop {hop.hop_number}: {hop.from_host} -> {hop.by_host} "
                f"({hop.with_protocol}) [{hop.ip_address or 'no IP'}]"
            )
            if hop.delay_seconds > 0:
                lines.append(f"    Delay: {hop.delay_seconds:.0f}s")

        lines.extend([
            "",
            "AUTHENTICATION",
            "-" * 40,
            f"SPF: {analysis.authentication.spf_result.value} ({analysis.authentication.spf_domain})",
            f"DKIM: {analysis.authentication.dkim_result.value} ({analysis.authentication.dkim_domain})",
            f"DMARC: {analysis.authentication.dmarc_result.value} ({analysis.authentication.dmarc_policy})",
            "",
            "THREAT ASSESSMENT",
            "-" * 40,
            f"Threat Level: {analysis.threat_level.value.upper()}",
            "",
        ])

        if analysis.threat_indicators:
            lines.append("Indicators:")
            for indicator in analysis.threat_indicators:
                lines.append(f"  - {indicator}")
        else:
            lines.append("No threat indicators found.")

        lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(lines)


# Global instance
header_analyzer = HeaderAnalyzer()

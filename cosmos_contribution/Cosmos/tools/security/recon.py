"""
cosmos Reconnaissance Engine

"Good news, everyone! I've built a reconnaissance tool for authorized security testing!"

IMPORTANT: Only use on systems you own or have explicit written permission to test.
"""

import asyncio
import socket
import ssl
import re
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger


class ReconType(Enum):
    """Types of reconnaissance."""
    PASSIVE = "passive"  # No direct interaction with target
    ACTIVE = "active"  # Direct probing (requires authorization)


@dataclass
class DNSRecord:
    """DNS record information."""
    record_type: str
    value: str
    ttl: int = 0


@dataclass
class SubdomainInfo:
    """Subdomain information."""
    subdomain: str
    ip_addresses: List[str] = field(default_factory=list)
    is_alive: bool = False


@dataclass
class ServiceInfo:
    """Service information from port scan."""
    port: int
    protocol: str
    service_name: str = ""
    version: str = ""
    banner: str = ""
    ssl_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconResult:
    """Complete reconnaissance result."""
    target: str
    scan_time: datetime = field(default_factory=datetime.now)
    recon_type: ReconType = ReconType.PASSIVE

    # DNS information
    ip_addresses: List[str] = field(default_factory=list)
    dns_records: List[DNSRecord] = field(default_factory=list)
    subdomains: List[SubdomainInfo] = field(default_factory=list)

    # WHOIS information
    whois_info: Dict[str, Any] = field(default_factory=dict)

    # Services
    services: List[ServiceInfo] = field(default_factory=list)

    # Technologies detected
    technologies: List[str] = field(default_factory=list)

    # Email addresses found
    emails: List[str] = field(default_factory=list)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "scan_time": self.scan_time.isoformat(),
            "recon_type": self.recon_type.value,
            "ip_addresses": self.ip_addresses,
            "dns_records": [
                {"type": r.record_type, "value": r.value, "ttl": r.ttl}
                for r in self.dns_records
            ],
            "subdomains": [
                {"subdomain": s.subdomain, "ips": s.ip_addresses, "alive": s.is_alive}
                for s in self.subdomains
            ],
            "whois": self.whois_info,
            "services": [
                {"port": s.port, "service": s.service_name, "version": s.version, "banner": s.banner}
                for s in self.services
            ],
            "technologies": self.technologies,
            "emails": self.emails,
        }


class ReconEngine:
    """
    Reconnaissance engine for authorized security testing.

    Capabilities:
    - DNS enumeration
    - Subdomain discovery
    - Service detection
    - Technology fingerprinting
    - OSINT gathering
    """

    # Common subdomains to check
    COMMON_SUBDOMAINS = [
        "www", "mail", "remote", "blog", "webmail", "server",
        "ns1", "ns2", "smtp", "secure", "vpn", "m", "shop",
        "ftp", "mail2", "test", "portal", "ns", "ww1", "host",
        "support", "dev", "web", "bbs", "ww42", "mx", "email",
        "cloud", "api", "staging", "beta", "admin", "login",
        "cdn", "static", "assets", "app", "mobile",
    ]

    # Technology signatures (simplified)
    TECH_SIGNATURES = {
        "WordPress": ["wp-content", "wp-includes", "WordPress"],
        "Drupal": ["Drupal", "sites/default/files"],
        "Joomla": ["Joomla", "/components/com_"],
        "Django": ["csrfmiddlewaretoken", "__admin__"],
        "Ruby on Rails": ["_session_id", "X-Rack-Cache"],
        "ASP.NET": ["__VIEWSTATE", "ASP.NET"],
        "nginx": ["nginx"],
        "Apache": ["Apache"],
        "IIS": ["IIS", "ASP.NET"],
        "Cloudflare": ["cloudflare", "cf-ray"],
        "AWS": ["amazonaws.com", "x-amz-"],
        "Google Cloud": ["googleapis.com"],
        "Azure": ["azure", "windows.net"],
    }

    def __init__(self):
        """Initialize recon engine."""
        pass

    async def scan(
        self,
        target: str,
        recon_type: ReconType = ReconType.PASSIVE,
        include_subdomains: bool = True,
        include_services: bool = False,
    ) -> ReconResult:
        """
        Perform reconnaissance on target.

        Args:
            target: Domain or IP to scan
            recon_type: Type of recon (passive/active)
            include_subdomains: Enumerate subdomains
            include_services: Scan for services (active only)

        Returns:
            ReconResult with findings
        """
        result = ReconResult(target=target, recon_type=recon_type)

        logger.info(f"Starting {recon_type.value} recon on {target}")

        # DNS resolution
        await self._resolve_dns(target, result)

        # DNS records
        await self._get_dns_records(target, result)

        # Subdomain enumeration
        if include_subdomains:
            await self._enumerate_subdomains(target, result)

        # Service scanning (active only)
        if include_services and recon_type == ReconType.ACTIVE:
            await self._scan_services(target, result)

        # Technology detection
        if recon_type == ReconType.ACTIVE:
            await self._detect_technologies(target, result)

        # WHOIS lookup
        await self._whois_lookup(target, result)

        logger.info(f"Recon complete: {len(result.subdomains)} subdomains, {len(result.services)} services")

        return result

    async def _resolve_dns(self, target: str, result: ReconResult):
        """Resolve target to IP addresses."""
        try:
            # A record
            ips = socket.gethostbyname_ex(target)[2]
            result.ip_addresses = ips
        except socket.gaierror:
            pass

    async def _get_dns_records(self, target: str, result: ReconResult):
        """Get DNS records for target."""
        try:
            import dns.resolver
        except ImportError:
            logger.debug("dnspython not installed")
            return

        record_types = ["A", "AAAA", "MX", "NS", "TXT", "SOA", "CNAME"]

        for record_type in record_types:
            try:
                answers = dns.resolver.resolve(target, record_type)
                for answer in answers:
                    result.dns_records.append(DNSRecord(
                        record_type=record_type,
                        value=str(answer),
                        ttl=answers.rrset.ttl if answers.rrset else 0,
                    ))
            except Exception:
                continue

    async def _enumerate_subdomains(self, target: str, result: ReconResult):
        """Enumerate subdomains."""
        found_subdomains: Set[str] = set()

        # Check common subdomains
        async def check_subdomain(subdomain: str):
            full_domain = f"{subdomain}.{target}"
            try:
                ips = socket.gethostbyname_ex(full_domain)[2]
                if ips:
                    return SubdomainInfo(
                        subdomain=full_domain,
                        ip_addresses=ips,
                        is_alive=True,
                    )
            except socket.gaierror:
                pass
            return None

        # Run checks concurrently
        tasks = [check_subdomain(sub) for sub in self.COMMON_SUBDOMAINS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for sub_result in results:
            if isinstance(sub_result, SubdomainInfo):
                result.subdomains.append(sub_result)
                found_subdomains.add(sub_result.subdomain)

    async def _scan_services(self, target: str, result: ReconResult):
        """Scan for services on common ports."""
        common_ports = [
            (21, "ftp"), (22, "ssh"), (23, "telnet"), (25, "smtp"),
            (53, "dns"), (80, "http"), (110, "pop3"), (143, "imap"),
            (443, "https"), (445, "smb"), (993, "imaps"), (995, "pop3s"),
            (3306, "mysql"), (3389, "rdp"), (5432, "postgresql"),
            (8080, "http-alt"), (8443, "https-alt"),
        ]

        # Resolve to IP first
        try:
            ip = socket.gethostbyname(target)
        except socket.gaierror:
            ip = target

        async def check_port(port: int, service_hint: str):
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(ip, port),
                    timeout=3.0,
                )

                service = ServiceInfo(
                    port=port,
                    protocol="tcp",
                    service_name=service_hint,
                )

                # Try to grab banner
                try:
                    writer.write(b"\r\n")
                    await writer.drain()
                    banner = await asyncio.wait_for(reader.read(1024), timeout=2.0)
                    service.banner = banner.decode("utf-8", errors="ignore").strip()[:200]

                    # Try to detect version from banner
                    version_match = re.search(r"[\d]+\.[\d]+(?:\.[\d]+)?", service.banner)
                    if version_match:
                        service.version = version_match.group()

                except Exception:
                    pass

                # Get SSL info for HTTPS ports
                if port in [443, 8443, 993, 995]:
                    try:
                        service.ssl_info = await self._get_ssl_info(ip, port)
                    except Exception:
                        pass

                writer.close()
                await writer.wait_closed()

                return service

            except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
                return None

        # Scan ports
        tasks = [check_port(port, service) for port, service in common_ports]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for service_result in results:
            if isinstance(service_result, ServiceInfo):
                result.services.append(service_result)

    async def _get_ssl_info(self, host: str, port: int) -> Dict[str, Any]:
        """Get SSL certificate information."""
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port, ssl=context),
                timeout=5.0,
            )

            ssl_object = writer.get_extra_info("ssl_object")
            if ssl_object:
                cert = ssl_object.getpeercert()
                if cert:
                    return {
                        "subject": dict(x[0] for x in cert.get("subject", [])),
                        "issuer": dict(x[0] for x in cert.get("issuer", [])),
                        "version": cert.get("version"),
                        "serial_number": cert.get("serialNumber"),
                        "not_before": cert.get("notBefore"),
                        "not_after": cert.get("notAfter"),
                        "san": cert.get("subjectAltName", []),
                    }

            writer.close()
            await writer.wait_closed()

        except Exception as e:
            logger.debug(f"SSL info error: {e}")

        return {}

    async def _detect_technologies(self, target: str, result: ReconResult):
        """Detect technologies used by target."""
        try:
            import httpx

            async with httpx.AsyncClient(verify=False) as client:
                for protocol in ["https", "http"]:
                    try:
                        response = await client.get(
                            f"{protocol}://{target}",
                            timeout=10.0,
                            follow_redirects=True,
                        )

                        # Check response headers and body
                        content = response.text.lower()
                        headers_str = str(response.headers).lower()

                        for tech, signatures in self.TECH_SIGNATURES.items():
                            for sig in signatures:
                                if sig.lower() in content or sig.lower() in headers_str:
                                    if tech not in result.technologies:
                                        result.technologies.append(tech)
                                    break

                        # Check server header
                        if "server" in response.headers:
                            server = response.headers["server"]
                            if server not in result.technologies:
                                result.technologies.append(f"Server: {server}")

                        break  # Success, no need to try HTTP

                    except Exception:
                        continue

        except ImportError:
            logger.debug("httpx not installed")
        except Exception as e:
            logger.debug(f"Technology detection error: {e}")

    async def _whois_lookup(self, target: str, result: ReconResult):
        """Perform WHOIS lookup."""
        try:
            import whois
            w = whois.whois(target)

            result.whois_info = {
                "registrar": w.registrar,
                "creation_date": str(w.creation_date) if w.creation_date else None,
                "expiration_date": str(w.expiration_date) if w.expiration_date else None,
                "name_servers": w.name_servers,
                "status": w.status,
                "org": w.org,
                "country": w.country,
            }

            # Extract emails if available
            if w.emails:
                result.emails = w.emails if isinstance(w.emails, list) else [w.emails]

        except ImportError:
            logger.debug("python-whois not installed")
        except Exception as e:
            logger.debug(f"WHOIS lookup error: {e}")

    def generate_report(self, result: ReconResult) -> str:
        """Generate human-readable recon report."""
        lines = [
            "=" * 60,
            f"RECONNAISSANCE REPORT: {result.target}",
            f"Type: {result.recon_type.value.upper()}",
            f"Time: {result.scan_time.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            "IP ADDRESSES",
            "-" * 40,
        ]

        for ip in result.ip_addresses:
            lines.append(f"  {ip}")

        lines.extend([
            "",
            "DNS RECORDS",
            "-" * 40,
        ])

        for record in result.dns_records:
            lines.append(f"  {record.record_type:6} {record.value}")

        if result.subdomains:
            lines.extend([
                "",
                f"SUBDOMAINS ({len(result.subdomains)} found)",
                "-" * 40,
            ])
            for sub in result.subdomains:
                status = "ALIVE" if sub.is_alive else "DOWN"
                lines.append(f"  {sub.subdomain} [{status}] {', '.join(sub.ip_addresses)}")

        if result.services:
            lines.extend([
                "",
                f"SERVICES ({len(result.services)} found)",
                "-" * 40,
            ])
            for svc in result.services:
                lines.append(f"  {svc.port}/tcp {svc.service_name} {svc.version}")
                if svc.banner:
                    lines.append(f"    Banner: {svc.banner[:60]}...")

        if result.technologies:
            lines.extend([
                "",
                "TECHNOLOGIES DETECTED",
                "-" * 40,
            ])
            for tech in result.technologies:
                lines.append(f"  - {tech}")

        if result.whois_info:
            lines.extend([
                "",
                "WHOIS INFORMATION",
                "-" * 40,
            ])
            for key, value in result.whois_info.items():
                if value:
                    lines.append(f"  {key}: {value}")

        lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(lines)


# Global instance
recon_engine = ReconEngine()

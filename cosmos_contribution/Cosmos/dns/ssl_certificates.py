"""
cosmos SSL Certificate Manager

"I've encrypted everything! Even my lab coat has a certificate!"

SSL/TLS certificate management with Let's Encrypt and monitoring.
"""

import asyncio
import ssl
import socket
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from loguru import logger

try:
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


class CertificateStatus(Enum):
    """Certificate status."""
    VALID = "valid"
    EXPIRING_SOON = "expiring_soon"
    EXPIRED = "expired"
    INVALID = "invalid"
    NOT_FOUND = "not_found"


@dataclass
class Certificate:
    """SSL certificate information."""
    domain: str
    issuer: str
    subject: str
    serial_number: str
    not_before: datetime
    not_after: datetime
    days_until_expiry: int
    status: CertificateStatus
    san: List[str] = field(default_factory=list)  # Subject Alternative Names
    fingerprint_sha256: str = ""
    is_wildcard: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "issuer": self.issuer,
            "subject": self.subject,
            "serial_number": self.serial_number,
            "not_before": self.not_before.isoformat(),
            "not_after": self.not_after.isoformat(),
            "days_until_expiry": self.days_until_expiry,
            "status": self.status.value,
            "san": self.san,
            "fingerprint_sha256": self.fingerprint_sha256,
            "is_wildcard": self.is_wildcard,
        }


@dataclass
class CertificateAlert:
    """Certificate expiry alert."""
    domain: str
    days_until_expiry: int
    severity: str  # critical, warning, info
    message: str
    created_at: datetime = field(default_factory=datetime.utcnow)


class SSLManager:
    """
    SSL/TLS certificate management for cosmos.

    Features:
    - Certificate inspection
    - Expiry monitoring
    - Let's Encrypt integration
    - Certificate chain validation
    - Automated renewal alerts
    """

    def __init__(
        self,
        storage_path: Path = None,
        warning_days: int = 30,
        critical_days: int = 7,
    ):
        self.storage_path = storage_path or Path("./data/certificates")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.warning_days = warning_days
        self.critical_days = critical_days
        self.monitored_domains: Dict[str, Certificate] = {}

    # =========================================================================
    # CERTIFICATE INSPECTION
    # =========================================================================

    def get_certificate(
        self,
        domain: str,
        port: int = 443,
        timeout: int = 10,
    ) -> Optional[Certificate]:
        """Get SSL certificate from a domain."""
        try:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

            with socket.create_connection((domain, port), timeout=timeout) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert_der = ssock.getpeercert(binary_form=True)

            if not HAS_CRYPTOGRAPHY:
                # Fallback to basic info without cryptography
                cert_dict = ssl.DER_cert_to_PEM_cert(cert_der)
                return self._parse_cert_basic(domain, cert_dict)

            cert = x509.load_der_x509_certificate(cert_der, default_backend())
            return self._parse_certificate(domain, cert)

        except ssl.SSLError as e:
            logger.error(f"SSL error for {domain}: {e}")
            return Certificate(
                domain=domain,
                issuer="",
                subject="",
                serial_number="",
                not_before=datetime.utcnow(),
                not_after=datetime.utcnow(),
                days_until_expiry=0,
                status=CertificateStatus.INVALID,
            )
        except socket.timeout:
            logger.error(f"Timeout connecting to {domain}")
            return None
        except Exception as e:
            logger.error(f"Failed to get certificate for {domain}: {e}")
            return None

    def _parse_certificate(self, domain: str, cert) -> Certificate:
        """Parse x509 certificate."""
        now = datetime.utcnow()
        not_after = cert.not_valid_after
        days_until_expiry = (not_after - now).days

        # Determine status
        if now > not_after:
            status = CertificateStatus.EXPIRED
        elif days_until_expiry <= self.critical_days:
            status = CertificateStatus.EXPIRING_SOON
        elif days_until_expiry <= self.warning_days:
            status = CertificateStatus.EXPIRING_SOON
        else:
            status = CertificateStatus.VALID

        # Get issuer and subject
        issuer = cert.issuer.rfc4514_string()
        subject = cert.subject.rfc4514_string()

        # Get SAN
        san = []
        try:
            san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
            san = [name.value for name in san_ext.value]
        except x509.ExtensionNotFound:
            pass

        # Get fingerprint
        from cryptography.hazmat.primitives import hashes
        fingerprint = cert.fingerprint(hashes.SHA256()).hex()

        # Check if wildcard
        is_wildcard = any(name.startswith("*.") for name in san) or \
                      subject.startswith("CN=*.")

        return Certificate(
            domain=domain,
            issuer=issuer,
            subject=subject,
            serial_number=str(cert.serial_number),
            not_before=cert.not_valid_before,
            not_after=not_after,
            days_until_expiry=days_until_expiry,
            status=status,
            san=san,
            fingerprint_sha256=fingerprint,
            is_wildcard=is_wildcard,
        )

    def _parse_cert_basic(self, domain: str, pem_cert: str) -> Certificate:
        """Basic certificate parsing without cryptography library."""
        # This is a fallback with limited info
        return Certificate(
            domain=domain,
            issuer="Unknown (cryptography library not installed)",
            subject="",
            serial_number="",
            not_before=datetime.utcnow(),
            not_after=datetime.utcnow() + timedelta(days=90),
            days_until_expiry=90,
            status=CertificateStatus.VALID,
        )

    # =========================================================================
    # CERTIFICATE MONITORING
    # =========================================================================

    def add_domain_to_monitor(self, domain: str, port: int = 443) -> bool:
        """Add a domain to monitoring."""
        cert = self.get_certificate(domain, port)
        if cert:
            self.monitored_domains[domain] = cert
            logger.info(f"Added {domain} to certificate monitoring")
            return True
        return False

    def remove_domain_from_monitor(self, domain: str) -> bool:
        """Remove a domain from monitoring."""
        if domain in self.monitored_domains:
            del self.monitored_domains[domain]
            return True
        return False

    def check_all_certificates(self) -> List[CertificateAlert]:
        """Check all monitored certificates."""
        alerts = []

        for domain in list(self.monitored_domains.keys()):
            cert = self.get_certificate(domain)
            if cert:
                self.monitored_domains[domain] = cert

                if cert.status == CertificateStatus.EXPIRED:
                    alerts.append(CertificateAlert(
                        domain=domain,
                        days_until_expiry=cert.days_until_expiry,
                        severity="critical",
                        message=f"Certificate for {domain} has EXPIRED!",
                    ))
                elif cert.days_until_expiry <= self.critical_days:
                    alerts.append(CertificateAlert(
                        domain=domain,
                        days_until_expiry=cert.days_until_expiry,
                        severity="critical",
                        message=f"Certificate for {domain} expires in {cert.days_until_expiry} days!",
                    ))
                elif cert.days_until_expiry <= self.warning_days:
                    alerts.append(CertificateAlert(
                        domain=domain,
                        days_until_expiry=cert.days_until_expiry,
                        severity="warning",
                        message=f"Certificate for {domain} expires in {cert.days_until_expiry} days",
                    ))

        return sorted(alerts, key=lambda a: a.days_until_expiry)

    def get_expiring_certificates(
        self,
        days: int = 30,
    ) -> List[Certificate]:
        """Get certificates expiring within specified days."""
        expiring = []
        for cert in self.monitored_domains.values():
            if cert.days_until_expiry <= days:
                expiring.append(cert)
        return sorted(expiring, key=lambda c: c.days_until_expiry)

    # =========================================================================
    # LET'S ENCRYPT INTEGRATION
    # =========================================================================

    async def request_certificate_certbot(
        self,
        domain: str,
        email: str,
        webroot: str = None,
        standalone: bool = False,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Request a certificate using certbot."""
        command = [
            "certbot", "certonly",
            "-d", domain,
            "--email", email,
            "--agree-tos",
            "--non-interactive",
        ]

        if dry_run:
            command.append("--dry-run")

        if standalone:
            command.append("--standalone")
        elif webroot:
            command.extend(["--webroot", "-w", webroot])
        else:
            command.append("--standalone")

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        success = process.returncode == 0
        if success:
            logger.info(f"Certificate {'requested (dry-run)' if dry_run else 'obtained'} for {domain}")
        else:
            logger.error(f"Certbot failed: {stderr.decode()}")

        return {
            "success": success,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "dry_run": dry_run,
        }

    async def renew_certificates_certbot(
        self,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Renew all certificates using certbot."""
        command = ["certbot", "renew", "--non-interactive"]

        if dry_run:
            command.append("--dry-run")

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        return {
            "success": process.returncode == 0,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "dry_run": dry_run,
        }

    # =========================================================================
    # CERTIFICATE CHAIN VALIDATION
    # =========================================================================

    def validate_certificate_chain(
        self,
        domain: str,
        port: int = 443,
    ) -> Dict[str, Any]:
        """Validate the complete certificate chain."""
        try:
            context = ssl.create_default_context()

            with socket.create_connection((domain, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    # If we get here, the chain is valid
                    return {
                        "domain": domain,
                        "valid": True,
                        "message": "Certificate chain is valid",
                    }

        except ssl.SSLCertVerificationError as e:
            return {
                "domain": domain,
                "valid": False,
                "message": str(e),
                "error_code": e.verify_code if hasattr(e, 'verify_code') else None,
            }
        except Exception as e:
            return {
                "domain": domain,
                "valid": False,
                "message": str(e),
            }

    # =========================================================================
    # BULK OPERATIONS
    # =========================================================================

    async def scan_domains(
        self,
        domains: List[str],
        port: int = 443,
    ) -> List[Certificate]:
        """Scan multiple domains concurrently."""
        results = []

        async def scan_one(domain: str):
            cert = self.get_certificate(domain, port)
            if cert:
                results.append(cert)

        tasks = [scan_one(domain) for domain in domains]
        await asyncio.gather(*tasks, return_exceptions=True)

        return results

    def generate_report(self) -> Dict[str, Any]:
        """Generate a certificate status report."""
        total = len(self.monitored_domains)
        valid = sum(1 for c in self.monitored_domains.values()
                   if c.status == CertificateStatus.VALID)
        expiring = sum(1 for c in self.monitored_domains.values()
                      if c.status == CertificateStatus.EXPIRING_SOON)
        expired = sum(1 for c in self.monitored_domains.values()
                     if c.status == CertificateStatus.EXPIRED)
        invalid = sum(1 for c in self.monitored_domains.values()
                     if c.status == CertificateStatus.INVALID)

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "total_monitored": total,
            "status_summary": {
                "valid": valid,
                "expiring_soon": expiring,
                "expired": expired,
                "invalid": invalid,
            },
            "certificates": [c.to_dict() for c in self.monitored_domains.values()],
            "expiring_within_30_days": [
                c.to_dict() for c in self.monitored_domains.values()
                if c.days_until_expiry <= 30
            ],
        }


# Singleton instance
ssl_manager = SSLManager()

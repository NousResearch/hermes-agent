"""End-to-end Tor route verification over TLS-validating SOCKS."""
import asyncio
import ipaddress
from dataclasses import dataclass, field

import httpx
from darkloom.policy import NetworkChannel, authorize
from darkloom.privacy import get_logger

logger = get_logger(__name__)
DEFAULT_CHECK_ENDPOINTS = (
    "https://check.torproject.org/api/ip",
    "https://api.ipify.org?format=json",
)


@dataclass
class VerificationResult:
    using_tor: bool
    exit_ip: str | None = None
    error: str | None = None
    observations: dict[str, str] = field(default_factory=dict)

    @property
    def is_anonymous(self) -> bool:
        return self.using_tor and self.exit_ip is not None


class TorVerifier:
    """Require Tor's structured assertion plus an independent observation."""

    def __init__(self, socks_proxy_url="socks5://127.0.0.1:9050", timeout=30.0,
                 endpoints=DEFAULT_CHECK_ENDPOINTS):
        self.socks_proxy_url = socks_proxy_url
        self.timeout = timeout
        self.endpoints = tuple(endpoints)

    @staticmethod
    def _valid_ip(value) -> str | None:
        try:
            return str(ipaddress.ip_address(value))
        except (ValueError, TypeError):
            return None

    def verify(self) -> VerificationResult:
        """Verify TLS, structured JSON, Tor status, and matching exit IPs."""
        observations = {}
        try:
            authorize(NetworkChannel.HTTP, proxy_url=self.socks_proxy_url)
            transport = httpx.HTTPTransport(proxy=self.socks_proxy_url)
            # Default verify=True validates the normal CA chain and hostname.
            with httpx.Client(transport=transport, timeout=self.timeout,
                              follow_redirects=False) as client:
                response = client.get(self.endpoints[0])
                response.raise_for_status()
                data = response.json()
                tor_ip = self._valid_ip(data.get("IP"))
                if data.get("IsTor") is not True or not tor_ip:
                    return VerificationResult(False, tor_ip, "Tor API did not confirm this route")
                observations[self.endpoints[0]] = tor_ip
                for endpoint in self.endpoints[1:]:
                    response = client.get(endpoint)
                    response.raise_for_status()
                    observed_ip = self._valid_ip(response.json().get("ip"))
                    if not observed_ip:
                        return VerificationResult(False, tor_ip, f"Invalid observation from {endpoint}", observations)
                    observations[endpoint] = observed_ip
                    if observed_ip != tor_ip:
                        return VerificationResult(False, tor_ip, "Independent exit observations disagree", observations)
            return VerificationResult(True, tor_ip, observations=observations)
        except Exception as exc:
            logger.error("Verification failed: %s", exc)
            return VerificationResult(False, error=str(exc), observations=observations)

    async def verify_async(self) -> VerificationResult:
        return await asyncio.to_thread(self.verify)

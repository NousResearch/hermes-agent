"""
cosmos DNS Manager

"I've conquered DNS! Now I control what names mean what!"

Multi-provider DNS management with support for all record types including MX.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from loguru import logger

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


class DNSRecordType(Enum):
    """DNS record types."""
    A = "A"
    AAAA = "AAAA"
    CNAME = "CNAME"
    MX = "MX"
    TXT = "TXT"
    NS = "NS"
    SOA = "SOA"
    SRV = "SRV"
    CAA = "CAA"
    PTR = "PTR"
    SPF = "SPF"  # Deprecated but still used
    DKIM = "DKIM"  # TXT record with specific format
    DMARC = "DMARC"  # TXT record with specific format


@dataclass
class DNSRecord:
    """DNS record definition."""
    id: str
    name: str  # subdomain or @ for root
    type: DNSRecordType
    content: str  # IP address, hostname, or text content
    ttl: int = 3600
    priority: int = 0  # For MX and SRV records
    proxied: bool = False  # Cloudflare-specific
    zone_id: str = ""
    zone_name: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "content": self.content,
            "ttl": self.ttl,
            "priority": self.priority,
            "proxied": self.proxied,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
        }


@dataclass
class DNSZone:
    """DNS zone definition."""
    id: str
    name: str
    status: str
    nameservers: List[str]
    created_at: datetime
    records_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "nameservers": self.nameservers,
            "created_at": self.created_at.isoformat(),
            "records_count": self.records_count,
        }


class DNSProvider(ABC):
    """Abstract DNS provider interface."""

    @abstractmethod
    async def list_zones(self) -> List[DNSZone]:
        """List all DNS zones."""
        pass

    @abstractmethod
    async def get_zone(self, zone_id: str) -> Optional[DNSZone]:
        """Get a specific zone."""
        pass

    @abstractmethod
    async def list_records(
        self,
        zone_id: str,
        record_type: DNSRecordType = None,
    ) -> List[DNSRecord]:
        """List DNS records in a zone."""
        pass

    @abstractmethod
    async def create_record(
        self,
        zone_id: str,
        record: DNSRecord,
    ) -> Optional[DNSRecord]:
        """Create a DNS record."""
        pass

    @abstractmethod
    async def update_record(
        self,
        zone_id: str,
        record_id: str,
        record: DNSRecord,
    ) -> Optional[DNSRecord]:
        """Update a DNS record."""
        pass

    @abstractmethod
    async def delete_record(
        self,
        zone_id: str,
        record_id: str,
    ) -> bool:
        """Delete a DNS record."""
        pass


class CloudflareDNSProvider(DNSProvider):
    """Cloudflare DNS provider implementation."""

    def __init__(self, api_token: str = None, email: str = None, api_key: str = None):
        import os
        self.api_token = api_token or os.getenv("CLOUDFLARE_API_TOKEN")
        self.email = email or os.getenv("CLOUDFLARE_EMAIL")
        self.api_key = api_key or os.getenv("CLOUDFLARE_API_KEY")
        self.base_url = "https://api.cloudflare.com/client/v4"
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {"Content-Type": "application/json"}
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"
            else:
                headers["X-Auth-Email"] = self.email
                headers["X-Auth-Key"] = self.api_key
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Any = None,
        params: Dict = None,
    ) -> Dict[str, Any]:
        session = await self._get_session()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        async with session.request(method, url, json=data, params=params) as response:
            result = await response.json()
            if not result.get("success", False):
                errors = result.get("errors", [])
                logger.error(f"Cloudflare API error: {errors}")
            return result

    async def list_zones(self) -> List[DNSZone]:
        result = await self._request("GET", "/zones")
        zones = []
        for z in result.get("result", []):
            zones.append(DNSZone(
                id=z["id"],
                name=z["name"],
                status=z["status"],
                nameservers=z.get("name_servers", []),
                created_at=datetime.fromisoformat(z["created_on"].replace("Z", "+00:00")),
            ))
        return zones

    async def get_zone(self, zone_id: str) -> Optional[DNSZone]:
        result = await self._request("GET", f"/zones/{zone_id}")
        if result.get("success"):
            z = result["result"]
            return DNSZone(
                id=z["id"],
                name=z["name"],
                status=z["status"],
                nameservers=z.get("name_servers", []),
                created_at=datetime.fromisoformat(z["created_on"].replace("Z", "+00:00")),
            )
        return None

    async def list_records(
        self,
        zone_id: str,
        record_type: DNSRecordType = None,
    ) -> List[DNSRecord]:
        params = {}
        if record_type:
            params["type"] = record_type.value

        result = await self._request("GET", f"/zones/{zone_id}/dns_records", params=params)
        records = []
        for r in result.get("result", []):
            records.append(DNSRecord(
                id=r["id"],
                name=r["name"],
                type=DNSRecordType(r["type"]),
                content=r["content"],
                ttl=r["ttl"],
                priority=r.get("priority", 0),
                proxied=r.get("proxied", False),
                zone_id=zone_id,
                zone_name=r.get("zone_name", ""),
            ))
        return records

    async def create_record(
        self,
        zone_id: str,
        record: DNSRecord,
    ) -> Optional[DNSRecord]:
        data = {
            "type": record.type.value,
            "name": record.name,
            "content": record.content,
            "ttl": record.ttl,
        }
        if record.type == DNSRecordType.MX:
            data["priority"] = record.priority
        if record.proxied is not None:
            data["proxied"] = record.proxied

        result = await self._request("POST", f"/zones/{zone_id}/dns_records", data=data)
        if result.get("success"):
            r = result["result"]
            return DNSRecord(
                id=r["id"],
                name=r["name"],
                type=DNSRecordType(r["type"]),
                content=r["content"],
                ttl=r["ttl"],
                priority=r.get("priority", 0),
                proxied=r.get("proxied", False),
                zone_id=zone_id,
            )
        return None

    async def update_record(
        self,
        zone_id: str,
        record_id: str,
        record: DNSRecord,
    ) -> Optional[DNSRecord]:
        data = {
            "type": record.type.value,
            "name": record.name,
            "content": record.content,
            "ttl": record.ttl,
        }
        if record.type == DNSRecordType.MX:
            data["priority"] = record.priority
        if record.proxied is not None:
            data["proxied"] = record.proxied

        result = await self._request("PUT", f"/zones/{zone_id}/dns_records/{record_id}", data=data)
        if result.get("success"):
            r = result["result"]
            return DNSRecord(
                id=r["id"],
                name=r["name"],
                type=DNSRecordType(r["type"]),
                content=r["content"],
                ttl=r["ttl"],
                priority=r.get("priority", 0),
                proxied=r.get("proxied", False),
                zone_id=zone_id,
            )
        return None

    async def delete_record(
        self,
        zone_id: str,
        record_id: str,
    ) -> bool:
        result = await self._request("DELETE", f"/zones/{zone_id}/dns_records/{record_id}")
        return result.get("success", False)


class Route53DNSProvider(DNSProvider):
    """AWS Route 53 DNS provider implementation."""

    def __init__(self):
        import os
        self.access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self._client = None

        try:
            import boto3
            self._client = boto3.client("route53")
        except ImportError:
            logger.warning("boto3 not installed, Route 53 unavailable")

    def _ensure_client(self):
        if not self._client:
            raise RuntimeError("Route 53 client not available")

    async def list_zones(self) -> List[DNSZone]:
        self._ensure_client()

        response = self._client.list_hosted_zones()
        zones = []
        for z in response.get("HostedZones", []):
            zones.append(DNSZone(
                id=z["Id"].split("/")[-1],
                name=z["Name"].rstrip("."),
                status="active",
                nameservers=[],
                created_at=datetime.utcnow(),
                records_count=z.get("ResourceRecordSetCount", 0),
            ))
        return zones

    async def get_zone(self, zone_id: str) -> Optional[DNSZone]:
        self._ensure_client()

        try:
            response = self._client.get_hosted_zone(Id=zone_id)
            z = response["HostedZone"]
            ns = response.get("DelegationSet", {}).get("NameServers", [])
            return DNSZone(
                id=z["Id"].split("/")[-1],
                name=z["Name"].rstrip("."),
                status="active",
                nameservers=ns,
                created_at=datetime.utcnow(),
                records_count=z.get("ResourceRecordSetCount", 0),
            )
        except Exception:
            return None

    async def list_records(
        self,
        zone_id: str,
        record_type: DNSRecordType = None,
    ) -> List[DNSRecord]:
        self._ensure_client()

        response = self._client.list_resource_record_sets(HostedZoneId=zone_id)
        records = []

        for r in response.get("ResourceRecordSets", []):
            rtype = r["Type"]
            if record_type and rtype != record_type.value:
                continue

            # Handle different record structures
            for value in r.get("ResourceRecords", []):
                records.append(DNSRecord(
                    id=f"{r['Name']}:{rtype}",
                    name=r["Name"].rstrip("."),
                    type=DNSRecordType(rtype),
                    content=value["Value"],
                    ttl=r.get("TTL", 300),
                    zone_id=zone_id,
                ))

        return records

    async def create_record(
        self,
        zone_id: str,
        record: DNSRecord,
    ) -> Optional[DNSRecord]:
        self._ensure_client()

        change = {
            "Action": "CREATE",
            "ResourceRecordSet": {
                "Name": record.name,
                "Type": record.type.value,
                "TTL": record.ttl,
                "ResourceRecords": [{"Value": record.content}],
            }
        }

        try:
            self._client.change_resource_record_sets(
                HostedZoneId=zone_id,
                ChangeBatch={"Changes": [change]}
            )
            record.zone_id = zone_id
            record.id = f"{record.name}:{record.type.value}"
            return record
        except Exception as e:
            logger.error(f"Failed to create Route 53 record: {e}")
            return None

    async def update_record(
        self,
        zone_id: str,
        record_id: str,
        record: DNSRecord,
    ) -> Optional[DNSRecord]:
        self._ensure_client()

        change = {
            "Action": "UPSERT",
            "ResourceRecordSet": {
                "Name": record.name,
                "Type": record.type.value,
                "TTL": record.ttl,
                "ResourceRecords": [{"Value": record.content}],
            }
        }

        try:
            self._client.change_resource_record_sets(
                HostedZoneId=zone_id,
                ChangeBatch={"Changes": [change]}
            )
            return record
        except Exception as e:
            logger.error(f"Failed to update Route 53 record: {e}")
            return None

    async def delete_record(
        self,
        zone_id: str,
        record_id: str,
    ) -> bool:
        self._ensure_client()

        # Record ID format: name:type
        name, rtype = record_id.split(":")

        # First get the current record to delete
        records = await self.list_records(zone_id, DNSRecordType(rtype))
        target_record = None
        for r in records:
            if r.name == name:
                target_record = r
                break

        if not target_record:
            return False

        change = {
            "Action": "DELETE",
            "ResourceRecordSet": {
                "Name": name,
                "Type": rtype,
                "TTL": target_record.ttl,
                "ResourceRecords": [{"Value": target_record.content}],
            }
        }

        try:
            self._client.change_resource_record_sets(
                HostedZoneId=zone_id,
                ChangeBatch={"Changes": [change]}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete Route 53 record: {e}")
            return False


class DNSManager:
    """
    Multi-provider DNS management for cosmos.

    Features:
    - Cloudflare, Route 53, Azure DNS support
    - All record types including MX, SPF, DKIM, DMARC
    - Bulk operations
    - DNS verification
    - Email configuration helpers
    """

    def __init__(self):
        self.providers: Dict[str, DNSProvider] = {}
        self._load_providers()

    def _load_providers(self):
        """Load configured DNS providers."""
        import os

        # Cloudflare
        if os.getenv("CLOUDFLARE_API_TOKEN") or os.getenv("CLOUDFLARE_API_KEY"):
            self.providers["cloudflare"] = CloudflareDNSProvider()
            logger.info("Loaded Cloudflare DNS provider")

        # Route 53
        if os.getenv("AWS_ACCESS_KEY_ID"):
            self.providers["route53"] = Route53DNSProvider()
            logger.info("Loaded Route 53 DNS provider")

    def get_provider(self, name: str) -> Optional[DNSProvider]:
        """Get a DNS provider by name."""
        return self.providers.get(name)

    def list_providers(self) -> List[str]:
        """List available DNS providers."""
        return list(self.providers.keys())

    # =========================================================================
    # HIGH-LEVEL OPERATIONS
    # =========================================================================

    async def list_all_zones(self) -> Dict[str, List[DNSZone]]:
        """List zones from all providers."""
        result = {}
        for name, provider in self.providers.items():
            try:
                zones = await provider.list_zones()
                result[name] = zones
            except Exception as e:
                logger.error(f"Failed to list zones from {name}: {e}")
                result[name] = []
        return result

    async def find_zone(self, domain: str) -> Optional[tuple]:
        """Find a zone by domain across all providers."""
        for provider_name, provider in self.providers.items():
            try:
                zones = await provider.list_zones()
                for zone in zones:
                    if domain.endswith(zone.name) or domain == zone.name:
                        return (provider_name, zone)
            except Exception as e:
                logger.error(f"Error searching {provider_name}: {e}")
        return None

    # =========================================================================
    # MX RECORD MANAGEMENT
    # =========================================================================

    async def get_mx_records(
        self,
        provider: str,
        zone_id: str,
    ) -> List[DNSRecord]:
        """Get MX records for a zone."""
        dns_provider = self.providers.get(provider)
        if not dns_provider:
            return []
        return await dns_provider.list_records(zone_id, DNSRecordType.MX)

    async def set_mx_records(
        self,
        provider: str,
        zone_id: str,
        mx_servers: List[Dict[str, Any]],
    ) -> bool:
        """
        Set MX records for a zone.

        mx_servers format:
        [
            {"server": "mx1.mail.com", "priority": 10},
            {"server": "mx2.mail.com", "priority": 20},
        ]
        """
        dns_provider = self.providers.get(provider)
        if not dns_provider:
            return False

        # Get zone for domain name
        zone = await dns_provider.get_zone(zone_id)
        if not zone:
            return False

        # Delete existing MX records
        existing = await dns_provider.list_records(zone_id, DNSRecordType.MX)
        for record in existing:
            await dns_provider.delete_record(zone_id, record.id)

        # Create new MX records
        for mx in mx_servers:
            record = DNSRecord(
                id="",
                name=zone.name,
                type=DNSRecordType.MX,
                content=mx["server"],
                priority=mx.get("priority", 10),
                ttl=mx.get("ttl", 3600),
            )
            await dns_provider.create_record(zone_id, record)

        logger.info(f"Set {len(mx_servers)} MX records for zone {zone_id}")
        return True

    # =========================================================================
    # EMAIL DNS CONFIGURATION
    # =========================================================================

    async def configure_email_dns(
        self,
        provider: str,
        zone_id: str,
        email_provider: str,
        domain: str = None,
    ) -> Dict[str, Any]:
        """
        Configure DNS for email providers.

        Supports: office365, google, zoho, custom
        """
        dns_provider = self.providers.get(provider)
        if not dns_provider:
            return {"success": False, "error": "Provider not found"}

        zone = await dns_provider.get_zone(zone_id)
        if not zone:
            return {"success": False, "error": "Zone not found"}

        domain = domain or zone.name
        records_created = []

        if email_provider == "office365":
            # MX Record
            mx = DNSRecord(
                id="", name=domain, type=DNSRecordType.MX,
                content=f"{domain.replace('.', '-')}.mail.protection.outlook.com",
                priority=0, ttl=3600
            )
            await dns_provider.create_record(zone_id, mx)
            records_created.append("MX")

            # Autodiscover CNAME
            autodiscover = DNSRecord(
                id="", name=f"autodiscover.{domain}", type=DNSRecordType.CNAME,
                content="autodiscover.outlook.com", ttl=3600
            )
            await dns_provider.create_record(zone_id, autodiscover)
            records_created.append("Autodiscover CNAME")

            # SPF Record
            spf = DNSRecord(
                id="", name=domain, type=DNSRecordType.TXT,
                content="v=spf1 include:spf.protection.outlook.com -all", ttl=3600
            )
            await dns_provider.create_record(zone_id, spf)
            records_created.append("SPF")

        elif email_provider == "google":
            # MX Records
            mx_servers = [
                {"server": "aspmx.l.google.com", "priority": 1},
                {"server": "alt1.aspmx.l.google.com", "priority": 5},
                {"server": "alt2.aspmx.l.google.com", "priority": 5},
                {"server": "alt3.aspmx.l.google.com", "priority": 10},
                {"server": "alt4.aspmx.l.google.com", "priority": 10},
            ]
            await self.set_mx_records(provider, zone_id, mx_servers)
            records_created.append("MX (5 records)")

            # SPF Record
            spf = DNSRecord(
                id="", name=domain, type=DNSRecordType.TXT,
                content="v=spf1 include:_spf.google.com ~all", ttl=3600
            )
            await dns_provider.create_record(zone_id, spf)
            records_created.append("SPF")

        elif email_provider == "zoho":
            # MX Records
            mx_servers = [
                {"server": "mx.zoho.com", "priority": 10},
                {"server": "mx2.zoho.com", "priority": 20},
                {"server": "mx3.zoho.com", "priority": 50},
            ]
            await self.set_mx_records(provider, zone_id, mx_servers)
            records_created.append("MX (3 records)")

            # SPF Record
            spf = DNSRecord(
                id="", name=domain, type=DNSRecordType.TXT,
                content="v=spf1 include:zoho.com ~all", ttl=3600
            )
            await dns_provider.create_record(zone_id, spf)
            records_created.append("SPF")

        logger.info(f"Configured email DNS for {email_provider} on {domain}")
        return {
            "success": True,
            "records_created": records_created,
            "email_provider": email_provider,
            "domain": domain,
        }

    async def add_dkim_record(
        self,
        provider: str,
        zone_id: str,
        selector: str,
        public_key: str,
    ) -> bool:
        """Add a DKIM record."""
        dns_provider = self.providers.get(provider)
        if not dns_provider:
            return False

        zone = await dns_provider.get_zone(zone_id)
        if not zone:
            return False

        record = DNSRecord(
            id="",
            name=f"{selector}._domainkey.{zone.name}",
            type=DNSRecordType.TXT,
            content=f"v=DKIM1; k=rsa; p={public_key}",
            ttl=3600,
        )

        result = await dns_provider.create_record(zone_id, record)
        return result is not None

    async def add_dmarc_record(
        self,
        provider: str,
        zone_id: str,
        policy: str = "none",  # none, quarantine, reject
        rua: str = None,  # Aggregate report email
        ruf: str = None,  # Forensic report email
        pct: int = 100,
    ) -> bool:
        """Add a DMARC record."""
        dns_provider = self.providers.get(provider)
        if not dns_provider:
            return False

        zone = await dns_provider.get_zone(zone_id)
        if not zone:
            return False

        content = f"v=DMARC1; p={policy}; pct={pct}"
        if rua:
            content += f"; rua=mailto:{rua}"
        if ruf:
            content += f"; ruf=mailto:{ruf}"

        record = DNSRecord(
            id="",
            name=f"_dmarc.{zone.name}",
            type=DNSRecordType.TXT,
            content=content,
            ttl=3600,
        )

        result = await dns_provider.create_record(zone_id, record)
        return result is not None

    # =========================================================================
    # DNS VERIFICATION
    # =========================================================================

    async def verify_dns_propagation(
        self,
        domain: str,
        record_type: DNSRecordType,
        expected_value: str,
        nameservers: List[str] = None,
    ) -> Dict[str, Any]:
        """Verify DNS propagation across nameservers."""
        import dns.resolver

        nameservers = nameservers or [
            "8.8.8.8",  # Google
            "1.1.1.1",  # Cloudflare
            "9.9.9.9",  # Quad9
        ]

        results = {}
        for ns in nameservers:
            try:
                resolver = dns.resolver.Resolver()
                resolver.nameservers = [ns]
                answers = resolver.resolve(domain, record_type.value)

                values = [str(rdata) for rdata in answers]
                results[ns] = {
                    "success": expected_value in values,
                    "values": values,
                }
            except Exception as e:
                results[ns] = {
                    "success": False,
                    "error": str(e),
                }

        propagated = all(r.get("success", False) for r in results.values())
        return {
            "domain": domain,
            "record_type": record_type.value,
            "expected": expected_value,
            "propagated": propagated,
            "results": results,
        }


# Singleton instance
dns_manager = DNSManager()

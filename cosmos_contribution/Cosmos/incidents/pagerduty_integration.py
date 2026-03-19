"""
cosmos PagerDuty Integration

"When the pager goes off, everyone runs away...
 except Zoidberg, he runs toward food."

PagerDuty integration for incident management.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from loguru import logger

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


@dataclass
class PagerDutyIncident:
    """PagerDuty incident representation."""
    id: str
    incident_number: int
    title: str
    status: str
    urgency: str
    priority: Optional[str]
    service_id: str
    service_name: str
    escalation_policy_id: str
    created_at: datetime
    last_status_change_at: datetime
    html_url: str
    assignments: List[Dict]
    acknowledgements: List[Dict]


class PagerDutyIntegration:
    """
    PagerDuty integration for incident management.

    Features:
    - Create and manage incidents
    - On-call schedule lookup
    - Escalation management
    - Webhook handling
    - Service health monitoring
    """

    def __init__(
        self,
        api_token: str,
        base_url: str = "https://api.pagerduty.com",
    ):
        if not HAS_HTTPX:
            raise ImportError("httpx is required for PagerDuty integration")

        self.api_token = api_token
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Token token={api_token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.pagerduty+json;version=2",
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Optional[Dict]:
        """Make an authenticated request to PagerDuty API."""
        url = f"{self.base_url}{endpoint}"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.request(
                    method,
                    url,
                    headers=self.headers,
                    **kwargs,
                )
                response.raise_for_status()

                if response.status_code == 204:
                    return {}
                return response.json()

            except httpx.HTTPStatusError as e:
                logger.error(f"PagerDuty API error: {e.response.status_code} - {e.response.text}")
                return None
            except Exception as e:
                logger.error(f"PagerDuty request failed: {e}")
                return None

    # =========================================================================
    # INCIDENTS
    # =========================================================================

    async def create_incident(
        self,
        service_id: str,
        title: str,
        body: str = None,
        urgency: str = "high",
        escalation_policy_id: str = None,
        priority_id: str = None,
        incident_key: str = None,
        from_email: str = None,
    ) -> Optional[PagerDutyIncident]:
        """Create a new incident."""
        payload = {
            "incident": {
                "type": "incident",
                "title": title,
                "service": {
                    "id": service_id,
                    "type": "service_reference",
                },
                "urgency": urgency,
            }
        }

        if body:
            payload["incident"]["body"] = {
                "type": "incident_body",
                "details": body,
            }

        if escalation_policy_id:
            payload["incident"]["escalation_policy"] = {
                "id": escalation_policy_id,
                "type": "escalation_policy_reference",
            }

        if priority_id:
            payload["incident"]["priority"] = {
                "id": priority_id,
                "type": "priority_reference",
            }

        if incident_key:
            payload["incident"]["incident_key"] = incident_key

        headers = self.headers.copy()
        if from_email:
            headers["From"] = from_email

        result = await self._request("POST", "/incidents", json=payload, headers=headers)

        if result:
            return self._parse_incident(result.get("incident", {}))
        return None

    async def get_incident(self, incident_id: str) -> Optional[PagerDutyIncident]:
        """Get an incident by ID."""
        result = await self._request("GET", f"/incidents/{incident_id}")

        if result:
            return self._parse_incident(result.get("incident", {}))
        return None

    async def list_incidents(
        self,
        statuses: List[str] = None,
        service_ids: List[str] = None,
        urgencies: List[str] = None,
        since: datetime = None,
        until: datetime = None,
        limit: int = 25,
    ) -> List[PagerDutyIncident]:
        """List incidents with filters."""
        params = {"limit": limit}

        if statuses:
            params["statuses[]"] = statuses
        if service_ids:
            params["service_ids[]"] = service_ids
        if urgencies:
            params["urgencies[]"] = urgencies
        if since:
            params["since"] = since.isoformat()
        if until:
            params["until"] = until.isoformat()

        result = await self._request("GET", "/incidents", params=params)

        if result:
            return [self._parse_incident(i) for i in result.get("incidents", [])]
        return []

    async def update_incident(
        self,
        incident_id: str,
        status: str = None,
        title: str = None,
        urgency: str = None,
        escalation_level: int = None,
        resolution: str = None,
        from_email: str = None,
    ) -> Optional[PagerDutyIncident]:
        """Update an incident."""
        payload = {"incident": {"type": "incident"}}

        if status:
            payload["incident"]["status"] = status
        if title:
            payload["incident"]["title"] = title
        if urgency:
            payload["incident"]["urgency"] = urgency
        if escalation_level:
            payload["incident"]["escalation_level"] = escalation_level
        if resolution:
            payload["incident"]["resolution"] = resolution

        headers = self.headers.copy()
        if from_email:
            headers["From"] = from_email

        result = await self._request(
            "PUT",
            f"/incidents/{incident_id}",
            json=payload,
            headers=headers,
        )

        if result:
            return self._parse_incident(result.get("incident", {}))
        return None

    async def acknowledge_incident(
        self,
        incident_id: str,
        from_email: str,
    ) -> bool:
        """Acknowledge an incident."""
        result = await self.update_incident(
            incident_id,
            status="acknowledged",
            from_email=from_email,
        )
        return result is not None

    async def resolve_incident(
        self,
        incident_id: str,
        from_email: str,
        resolution: str = None,
    ) -> bool:
        """Resolve an incident."""
        result = await self.update_incident(
            incident_id,
            status="resolved",
            resolution=resolution,
            from_email=from_email,
        )
        return result is not None

    async def add_note(
        self,
        incident_id: str,
        content: str,
        from_email: str,
    ) -> bool:
        """Add a note to an incident."""
        payload = {
            "note": {
                "content": content,
            }
        }

        headers = self.headers.copy()
        headers["From"] = from_email

        result = await self._request(
            "POST",
            f"/incidents/{incident_id}/notes",
            json=payload,
            headers=headers,
        )

        return result is not None

    async def get_notes(self, incident_id: str) -> List[Dict]:
        """Get notes for an incident."""
        result = await self._request("GET", f"/incidents/{incident_id}/notes")

        return result.get("notes", []) if result else []

    def _parse_incident(self, data: Dict) -> PagerDutyIncident:
        """Parse PagerDuty incident data."""
        return PagerDutyIncident(
            id=data["id"],
            incident_number=data.get("incident_number", 0),
            title=data.get("title", ""),
            status=data.get("status", ""),
            urgency=data.get("urgency", ""),
            priority=data.get("priority", {}).get("summary") if data.get("priority") else None,
            service_id=data.get("service", {}).get("id", ""),
            service_name=data.get("service", {}).get("summary", ""),
            escalation_policy_id=data.get("escalation_policy", {}).get("id", ""),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")) if data.get("created_at") else datetime.utcnow(),
            last_status_change_at=datetime.fromisoformat(data["last_status_change_at"].replace("Z", "+00:00")) if data.get("last_status_change_at") else datetime.utcnow(),
            html_url=data.get("html_url", ""),
            assignments=data.get("assignments", []),
            acknowledgements=data.get("acknowledgements", []),
        )

    # =========================================================================
    # ON-CALL
    # =========================================================================

    async def get_oncalls(
        self,
        schedule_ids: List[str] = None,
        escalation_policy_ids: List[str] = None,
        since: datetime = None,
        until: datetime = None,
    ) -> List[Dict]:
        """Get current on-call users."""
        params = {}

        if schedule_ids:
            params["schedule_ids[]"] = schedule_ids
        if escalation_policy_ids:
            params["escalation_policy_ids[]"] = escalation_policy_ids
        if since:
            params["since"] = since.isoformat()
        if until:
            params["until"] = until.isoformat()

        result = await self._request("GET", "/oncalls", params=params)

        return result.get("oncalls", []) if result else []

    async def get_current_oncall(
        self,
        escalation_policy_id: str,
    ) -> Optional[Dict]:
        """Get current on-call user for an escalation policy."""
        oncalls = await self.get_oncalls(
            escalation_policy_ids=[escalation_policy_id],
        )

        if oncalls:
            # Return first on-call at level 1
            for oncall in oncalls:
                if oncall.get("escalation_level") == 1:
                    return oncall.get("user")

        return None

    # =========================================================================
    # SERVICES
    # =========================================================================

    async def list_services(
        self,
        team_ids: List[str] = None,
        include: List[str] = None,
    ) -> List[Dict]:
        """List services."""
        params = {}

        if team_ids:
            params["team_ids[]"] = team_ids
        if include:
            params["include[]"] = include

        result = await self._request("GET", "/services", params=params)

        return result.get("services", []) if result else []

    async def get_service(self, service_id: str) -> Optional[Dict]:
        """Get a service by ID."""
        result = await self._request("GET", f"/services/{service_id}")

        return result.get("service") if result else None

    async def create_service(
        self,
        name: str,
        escalation_policy_id: str,
        description: str = None,
        auto_resolve_timeout: int = 14400,
        acknowledgement_timeout: int = 1800,
    ) -> Optional[Dict]:
        """Create a new service."""
        payload = {
            "service": {
                "type": "service",
                "name": name,
                "escalation_policy": {
                    "id": escalation_policy_id,
                    "type": "escalation_policy_reference",
                },
                "auto_resolve_timeout": auto_resolve_timeout,
                "acknowledgement_timeout": acknowledgement_timeout,
            }
        }

        if description:
            payload["service"]["description"] = description

        result = await self._request("POST", "/services", json=payload)

        return result.get("service") if result else None

    # =========================================================================
    # ESCALATION POLICIES
    # =========================================================================

    async def list_escalation_policies(self) -> List[Dict]:
        """List escalation policies."""
        result = await self._request("GET", "/escalation_policies")

        return result.get("escalation_policies", []) if result else []

    async def get_escalation_policy(self, policy_id: str) -> Optional[Dict]:
        """Get an escalation policy by ID."""
        result = await self._request("GET", f"/escalation_policies/{policy_id}")

        return result.get("escalation_policy") if result else None

    # =========================================================================
    # SCHEDULES
    # =========================================================================

    async def list_schedules(self) -> List[Dict]:
        """List schedules."""
        result = await self._request("GET", "/schedules")

        return result.get("schedules", []) if result else []

    async def get_schedule(
        self,
        schedule_id: str,
        since: datetime = None,
        until: datetime = None,
    ) -> Optional[Dict]:
        """Get a schedule by ID."""
        params = {}
        if since:
            params["since"] = since.isoformat()
        if until:
            params["until"] = until.isoformat()

        result = await self._request(
            "GET",
            f"/schedules/{schedule_id}",
            params=params,
        )

        return result.get("schedule") if result else None

    # =========================================================================
    # USERS
    # =========================================================================

    async def list_users(
        self,
        team_ids: List[str] = None,
    ) -> List[Dict]:
        """List users."""
        params = {}
        if team_ids:
            params["team_ids[]"] = team_ids

        result = await self._request("GET", "/users", params=params)

        return result.get("users", []) if result else []

    async def get_user(self, user_id: str) -> Optional[Dict]:
        """Get a user by ID."""
        result = await self._request("GET", f"/users/{user_id}")

        return result.get("user") if result else None

    # =========================================================================
    # WEBHOOKS
    # =========================================================================

    def parse_webhook(self, payload: Dict) -> List[Dict]:
        """Parse a PagerDuty webhook payload."""
        events = []

        for message in payload.get("messages", []):
            event = {
                "type": message.get("event"),
                "incident": message.get("incident", {}),
                "log_entries": message.get("log_entries", []),
                "created_on": message.get("created_on"),
            }
            events.append(event)

        return events

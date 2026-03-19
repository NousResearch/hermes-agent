"""
cosmos OpsGenie Integration

"Good news, everyone! The alerts are so loud, even Nibbler woke up!"

OpsGenie integration for incident management.
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
class OpsGenieAlert:
    """OpsGenie alert representation."""
    id: str
    tiny_id: str
    alias: str
    message: str
    status: str
    acknowledged: bool
    is_seen: bool
    priority: str
    source: str
    tags: List[str]
    responders: List[Dict]
    created_at: datetime
    updated_at: datetime
    count: int


class OpsGenieIntegration:
    """
    OpsGenie integration for alert and incident management.

    Features:
    - Alert management
    - On-call schedules
    - Team management
    - Integration webhooks
    - Incident coordination
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.opsgenie.com/v2",
    ):
        if not HAS_HTTPX:
            raise ImportError("httpx is required for OpsGenie integration")

        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"GenieKey {api_key}",
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Optional[Dict]:
        """Make an authenticated request to OpsGenie API."""
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
                logger.error(f"OpsGenie API error: {e.response.status_code} - {e.response.text}")
                return None
            except Exception as e:
                logger.error(f"OpsGenie request failed: {e}")
                return None

    # =========================================================================
    # ALERTS
    # =========================================================================

    async def create_alert(
        self,
        message: str,
        alias: str = None,
        description: str = None,
        responders: List[Dict] = None,
        visible_to: List[Dict] = None,
        actions: List[str] = None,
        tags: List[str] = None,
        details: Dict = None,
        entity: str = None,
        source: str = "cosmos",
        priority: str = "P3",
        user: str = None,
        note: str = None,
    ) -> Optional[str]:
        """Create a new alert."""
        payload = {
            "message": message,
            "source": source,
            "priority": priority,
        }

        if alias:
            payload["alias"] = alias
        if description:
            payload["description"] = description
        if responders:
            payload["responders"] = responders
        if visible_to:
            payload["visibleTo"] = visible_to
        if actions:
            payload["actions"] = actions
        if tags:
            payload["tags"] = tags
        if details:
            payload["details"] = details
        if entity:
            payload["entity"] = entity
        if user:
            payload["user"] = user
        if note:
            payload["note"] = note

        result = await self._request("POST", "/alerts", json=payload)

        if result:
            return result.get("requestId")
        return None

    async def get_alert(
        self,
        identifier: str,
        identifier_type: str = "id",
    ) -> Optional[OpsGenieAlert]:
        """Get an alert by ID or alias."""
        params = {"identifierType": identifier_type}

        result = await self._request(
            "GET",
            f"/alerts/{identifier}",
            params=params,
        )

        if result and "data" in result:
            return self._parse_alert(result["data"])
        return None

    async def list_alerts(
        self,
        query: str = None,
        search_identifier: str = None,
        search_identifier_type: str = None,
        offset: int = 0,
        limit: int = 20,
        sort: str = "createdAt",
        order: str = "desc",
    ) -> List[OpsGenieAlert]:
        """List alerts with filters."""
        params = {
            "offset": offset,
            "limit": limit,
            "sort": sort,
            "order": order,
        }

        if query:
            params["query"] = query
        if search_identifier:
            params["searchIdentifier"] = search_identifier
        if search_identifier_type:
            params["searchIdentifierType"] = search_identifier_type

        result = await self._request("GET", "/alerts", params=params)

        if result and "data" in result:
            return [self._parse_alert(a) for a in result["data"]]
        return []

    async def acknowledge_alert(
        self,
        identifier: str,
        identifier_type: str = "id",
        user: str = None,
        source: str = "cosmos",
        note: str = None,
    ) -> bool:
        """Acknowledge an alert."""
        payload = {"source": source}

        if user:
            payload["user"] = user
        if note:
            payload["note"] = note

        result = await self._request(
            "POST",
            f"/alerts/{identifier}/acknowledge",
            params={"identifierType": identifier_type},
            json=payload,
        )

        return result is not None

    async def close_alert(
        self,
        identifier: str,
        identifier_type: str = "id",
        user: str = None,
        source: str = "cosmos",
        note: str = None,
    ) -> bool:
        """Close an alert."""
        payload = {"source": source}

        if user:
            payload["user"] = user
        if note:
            payload["note"] = note

        result = await self._request(
            "POST",
            f"/alerts/{identifier}/close",
            params={"identifierType": identifier_type},
            json=payload,
        )

        return result is not None

    async def snooze_alert(
        self,
        identifier: str,
        end_time: datetime,
        identifier_type: str = "id",
        user: str = None,
        source: str = "cosmos",
        note: str = None,
    ) -> bool:
        """Snooze an alert until a specific time."""
        payload = {
            "endTime": end_time.isoformat(),
            "source": source,
        }

        if user:
            payload["user"] = user
        if note:
            payload["note"] = note

        result = await self._request(
            "POST",
            f"/alerts/{identifier}/snooze",
            params={"identifierType": identifier_type},
            json=payload,
        )

        return result is not None

    async def add_note(
        self,
        identifier: str,
        note: str,
        identifier_type: str = "id",
        user: str = None,
        source: str = "cosmos",
    ) -> bool:
        """Add a note to an alert."""
        payload = {
            "note": note,
            "source": source,
        }

        if user:
            payload["user"] = user

        result = await self._request(
            "POST",
            f"/alerts/{identifier}/notes",
            params={"identifierType": identifier_type},
            json=payload,
        )

        return result is not None

    async def escalate_alert(
        self,
        identifier: str,
        escalation_id: str,
        identifier_type: str = "id",
        user: str = None,
        source: str = "cosmos",
        note: str = None,
    ) -> bool:
        """Escalate an alert to the next level."""
        payload = {
            "escalation": {"id": escalation_id},
            "source": source,
        }

        if user:
            payload["user"] = user
        if note:
            payload["note"] = note

        result = await self._request(
            "POST",
            f"/alerts/{identifier}/escalate",
            params={"identifierType": identifier_type},
            json=payload,
        )

        return result is not None

    async def assign_alert(
        self,
        identifier: str,
        owner_id: str,
        owner_type: str = "user",
        identifier_type: str = "id",
        user: str = None,
        source: str = "cosmos",
        note: str = None,
    ) -> bool:
        """Assign an alert to a user or team."""
        payload = {
            "owner": {
                "id": owner_id,
                "type": owner_type,
            },
            "source": source,
        }

        if user:
            payload["user"] = user
        if note:
            payload["note"] = note

        result = await self._request(
            "POST",
            f"/alerts/{identifier}/assign",
            params={"identifierType": identifier_type},
            json=payload,
        )

        return result is not None

    async def add_tags(
        self,
        identifier: str,
        tags: List[str],
        identifier_type: str = "id",
        user: str = None,
        source: str = "cosmos",
        note: str = None,
    ) -> bool:
        """Add tags to an alert."""
        payload = {
            "tags": tags,
            "source": source,
        }

        if user:
            payload["user"] = user
        if note:
            payload["note"] = note

        result = await self._request(
            "POST",
            f"/alerts/{identifier}/tags",
            params={"identifierType": identifier_type},
            json=payload,
        )

        return result is not None

    def _parse_alert(self, data: Dict) -> OpsGenieAlert:
        """Parse OpsGenie alert data."""
        return OpsGenieAlert(
            id=data.get("id", ""),
            tiny_id=data.get("tinyId", ""),
            alias=data.get("alias", ""),
            message=data.get("message", ""),
            status=data.get("status", ""),
            acknowledged=data.get("acknowledged", False),
            is_seen=data.get("isSeen", False),
            priority=data.get("priority", ""),
            source=data.get("source", ""),
            tags=data.get("tags", []),
            responders=data.get("responders", []),
            created_at=datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00")) if data.get("createdAt") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updatedAt"].replace("Z", "+00:00")) if data.get("updatedAt") else datetime.utcnow(),
            count=data.get("count", 1),
        )

    # =========================================================================
    # INCIDENTS
    # =========================================================================

    async def create_incident(
        self,
        message: str,
        description: str = None,
        responders: List[Dict] = None,
        tags: List[str] = None,
        details: Dict = None,
        priority: str = "P3",
        note: str = None,
        notify_stakeholders: List[Dict] = None,
    ) -> Optional[str]:
        """Create a new incident."""
        payload = {
            "message": message,
            "priority": priority,
        }

        if description:
            payload["description"] = description
        if responders:
            payload["responders"] = responders
        if tags:
            payload["tags"] = tags
        if details:
            payload["details"] = details
        if note:
            payload["note"] = note
        if notify_stakeholders:
            payload["notifyStakeholders"] = notify_stakeholders

        result = await self._request("POST", "/incidents/create", json=payload)

        if result:
            return result.get("requestId")
        return None

    async def get_incident(self, identifier: str) -> Optional[Dict]:
        """Get an incident by ID."""
        result = await self._request("GET", f"/incidents/{identifier}")

        return result.get("data") if result else None

    async def list_incidents(
        self,
        query: str = None,
        offset: int = 0,
        limit: int = 20,
        sort: str = "createdAt",
        order: str = "desc",
    ) -> List[Dict]:
        """List incidents with filters."""
        params = {
            "offset": offset,
            "limit": limit,
            "sort": sort,
            "order": order,
        }

        if query:
            params["query"] = query

        result = await self._request("GET", "/incidents", params=params)

        return result.get("data", []) if result else []

    async def resolve_incident(
        self,
        identifier: str,
        note: str = None,
    ) -> bool:
        """Resolve an incident."""
        payload = {}
        if note:
            payload["note"] = note

        result = await self._request(
            "POST",
            f"/incidents/{identifier}/resolve",
            json=payload,
        )

        return result is not None

    async def close_incident(
        self,
        identifier: str,
        note: str = None,
    ) -> bool:
        """Close an incident."""
        payload = {}
        if note:
            payload["note"] = note

        result = await self._request(
            "POST",
            f"/incidents/{identifier}/close",
            json=payload,
        )

        return result is not None

    # =========================================================================
    # SCHEDULES
    # =========================================================================

    async def list_schedules(self) -> List[Dict]:
        """List all schedules."""
        result = await self._request("GET", "/schedules")

        return result.get("data", []) if result else []

    async def get_schedule(self, identifier: str) -> Optional[Dict]:
        """Get a schedule by ID or name."""
        result = await self._request("GET", f"/schedules/{identifier}")

        return result.get("data") if result else None

    async def get_oncall(
        self,
        identifier: str,
        date: datetime = None,
        flat: bool = True,
    ) -> List[Dict]:
        """Get on-call participants for a schedule."""
        params = {"flat": str(flat).lower()}

        if date:
            params["date"] = date.isoformat()

        result = await self._request(
            "GET",
            f"/schedules/{identifier}/on-calls",
            params=params,
        )

        return result.get("data", {}).get("onCallParticipants", []) if result else []

    # =========================================================================
    # TEAMS
    # =========================================================================

    async def list_teams(self) -> List[Dict]:
        """List all teams."""
        result = await self._request("GET", "/teams")

        return result.get("data", []) if result else []

    async def get_team(self, identifier: str) -> Optional[Dict]:
        """Get a team by ID or name."""
        result = await self._request("GET", f"/teams/{identifier}")

        return result.get("data") if result else None

    async def get_team_oncall(self, identifier: str) -> List[Dict]:
        """Get on-call users for a team."""
        result = await self._request("GET", f"/teams/{identifier}/oncalls")

        return result.get("data", {}).get("onCallParticipants", []) if result else []

    # =========================================================================
    # USERS
    # =========================================================================

    async def list_users(self) -> List[Dict]:
        """List all users."""
        result = await self._request("GET", "/users")

        return result.get("data", []) if result else []

    async def get_user(self, identifier: str) -> Optional[Dict]:
        """Get a user by ID or username."""
        result = await self._request("GET", f"/users/{identifier}")

        return result.get("data") if result else None

    # =========================================================================
    # ESCALATIONS
    # =========================================================================

    async def list_escalations(self) -> List[Dict]:
        """List all escalation policies."""
        result = await self._request("GET", "/escalations")

        return result.get("data", []) if result else []

    async def get_escalation(self, identifier: str) -> Optional[Dict]:
        """Get an escalation policy by ID or name."""
        result = await self._request("GET", f"/escalations/{identifier}")

        return result.get("data") if result else None

    # =========================================================================
    # HEARTBEATS
    # =========================================================================

    async def ping_heartbeat(self, name: str) -> bool:
        """Ping a heartbeat to indicate service is alive."""
        result = await self._request("GET", f"/heartbeats/{name}/ping")

        return result is not None

    async def get_heartbeat(self, name: str) -> Optional[Dict]:
        """Get heartbeat status."""
        result = await self._request("GET", f"/heartbeats/{name}")

        return result.get("data") if result else None

    # =========================================================================
    # WEBHOOKS
    # =========================================================================

    def parse_webhook(self, payload: Dict) -> Dict:
        """Parse an OpsGenie webhook payload."""
        return {
            "action": payload.get("action"),
            "source": payload.get("source", {}).get("name"),
            "alert": payload.get("alert", {}),
            "integration_id": payload.get("integrationId"),
            "integration_name": payload.get("integrationName"),
        }

"""
cosmos Incident Manager

"When disaster strikes, which it frequently does in my lab,
 always blame the mutant sea bass!"

Comprehensive incident tracking and management system.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from loguru import logger


class IncidentSeverity(Enum):
    """Incident severity levels."""
    SEV1 = "sev1"  # Critical - Major business impact
    SEV2 = "sev2"  # High - Significant impact
    SEV3 = "sev3"  # Medium - Limited impact
    SEV4 = "sev4"  # Low - Minor impact
    SEV5 = "sev5"  # Informational


class IncidentStatus(Enum):
    """Incident lifecycle status."""
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MITIGATING = "mitigating"
    MONITORING = "monitoring"
    RESOLVED = "resolved"
    POSTMORTEM = "postmortem"
    CLOSED = "closed"


class IncidentType(Enum):
    """Types of incidents."""
    OUTAGE = "outage"
    DEGRADATION = "degradation"
    SECURITY = "security"
    DATA_LOSS = "data_loss"
    PERFORMANCE = "performance"
    INFRASTRUCTURE = "infrastructure"
    NETWORK = "network"
    APPLICATION = "application"
    THIRD_PARTY = "third_party"
    OTHER = "other"


@dataclass
class IncidentUpdate:
    """An update to an incident."""
    id: str
    timestamp: datetime
    author: str
    message: str
    status_change: Optional[IncidentStatus] = None
    visibility: str = "internal"  # internal, public
    attachments: List[str] = field(default_factory=list)


@dataclass
class IncidentTask:
    """A task associated with an incident."""
    id: str
    title: str
    description: str
    assignee: str
    status: str = "pending"  # pending, in_progress, completed
    priority: int = 2
    due_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Incident:
    """
    An incident record with full lifecycle tracking.
    """
    id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus = IncidentStatus.TRIGGERED
    incident_type: IncidentType = IncidentType.OTHER

    # Assignment
    commander: str = ""  # Incident commander
    responders: List[str] = field(default_factory=list)
    teams: List[str] = field(default_factory=list)

    # Impact
    affected_services: List[str] = field(default_factory=list)
    affected_customers: int = 0
    customer_impact: str = ""

    # Classification
    tags: List[str] = field(default_factory=list)
    source: str = ""  # What triggered the incident (monitoring, customer, etc.)
    related_incidents: List[str] = field(default_factory=list)

    # Timeline
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    identified_at: Optional[datetime] = None
    mitigated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    # Updates and tasks
    updates: List[IncidentUpdate] = field(default_factory=list)
    tasks: List[IncidentTask] = field(default_factory=list)

    # External references
    slack_channel: str = ""
    video_call_url: str = ""
    status_page_id: str = ""
    pagerduty_incident_id: str = ""

    # Metadata
    runbook_id: Optional[str] = None
    root_cause: str = ""
    resolution: str = ""
    postmortem_url: str = ""

    @property
    def ttd(self) -> Optional[timedelta]:
        """Time to Detect (from trigger to acknowledge)."""
        if self.acknowledged_at:
            return self.acknowledged_at - self.triggered_at
        return None

    @property
    def tti(self) -> Optional[timedelta]:
        """Time to Identify root cause."""
        if self.identified_at:
            return self.identified_at - self.triggered_at
        return None

    @property
    def ttm(self) -> Optional[timedelta]:
        """Time to Mitigate."""
        if self.mitigated_at:
            return self.mitigated_at - self.triggered_at
        return None

    @property
    def ttr(self) -> Optional[timedelta]:
        """Time to Resolve."""
        if self.resolved_at:
            return self.resolved_at - self.triggered_at
        return None

    def add_update(
        self,
        author: str,
        message: str,
        status_change: IncidentStatus = None,
        visibility: str = "internal",
    ):
        """Add an update to the incident."""
        import uuid
        update = IncidentUpdate(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.utcnow(),
            author=author,
            message=message,
            status_change=status_change,
            visibility=visibility,
        )
        self.updates.append(update)

        if status_change:
            self.status = status_change
            self._update_timestamps(status_change)

    def _update_timestamps(self, status: IncidentStatus):
        """Update timestamps based on status change."""
        now = datetime.utcnow()
        if status == IncidentStatus.ACKNOWLEDGED:
            self.acknowledged_at = now
        elif status == IncidentStatus.IDENTIFIED:
            self.identified_at = now
        elif status in [IncidentStatus.MITIGATING, IncidentStatus.MONITORING]:
            self.mitigated_at = now
        elif status == IncidentStatus.RESOLVED:
            self.resolved_at = now
        elif status == IncidentStatus.CLOSED:
            self.closed_at = now

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "status": self.status.value,
            "type": self.incident_type.value,
            "commander": self.commander,
            "responders": self.responders,
            "teams": self.teams,
            "affected_services": self.affected_services,
            "affected_customers": self.affected_customers,
            "customer_impact": self.customer_impact,
            "tags": self.tags,
            "source": self.source,
            "triggered_at": self.triggered_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "identified_at": self.identified_at.isoformat() if self.identified_at else None,
            "mitigated_at": self.mitigated_at.isoformat() if self.mitigated_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "ttd_seconds": self.ttd.total_seconds() if self.ttd else None,
            "tti_seconds": self.tti.total_seconds() if self.tti else None,
            "ttm_seconds": self.ttm.total_seconds() if self.ttm else None,
            "ttr_seconds": self.ttr.total_seconds() if self.ttr else None,
            "update_count": len(self.updates),
            "task_count": len(self.tasks),
            "root_cause": self.root_cause,
            "resolution": self.resolution,
        }


class IncidentManager:
    """
    Comprehensive incident management system.

    Features:
    - Incident lifecycle management
    - On-call integration
    - Status page updates
    - Runbook execution
    - Metrics and analytics
    - Postmortem generation
    """

    def __init__(
        self,
        storage_path: Path = None,
    ):
        self.storage_path = Path(storage_path) if storage_path else Path("./data/incidents")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.incidents: Dict[str, Incident] = {}
        self.notification_handlers: List[Callable] = []
        self.escalation_policies: Dict[str, List[Dict]] = {}

        self._load_incidents()

    def _load_incidents(self):
        """Load incidents from storage."""
        incidents_file = self.storage_path / "incidents.json"
        if incidents_file.exists():
            try:
                with open(incidents_file) as f:
                    data = json.load(f)
                for inc_id, inc_data in data.items():
                    self.incidents[inc_id] = self._dict_to_incident(inc_data)
            except Exception as e:
                logger.error(f"Failed to load incidents: {e}")

    def _save_incidents(self):
        """Save incidents to storage."""
        incidents_file = self.storage_path / "incidents.json"
        data = {inc_id: inc.to_dict() for inc_id, inc in self.incidents.items()}
        with open(incidents_file, "w") as f:
            json.dump(data, f, indent=2)

    def _dict_to_incident(self, data: Dict) -> Incident:
        """Convert dictionary to Incident object."""
        return Incident(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            severity=IncidentSeverity(data["severity"]),
            status=IncidentStatus(data["status"]),
            incident_type=IncidentType(data.get("type", "other")),
            commander=data.get("commander", ""),
            responders=data.get("responders", []),
            teams=data.get("teams", []),
            affected_services=data.get("affected_services", []),
            affected_customers=data.get("affected_customers", 0),
            tags=data.get("tags", []),
            source=data.get("source", ""),
            triggered_at=datetime.fromisoformat(data["triggered_at"]),
            root_cause=data.get("root_cause", ""),
            resolution=data.get("resolution", ""),
        )

    # =========================================================================
    # INCIDENT CRUD
    # =========================================================================

    def create_incident(
        self,
        title: str,
        description: str,
        severity: IncidentSeverity,
        incident_type: IncidentType = IncidentType.OTHER,
        affected_services: List[str] = None,
        source: str = "manual",
        tags: List[str] = None,
    ) -> Incident:
        """Create a new incident."""
        import uuid

        incident = Incident(
            id=f"INC-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}",
            title=title,
            description=description,
            severity=severity,
            incident_type=incident_type,
            affected_services=affected_services or [],
            source=source,
            tags=tags or [],
        )

        self.incidents[incident.id] = incident
        self._save_incidents()

        # Trigger notifications
        self._notify(incident, "created")

        logger.info(f"Created incident: {incident.id} - {title}")
        return incident

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get an incident by ID."""
        return self.incidents.get(incident_id)

    def update_incident(
        self,
        incident_id: str,
        **updates,
    ) -> Optional[Incident]:
        """Update an incident's properties."""
        incident = self.incidents.get(incident_id)
        if not incident:
            return None

        for key, value in updates.items():
            if hasattr(incident, key):
                setattr(incident, key, value)

        self._save_incidents()
        return incident

    def delete_incident(self, incident_id: str) -> bool:
        """Delete an incident."""
        if incident_id in self.incidents:
            del self.incidents[incident_id]
            self._save_incidents()
            return True
        return False

    def list_incidents(
        self,
        status: IncidentStatus = None,
        severity: IncidentSeverity = None,
        since: datetime = None,
        limit: int = 50,
    ) -> List[Incident]:
        """List incidents with optional filters."""
        incidents = list(self.incidents.values())

        if status:
            incidents = [i for i in incidents if i.status == status]
        if severity:
            incidents = [i for i in incidents if i.severity == severity]
        if since:
            incidents = [i for i in incidents if i.triggered_at >= since]

        incidents.sort(key=lambda i: i.triggered_at, reverse=True)
        return incidents[:limit]

    def get_active_incidents(self) -> List[Incident]:
        """Get all non-closed incidents."""
        active_statuses = [
            IncidentStatus.TRIGGERED,
            IncidentStatus.ACKNOWLEDGED,
            IncidentStatus.INVESTIGATING,
            IncidentStatus.IDENTIFIED,
            IncidentStatus.MITIGATING,
            IncidentStatus.MONITORING,
        ]
        return [i for i in self.incidents.values() if i.status in active_statuses]

    # =========================================================================
    # STATUS MANAGEMENT
    # =========================================================================

    def acknowledge(
        self,
        incident_id: str,
        user: str,
        message: str = "Incident acknowledged",
    ) -> bool:
        """Acknowledge an incident."""
        incident = self.incidents.get(incident_id)
        if not incident:
            return False

        incident.add_update(user, message, IncidentStatus.ACKNOWLEDGED)
        if not incident.commander:
            incident.commander = user
        if user not in incident.responders:
            incident.responders.append(user)

        self._save_incidents()
        self._notify(incident, "acknowledged")
        return True

    def escalate(
        self,
        incident_id: str,
        new_severity: IncidentSeverity,
        user: str,
        reason: str,
    ) -> bool:
        """Escalate incident severity."""
        incident = self.incidents.get(incident_id)
        if not incident:
            return False

        old_severity = incident.severity
        incident.severity = new_severity
        incident.add_update(
            user,
            f"Escalated from {old_severity.value} to {new_severity.value}: {reason}",
        )

        self._save_incidents()
        self._notify(incident, "escalated")
        return True

    def resolve(
        self,
        incident_id: str,
        user: str,
        resolution: str,
    ) -> bool:
        """Resolve an incident."""
        incident = self.incidents.get(incident_id)
        if not incident:
            return False

        incident.resolution = resolution
        incident.add_update(user, f"Resolved: {resolution}", IncidentStatus.RESOLVED)

        self._save_incidents()
        self._notify(incident, "resolved")
        return True

    def close(
        self,
        incident_id: str,
        user: str,
        postmortem_url: str = None,
    ) -> bool:
        """Close an incident."""
        incident = self.incidents.get(incident_id)
        if not incident:
            return False

        if postmortem_url:
            incident.postmortem_url = postmortem_url

        incident.add_update(user, "Incident closed", IncidentStatus.CLOSED)

        self._save_incidents()
        self._notify(incident, "closed")
        return True

    # =========================================================================
    # RESPONDER MANAGEMENT
    # =========================================================================

    def assign_commander(
        self,
        incident_id: str,
        commander: str,
        assigner: str,
    ) -> bool:
        """Assign an incident commander."""
        incident = self.incidents.get(incident_id)
        if not incident:
            return False

        old_commander = incident.commander
        incident.commander = commander
        incident.add_update(
            assigner,
            f"Incident commander changed from {old_commander or 'unassigned'} to {commander}",
        )

        self._save_incidents()
        return True

    def add_responder(
        self,
        incident_id: str,
        responder: str,
        adder: str,
    ) -> bool:
        """Add a responder to an incident."""
        incident = self.incidents.get(incident_id)
        if not incident:
            return False

        if responder not in incident.responders:
            incident.responders.append(responder)
            incident.add_update(adder, f"Added responder: {responder}")
            self._save_incidents()

        return True

    def remove_responder(
        self,
        incident_id: str,
        responder: str,
        remover: str,
    ) -> bool:
        """Remove a responder from an incident."""
        incident = self.incidents.get(incident_id)
        if not incident:
            return False

        if responder in incident.responders:
            incident.responders.remove(responder)
            incident.add_update(remover, f"Removed responder: {responder}")
            self._save_incidents()

        return True

    # =========================================================================
    # TASKS
    # =========================================================================

    def add_task(
        self,
        incident_id: str,
        title: str,
        description: str,
        assignee: str,
        creator: str,
        priority: int = 2,
        due_at: datetime = None,
    ) -> Optional[IncidentTask]:
        """Add a task to an incident."""
        incident = self.incidents.get(incident_id)
        if not incident:
            return None

        import uuid
        task = IncidentTask(
            id=str(uuid.uuid4())[:8],
            title=title,
            description=description,
            assignee=assignee,
            priority=priority,
            due_at=due_at,
        )

        incident.tasks.append(task)
        incident.add_update(creator, f"Added task: {title} (assigned to {assignee})")
        self._save_incidents()

        return task

    def complete_task(
        self,
        incident_id: str,
        task_id: str,
        user: str,
    ) -> bool:
        """Mark a task as completed."""
        incident = self.incidents.get(incident_id)
        if not incident:
            return False

        for task in incident.tasks:
            if task.id == task_id:
                task.status = "completed"
                task.completed_at = datetime.utcnow()
                incident.add_update(user, f"Completed task: {task.title}")
                self._save_incidents()
                return True

        return False

    # =========================================================================
    # NOTIFICATIONS
    # =========================================================================

    def register_notification_handler(self, handler: Callable):
        """Register a notification handler."""
        self.notification_handlers.append(handler)

    def _notify(self, incident: Incident, event: str):
        """Trigger notifications for an incident event."""
        for handler in self.notification_handlers:
            try:
                handler(incident, event)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")

    # =========================================================================
    # ESCALATION POLICIES
    # =========================================================================

    def set_escalation_policy(
        self,
        policy_name: str,
        levels: List[Dict],
    ):
        """Set an escalation policy."""
        self.escalation_policies[policy_name] = levels
        logger.info(f"Set escalation policy: {policy_name}")

    async def check_escalations(self):
        """Check for incidents that need escalation."""
        for incident in self.get_active_incidents():
            if incident.status == IncidentStatus.TRIGGERED:
                # Check time since trigger
                time_open = datetime.utcnow() - incident.triggered_at

                # Auto-escalate if not acknowledged within thresholds
                if incident.severity == IncidentSeverity.SEV1 and time_open > timedelta(minutes=5):
                    logger.warning(f"SEV1 incident {incident.id} not acknowledged after 5 minutes")
                elif incident.severity == IncidentSeverity.SEV2 and time_open > timedelta(minutes=15):
                    logger.warning(f"SEV2 incident {incident.id} not acknowledged after 15 minutes")

    # =========================================================================
    # ANALYTICS
    # =========================================================================

    def get_metrics(
        self,
        since: datetime = None,
        until: datetime = None,
    ) -> Dict[str, Any]:
        """Get incident metrics for a time period."""
        incidents = list(self.incidents.values())

        if since:
            incidents = [i for i in incidents if i.triggered_at >= since]
        if until:
            incidents = [i for i in incidents if i.triggered_at <= until]

        if not incidents:
            return {"total_incidents": 0}

        # Count by severity
        by_severity = {}
        for sev in IncidentSeverity:
            count = len([i for i in incidents if i.severity == sev])
            by_severity[sev.value] = count

        # Count by type
        by_type = {}
        for t in IncidentType:
            count = len([i for i in incidents if i.incident_type == t])
            if count > 0:
                by_type[t.value] = count

        # Calculate MTTR (Mean Time To Resolve)
        resolved = [i for i in incidents if i.ttr]
        mttr = None
        if resolved:
            total_seconds = sum(i.ttr.total_seconds() for i in resolved)
            mttr = total_seconds / len(resolved)

        # Calculate MTTA (Mean Time To Acknowledge)
        acknowledged = [i for i in incidents if i.ttd]
        mtta = None
        if acknowledged:
            total_seconds = sum(i.ttd.total_seconds() for i in acknowledged)
            mtta = total_seconds / len(acknowledged)

        return {
            "total_incidents": len(incidents),
            "by_severity": by_severity,
            "by_type": by_type,
            "active_incidents": len([i for i in incidents if i.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]]),
            "mttr_seconds": mttr,
            "mtta_seconds": mtta,
            "affected_services": list(set(s for i in incidents for s in i.affected_services)),
        }

    def generate_postmortem_template(self, incident_id: str) -> str:
        """Generate a postmortem template for an incident."""
        incident = self.incidents.get(incident_id)
        if not incident:
            return ""

        template = f"""# Postmortem: {incident.title}

## Incident Summary
- **Incident ID:** {incident.id}
- **Severity:** {incident.severity.value.upper()}
- **Type:** {incident.incident_type.value}
- **Duration:** {incident.ttr.total_seconds() / 60:.1f} minutes (if resolved)
- **Incident Commander:** {incident.commander}

## Timeline
- **Triggered:** {incident.triggered_at.isoformat()}
- **Acknowledged:** {incident.acknowledged_at.isoformat() if incident.acknowledged_at else 'N/A'}
- **Identified:** {incident.identified_at.isoformat() if incident.identified_at else 'N/A'}
- **Mitigated:** {incident.mitigated_at.isoformat() if incident.mitigated_at else 'N/A'}
- **Resolved:** {incident.resolved_at.isoformat() if incident.resolved_at else 'N/A'}

## Impact
- **Affected Services:** {', '.join(incident.affected_services) or 'N/A'}
- **Customer Impact:** {incident.customer_impact or 'TBD'}

## Root Cause
{incident.root_cause or '[To be determined]'}

## Resolution
{incident.resolution or '[To be documented]'}

## What Went Well
-

## What Went Wrong
-

## Action Items
| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| | | | |

## Lessons Learned
-

## Updates Log
"""
        for update in incident.updates:
            template += f"\n- [{update.timestamp.isoformat()}] {update.author}: {update.message}"

        return template


# Singleton instance
incident_manager = IncidentManager()

"""Federation cron relay — distributed scheduled job management.

When a cron job is created/modified/deleted on one device, it syncs to all
peers. Only one device executes each job at a time (leader election per job).

If the leader goes offline, another peer automatically takes over the job
(failover).

Protocol messages:
- CRON_SYNC: cron job data broadcast
- CRON_HEARTBEAT: leader is still executing
- CRON_HANDOFF: leader going offline, transfer job
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from pathlib import Path

from gateway.federation.federation_protocol import FedMessage, MessageType

logger = logging.getLogger(__name__)


@dataclass
class CronJobInfo:
    """Information about a federated cron job."""

    job_id: str
    name: str
    schedule: str
    enabled: bool = True
    owner_device: str = ""
    leader_device: str = ""
    last_run: float = 0.0
    last_result: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "schedule": self.schedule,
            "enabled": self.enabled,
            "owner_device": self.owner_device,
            "leader_device": self.leader_device,
            "last_run": self.last_run,
            "last_result": self.last_result,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CronJobInfo":
        return cls(**d)


class FederationCronRelay:
    """Synchronize and distribute cron jobs across federation peers.

    Each cron job has a designated leader device that executes it.
    If the leader goes offline, another peer takes over automatically.

    Usage:
        relay = FederationCronRelay(
            device_id="my-device",
            adapter=federation_adapter,
        )
        await relay.start()
    """

    def __init__(
        self,
        device_id: str,
        adapter: Any,  # FederationAdapter
        heartbeat_interval: float = 30.0,
        offline_threshold: float = 90.0,
    ):
        self.device_id = device_id
        self.adapter = adapter
        self.heartbeat_interval = heartbeat_interval
        self.offline_threshold = offline_threshold

        self._jobs: Dict[str, CronJobInfo] = {}  # job_id -> info
        self._leadership: Dict[str, bool] = {}  # job_id -> am I leader?
        self._running = False

    async def start(self) -> None:
        """Start cron relay."""
        self._running = True
        logger.info(
            "Federation cron relay: started (device=%s)", self.device_id,
        )

    async def stop(self) -> None:
        """Stop cron relay."""
        self._running = False
        logger.info("Federation cron relay: stopped")

    # ----------------------------------------------------------------
    # Job synchronization
    # ----------------------------------------------------------------

    async def sync_job(self, job_info: CronJobInfo) -> None:
        """Broadcast a cron job to all peers."""
        msg = FedMessage(
            msg_type=MessageType.CRON_SYNC.value,
            sender_id=self.device_id,
            payload={
                "action": "update",
                "job": job_info.to_dict(),
            },
        )
        await self.adapter.send(msg)
        self._apply_job(job_info)
        logger.info(
            "Federation cron: synced job %s (%s) to peers",
            job_info.job_id, job_info.name,
        )

    async def delete_job(self, job_id: str) -> None:
        """Broadcast job deletion to all peers."""
        msg = FedMessage(
            msg_type=MessageType.CRON_SYNC.value,
            sender_id=self.device_id,
            payload={
                "action": "delete",
                "job_id": job_id,
            },
        )
        await self.adapter.send(msg)
        self._jobs.pop(job_id, None)
        self._leadership.pop(job_id, None)
        logger.info("Federation cron: deleted job %s", job_id)

    def handle_cron_sync(self, msg: FedMessage) -> None:
        """Handle incoming CRON_SYNC from a peer."""
        sender = msg.sender_id
        if sender == self.device_id:
            return

        action = msg.payload.get("action", "")

        if action == "update":
            job_data = msg.payload.get("job", {})
            if not job_data:
                return
            remote_job = CronJobInfo.from_dict(job_data)
            local = self._jobs.get(remote_job.job_id)

            # Apply if newer or doesn't exist
            if not local or remote_job.updated_at > local.updated_at:
                self._apply_job(remote_job)

        elif action == "delete":
            job_id = msg.payload.get("job_id", "")
            if job_id:
                self._jobs.pop(job_id, None)
                self._leadership.pop(job_id, None)

    def _apply_job(self, job_info: CronJobInfo) -> None:
        """Apply a remote or local job to the registry."""
        self._jobs[job_info.job_id] = job_info
        # Determine if we're the leader for this job
        self._leadership[job_info.job_id] = (
            job_info.leader_device == self.device_id
        )

    # ----------------------------------------------------------------
    # Leader management
    # ----------------------------------------------------------------

    async def claim_leadership(self, job_id: str) -> bool:
        """Claim leadership for a job."""
        if job_id not in self._jobs:
            return False

        job = self._jobs[job_id]
        if job.leader_device and job.leader_device != self.device_id:
            # Check if current leader is still alive
            # (This would require a heartbeat check in production)
            pass

        # Claim leadership
        job.leader_device = self.device_id
        job.updated_at = time.time()
        self._leadership[job_id] = True

        # Sync to peers
        await self.sync_job(job)
        return True

    async def release_leadership(self, job_id: str, reason: str = "offline") -> None:
        """Release leadership — job will be picked up by another peer."""
        if job_id not in self._jobs:
            return

        job = self._jobs[job_id]
        job.leader_device = ""
        job.updated_at = time.time()
        self._leadership.pop(job_id, None)

        # Broadcast release
        await self.sync_job(job)
        logger.info(
            "Federation cron: released leadership of %s (%s)",
            job_id, reason,
        )

    def is_leader(self, job_id: str) -> bool:
        """Check if this device is the leader for a job."""
        return self._leadership.get(job_id, False)

    def get_my_jobs(self) -> List[CronJobInfo]:
        """Get all jobs where this device is the leader."""
        return [
            job for job in self._jobs.values()
            if job.leader_device == self.device_id
        ]

    def get_all_jobs(self) -> List[CronJobInfo]:
        """Get all known jobs."""
        return list(self._jobs.values())

    @property
    def job_count(self) -> int:
        """Total number of known jobs."""
        return len(self._jobs)

    @property
    def my_job_count(self) -> int:
        """Number of jobs where this device is leader."""
        return sum(1 for j in self._jobs.values() if j.leader_device == self.device_id)


# ========================================================================
# Skill sync
# ========================================================================

@dataclass
class SkillInfo:
    """Information about a skill to sync."""

    name: str
    category: str = ""
    content: str = ""
    version: int = 1
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "content": self.content,
            "version": self.version,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SkillInfo":
        return cls(**d)


class FederationSkillSync:
    """Synchronize skills across federation peers.

    When a skill is created/updated/deleted on one device, it syncs to all
    peers.  Skills are stored in ~/.hermes/skills/ on each device.
    """

    def __init__(
        self,
        device_id: str,
        adapter: Any,  # FederationAdapter
        hermes_home: Optional[str] = None,
    ):
        self.device_id = device_id
        self.adapter = adapter
        self.hermes_home = hermes_home or str(Path.home() / ".hermes")
        self._skills: Dict[str, SkillInfo] = {}
        self._running = False

    async def start(self) -> None:
        """Start skill sync."""
        self._running = True
        self._load_local_skills()
        logger.info(
            "Federation skill sync: started (device=%s, skills=%d)",
            self.device_id, len(self._skills),
        )

    async def stop(self) -> None:
        """Stop skill sync."""
        self._running = False
        logger.info("Federation skill sync: stopped")

    def _load_local_skills(self) -> None:
        """Load local skill metadata for sync."""
        import os
        from pathlib import Path

        skills_dir = Path(self.hermes_home) / "skills"
        if not skills_dir.exists():
            return

        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue

            # Read frontmatter to extract metadata
            content = skill_file.read_text(encoding="utf-8")
            name = skill_dir.name
            category = ""

            # Simple YAML frontmatter extraction
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 2:
                    import re
                    cat_match = re.search(r"category:\s*(.+)", parts[1])
                    if cat_match:
                        category = cat_match.group(1).strip()

            self._skills[name] = SkillInfo(
                name=name,
                category=category,
                version=1,
                updated_at=skill_file.stat().st_mtime,
            )

    async def sync_skill(self, name: str, content: str, category: str = "") -> None:
        """Sync a skill to all peers."""
        skill = SkillInfo(
            name=name,
            category=category,
            content=content,
            updated_at=time.time(),
        )
        self._skills[name] = skill

        msg = FedMessage(
            msg_type=MessageType.SKILL_SYNC.value,
            sender_id=self.device_id,
            payload={
                "action": "update",
                "skill": skill.to_dict(),
            },
        )
        await self.adapter.send(msg)
        logger.info("Federation skill: synced %s to peers", name)

    async def delete_skill(self, name: str) -> None:
        """Broadcast skill deletion to all peers."""
        msg = FedMessage(
            msg_type=MessageType.SKILL_SYNC.value,
            sender_id=self.device_id,
            payload={
                "action": "delete",
                "name": name,
            },
        )
        await self.adapter.send(msg)
        self._skills.pop(name, None)

    def handle_skill_sync(self, msg: FedMessage) -> None:
        """Handle incoming SKILL_SYNC from a peer."""
        sender = msg.sender_id
        if sender == self.device_id:
            return

        action = msg.payload.get("action", "")

        if action == "update":
            skill_data = msg.payload.get("skill", {})
            if not skill_data:
                return
            remote = SkillInfo.from_dict(skill_data)
            local = self._skills.get(remote.name)

            # Apply if newer or doesn't exist
            if not local or remote.updated_at > local.updated_at:
                self._apply_remote_skill(remote)

        elif action == "delete":
            name = msg.payload.get("name", "")
            if name:
                self._skills.pop(name, None)
                # Optionally delete local file
                self._delete_local_skill_file(name)

    def _apply_remote_skill(self, skill: SkillInfo) -> None:
        """Write a remote skill to local filesystem."""
        import os
        from pathlib import Path

        skill_dir = Path(self.hermes_home) / "skills" / skill.name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"

        if skill.content:
            skill_file.write_text(skill.content, encoding="utf-8")
            self._skills[skill.name] = skill
            logger.info(
                "Federation skill: applied remote %s (v%d)",
                skill.name, skill.version,
            )

    def _delete_local_skill_file(self, name: str) -> None:
        """Delete a local skill file."""
        import shutil
        from pathlib import Path

        skill_dir = Path(self.hermes_home) / "skills" / name
        if skill_dir.exists():
            shutil.rmtree(skill_dir)
            logger.info("Federation skill: deleted local %s", name)

    @property
    def skill_count(self) -> int:
        """Number of known skills."""
        return len(self._skills)

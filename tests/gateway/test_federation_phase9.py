"""Tests for federation Phase 9 — cron relay + skill sync."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gateway.federation.federation_cron_relay import (
    CronJobInfo,
    FederationCronRelay,
    FederationSkillSync,
    SkillInfo,
)


# ========================================================================
# CronJobInfo tests
# ========================================================================

class TestCronJobInfo:
    def test_roundtrip(self):
        job = CronJobInfo(
            job_id="job-001",
            name="daily backup",
            schedule="0 2 * * *",
            leader_device="dev-a",
        )
        d = job.to_dict()
        restored = CronJobInfo.from_dict(d)
        assert restored.job_id == "job-001"
        assert restored.name == "daily backup"
        assert restored.schedule == "0 2 * * *"
        assert restored.leader_device == "dev-a"


# ========================================================================
# FederationCronRelay tests
# ========================================================================

class TestFederationCronRelay:
    def _make_relay(self):
        adapter = MagicMock()
        from unittest.mock import AsyncMock
        adapter.send = AsyncMock()
        return FederationCronRelay(
            device_id="dev-a",
            adapter=adapter,
        )

    def test_init(self):
        relay = self._make_relay()
        assert relay.device_id == "dev-a"
        assert relay.job_count == 0

    def test_sync_job_broadcasts(self):
        relay = self._make_relay()
        job = CronJobInfo(
            job_id="j-001",
            name="test job",
            schedule="*/5 * * * *",
            leader_device="dev-a",
        )
        import asyncio
        asyncio.get_event_loop().run_until_complete(relay.sync_job(job))
        relay.adapter.send.assert_called_once()

    def test_apply_job(self):
        relay = self._make_relay()
        job = CronJobInfo(
            job_id="j-002",
            name="another job",
            schedule="0 * * * *",
            leader_device="dev-b",
        )
        relay._apply_job(job)
        assert relay.job_count == 1
        assert relay.is_leader("j-002") is False  # Not our job

    def test_claim_leadership(self):
        relay = self._make_relay()
        job = CronJobInfo(
            job_id="j-003",
            name="claimable job",
            schedule="0 * * * *",
        )
        relay._jobs["j-003"] = job

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            relay.claim_leadership("j-003")
        )
        assert relay.is_leader("j-003") is True

    def test_release_leadership(self):
        relay = self._make_relay()
        job = CronJobInfo(
            job_id="j-004",
            name="release job",
            schedule="0 * * * *",
            leader_device="dev-a",
        )
        relay._jobs["j-004"] = job
        relay._leadership["j-004"] = True

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            relay.release_leadership("j-004")
        )
        assert relay.is_leader("j-004") is False

    def test_handle_cron_sync_update(self):
        relay = self._make_relay()
        msg = MagicMock()
        msg.sender_id = "dev-b"
        msg.payload = {
            "action": "update",
            "job": {
                "job_id": "j-005",
                "name": "remote job",
                "schedule": "0 3 * * *",
                "enabled": True,
                "leader_device": "dev-b",
            },
        }
        relay.handle_cron_sync(msg)
        assert relay.job_count == 1
        assert "j-005" in relay._jobs

    def test_handle_cron_sync_delete(self):
        relay = self._make_relay()
        job = CronJobInfo(
            job_id="j-006",
            name="deletable job",
            schedule="0 * * * *",
        )
        relay._jobs["j-006"] = job
        relay._leadership["j-006"] = True

        msg = MagicMock()
        msg.sender_id = "dev-b"
        msg.payload = {
            "action": "delete",
            "job_id": "j-006",
        }
        relay.handle_cron_sync(msg)
        assert relay.job_count == 0

    def test_get_my_jobs(self):
        relay = self._make_relay()
        relay._jobs["j-007"] = CronJobInfo(
            job_id="j-007", name="my job", schedule="0 * * * *",
            leader_device="dev-a",
        )
        relay._jobs["j-008"] = CronJobInfo(
            job_id="j-008", name="other job", schedule="0 * * * *",
            leader_device="dev-b",
        )
        my_jobs = relay.get_my_jobs()
        assert len(my_jobs) == 1
        assert my_jobs[0].job_id == "j-007"

    def test_job_counts(self):
        relay = self._make_relay()
        relay._jobs["j-009"] = CronJobInfo(
            job_id="j-009", name="job1", schedule="0 * * * *",
            leader_device="dev-a",
        )
        relay._jobs["j-010"] = CronJobInfo(
            job_id="j-010", name="job2", schedule="0 * * * *",
            leader_device="dev-b",
        )
        assert relay.job_count == 2
        assert relay.my_job_count == 1


# ========================================================================
# SkillInfo tests
# ========================================================================

class TestSkillInfo:
    def test_roundtrip(self):
        skill = SkillInfo(
            name="test-skill",
            category="devops",
            content="---\ncategory: devops\n---\nSkill content",
        )
        d = skill.to_dict()
        restored = SkillInfo.from_dict(d)
        assert restored.name == "test-skill"
        assert restored.category == "devops"


# ========================================================================
# FederationSkillSync tests
# ========================================================================

class TestFederationSkillSync:
    def _make_sync(self, tmp_path: Path):
        adapter = MagicMock()
        from unittest.mock import AsyncMock
        adapter.send = AsyncMock()
        sync = FederationSkillSync(
            device_id="dev-a",
            adapter=adapter,
            hermes_home=str(tmp_path),
        )
        return sync

    def test_init(self, tmp_path):
        sync = self._make_sync(tmp_path)
        assert sync.device_id == "dev-a"
        assert sync.skill_count == 0

    def test_load_local_skills(self, tmp_path):
        skills_dir = tmp_path / "skills" / "test-skill"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text(
            "---\ncategory: devops\n---\n# Test Skill\nContent here"
        )
        sync = self._make_sync(tmp_path)
        sync._load_local_skills()
        assert sync.skill_count == 1
        assert "test-skill" in sync._skills

    def test_apply_remote_skill(self, tmp_path):
        sync = self._make_sync(tmp_path)
        skill = SkillInfo(
            name="remote-skill",
            category="ops",
            content="---\ncategory: ops\n---\nRemote skill content",
        )
        sync._apply_remote_skill(skill)
        skill_file = tmp_path / "skills" / "remote-skill" / "SKILL.md"
        assert skill_file.exists()
        assert "Remote skill content" in skill_file.read_text()

    def test_handle_skill_sync_update(self, tmp_path):
        sync = self._make_sync(tmp_path)
        msg = MagicMock()
        msg.sender_id = "dev-b"
        msg.payload = {
            "action": "update",
            "skill": {
                "name": "remote-skill",
                "category": "test",
                "content": "Remote skill content",
                "version": 1,
            },
        }
        sync.handle_skill_sync(msg)
        assert "remote-skill" in sync._skills

    def test_handle_skill_sync_delete(self, tmp_path):
        sync = self._make_sync(tmp_path)
        skills_dir = tmp_path / "skills" / "to-delete"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("local content")
        sync._load_local_skills()
        assert sync.skill_count == 1

        msg = MagicMock()
        msg.sender_id = "dev-b"
        msg.payload = {
            "action": "delete",
            "name": "to-delete",
        }
        sync.handle_skill_sync(msg)
        assert sync.skill_count == 0
        assert not (tmp_path / "skills" / "to-delete").exists()

    def test_delete_local_skill_file(self, tmp_path):
        sync = self._make_sync(tmp_path)
        skills_dir = tmp_path / "skills" / "old-skill"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("old content")

        sync._delete_local_skill_file("old-skill")
        assert not (tmp_path / "skills" / "old-skill").exists()

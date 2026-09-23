from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from gateway.action_journal import ActionJournal, MutationEvent, MutationStatus, MutationType
from gateway.config import Platform
from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_one_shot_uses_bound_profile_and_private_telegram_run(tmp_path) -> None:
    key = uuid4()
    journal = ActionJournal(tmp_path / "actions.sqlite3", profile_key="profile-a")
    journal.append(
        MutationEvent(
            source_event_key=key,
            status=MutationStatus.SUCCEEDED,
            action_type=MutationType.CALENDAR,
            title="Google Calendar create event",
            description="Google Calendar create event succeeded.",
            provider="Google Calendar",
            operation="create_event",
            occurred_at=datetime.now(UTC),
            context="Google Calendar create event succeeded.",
            requires_receipt=True,
        )
    )
    runner = object.__new__(GatewayRunner)
    runner._becky_loops_config = SimpleNamespace(chat_id="-100123456789")
    runner._becky_action_journal = journal
    runner._becky_profile_name = "profile-a"
    runner._get_proxy_url = lambda: None
    captured = {}

    async def fake_run_agent(**kwargs):
        captured.update(kwargs)
        return {
            "turn_tool_events": [
                {
                    "name": "mcp__google_calendar__create_event",
                    "requested_name": "mcp__google_calendar__create_event",
                    "success": True,
                },
                {"name": "terminal", "requested_name": "terminal", "success": False},
            ]
        }

    runner._run_agent = fake_run_agent
    result = await runner._execute_becky_one_shot(
        title="Calendar",
        text="Add the appointment",
        idempotency_key=key,
        note_default="obsidian",
        policy_version="1",
    )

    assert result.disposition == "succeeded"
    assert captured["private_run"] is True
    source = captured["source"]
    assert source.platform is Platform.TELEGRAM
    assert source.chat_id == "-100123456789"
    assert source.profile == "profile-a"
